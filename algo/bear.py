import copy
import torch

from network.bear import Actor, Critic, VAE
from utils import mmd_loss_gaussian, mmd_loss_laplacian
from algo.algo import Algo


class BEAR(Algo):
    def __init__(self, summary_dir, env_name, seed, device, max_timesteps, eval_freq,
                 buffer_dir, num_qs, state_dim, action_dim,
                 max_action, batch_size=100, gamma=0.99, tau=0.005, lmbda=0.75,
                 delta_conf=0.1, mode='auto', num_samples=10,
                 mmd_sigma=10.0, lagrange_thresh=10.0, kernel_type='laplacian'):

        super().__init__(
            env_name, seed, buffer_dir, summary_dir, max_timesteps, eval_freq,
            batch_size, state_dim, action_dim, device, gamma, tau, lmbda)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.delta_conf = delta_conf
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples = 121#num_samples
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.kernel_type = kernel_type

        if self.mode == 'auto':
            self.log_lagrange = torch.randn((), requires_grad=True, device=device)
            self.lagrange_optimizer = torch.optim.Adam([self.log_lagrange, ], lr=1e-3)

        self.epoch = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(
                state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(axis=0)
        return action[ind].cpu().data.numpy().flatten()

    def train(self, iterations):
        for it in range(iterations):
            state, action, next_state, reward, not_done = self.store.sample()

            self.update_vae(state, action)

            next_state = torch.repeat_interleave(next_state, 10, 0)
            with torch.no_grad():
                next_action = self.actor_target(next_state)

            self.update_critic(state, action, next_state, next_action, reward, not_done)

            _, raw_sampled_actions = self.vae.decode_multiple(
                state, num_decode=self.num_samples)  # B x N x d
            actor_actions, raw_actor_actions = self.actor.sample_multiple(
                state, self.num_samples)

            if self.kernel_type == 'gaussian':
                mmd_loss = mmd_loss_gaussian(
                    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
            else:
                mmd_loss = mmd_loss_laplacian(
                    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

            curr_q1, curr_q2 = self.critic(
                state.repeat(self.num_samples, 1),
                actor_actions.view(-1, actor_actions.shape[-1]))
            curr_q1 = curr_q1.view(self.batch_size, self.num_samples).mean(axis=1)
            curr_q2 = curr_q2.view(self.batch_size, self.num_samples).mean(axis=1)
            min_curr_q = torch.min(curr_q1, curr_q2)

            if self.mode == 'auto':
                alpha = self.log_lagrange.exp()
            else:
                alpha = 100.0

            actor_loss = (
                - min_curr_q
                + alpha * mmd_loss
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.mode == 'auto':
                self.update_lagrange(min_curr_q, mmd_loss)

            self.update_targets()

        self.epoch = self.epoch + 1

    def update_lagrange(self, min_curr_q, mmd_loss):
        thresh = 0.05
        lagrange_loss = - (
            - min_curr_q.detach()
            + self.log_lagrange.exp() * (mmd_loss.detach() - thresh)
        ).mean()

        self.lagrange_optimizer.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optimizer.step()
        self.log_lagrange.data.clamp_(min=-5.0, max=self.lagrange_thresh)

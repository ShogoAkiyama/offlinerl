import copy
import torch
import torch.nn.functional as F

from network.bear import Actor, Critic, VAE
from utils import mmd_loss_gaussian, mmd_loss_laplacian
from algo.algo import Algo


class BRAC(Algo):
    def __init__(self, summary_dir, env_name, seed, device, max_timesteps, eval_freq,
                 buffer_dir, num_qs, state_dim, action_dim,
                 max_action, batch_size=100, gamma=0.99, tau=0.005, lmbda=0.75,
                 train_alpha=False, num_samples=10,
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

        self.train_alpha = train_alpha
        self.num_qs = num_qs
        self.num_samples = num_samples
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.kernel_type = kernel_type

        if self.train_alpha:
            self.log_lagrange = torch.randn((), requires_grad=True, device=device)
            self.lagrange_optimizer = torch.optim.Adam([self.log_lagrange, ], lr=1e-3)

        self.use_kl = True
        self.ite = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(
                state.reshape(1, -1)).repeat(10, 1).to(self.device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(axis=0)
        return action[ind].cpu().data.numpy().flatten()

    def train(self, iterations):
        for it in range(iterations):
            state, action, next_state, reward, not_done = self.store.sample()

            self.update_vae(state, action)

            self.update_critic(
                state, action, next_state,
                reward, not_done)

            self.update_actor(state)

            # if self.train_alpha:
            #     self.update_lagrange(min_curr_q, mmd_loss)

            self.update_targets()
            self.ite += 1

    def update_critic(self, state, action, next_state, reward, not_done):

        next_state_repeat = torch.repeat_interleave(next_state, 10, 0)
        with torch.no_grad():
            next_action_repeat = self.actor_target(next_state_repeat)

        _, raw_sampled_actions = self.vae.decode_multiple(
            next_state, num_decode=self.num_samples)  # B x N x d
        actor_actions, raw_actor_actions = self.actor.sample_multiple(
            next_state, self.num_samples)

        if self.use_kl:
            mmd_loss = self.kl_loss(raw_sampled_actions, state)
        else:
            if self.kernel_type == 'gaussian':
                mmd_loss = mmd_loss_gaussian(
                    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
            else:
                mmd_loss = mmd_loss_laplacian(
                    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

        if self.train_alpha:
            alpha = self.log_lagrange.exp()
        else:
            alpha = 0.1

        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(
                next_state_repeat, next_action_repeat)

            next_q = torch.min(next_q1, next_q2)
            next_q = next_q.reshape(self.batch_size, -1).max(axis=1)[0] - alpha*mmd_loss
            next_q = next_q.reshape(-1, 1)

            target_q = reward + not_done * self.gamma * next_q

        curr_q1, curr_q2 = self.critic(state, action)

        # >> norms
        norms = 0
        for param in self.critic.parameters():
            norms += torch.sum(torch.square(param))
        # >> norms

        critic_loss = (
            F.mse_loss(curr_q1, target_q)
            + F.mse_loss(curr_q2, target_q)
            + 0. * norms
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, state):
        _, raw_sampled_actions = self.vae.decode_multiple(
            state, num_decode=self.num_samples)  # B x N x d
        actor_actions, raw_actor_actions = self.actor.sample_multiple(
            state, self.num_samples)

        if self.use_kl:
            mmd_loss = self.kl_loss(raw_sampled_actions, state)
        else:
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

        if self.train_alpha:
            alpha = self.log_lagrange.exp()
        else:
            alpha = 0.1

        norms = 0
        for param in self.actor.parameters():
            norms += torch.sum(torch.square(param))

        actor_loss = (
            - min_curr_q
            + alpha * mmd_loss
            + 0. * norms
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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

    def kl_loss(self, samples1, state):
        """We just do likelihood, we make sure that the policy is close to the
           data in terms of the KL."""
        state_rep = state[:, None].repeat(1, self.num_samples, 1).view(-1, state.size(1))
        samples1_reshape = samples1.view(state_rep.shape[0], -1)
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(-1, self.num_samples)
        return (-samples1_log_prob).mean(axis=1)

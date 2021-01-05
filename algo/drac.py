import copy
import torch
import torch.nn.functional as F

from network.bcq import Actor, Critic, VAE
from algo.algo import Algo
from utils import mmd_loss_gaussian, mmd_loss_laplacian


class DRAC(Algo):
    def __init__(self, env_name, seed, buffer_dir, summary_dir, max_timesteps,
                 eval_freq, batch_size, state_dim, action_dim, max_action,
                 device, gamma=0.99, tau=0.005, lmbda=0.75, phi=0.05):

        super().__init__(
            env_name, seed, buffer_dir, summary_dir, max_timesteps, eval_freq,
            batch_size, state_dim, action_dim, device, gamma, tau, lmbda)

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

    def train(self, iterations):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = self.store.sample()

            self.update_vae(state, action)

            next_state = torch.repeat_interleave(next_state, 10, 0)
            with torch.no_grad():
                next_action = self.actor_target(next_state, self.vae.decode(next_state))

            self.update_critic(state, action, next_state, next_action, reward, not_done)

            self.update_actor(state)

            self.update_targets()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(
                state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def update_critic(self, state, action, next_state, next_action, reward, not_done):
        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(
                next_state, next_action)

            next_q = self.lmbda * torch.min(
                next_q1, next_q2) + (1. - self.lmbda) * torch.max(next_q1, next_q2)
            next_q = next_q.reshape(self.batch_size, -1).max(1)[0].reshape(-1, 1)

            target_q = reward + not_done * self.gamma * next_q

        curr_q1, curr_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)

        # >>> loss value_aug
        # curr_q1_aug, curr_q2_aug = self.critic(state_aug, action)
        # critic_loss_aug = F.mse_loss(curr_q1.detach(), curr_q1_aug) + F.mse_loss(curr_q2.detach(), curr_q2_aug)
        # >>> loss value_aug

        loss = critic_loss #+ 10.*critic_loss_aug

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, state):
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)

        # state_aug = torch.repeat_interleave(state, 100, 0)
        # noise = torch.normal(0, 1, size=state_aug.size()).to(self.device)
        # state_aug += noise
        # actions_aug = self.vae.decode(state_aug)
        # _, raw_sampled_actions = self.vae.decode_multiple(
        #     state, num_decode=100)  # B x N x d
        # perturbed_actions_aug = self.actor(state_aug, actions_aug)
        # perturbed_actions_aug = perturbed_actions_aug.view(self.batch_size, 100, -1)
        # mmd_loss = mmd_loss_gaussian(
        #     raw_sampled_actions, perturbed_actions_aug, sigma=20)

        actor_loss = (
            -self.critic.q1(state, perturbed_actions)
            # + 0.1 * mmd_loss.view(-1, 1)
        ).mean()
        # actor_loss += -self.critic.q2(state, perturbed_actions).mean()

        # >>> loss value_aug
        # sampled_actions = self.vae.decode(state_aug)
        # perturbed_actions = self.actor(state_aug, sampled_actions)
        # actor_loss_aug = -self.critic.q1(state_aug, perturbed_actions).mean()
        # >>> loss value_aug

        loss = actor_loss #+ 10. * actor_loss_aug

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

import copy
import torch

from network.bcq import Actor, Critic, VAE
from algo.algo import Algo


class BCQ(Algo):
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

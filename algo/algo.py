import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from storage import ReplayBuffer


class Algo(object):
    def __init__(self, env_name, seed, buffer_dir, summary_dir, max_timesteps, eval_freq,
                 batch_size, state_dim, action_dim, device,
                 gamma, tau, lmbda):

        self.env_name = env_name
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps
        self.eval_freq = eval_freq

        self.gamma = gamma
        self.tau = tau
        self.lmbda = lmbda

        self.store = ReplayBuffer(batch_size, state_dim, action_dim, device)
        self.store.load(buffer_dir)

        self.training_iters = 0
        self.writer = SummaryWriter(log_dir=summary_dir)

    def run(self):
        while self.training_iters < self.max_timesteps:
            self.train(iterations=int(self.eval_freq))

            self.eval_policy(self.env_name, self.seed)

            self.training_iters += self.eval_freq
            print(f"Training iterations: {self.training_iters}")

    def eval_policy(self, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        avg_q = 0
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action, q = self.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
                avg_q += q

        avg_reward /= eval_episodes
        avg_q /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")

        self.writer.add_scalar(
            'eval/return', avg_reward, self.training_iters)
        self.writer.add_scalar(
            'eval/Estimate Q', avg_q, self.training_iters)

    def update_vae(self, state, action):
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        # >> norms
        norms = 0
        for param in self.vae.parameters():
            norms += torch.sum(torch.square(param))
        # >> norms

        loss = (
            vae_loss
            # + 1e-4 * norms
        )

        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

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

        # # >> norms
        # norms = 0
        # for param in self.critic.parameters():
        #     norms += torch.sum(torch.square(param))
        # # >> norms

        loss = (
            critic_loss
            # + 1e-5 * norms
        )

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, state):
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)

        actor_loss = -self.critic.q1(state, perturbed_actions).mean()

        # # >> norms
        # norms = 0
        # for param in self.critic.parameters():
        #     norms += torch.sum(torch.square(param))
        # # >> norms

        loss = (
            actor_loss
            # + 1e-5 * norms
        )

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update_targets(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

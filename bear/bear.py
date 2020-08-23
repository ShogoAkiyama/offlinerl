import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from bear.network import Actor, Critic, VAE
from storage import ReplayBuffer
from utils import mmd_loss_gaussian, mmd_loss_laplacian


class BEAR(object):
    def __init__(self, summary_dir, env_name, device, buffer_dir, num_qs, state_dim, action_dim,
                 max_action, batch_size=100, gamma=0.99, tau=0.005,
                 delta_conf=0.1, mode='auto', num_samples=10,
                 mmd_sigma=10.0, lagrange_thresh=10.0, kernel_type='laplacian'):

        self.env_name = env_name
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(num_qs, state_dim, action_dim).to(device)
        self.critic_target = Critic(num_qs, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.delta_conf = delta_conf
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples = num_samples
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.kernel_type = kernel_type

        if self.mode == 'auto':
            self.log_lagrange = torch.randn((), requires_grad=True, device=device)
            self.lagrange_optimizer = torch.optim.Adam([self.log_lagrange, ], lr=1e-3)

        self.epoch = 0

        self.store = ReplayBuffer(batch_size, state_dim, action_dim, device)
        self.store.load(buffer_dir)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.training_iters = 0
        self.writer = SummaryWriter(log_dir=summary_dir)

    def run(self, max_timesteps, eval_freq):
        while self.training_iters < max_timesteps:
            self.train(iterations=int(eval_freq))
            self.eval_policy()
            self.training_iters += eval_freq
            print("Training iterations: " + str(self.training_iters))

    def eval_policy(self, eval_episodes=10):
        eval_env = gym.make(self.env_name)

        avg_reward = 0.
        all_rewards = []
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
            all_rewards.append(avg_reward)

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        print("---------------------------------------")

        self.writer.add_scalar(
            'eval/return', avg_reward, self.training_iters)

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
            state, action, next_state, reward, done = self.store.sample()

            # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * kl_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            with torch.no_grad():
                next_state = torch.repeat_interleave(next_state, 10, 0)

                next_q1, next_q2 = self.critic_target(
                    next_state, self.actor_target(next_state))

                next_q = 0.75 * torch.min(next_q1, next_q2) + 0.25 * torch.max(next_q1, next_q2)
                next_q = next_q.view(self.batch_size, -1).max(1)[0].view(-1, 1)
                target_q = reward + done * self.gamma * next_q

            curr_q1, curr_q2 = self.critic(state, action, with_var=False)

            critic_loss = (
                F.mse_loss(curr_q1, target_q)
                + F.mse_loss(curr_q2, target_q)
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Action Training
            _, raw_sampled_actions = self.vae.decode_multiple(
                state, num_decode=self.num_samples)  # B x N x d
            actor_actions, raw_actor_actions = self.actor.sample_multiple(
                state, self.num_samples)

            # MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
            if self.kernel_type == 'gaussian':
                mmd_loss = mmd_loss_gaussian(
                    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
            else:
                mmd_loss = mmd_loss_laplacian(
                    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

            critic_qs = self.critic.q_all(
                state.repeat(self.num_samples, 1),
                actor_actions.permute(1, 0, 2).contiguous().view(-1, self.action_dim)
            )
            critic_qs = critic_qs.view(
                self.num_qs, self.num_samples, self.batch_size, 1).mean(1)

            min_critic_qs = critic_qs.min(axis=0)[0]

            if self.epoch >= 20:
                if self.mode == 'auto':
                    actor_loss = (
                        - min_critic_qs
                        + self.log_lagrange.exp() * mmd_loss
                    ).mean()
                else:
                    actor_loss = (
                        - min_critic_qs
                        + 100.0 * mmd_loss
                    ).mean()
            else:
                if self.mode == 'auto':
                    actor_loss = (self.log_lagrange.exp() * mmd_loss).mean()
                else:
                    actor_loss = 100.0 * mmd_loss.mean()

            self.actor_optimizer.zero_grad()
            if self.mode == 'auto':
                actor_loss.backward(retain_graph=True)
            else:
                actor_loss.backward()
            # torch.nn.utils.clip_grad_norm(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Threshold for the lagrange multiplier
            thresh = 0.05
            if self.mode == 'auto':
                lagrange_loss = - (
                    - min_critic_qs.detach()
                    + self.log_lagrange.exp() * (mmd_loss.detach() - thresh)
                ).mean()

                self.lagrange_optimizer.zero_grad()
                lagrange_loss.backward()
                self.lagrange_optimizer.step()
                self.log_lagrange.data.clamp_(min=-5.0, max=self.lagrange_thresh)

                # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.epoch = self.epoch + 1

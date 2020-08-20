import numpy as np
import copy
import gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from storage import ReplayBuffer
from bcq.network import Actor, Critic, VAE


class BCQ(object):
	def __init__(self, summary_dir, env_name, seed, buffer_dir, max_timesteps, eval_freq,
				 batch_size, state_dim, action_dim, max_action, device,
				 discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):

		self.env_name = env_name
		self.seed = seed
		self.max_timesteps = max_timesteps
		self.eval_freq = eval_freq
		self.batch_size = batch_size

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		latent_dim = action_dim * 2
		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device

		self.store = ReplayBuffer(state_dim, action_dim, device)
		self.store.load(buffer_dir)

		self.training_iters = 0
		self.writer = SummaryWriter(log_dir=summary_dir)

	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(
				state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()

	def run(self):
		while self.training_iters < self.max_timesteps:
			self.train(self.store, iterations=int(self.eval_freq))

			self.eval_policy(self.env_name, self.seed)

			self.training_iters += self.eval_freq
			print(f"Training iterations: {self.training_iters}")

	def eval_policy(self, env_name, seed, eval_episodes=10):
		eval_env = gym.make(env_name)
		eval_env.seed(seed + 100)

		avg_reward = 0.
		for _ in range(eval_episodes):
			state, done = eval_env.reset(), False
			while not done:
				action = self.select_action(np.array(state))
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward

		avg_reward /= eval_episodes

		print("---------------------------------------")
		print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
		print("---------------------------------------")

		self.writer.add_scalar(
			'eval/return', avg_reward, self.training_iters)
		return avg_reward

	def train(self, replay_buffer, iterations):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * kl_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				next_q1, next_q2 = self.critic_target(
					next_state,
					self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				next_q = self.lmbda * torch.min(
					next_q1, next_q2) + (1. - self.lmbda) * torch.max(next_q1, next_q2)
				# Take max over each action sampled from the VAE
				next_q = next_q.reshape(self.batch_size, -1).max(1)[0].reshape(-1, 1)

				target_q = reward + not_done * self.discount * next_q

			curr_q1, curr_q2 = self.critic(state, action)
			critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

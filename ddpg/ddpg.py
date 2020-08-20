import os
import copy
import torch
import torch.nn.functional as F

from ddpg.network import Actor, Critic


class DDPG:
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.discount = discount
		self.tau = tau
		self.device = device

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, state, action, next_state, reward, not_done):
		# Compute the target Q value
		next_q = self.critic_target(next_state, self.actor_target(next_state))
		target_q = reward + (not_done * self.discount * next_q).detach()

		# Get current Q estimate
		curr_q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(curr_q, target_q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, dirname):
		if not os.path.exists(os.path.join(dirname)):
			os.makedirs(dirname)
		critic_dir = os.path.join(dirname, "critic")
		actor_dir = os.path.join(dirname, "actor")
		torch.save(self.critic.state_dict(), critic_dir)
		torch.save(self.actor.state_dict(), actor_dir)

	def load(self, filename):
		self.critic.load_state_dict(torch.load(
			os.path.join(filename, "critic")))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(
			os.path.join(filename, "actor")))
		self.actor_target = copy.deepcopy(self.actor)

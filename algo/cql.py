import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
import math
import torch.nn as nn

from network.cql import VectorEncoder, NormalPolicy, VectorEncoderWithAction, ContinuousQFunction, EnsembleContinuousQFunction
from utils import mmd_loss_gaussian, mmd_loss_laplacian
from algo.algo import Algo


def create_continuous_q_function(observation_shape,
                                 action_size,
                                 n_ensembles=2,
                                 use_batch_norm=False,
                                 bootstrap=False,
                                 share_encoder=False):

    q_funcs = []
    for _ in range(n_ensembles):
        if not share_encoder:
            encoder = VectorEncoderWithAction(
                observation_shape, action_size,
                use_batch_norm=use_batch_norm)
        q_func = ContinuousQFunction(encoder)

        q_funcs.append(q_func)
    return EnsembleContinuousQFunction(q_funcs, bootstrap)


def create_normal_policy(observation_shape, action_size):
    encoder = VectorEncoder(observation_shape)
    return NormalPolicy(encoder, action_size)


class CQL(Algo):
    def __init__(self, summary_dir, env_name, seed, device, max_timesteps, eval_freq,
                 buffer_dir, num_qs, state_dim, action_dim,
                 max_action, batch_size=100, gamma=0.99, tau=0.005, lmbda=0.75,
                 train_alpha=False, num_samples=10,
                 mmd_sigma=10.0, lagrange_thresh=10.0, kernel_type='laplacian'):

        super().__init__(
            env_name, seed, buffer_dir, summary_dir, max_timesteps, eval_freq,
            batch_size, state_dim, action_dim, device, gamma, tau, lmbda)

        bootstrap = True
        self.q_func = create_continuous_q_function(
            state_dim, action_dim, n_ensembles=2, bootstrap=bootstrap).to(device)
        self.policy = create_normal_policy(state_dim, action_dim).to(device)
        self.targ_q_func = copy.deepcopy(self.q_func)
        self.targ_policy = copy.deepcopy(self.policy)

        self.critic_optim = Adam(self.q_func.parameters(), lr=3e-4, eps=1e-8)
        self.actor_optim = Adam(self.policy.parameters(), lr=3e-5, eps=1e-8)

        # build alpha
        initial_val = math.log(5.0)
        data = torch.full((1, 1), initial_val, device=self.device)
        self.log_alpha = nn.Parameter(data)
        self.alpha_optim = Adam([self.log_alpha], lr=3e-4)

        # build temperature
        initial_val = math.log(1.0)
        data = torch.full((1, 1), initial_val, device=self.device)
        self.log_temp = nn.Parameter(data)
        self.temp_optim = Adam([self.log_temp], 3e-5)

        self.action_dim = action_dim
        self.n_action_samples = 10
        self.n_critics = 2
        self.alpha_threshold = 10.0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            # action = self.policy(state, deterministic=True)
            action = self.policy.best_action(state)
        return action.cpu().data.numpy().flatten()

    def train(self, iterations):
        for it in range(iterations):
            state, action, next_state, reward, not_done = self.store.sample()

            critic_loss = self.update_critic(state, action, next_state, reward, not_done)
            actor_loss = self.update_actor(state)
            temp_loss, temp = self.update_temp(state)
            alpha_loss, alpha = self.update_alpha(state, action)

            self.update_critic_target()
            self.update_actor_target()

        print("critic_loss:", critic_loss)
        print("actor_loss:",  actor_loss)
        print("temp_loss:",   temp_loss)
        print("temp:",   temp)
        print("alpha_loss:",  alpha_loss)
        print("alpha:",  alpha)

    def update_critic(self, state, action, next_state, reward, not_done):
        # calc targetQ
        with torch.no_grad():
            next_action, log_prob = self.policy.sample(next_state, with_log_prob=True)
            entropy = self.log_temp.exp() * log_prob
            next_q = self.targ_q_func.compute_target(next_state, next_action) - entropy
        next_q *= not_done

        # calc critic loss
        loss = self.q_func.compute_error(state, action, reward, next_q, 0.99)
        conservative_loss = self._compute_conservative_loss(state, action)
        loss += conservative_loss

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.item()

    def update_actor(self, state):
        action, log_prob = self.policy(state, with_log_prob=True)
        entropy = self.log_temp.exp() * log_prob
        q_t = self.q_func(state, action, 'min')
        loss = (entropy - q_t).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.item()

    def update_temp(self, state):
        with torch.no_grad():
            _, log_prob = self.policy.sample(state, with_log_prob=True)
            targ_temp = log_prob - self.action_dim

        loss = -(self.log_temp.exp() * targ_temp).mean()

        self.temp_optim.zero_grad()
        loss.backward()
        self.temp_optim.step()

        return loss.item(), self.log_temp.exp().item()

    def update_alpha(self, state, action):
        loss = -self._compute_conservative_loss(state, action)

        self.alpha_optim.zero_grad()
        loss.backward()
        self.alpha_optim.step()

        return loss.item(), self.log_alpha.exp().item()

    def _compute_conservative_loss(self, obs_t, act_t):

        with torch.no_grad():
            policy_actions, n_log_probs = self.policy.sample_n(
                obs_t, self.n_action_samples, with_log_prob=True)

        repeated_obs_t = obs_t.expand(self.n_action_samples, *obs_t.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs_t = repeated_obs_t.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs_t = transposed_obs_t.reshape(-1, *obs_t.shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_dim)

        # estimate action-values for policy actions
        policy_values = self.q_func(flat_obs_t, flat_policy_acts, 'none')
        policy_values = policy_values.view(self.n_critics, obs_t.shape[0],
                                           self.n_action_samples, 1)
        log_probs = n_log_probs.view(1, -1, self.n_action_samples, 1)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        random_actions = torch.zeros_like(flat_policy_acts).uniform_(-1.0, 1.0)
        random_values = self.q_func(flat_obs_t, random_actions, 'none')
        random_values = random_values.view(self.n_critics, obs_t.shape[0],
                                           self.n_action_samples, 1)

        # get maximum value to avoid overflow
        base = torch.max(policy_values.max(), random_values.max()).detach()

        # compute logsumexp
        policy_meanexp = (policy_values - base - log_probs).exp().mean(dim=2)
        random_meanexp = (random_values - base).exp().mean(dim=2) / 0.5
        # small constant value seems to be necessary to avoid nan
        logsumexp = (0.5 * random_meanexp + 0.5 * policy_meanexp + 1e-10).log()
        logsumexp += base

        # estimate action-values for data actions
        data_values = self.q_func(obs_t, act_t, 'none')

        element_wise_loss = logsumexp - data_values - self.alpha_threshold

        # this clipping seems to stabilize training
        clipped_alpha = self.log_alpha.clamp(-10.0, 2.0).exp()

        return (clipped_alpha * element_wise_loss).sum(dim=0).mean()

    def update_critic_target(self):
        soft_sync(self.targ_q_func, self.q_func, self.tau)

    def update_actor_target(self):
        soft_sync(self.targ_policy, self.policy, self.tau)

    # def update_targets(self):
    #     for param, target_param in zip(self.q_func.parameters(), self.targ_q_func.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #
    #     for param, target_param in zip(self.policy.parameters(), self.targ_policy.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def soft_sync(targ_model, model, tau):
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)

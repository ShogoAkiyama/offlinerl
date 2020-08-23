import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5*torch.log(one_plus_x / one_minus_x)


class Actor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.randn(
            std_a.size(), device=state.device)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a.unsqueeze(1) + \
            std_a.unsqueeze(1) * torch.randn(
            (std_a.size(0), num_sample, std_a.size(1)),
            device=state.device).clamp(-0.5, 0.5)

        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action ** 2).clamp(min=1e-6).log().sum(-1)
        return log_pis


class Critic(nn.Module):
    """ Critic which does have a network of 4 Q-functions"""

    def __init__(self, num_qs, state_dim, action_dim):
        super(Critic, self).__init__()

        self.num_qs = num_qs

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action, with_var=False):

        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0), ], 0)   # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def q_all(self, state, action, with_var=False):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0), ], 0)
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""

    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn(std.size(), device=state.device)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn(
                (state.shape[0], self.latent_dim),
                device=state.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            z = torch.randn(
                (state.shape[0], num_decode, self.latent_dim),
                device=state.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)), self.d3(a)

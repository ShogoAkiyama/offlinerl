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
        self.linear1 = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = self.linear1(state)
        mean = self.mean(a)
        log_std = self.log_std(a)

        std = torch.exp(log_std)
        z = mean + std * torch.randn(
            std.size(), device=state.device)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = self.linear1(state)
        mean = self.mean(a)
        log_std = self.log_std(a)

        std = torch.exp(log_std)
        z = mean.unsqueeze(1) + \
            std.unsqueeze(1) * torch.randn(
            (std.size(0), num_sample, std.size(1)),
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

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        q1 = self.linear1(x)
        q2 = self.linear2(x)

        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], 1)
        q1 = self.linear1(x)
        return q1


class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""

    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU()
        )

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Linear(750, action_dim)
        )

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        z = self.encoder(x)

        mean = self.mean(z)
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

        y = torch.cat([state, z], 1)
        y = self.decoder(y)
        return self.max_action * torch.tanh(y)

    def decode_multiple(self, state, z=None, num_decode=10):
        if z is None:
            z = torch.randn(
                (state.shape[0], num_decode, self.latent_dim),
                device=state.device).clamp(-0.5, 0.5)

        # x = torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], axis=2)
        x = torch.cat([state.unsqueeze(1).repeat(1, num_decode, 1), z], axis=2)
        x = self.decoder(x)
        return self.max_action * torch.tanh(x), x

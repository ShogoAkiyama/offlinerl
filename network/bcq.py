import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim)
        )

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.linear(x)
        x = self.phi * self.max_action * torch.tanh(x)
        return (x + action).clamp(-self.max_action, self.max_action)


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
        q1 = self.linear2(x)
        return q1


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
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
        self.device = device

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        z = self.encoder(x)

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(
                self.device).clamp(-0.5, 0.5)

        y = torch.cat([state, z], 1)
        y = self.decoder(y)
        return self.max_action * torch.tanh(y)

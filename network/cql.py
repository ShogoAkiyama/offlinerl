import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Categorical


def _squash_action(dist, raw_action):
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
    return squashed_action, log_prob


def _reduce(value, reduction_type):
    if reduction_type == 'mean':
        return value.mean()
    elif reduction_type == 'sum':
        return value.sum()
    elif reduction_type == 'none':
        return value.view(-1, 1)
    raise ValueError('invalid reduction type.')


def _reduce_ensemble(y, reduction='min', dim=0, lam=0.75):
    if reduction == 'min':
        return y.min(dim=dim).values
    elif reduction == 'max':
        return y.max(dim=dim).values
    elif reduction == 'mean':
        return y.mean(dim=dim)
    elif reduction == 'none':
        return y
    elif reduction == 'mix':
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    else:
        raise ValueError


class NormalPolicy(nn.Module):
    def __init__(self, encoder, action_size):
        super().__init__()
        self.action_size = action_size
        self.encoder = encoder
        self.mu = nn.Linear(encoder.feature_size, action_size)
        self.logstd = nn.Linear(encoder.feature_size, action_size)

    def dist(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x, deterministic=False, with_log_prob=False):
        if deterministic:
            # to avoid errors at ONNX export because broadcast_tensors in
            # Normal distribution is not supported by ONNX
            action = self.mu(self.encoder(x))
        else:
            dist = self.dist(x)
            action = dist.rsample()

        if with_log_prob:
            return _squash_action(dist, action)

        return torch.tanh(action)

    def sample(self, x, with_log_prob=False):
        return self.forward(x, with_log_prob=with_log_prob)

    def sample_n(self, x, n, with_log_prob=False):
        dist = self.dist(x)

        action = dist.rsample((n, ))

        squashed_action_T, log_prob_T = _squash_action(dist, action)

        # (n, batch, action) -> (batch, n, action)
        squashed_action = squashed_action_T.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob_T.transpose(0, 1)

        if with_log_prob:
            return squashed_action, log_prob

        return squashed_action

    def best_action(self, x):
        return self.forward(x, deterministic=True, with_log_prob=False)


class ContinuousQFunction(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.action_size = encoder.action_size
        self.fc = nn.Linear(encoder.feature_size, 1)

    def forward(self, x, action):
        h = self.encoder(x, action)
        return self.fc(h)

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        q_t = self.forward(obs_t, act_t)
        y = rew_tp1 + gamma * q_tp1
        loss = F.mse_loss(q_t, y, reduction='none')
        return _reduce(loss, reduction)

    def compute_target(self, x, action):
        return self.forward(x, action)


class VectorEncoder(nn.Module):
    def __init__(self,
                 observation_shape,
                 hidden_units=None,
                 use_batch_norm=False,
                 activation=torch.relu):
        super().__init__()
        self.observation_shape = observation_shape

        if hidden_units is None:
            hidden_units = [256, 256]

        self.use_batch_norm = use_batch_norm
        self.feature_size = hidden_units[-1]
        self.activation = activation

        in_units = [observation_shape] + hidden_units[:-1]
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for in_unit, out_unit in zip(in_units, hidden_units):
            self.fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(out_unit))

    def forward(self, x):
        h = x
        for i in range(len(self.fcs)):
            h = self.activation(self.fcs[i](h))
            if self.use_batch_norm:
                h = self.bns[i](h)
        return h


class VectorEncoderWithAction(VectorEncoder):
    def __init__(self,
                 observation_shape,
                 action_size,
                 hidden_units=None,
                 use_batch_norm=False,
                 activation=torch.relu):
        self.action_size = action_size
        concat_shape = observation_shape + action_size
        super().__init__(concat_shape, hidden_units, use_batch_norm,
                         activation)
        self.observation_shape = observation_shape

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        return super().forward(x)



class EnsembleQFunction(nn.Module):
    def __init__(self, q_funcs, bootstrap=False):
        super().__init__()
        self.action_size = q_funcs[0].action_size
        self.q_funcs = nn.ModuleList(q_funcs)
        self.bootstrap = bootstrap and len(q_funcs) > 1

    def compute_error(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        td_sum = 0.0
        for i, q_func in enumerate(self.q_funcs):
            loss = q_func.compute_error(obs_t,
                                        act_t,
                                        rew_tp1,
                                        q_tp1,
                                        gamma,
                                        reduction='none')

            if self.bootstrap:
                mask = torch.randint(0, 2, loss.shape, device=obs_t.device)
                loss *= mask.float()

            td_sum += loss.mean()

        return td_sum

    def compute_target(self, x, action, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            target = q_func.compute_target(x, action)
            values.append(target.view(1, x.shape[0], -1))

        values = torch.cat(values, dim=0)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(self, x, action, reduction='mean'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)


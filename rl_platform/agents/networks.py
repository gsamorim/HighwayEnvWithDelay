"""Neural network building blocks for the SAC agent.

Kept generic (plain MLPs over flat vector observations) so the same modules
can be reused by future off-policy agents (TD3, etc.) added to the registry.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(sizes: list[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy (the standard SAC actor).

    Outputs actions in [-1, 1]^act_dim; the environment wrapper/agent is
    responsible for rescaling to the true action bounds if they differ.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] = [256, 256]):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pre_tanh = mu
        else:
            dist = Normal(mu, std)
            pre_tanh = dist.rsample()

        action = torch.tanh(pre_tanh)

        logp = None
        if with_logprob:
            dist = Normal(mu, std)
            logp = dist.log_prob(pre_tanh).sum(axis=-1)
            # tanh squashing correction (SAC paper, appendix C)
            logp -= (2 * (torch.log(torch.tensor(2.0, device=obs.device)) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(axis=-1)

        return action, logp


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] = [256, 256]):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Two independent Q networks (clipped double-Q trick used by SAC/TD3)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] = [256, 256]):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.q1(obs, act), self.q2(obs, act)

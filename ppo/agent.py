"""Actor-critic networks for PPO on continuous and discrete action spaces."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str, out_std: float) -> nn.Sequential:
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), act()]
        last = h
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    """Separate actor and critic MLPs.

    Continuous actions use a diagonal Gaussian with a state-independent log-std
    (the standard PPO parameterisation for MuJoCo-style control). Discrete actions
    use a categorical head.
    """

    def __init__(self, obs_dim: int, action_space, hidden: tuple[int, ...] = (64, 64), activation: str = "tanh"):
        super().__init__()
        self.discrete = not hasattr(action_space, "high")
        act_dim = int(action_space.n) if self.discrete else int(np.prod(action_space.shape))
        self.critic = mlp(obs_dim, hidden, 1, activation, out_std=1.0)
        self.actor = mlp(obs_dim, hidden, act_dim, activation, out_std=0.01)
        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def distribution(self, x: torch.Tensor):
        out = self.actor(x)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.expand_as(out).exp())

    def action_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        dist = self.distribution(x)
        if action is None:
            action = dist.sample()
        if self.discrete:
            logp, ent = dist.log_prob(action), dist.entropy()
        else:
            logp, ent = dist.log_prob(action).sum(-1), dist.entropy().sum(-1)
        return action, logp, ent, self.critic(x)

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        out = self.actor(x)
        return out.argmax(-1) if self.discrete else out

"""Network definitions for actor/critic models."""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Create a small 2-layer MLP with ReLU activations."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class Actor(nn.Module):
    """Actor network mapping observations to bounded continuous actions.

    Input shape:  (B, obs_dim) or (obs_dim,)
    Output shape: (B, act_dim) or (act_dim,)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.net = _build_mlp(obs_dim, hidden_dim, act_dim)
        self.out_act = nn.Tanh()  # bounds actions to [-1, 1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute policy action for a batch of observations."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs last dim must be {self.obs_dim}, got {obs.shape[-1]}")
        return self.out_act(self.net(obs))


class Critic(nn.Module):
    """Critic network estimating scalar Q(s, a).

    Input shapes:
    - obs: (B, obs_dim) or (obs_dim,)
    - act: (B, act_dim) or (act_dim,)
    Output shape:
    - q:   (B, 1) or (1,)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.net = _build_mlp(obs_dim + act_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Estimate Q-values for observation-action pairs."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs last dim must be {self.obs_dim}, got {obs.shape[-1]}")
        if act.shape[-1] != self.act_dim:
            raise ValueError(f"act last dim must be {self.act_dim}, got {act.shape[-1]}")
        x = torch.cat([obs, act], dim=-1)  # (B, obs_dim + act_dim)
        return self.net(x)

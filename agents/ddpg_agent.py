"""DDPG agent implementation for continuous-control tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agents.networks import Actor, Critic
from agents.replay_buffer import TransitionBatch


@dataclass(slots=True)
class DDPGHyperParams:
    """Hyperparameters used by :class:`DDPGAgent`."""

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    hidden_dim: int = 256
    policy_noise_std: float = 0.1


class DDPGAgent:
    """Minimal DDPG agent with target networks and checkpoint hooks."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        device: str = "cpu",
        hparams: DDPGHyperParams | None = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.hparams = hparams or DDPGHyperParams()
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        self.actor = Actor(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)
        self.critic = Critic(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.critic_lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select an action from the current policy.

        Args:
            obs: Observation with shape ``(obs_dim,)``.
            add_noise: Whether to add Gaussian exploration noise.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, self.obs_dim)
        act_t = self.actor(obs_t)  # (1, act_dim)
        action = act_t.cpu().numpy().reshape(self.act_dim)

        if add_noise:
            noise = np.random.normal(loc=0.0, scale=self.hparams.policy_noise_std, size=self.act_dim)
            action = action + noise.astype(np.float32)

        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def update_step(self, batch: TransitionBatch) -> dict[str, float]:
        """Run one DDPG update step using a sampled replay batch."""
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)  # (B, obs_dim)
        acts = torch.as_tensor(batch.acts, dtype=torch.float32, device=self.device)  # (B, act_dim)
        rews = torch.as_tensor(batch.rews, dtype=torch.float32, device=self.device)  # (B, 1)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)  # (B, obs_dim)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)  # (B, 1)

        with torch.no_grad():
            next_acts = self.actor_target(next_obs)  # (B, act_dim)
            target_q = self.critic_target(next_obs, next_acts)  # (B, 1)
            y = rews + (1.0 - dones) * self.hparams.gamma * target_q  # (B, 1)

        current_q = self.critic(obs, acts)  # (B, 1)
        critic_loss = F.mse_loss(current_q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        pred_actions = self.actor(obs)  # (B, act_dim)
        actor_loss = -self.critic(obs, pred_actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()

        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

    @torch.no_grad()
    def soft_update(self) -> None:
        """Soft-update target networks using ``tau``."""
        tau = self.hparams.tau
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return serializable checkpoint payload."""
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hparams": asdict(self.hparams),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Load model/optimizer state from a checkpoint payload."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])

    def save_checkpoint(self, path: str | Path) -> None:
        """Save checkpoint payload to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), out_path)

    def load_checkpoint(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        """Load checkpoint payload from disk."""
        in_path = Path(path)
        payload = torch.load(in_path, map_location=map_location or self.device)
        self.load_checkpoint_state(payload)


__all__ = ["DDPGAgent", "DDPGHyperParams"]

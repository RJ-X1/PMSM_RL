"""Vanilla TD3 agent implementation for continuous-control tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agents.networks import Actor, TwinCritic
from agents.replay_buffer import TransitionBatch


@dataclass(slots=True)
class TD3HyperParams:
    """Hyperparameters used by :class:`TD3Agent`."""

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    hidden_dim: int = 256
    exploration_noise: float = 0.10
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    policy_delay: int = 2


class TD3Agent:
    """Vanilla TD3 agent with target policy smoothing and delayed actor updates."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        device: str = "cpu",
        hparams: TD3HyperParams | None = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.hparams = hparams or TD3HyperParams()
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self._update_step_count = 0

        self.actor = Actor(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)
        self.critics = TwinCritic(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)
        self.critics_target = TwinCritic(self.obs_dim, self.act_dim, hidden_dim=self.hparams.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critics_target.load_state_dict(self.critics.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=self.hparams.critic_lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select an action from the current policy."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, self.obs_dim)
        action = self.actor(obs_t).cpu().numpy().reshape(self.act_dim)
        if add_noise:
            noise = np.random.normal(
                loc=0.0,
                scale=float(self.hparams.exploration_noise),
                size=self.act_dim,
            ).astype(np.float32)
            action = action + noise
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def update_step(self, batch: TransitionBatch) -> dict[str, float]:
        """Run one TD3 update step using a sampled replay batch."""
        self._update_step_count += 1

        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(batch.acts, dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(batch.rews, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(acts) * float(self.hparams.target_policy_noise)
            noise = noise.clamp(
                -float(self.hparams.target_noise_clip),
                float(self.hparams.target_noise_clip),
            )
            next_acts = self.actor_target(next_obs) + noise
            next_acts = next_acts.clamp(self.action_low, self.action_high)

            target_q1, target_q2 = self.critics_target(next_obs, next_acts)
            target_q = torch.min(target_q1, target_q2)
            y = rews + (1.0 - dones) * float(self.hparams.gamma) * target_q

        current_q1, current_q2 = self.critics(obs, acts)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss_value = 0.0
        actor_updated = 0.0
        if self._update_step_count % int(self.hparams.policy_delay) == 0:
            pred_actions = self.actor(obs)
            actor_loss = -self.critics.q1_forward(obs, pred_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.soft_update()
            actor_loss_value = float(actor_loss.item())
            actor_updated = 1.0

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": float(critic_loss.item()),
            "actor_updated": actor_updated,
        }

    @torch.no_grad()
    def soft_update(self) -> None:
        """Soft-update target networks using ``tau``."""
        tau = float(self.hparams.tau)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        for target_param, param in zip(self.critics_target.parameters(), self.critics.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return serializable checkpoint payload."""
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hparams": asdict(self.hparams),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": self.critics.state_dict(),
            "critics_target": self.critics_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_step_count": int(self._update_step_count),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Load model/optimizer state from a checkpoint payload."""
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critics.load_state_dict(state["critics"])
        self.critics_target.load_state_dict(state["critics_target"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self._update_step_count = int(state.get("update_step_count", 0))

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


__all__ = ["TD3Agent", "TD3HyperParams"]

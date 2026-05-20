"""Cascade position-speed-current baseline for gun servo tracking.

The new research direction uses RL as the position outer loop.  This baseline
keeps a classical position PID interface that emits the same normalized action
as the RL policy: a speed-command correction in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from baselines.pid_position_controller import PIDPositionController, PIDPositionControllerConfig
from baselines.pi_speed_controller import PISpeedController, PISpeedControllerConfig
from envs.obs_parser import resolve_signal_indices


@dataclass(slots=True)
class CascadeServoControllerConfig:
    """Baseline cascade gains and normalizing scales."""

    position: PIDPositionControllerConfig = field(default_factory=PIDPositionControllerConfig)
    speed: PISpeedControllerConfig = field(default_factory=PISpeedControllerConfig)
    theta_scale: float = float(np.deg2rad(30.0))
    omega_scale: float = float(np.deg2rad(90.0))
    dt: float = 0.002


class CascadeServoController:
    """Position PID outer loop compatible with the gun-servo env action API."""

    REQUIRED_SIGNALS = ("e_theta", "e_omega")

    def __init__(
        self,
        *,
        signal_names: Sequence[str],
        config: CascadeServoControllerConfig | None = None,
    ) -> None:
        self.signal_names = list(signal_names)
        self.config = config or CascadeServoControllerConfig()
        self.position_pid = PIDPositionController(self.config.position)
        self.speed_pi = PISpeedController(self.config.speed)
        self._required_indices = resolve_signal_indices(self.signal_names, self.REQUIRED_SIGNALS)

    def reset(self) -> None:
        self.position_pid.reset()
        self.speed_pi.reset()

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Return normalized speed correction action for GunServoPositionEnv."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            self.reset()
            return np.zeros(1, dtype=np.float32)
        e_theta = float(x[self._required_indices["e_theta"]]) * float(self.config.theta_scale)
        e_omega = float(x[self._required_indices["e_omega"]]) * float(self.config.omega_scale)
        delta_omega = self.position_pid.compute_delta_omega(
            e_theta=e_theta,
            e_omega=e_omega,
            dt=float(self.config.dt),
        )
        limit = max(float(self.config.position.max_delta_omega), 1e-9)
        return np.asarray([np.clip(delta_omega / limit, -1.0, 1.0)], dtype=np.float32)

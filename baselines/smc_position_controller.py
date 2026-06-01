"""Sliding-mode position outer-loop baseline for gun servo tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.obs_parser import resolve_signal_indices


@dataclass(slots=True)
class SMCPositionControllerConfig:
    """SMC gains and command limits for load-side position control."""

    lambda_smc: float = 6.0
    k_e: float = 4.0
    k_s: float = float(np.deg2rad(4.0))
    phi: float = 0.06
    k_d: float = 0.25
    max_delta_omega: float = float(np.deg2rad(30.0))


class SMCPositionController:
    """Position SMC controller that outputs a load-side speed correction."""

    def __init__(self, config: SMCPositionControllerConfig | None = None) -> None:
        self.config = config or SMCPositionControllerConfig()

    def reset(self) -> None:
        """Reset controller state.

        The saturated SMC law is memoryless, but the method keeps the same
        lifecycle contract as the PID and RL wrappers.
        """

    def compute_delta_omega(self, *, e_theta: float, e_omega: float) -> float:
        """Return the load-side speed correction in rad/s."""
        cfg = self.config
        sliding = float(e_omega) + float(cfg.lambda_smc) * float(e_theta)
        boundary = max(float(cfg.phi), 1e-9)
        sat_s = float(np.clip(sliding / boundary, -1.0, 1.0))
        raw = (
            float(cfg.k_e) * float(e_theta)
            + float(cfg.k_s) * sat_s
            + float(cfg.k_d) * float(e_omega)
        )
        limit = max(float(cfg.max_delta_omega), 1e-9)
        return float(np.clip(raw, -limit, limit))


@dataclass(slots=True)
class SMCPositionActionControllerConfig:
    """Observation scales for converting SMC speed correction to env action."""

    position: SMCPositionControllerConfig
    theta_scale: float = float(np.deg2rad(30.0))
    omega_scale: float = float(np.deg2rad(30.0))


class SMCPositionActionController:
    """SMC outer loop compatible with the gun-servo env action API."""

    REQUIRED_SIGNALS = ("e_theta", "e_omega")

    def __init__(
        self,
        *,
        signal_names: Sequence[str],
        config: SMCPositionActionControllerConfig,
    ) -> None:
        self.signal_names = list(signal_names)
        self.config = config
        self.position_smc = SMCPositionController(self.config.position)
        self._required_indices = resolve_signal_indices(self.signal_names, self.REQUIRED_SIGNALS)

    def reset(self) -> None:
        self.position_smc.reset()

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Return normalized speed-correction action for GunServoPositionEnv."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            self.reset()
            return np.zeros(1, dtype=np.float32)
        e_theta = float(x[self._required_indices["e_theta"]]) * float(self.config.theta_scale)
        e_omega = float(x[self._required_indices["e_omega"]]) * float(self.config.omega_scale)
        delta_omega = self.position_smc.compute_delta_omega(e_theta=e_theta, e_omega=e_omega)
        limit = max(float(self.config.position.max_delta_omega), 1e-9)
        return np.asarray([np.clip(delta_omega / limit, -1.0, 1.0)], dtype=np.float32)

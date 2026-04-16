"""PI baseline aligned with the live PMSM observation semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.obs_parser import resolve_signal_indices


@dataclass(slots=True)
class PIControllerConfig:
    """PI gains and angle scaling for dq current control."""

    kp_d: float = 0.5
    ki_d: float = 0.05
    kp_q: float = 0.5
    ki_q: float = 0.05
    dq_voltage_limit: float = 1.0
    action_limit: float = 1.0
    epsilon_scale: float = float(np.pi)


class PICurrentController:
    """dq-current PI controller with inverse dq->abc voltage projection."""

    REQUIRED_SIGNALS = ("i_d", "i_q", "ref_i_d", "ref_i_q", "epsilon")

    def __init__(
        self,
        action_dim: int,
        *,
        signal_names: Sequence[str] | None = None,
        config: PIControllerConfig | None = None,
    ) -> None:
        self.action_dim = int(action_dim)
        if self.action_dim != 3:
            raise ValueError(f"PI baseline expects 3D abc voltage actions, got action_dim={self.action_dim}")

        self.config = config or PIControllerConfig()
        self.signal_names = list(signal_names or [])
        self._required_indices = resolve_signal_indices(self.signal_names, self.REQUIRED_SIGNALS)
        self._integrator_d = 0.0
        self._integrator_q = 0.0

    def reset(self) -> None:
        """Reset PI integrators."""
        self._integrator_d = 0.0
        self._integrator_q = 0.0

    def _read_required_signal(self, obs: np.ndarray, signal_name: str) -> float:
        return float(obs[self._required_indices[signal_name]])

    def _inverse_park_to_abc(self, u_d: float, u_q: float, epsilon: float) -> np.ndarray:
        theta = float(epsilon) * float(self.config.epsilon_scale)
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))

        # Convert dq voltages to alpha-beta, then to three-phase abc voltages.
        u_alpha = cos_theta * float(u_d) - sin_theta * float(u_q)
        u_beta = sin_theta * float(u_d) + cos_theta * float(u_q)
        sqrt3_over_2 = float(np.sqrt(3.0) / 2.0)
        return np.asarray(
            [
                u_alpha,
                -0.5 * u_alpha + sqrt3_over_2 * u_beta,
                -0.5 * u_alpha - sqrt3_over_2 * u_beta,
            ],
            dtype=np.float32,
        )

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute normalized abc voltage actions from dq current tracking error."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError("Observation vector is empty")

        i_d = self._read_required_signal(x, "i_d")
        i_q = self._read_required_signal(x, "i_q")
        ref_i_d = self._read_required_signal(x, "ref_i_d")
        ref_i_q = self._read_required_signal(x, "ref_i_q")
        epsilon = self._read_required_signal(x, "epsilon")

        error_d = ref_i_d - i_d
        error_q = ref_i_q - i_q
        self._integrator_d += error_d
        self._integrator_q += error_q

        u_d = self.config.kp_d * error_d + self.config.ki_d * self._integrator_d
        u_q = self.config.kp_q * error_q + self.config.ki_q * self._integrator_q
        u_d = float(np.clip(u_d, -self.config.dq_voltage_limit, self.config.dq_voltage_limit))
        u_q = float(np.clip(u_q, -self.config.dq_voltage_limit, self.config.dq_voltage_limit))

        action = self._inverse_park_to_abc(u_d=u_d, u_q=u_q, epsilon=epsilon)
        return np.clip(action, -self.config.action_limit, self.config.action_limit).astype(np.float32, copy=False)

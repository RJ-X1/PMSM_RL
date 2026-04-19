"""PI current-control baseline for custom PMSM and legacy GEM environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.motor_model import PMSMMotorParams, clip_voltage_vector
from envs.obs_parser import resolve_signal_indices


@dataclass(slots=True)
class PIControllerConfig:
    """Tunable PI gains and limits for dq current control."""

    kp_d: float = 6.0
    ki_d: float = 800.0
    kp_q: float = 6.0
    ki_q: float = 800.0
    integrator_limit_d: float = 120.0
    integrator_limit_q: float = 120.0
    voltage_limit: float | None = None
    action_limit: float = 1.0
    anti_windup: str = "conditional_integration"
    use_resistance_compensation: bool = True
    use_decoupling: bool = True
    use_back_emf_compensation: bool = True
    epsilon_scale: float = float(np.pi)


class PICurrentController:
    """dq-current PI controller with custom-env and legacy GEM compatibility."""

    REQUIRED_SIGNALS_CUSTOM = ("i_d", "i_q", "ref_i_d", "ref_i_q", "omega_m")
    REQUIRED_SIGNALS_LEGACY = ("i_d", "i_q", "ref_i_d", "ref_i_q", "epsilon")

    def __init__(
        self,
        action_dim: int,
        *,
        signal_names: Sequence[str] | None = None,
        config: PIControllerConfig | None = None,
        motor_params: PMSMMotorParams | None = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.signal_names = list(signal_names or [])
        self.config = config or PIControllerConfig()
        self.motor_params = motor_params
        self._anti_windup_mode = str(self.config.anti_windup).strip().lower()
        if self._anti_windup_mode not in {"none", "clamp", "conditional", "conditional_integration"}:
            raise ValueError(f"Unsupported anti_windup mode: {self.config.anti_windup}")

        self.mode = self._resolve_mode()
        required_signals = (
            self.REQUIRED_SIGNALS_CUSTOM if self.mode == "custom_dq" else self.REQUIRED_SIGNALS_LEGACY
        )
        self._required_indices = resolve_signal_indices(self.signal_names, required_signals)
        self._integrator_d = 0.0
        self._integrator_q = 0.0

    def _resolve_mode(self) -> str:
        signal_set = set(str(name) for name in self.signal_names)
        if self.action_dim == 2 and all(name in signal_set for name in self.REQUIRED_SIGNALS_CUSTOM):
            return "custom_dq"
        if self.action_dim == 3 and all(name in signal_set for name in self.REQUIRED_SIGNALS_LEGACY):
            return "legacy_abc"

        if self.action_dim == 2:
            missing = [name for name in self.REQUIRED_SIGNALS_CUSTOM if name not in signal_set]
            raise ValueError(
                "Custom dq PI controller requires 2D action space and signals "
                f"{self.REQUIRED_SIGNALS_CUSTOM}; missing={missing}"
            )

        if self.action_dim == 3:
            missing = [name for name in self.REQUIRED_SIGNALS_LEGACY if name not in signal_set]
            raise ValueError(
                "Legacy GEM PI controller requires 3D action space and signals "
                f"{self.REQUIRED_SIGNALS_LEGACY}; missing={missing}"
            )

        raise ValueError(f"Unsupported PI action dimension: {self.action_dim}")

    def reset(self) -> None:
        """Reset PI integrator states."""
        self._integrator_d = 0.0
        self._integrator_q = 0.0

    def _read_required_signal(self, obs: np.ndarray, signal_name: str) -> float:
        return float(obs[self._required_indices[signal_name]])

    def _resolve_voltage_limit(self) -> float:
        if self.config.voltage_limit is not None:
            return float(self.config.voltage_limit)
        if self.mode == "custom_dq" and self.motor_params is not None:
            return float(self.motor_params.Umax)
        return 1.0

    def _candidate_integrators(self, *, error_d: float, error_q: float, dt: float) -> tuple[float, float]:
        candidate_d = self._integrator_d + float(self.config.ki_d) * float(error_d) * float(dt)
        candidate_q = self._integrator_q + float(self.config.ki_q) * float(error_q) * float(dt)
        candidate_d = float(
            np.clip(candidate_d, -float(self.config.integrator_limit_d), float(self.config.integrator_limit_d))
        )
        candidate_q = float(
            np.clip(candidate_q, -float(self.config.integrator_limit_q), float(self.config.integrator_limit_q))
        )
        return candidate_d, candidate_q

    def _apply_anti_windup(
        self,
        *,
        candidate_d: float,
        candidate_q: float,
        raw_u_dq: np.ndarray,
        saturated_u_dq: np.ndarray,
        error_d: float,
        error_q: float,
    ) -> None:
        if self._anti_windup_mode in {"none", "clamp"}:
            self._integrator_d = candidate_d
            self._integrator_q = candidate_q
            return

        if np.allclose(raw_u_dq, saturated_u_dq, rtol=1e-6, atol=1e-6):
            self._integrator_d = candidate_d
            self._integrator_q = candidate_q
            return

        correction = np.asarray(raw_u_dq, dtype=np.float64) - np.asarray(saturated_u_dq, dtype=np.float64)
        if float(correction[0]) * float(error_d) <= 0.0:
            self._integrator_d = candidate_d
        if float(correction[1]) * float(error_q) <= 0.0:
            self._integrator_q = candidate_q

    def _custom_feedforward(self, *, i_d: float, i_q: float, omega_m: float) -> np.ndarray:
        if self.motor_params is None:
            raise ValueError("Custom dq PI controller requires motor_params for feedforward terms")

        omega_e = float(self.motor_params.p) * float(omega_m)
        u_d_ff = 0.0
        u_q_ff = 0.0

        if bool(self.config.use_resistance_compensation):
            u_d_ff += float(self.motor_params.Rs) * float(i_d)
            u_q_ff += float(self.motor_params.Rs) * float(i_q)

        if bool(self.config.use_decoupling):
            u_d_ff -= omega_e * float(self.motor_params.Lq) * float(i_q)
            u_q_ff += omega_e * float(self.motor_params.Ld) * float(i_d)

        if bool(self.config.use_back_emf_compensation):
            u_q_ff += omega_e * float(self.motor_params.psi_f)

        return np.asarray([u_d_ff, u_q_ff], dtype=np.float64)

    def _compose_dq_voltage(
        self,
        *,
        error_d: float,
        error_q: float,
        dt: float,
        feedforward_dq: np.ndarray,
    ) -> np.ndarray:
        candidate_d, candidate_q = self._candidate_integrators(error_d=error_d, error_q=error_q, dt=dt)
        raw_u_dq = np.asarray(
            [
                float(feedforward_dq[0]) + float(self.config.kp_d) * float(error_d) + candidate_d,
                float(feedforward_dq[1]) + float(self.config.kp_q) * float(error_q) + candidate_q,
            ],
            dtype=np.float64,
        )
        saturated_u_dq = clip_voltage_vector(raw_u_dq, limit=self._resolve_voltage_limit())
        self._apply_anti_windup(
            candidate_d=candidate_d,
            candidate_q=candidate_q,
            raw_u_dq=raw_u_dq,
            saturated_u_dq=saturated_u_dq,
            error_d=error_d,
            error_q=error_q,
        )

        final_u_dq = np.asarray(
            [
                float(feedforward_dq[0]) + float(self.config.kp_d) * float(error_d) + self._integrator_d,
                float(feedforward_dq[1]) + float(self.config.kp_q) * float(error_q) + self._integrator_q,
            ],
            dtype=np.float64,
        )
        return clip_voltage_vector(final_u_dq, limit=self._resolve_voltage_limit())

    def _inverse_park_to_abc(self, *, u_d: float, u_q: float, epsilon: float) -> np.ndarray:
        theta = float(epsilon) * float(self.config.epsilon_scale)
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))
        u_alpha = cos_theta * float(u_d) - sin_theta * float(u_q)
        u_beta = sin_theta * float(u_d) + cos_theta * float(u_q)
        sqrt3_over_2 = float(np.sqrt(3.0) / 2.0)
        return np.asarray(
            [
                u_alpha,
                -0.5 * u_alpha + sqrt3_over_2 * u_beta,
                -0.5 * u_alpha - sqrt3_over_2 * u_beta,
            ],
            dtype=np.float64,
        )

    def _compute_custom_action(self, obs: np.ndarray) -> np.ndarray:
        i_d = self._read_required_signal(obs, "i_d")
        i_q = self._read_required_signal(obs, "i_q")
        ref_i_d = self._read_required_signal(obs, "ref_i_d")
        ref_i_q = self._read_required_signal(obs, "ref_i_q")
        omega_m = self._read_required_signal(obs, "omega_m")

        error_d = float(ref_i_d) - float(i_d)
        error_q = float(ref_i_q) - float(i_q)
        feedforward_dq = self._custom_feedforward(i_d=i_d, i_q=i_q, omega_m=omega_m)
        u_dq = self._compose_dq_voltage(
            error_d=error_d,
            error_q=error_q,
            dt=float(self.motor_params.Ts if self.motor_params is not None else 1e-4),
            feedforward_dq=feedforward_dq,
        )
        voltage_limit = max(self._resolve_voltage_limit(), 1e-6)
        action = np.clip(u_dq / voltage_limit, -float(self.config.action_limit), float(self.config.action_limit))
        return action.astype(np.float32, copy=False)

    def _compute_legacy_action(self, obs: np.ndarray) -> np.ndarray:
        i_d = self._read_required_signal(obs, "i_d")
        i_q = self._read_required_signal(obs, "i_q")
        ref_i_d = self._read_required_signal(obs, "ref_i_d")
        ref_i_q = self._read_required_signal(obs, "ref_i_q")
        epsilon = self._read_required_signal(obs, "epsilon")

        error_d = float(ref_i_d) - float(i_d)
        error_q = float(ref_i_q) - float(i_q)
        u_dq = self._compose_dq_voltage(
            error_d=error_d,
            error_q=error_q,
            dt=1.0,
            feedforward_dq=np.zeros(2, dtype=np.float64),
        )
        action = self._inverse_park_to_abc(u_d=float(u_dq[0]), u_q=float(u_dq[1]), epsilon=epsilon)
        action = np.clip(action, -float(self.config.action_limit), float(self.config.action_limit))
        return action.astype(np.float32, copy=False)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute normalized control action from the current observation."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError("Observation vector is empty")
        if not np.isfinite(x).all():
            self.reset()
            return np.zeros((self.action_dim,), dtype=np.float32)

        action = self._compute_custom_action(x) if self.mode == "custom_dq" else self._compute_legacy_action(x)
        if not np.isfinite(action).all():
            self.reset()
            return np.zeros((self.action_dim,), dtype=np.float32)
        return action

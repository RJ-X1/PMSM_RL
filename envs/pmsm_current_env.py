"""Custom dq-axis PMSM current-control environment."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from envs.motor_model import (
    PMSMEnvParams,
    PMSMMotorParams,
    PMSMState,
    clip_voltage_vector,
    electromagnetic_torque,
    rk4_step,
)


class PMSMCurrentControlEnv(gym.Env[np.ndarray, np.ndarray]):
    """Paper-style PMSM dq current-control environment with a fixed flat observation.

    The observation always follows:
    ``[i_d, i_q, omega_m, ref_i_d, ref_i_q, e_d, e_q, prev_u_d, prev_u_q, T_L]``

    The action is a normalized dq voltage command:
    ``[a_d, a_q] in [-1, 1]^2``.

    Reward terms are normalized by ``Imax`` and ``Umax`` so the paper-style
    baseline remains numerically stable across different motor parameter sets.
    """

    metadata = {"render_modes": []}

    OBSERVATION_NAMES = (
        "i_d",
        "i_q",
        "omega_m",
        "ref_i_d",
        "ref_i_q",
        "e_d",
        "e_q",
        "prev_u_d",
        "prev_u_q",
        "T_L",
    )
    ACTION_NAMES = ("act_u_d", "act_u_q")
    REWARD_TERM_NAMES = (
        "reward_tracking_d",
        "reward_tracking_q",
        "reward_voltage_effort",
        "reward_delta_action",
        "reward_limit_penalty",
    )

    def __init__(
        self,
        *,
        env_id: str = "Custom-PMSM-Current-v0",
        motor_params: PMSMMotorParams | None = None,
        env_params: PMSMEnvParams | None = None,
        ref_i_d: float = 0.0,
        ref_i_q: float = 10.0,
        randomize_reference: bool = False,
        ref_i_d_range: Iterable[float] = (0.0, 0.0),
        ref_i_q_range: Iterable[float] = (10.0, 10.0),
        init_i_d_range: Iterable[float] = (0.0, 0.0),
        init_i_q_range: Iterable[float] = (0.0, 0.0),
        init_omega_m_range: Iterable[float] = (0.0, 0.0),
        load_torque: float = 0.0,
        load_torque_range: Iterable[float] = (0.0, 0.0),
        observation_noise_std: float = 0.0,
        process_noise_std: float = 0.0,
        reward_mode: str = "vanilla_td3_reward",
        reward_w_ed: float | None = None,
        reward_w_eq: float | None = None,
        reward_w_u: float | None = None,
        reward_w_da: float | None = None,
        reward_w_lim: float | None = None,
        reward_error_weight: float | None = None,
        reward_voltage_weight: float | None = None,
        reward_delta_action_weight: float | None = None,
        reward_limit_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.env_id = str(env_id)
        self.is_custom_pmsm_env = True

        self.motor_params = motor_params or PMSMMotorParams()
        self.env_params = env_params or PMSMEnvParams()

        self.ref_i_d = float(ref_i_d)
        self.ref_i_q = float(ref_i_q)
        self.randomize_reference = bool(randomize_reference)
        self.ref_i_d_range = self._as_range(ref_i_d_range, default=(self.ref_i_d, self.ref_i_d))
        self.ref_i_q_range = self._as_range(ref_i_q_range, default=(self.ref_i_q, self.ref_i_q))
        self.init_i_d_range = self._as_range(init_i_d_range, default=(0.0, 0.0))
        self.init_i_q_range = self._as_range(init_i_q_range, default=(0.0, 0.0))
        self.init_omega_m_range = self._as_range(init_omega_m_range, default=(0.0, 0.0))

        self.fixed_load_torque = float(load_torque)
        self.load_torque_range = self._as_range(
            load_torque_range,
            default=(self.fixed_load_torque, self.fixed_load_torque),
        )

        self.observation_noise_std = float(observation_noise_std)
        self.process_noise_std = float(process_noise_std)
        legacy_error_weight = 0.40 if reward_error_weight is None else float(reward_error_weight)
        self.reward_mode = str(reward_mode).strip().lower()
        self.reward_w_ed = float(legacy_error_weight if reward_w_ed is None else reward_w_ed)
        self.reward_w_eq = float(legacy_error_weight if reward_w_eq is None else reward_w_eq)
        self.reward_w_u = float(0.08 if reward_voltage_weight is None and reward_w_u is None else (
            reward_voltage_weight if reward_w_u is None else reward_w_u
        ))
        self.reward_w_da = float(0.07 if reward_delta_action_weight is None and reward_w_da is None else (
            reward_delta_action_weight if reward_w_da is None else reward_w_da
        ))
        self.reward_w_lim = float(0.05 if reward_limit_weight is None and reward_w_lim is None else (
            reward_limit_weight if reward_w_lim is None else reward_w_lim
        ))

        self.current_limit = self.env_params.resolved_current_limit(self.motor_params)
        self.observation_names = list(self.OBSERVATION_NAMES)
        self.action_names = list(self.ACTION_NAMES)
        self.reward_term_names = list(self.REWARD_TERM_NAMES)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.full((len(self.OBSERVATION_NAMES),), -np.inf, dtype=np.float32),
            high=np.full((len(self.OBSERVATION_NAMES),), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = PMSMState()
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.prev_u_dq = np.zeros(2, dtype=np.float64)
        self.load_torque = self.fixed_load_torque
        self.elapsed_steps = 0

    @staticmethod
    def _as_range(values: Iterable[float], *, default: tuple[float, float]) -> tuple[float, float]:
        arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return float(default[0]), float(default[1])
        if arr.size != 2:
            raise ValueError(f"Expected a 2-value range, got {arr.tolist()}")
        lo, hi = float(arr[0]), float(arr[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low == high:
            return float(low)
        return float(self.np_random.uniform(low, high))

    def _sample_reference(self) -> tuple[float, float]:
        if not self.randomize_reference:
            return float(self.ref_i_d), float(self.ref_i_q)
        return (
            self._sample_uniform(self.ref_i_d_range),
            self._sample_uniform(self.ref_i_q_range),
        )

    def _apply_process_noise(self, state: PMSMState) -> PMSMState:
        if self.process_noise_std <= 0.0:
            return state
        noisy = state.as_array(dtype=np.float64)
        noisy += self.np_random.normal(0.0, self.process_noise_std, size=3)
        return PMSMState.from_array(noisy)

    def _measured_state(self) -> tuple[float, float, float]:
        values = self.state.as_array(dtype=np.float64)
        if self.observation_noise_std > 0.0:
            values += self.np_random.normal(0.0, self.observation_noise_std, size=3)
        i_d, i_q, omega_m = values
        return float(i_d), float(i_q), float(omega_m)

    def _build_observation(self) -> np.ndarray:
        i_d, i_q, omega_m = self._measured_state()
        e_d = float(self.ref_i_d) - i_d
        e_q = float(self.ref_i_q) - i_q
        obs = np.asarray(
            [
                i_d,
                i_q,
                omega_m,
                self.ref_i_d,
                self.ref_i_q,
                e_d,
                e_q,
                self.prev_u_dq[0],
                self.prev_u_dq[1],
                self.load_torque,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
        return obs

    def _build_info(self, *, reward: float = 0.0, done_reason: str = "") -> dict[str, Any]:
        current_mag = float(np.linalg.norm([self.state.i_d, self.state.i_q]))
        torque_e = float(electromagnetic_torque(self.state, self.motor_params))
        return {
            "env_id": self.env_id,
            "done_reason": done_reason,
            "reward_mode": self.reward_mode,
            "elapsed_steps": int(self.elapsed_steps),
            "i_d": float(self.state.i_d),
            "i_q": float(self.state.i_q),
            "omega_m": float(self.state.omega_m),
            "ref_i_d": float(self.ref_i_d),
            "ref_i_q": float(self.ref_i_q),
            "prev_u_d": float(self.prev_u_dq[0]),
            "prev_u_q": float(self.prev_u_dq[1]),
            "load_torque": float(self.load_torque),
            "current_magnitude": current_mag,
            "current_limit": float(self.current_limit),
            "torque_e": torque_e,
            "reward": float(reward),
        }

    def _vanilla_limit_penalty(self, current_mag: float) -> float:
        """Preserve the original baseline current-limit penalty behavior."""
        return max(0.0, float(current_mag) / float(self.current_limit) - 1.0)

    def _constraint_aware_limit_penalty(self) -> float:
        """Per-axis squared current-limit penalty used by the enhanced reward mode."""
        i_scale = max(float(self.motor_params.Imax), 1e-6)
        i_d_penalty = max(0.0, abs(float(self.state.i_d)) / i_scale - 1.0) ** 2
        i_q_penalty = max(0.0, abs(float(self.state.i_q)) / i_scale - 1.0) ** 2
        return float(i_d_penalty + i_q_penalty)

    def _reward_terms(
        self,
        *,
        e_d: float,
        e_q: float,
        u_dq: np.ndarray,
        delta_a: np.ndarray,
        current_mag: float,
    ) -> tuple[float, OrderedDict[str, float]]:
        i_scale = max(float(self.motor_params.Imax), 1e-6)
        u_scale = max(float(self.motor_params.Umax), 1e-6)
        tracking_d = float(abs(float(e_d)) / i_scale)
        tracking_q = float(abs(float(e_q)) / i_scale)
        voltage_effort = float(np.dot(u_dq / u_scale, u_dq / u_scale))
        delta_action = float(np.dot(delta_a, delta_a))
        if self.reward_mode == "constraint_aware_reward":
            limit_penalty = self._constraint_aware_limit_penalty()
        elif self.reward_mode == "vanilla_td3_reward":
            limit_penalty = self._vanilla_limit_penalty(current_mag)
        else:
            raise ValueError(f"Unsupported reward mode: {self.reward_mode}")
        reward = (
            1.0
            - self.reward_w_ed * tracking_d
            - self.reward_w_eq * tracking_q
            - self.reward_w_u * voltage_effort
            - self.reward_w_da * delta_action
            - self.reward_w_lim * limit_penalty
        )
        terms = OrderedDict(
            reward_tracking_d=tracking_d,
            reward_tracking_q=tracking_q,
            reward_voltage_effort=voltage_effort,
            reward_delta_action=delta_action,
            reward_limit_penalty=float(limit_penalty),
        )
        safe_reward = float(np.nan_to_num(reward, nan=-1.0, posinf=-1.0, neginf=-1.0))
        return safe_reward, terms

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options

        self.state = PMSMState(
            i_d=self._sample_uniform(self.init_i_d_range),
            i_q=self._sample_uniform(self.init_i_q_range),
            omega_m=self._sample_uniform(self.init_omega_m_range),
        )
        self.ref_i_d, self.ref_i_q = self._sample_reference()
        self.load_torque = self._sample_uniform(self.load_torque_range)
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.prev_u_dq = np.zeros(2, dtype=np.float64)
        self.elapsed_steps = 0

        obs = self._build_observation()
        if self.env_params.terminate_on_nonfinite and not np.isfinite(obs).all():
            raise FloatingPointError("Custom PMSM environment reset produced non-finite observation")
        return obs, self._build_info(done_reason="")

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.isfinite(action_arr).all():
            safe_obs = self._build_observation()
            return safe_obs, -1.0, True, False, self._build_info(done_reason="non_finite_action")

        clipped_action = np.clip(action_arr, -1.0, 1.0)
        u_dq = clip_voltage_vector(
            clipped_action * float(self.motor_params.Umax),
            limit=float(self.motor_params.Umax),
        )
        delta_a = clipped_action - self.prev_action

        done_reason = ""
        terminated = False
        truncated = False

        try:
            next_state = rk4_step(
                self.state,
                u_dq=u_dq,
                load_torque=self.load_torque,
                params=self.motor_params,
            )
            next_state = self._apply_process_noise(next_state)
        except FloatingPointError:
            next_state = self.state
            terminated = True
            done_reason = "non_finite_state"

        self.state = next_state
        self.prev_action = clipped_action
        self.prev_u_dq = u_dq
        self.elapsed_steps += 1

        current_mag = float(np.linalg.norm([self.state.i_d, self.state.i_q]))
        e_d = float(self.ref_i_d) - float(self.state.i_d)
        e_q = float(self.ref_i_q) - float(self.state.i_q)
        reward, reward_terms = self._reward_terms(
            e_d=e_d,
            e_q=e_q,
            u_dq=u_dq,
            delta_a=delta_a,
            current_mag=current_mag,
        )

        if current_mag > float(self.current_limit) and self.env_params.terminate_on_overcurrent:
            terminated = True
            done_reason = done_reason or "overcurrent"
            reward -= 1.0

        if self.elapsed_steps >= int(self.env_params.episode_steps):
            truncated = not terminated
            if not done_reason and truncated:
                done_reason = "episode_limit"

        obs = self._build_observation()
        if self.env_params.terminate_on_nonfinite and not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
            terminated = True
            done_reason = done_reason or "non_finite_observation"
            reward = -1.0

        info = self._build_info(reward=reward, done_reason=done_reason)
        info["reward_terms"] = dict(reward_terms)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        """No-op render hook; the environment is currently numeric-only."""
        return None

    def close(self) -> None:
        """No-op close hook for Gymnasium compatibility."""
        return None

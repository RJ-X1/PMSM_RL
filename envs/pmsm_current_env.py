"""Custom dq-axis PMSM current-control environment."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
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
        domain_randomization: dict[str, Any] | None = None,
        apply_domain_randomization: bool = False,
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

        self.nominal_motor_params = motor_params or PMSMMotorParams()
        self.motor_params = replace(self.nominal_motor_params)
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
        self.base_current_measurement_noise_std = float(observation_noise_std)
        self.base_speed_measurement_noise_std = float(observation_noise_std)
        self.current_measurement_noise_std = float(self.base_current_measurement_noise_std)
        self.speed_measurement_noise_std = float(self.base_speed_measurement_noise_std)
        self.process_noise_std = float(process_noise_std)
        self.domain_randomization = self._normalize_domain_randomization_config(domain_randomization)
        self.apply_domain_randomization = bool(apply_domain_randomization)
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
        self.load_schedule_active = False
        self.next_load_change_step: int | None = None
        self.last_randomization_sample: dict[str, float | bool | int] = {}
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

    @staticmethod
    def _as_int_range(values: Iterable[int], *, default: tuple[int, int]) -> tuple[int, int]:
        arr = np.asarray(list(values), dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return int(default[0]), int(default[1])
        if arr.size != 2:
            raise ValueError(f"Expected a 2-value integer range, got {arr.tolist()}")
        lo, hi = int(arr[0]), int(arr[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    def _normalize_domain_randomization_config(
        self,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        defaults = {
            "enabled": False,
            "rs_scale_range": (1.0, 1.0),
            "ld_scale_range": (1.0, 1.0),
            "lq_scale_range": (1.0, 1.0),
            "psi_f_scale_range": (1.0, 1.0),
            "j_scale_range": (1.0, 1.0),
            "vdc_scale_range": (1.0, 1.0),
            "sigma_i_range": (0.0, 0.0),
            "sigma_omega_range": (0.0, 0.0),
            "sigma_omega_rpm_range": (0.0, 0.0),
            "load_torque_range": (self.fixed_load_torque, self.fixed_load_torque),
            "load_change_interval_steps_range": (0, 0),
        }
        raw = dict(config or {})
        return {
            "enabled": bool(raw.get("enabled", defaults["enabled"])),
            "rs_scale_range": self._as_range(
                raw.get("rs_scale_range", defaults["rs_scale_range"]),
                default=defaults["rs_scale_range"],
            ),
            "ld_scale_range": self._as_range(
                raw.get("ld_scale_range", defaults["ld_scale_range"]),
                default=defaults["ld_scale_range"],
            ),
            "lq_scale_range": self._as_range(
                raw.get("lq_scale_range", defaults["lq_scale_range"]),
                default=defaults["lq_scale_range"],
            ),
            "psi_f_scale_range": self._as_range(
                raw.get("psi_f_scale_range", defaults["psi_f_scale_range"]),
                default=defaults["psi_f_scale_range"],
            ),
            "j_scale_range": self._as_range(
                raw.get("j_scale_range", defaults["j_scale_range"]),
                default=defaults["j_scale_range"],
            ),
            "vdc_scale_range": self._as_range(
                raw.get("vdc_scale_range", defaults["vdc_scale_range"]),
                default=defaults["vdc_scale_range"],
            ),
            "sigma_i_range": self._as_range(
                raw.get("sigma_i_range", defaults["sigma_i_range"]),
                default=defaults["sigma_i_range"],
            ),
            "sigma_omega_range": self._as_range(
                raw.get("sigma_omega_range", defaults["sigma_omega_range"]),
                default=defaults["sigma_omega_range"],
            ),
            "sigma_omega_rpm_range": self._as_range(
                raw.get("sigma_omega_rpm_range", defaults["sigma_omega_rpm_range"]),
                default=defaults["sigma_omega_rpm_range"],
            ),
            "load_torque_range": self._as_range(
                raw.get("load_torque_range", defaults["load_torque_range"]),
                default=defaults["load_torque_range"],
            ),
            "load_change_interval_steps_range": self._as_int_range(
                raw.get("load_change_interval_steps_range", defaults["load_change_interval_steps_range"]),
                default=defaults["load_change_interval_steps_range"],
            ),
        }

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low == high:
            return float(low)
        return float(self.np_random.uniform(low, high))

    def _sample_int(self, value_range: tuple[int, int]) -> int:
        low, high = int(value_range[0]), int(value_range[1])
        if low >= high:
            return int(low)
        return int(self.np_random.integers(low, high + 1))

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

    def _sample_domain_randomization(self) -> None:
        """Sample active motor/noise/load parameters for one episode."""
        dr = self.domain_randomization
        if not (self.apply_domain_randomization and bool(dr.get("enabled", False))):
            self.motor_params = replace(self.nominal_motor_params)
            self.current_measurement_noise_std = float(self.base_current_measurement_noise_std)
            self.speed_measurement_noise_std = float(self.base_speed_measurement_noise_std)
            self.load_torque = self._sample_uniform(self.load_torque_range)
            self.load_schedule_active = False
            self.next_load_change_step = None
            self.current_limit = self.env_params.resolved_current_limit(self.motor_params)
            sigma_omega_rpm = float(
                self.speed_measurement_noise_std * (60.0 / (2.0 * np.pi))
            )
            self.last_randomization_sample = {
                "domain_randomization_active": False,
                "Rs": float(self.motor_params.Rs),
                "Ld": float(self.motor_params.Ld),
                "Lq": float(self.motor_params.Lq),
                "psi_f": float(self.motor_params.psi_f),
                "J": float(self.motor_params.J),
                "Vdc": float(self.motor_params.Vdc),
                "Umax": float(self.motor_params.Umax),
                "sigma_i": float(self.current_measurement_noise_std),
                "sigma_omega_rpm": sigma_omega_rpm,
                "load_torque": float(self.load_torque),
                "load_schedule_active": False,
                "load_change_interval_steps": 0,
            }
            return

        rs_scale = self._sample_uniform(dr["rs_scale_range"])
        ld_scale = self._sample_uniform(dr["ld_scale_range"])
        lq_scale = self._sample_uniform(dr["lq_scale_range"])
        psi_f_scale = self._sample_uniform(dr["psi_f_scale_range"])
        j_scale = self._sample_uniform(dr["j_scale_range"])
        vdc_scale = self._sample_uniform(dr["vdc_scale_range"])

        self.motor_params = replace(
            self.nominal_motor_params,
            Rs=float(self.nominal_motor_params.Rs) * rs_scale,
            Ld=float(self.nominal_motor_params.Ld) * ld_scale,
            Lq=float(self.nominal_motor_params.Lq) * lq_scale,
            psi_f=float(self.nominal_motor_params.psi_f) * psi_f_scale,
            J=float(self.nominal_motor_params.J) * j_scale,
            Vdc=float(self.nominal_motor_params.Vdc) * vdc_scale,
            Umax=float(self.nominal_motor_params.Umax) * vdc_scale,
        )
        self.current_limit = self.env_params.resolved_current_limit(self.motor_params)
        self.current_measurement_noise_std = self._sample_uniform(dr["sigma_i_range"])
        self.speed_measurement_noise_std = self._sample_uniform(dr["sigma_omega_range"])
        self.load_torque = self._sample_uniform(dr["load_torque_range"])

        change_interval_range = dr["load_change_interval_steps_range"]
        self.load_schedule_active = int(change_interval_range[1]) > 0
        if self.load_schedule_active:
            self.next_load_change_step = self._sample_int(change_interval_range)
        else:
            self.next_load_change_step = None

        sigma_omega_rpm = float(
            self.speed_measurement_noise_std * (60.0 / (2.0 * np.pi))
        )
        self.last_randomization_sample = {
            "domain_randomization_active": True,
            "Rs": float(self.motor_params.Rs),
            "Ld": float(self.motor_params.Ld),
            "Lq": float(self.motor_params.Lq),
            "psi_f": float(self.motor_params.psi_f),
            "J": float(self.motor_params.J),
            "Vdc": float(self.motor_params.Vdc),
            "Umax": float(self.motor_params.Umax),
            "sigma_i": float(self.current_measurement_noise_std),
            "sigma_omega_rpm": float(sigma_omega_rpm),
            "load_torque": float(self.load_torque),
            "load_schedule_active": bool(self.load_schedule_active),
            "load_change_interval_steps": int(self.next_load_change_step or 0),
        }

    def _maybe_update_load_torque_schedule(self) -> None:
        """Apply piecewise-constant load changes when DR enables a schedule."""
        if not self.load_schedule_active or self.next_load_change_step is None:
            return
        if self.elapsed_steps < int(self.next_load_change_step):
            return
        self.load_torque = self._sample_uniform(self.domain_randomization["load_torque_range"])
        interval = self._sample_int(self.domain_randomization["load_change_interval_steps_range"])
        self.next_load_change_step = int(self.elapsed_steps + max(interval, 1))
        self.last_randomization_sample["load_torque"] = float(self.load_torque)
        self.last_randomization_sample["load_change_interval_steps"] = int(interval)

    def _measured_state(self) -> tuple[float, float, float]:
        values = self.state.as_array(dtype=np.float64)
        if self.current_measurement_noise_std > 0.0:
            values[:2] += self.np_random.normal(0.0, self.current_measurement_noise_std, size=2)
        if self.speed_measurement_noise_std > 0.0:
            values[2] += float(self.np_random.normal(0.0, self.speed_measurement_noise_std))
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
            "domain_randomization_active": bool(self.last_randomization_sample.get("domain_randomization_active", False)),
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
            "active_Rs": float(self.motor_params.Rs),
            "active_Ld": float(self.motor_params.Ld),
            "active_Lq": float(self.motor_params.Lq),
            "active_psi_f": float(self.motor_params.psi_f),
            "active_J": float(self.motor_params.J),
            "active_Vdc": float(self.motor_params.Vdc),
            "active_Umax": float(self.motor_params.Umax),
            "current_measurement_noise_std": float(self.current_measurement_noise_std),
            "speed_measurement_noise_std": float(self.speed_measurement_noise_std),
            "speed_measurement_noise_rpm": float(
                self.speed_measurement_noise_std * (60.0 / (2.0 * np.pi))
            ),
            "load_schedule_active": bool(self.load_schedule_active),
            "next_load_change_step": None if self.next_load_change_step is None else int(self.next_load_change_step),
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
        self._sample_domain_randomization()
        self.ref_i_d, self.ref_i_q = self._sample_reference()
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
        self._maybe_update_load_torque_schedule()
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

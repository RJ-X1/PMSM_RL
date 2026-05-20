"""Custom gun/fire-control servo position environment.

This environment is the first minimal simulation for the project's new focus:
reinforcement learning control of a gun-servo position outer loop.  The old
PMSM current-control environment remains available for backward compatibility.
Here the RL policy outputs a one-dimensional speed-command correction, not dq
voltage or PWM.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from baselines.pi_speed_controller import PISpeedController, PISpeedControllerConfig
from envs.gun_load_model import GunLoadParams, GunLoadState, SingleInertiaGunLoad
from envs.obs_parser import GUN_SERVO_ACTION_NAMES, GUN_SERVO_OBSERVATION_NAMES
from envs.trajectory_generator import GunServoTrajectoryGenerator, TrajectoryPoint

RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


class GunServoPositionEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-axis gun servo tracking task with RL as the position outer loop."""

    metadata = {"render_modes": []}
    OBSERVATION_NAMES = GUN_SERVO_OBSERVATION_NAMES
    ACTION_NAMES = GUN_SERVO_ACTION_NAMES
    REWARD_TERM_NAMES = (
        "reward_theta",
        "reward_omega",
        "reward_iq",
        "reward_action_smoothness",
        "reward_saturation",
        "reward_terminal",
        "reward_total",
    )

    def __init__(
        self,
        *,
        env_id: str = "Custom-GunServo-Position-v0",
        motor: dict[str, Any] | None = None,
        load: dict[str, Any] | None = None,
        reference: dict[str, Any] | None = None,
        rl_action: dict[str, Any] | None = None,
        reward: dict[str, Any] | None = None,
        domain_randomization: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        speed_controller: dict[str, Any] | None = None,
        apply_domain_randomization: bool = False,
    ) -> None:
        super().__init__()
        self.env_id = str(env_id)
        self.is_gun_servo_position_env = True

        self.motor_cfg = dict(motor or {})
        self.load_cfg = dict(load or {})
        self.reference_cfg = dict(reference or {})
        self.rl_action_cfg = dict(rl_action or {})
        self.reward_cfg = dict(reward or {})
        self.domain_randomization = self._normalize_domain_randomization(domain_randomization or {})
        self.environment_cfg = dict(environment or {})
        self.apply_domain_randomization = bool(apply_domain_randomization)

        self.p = float(self.motor_cfg.get("p", 4.0))
        self.psi_f = float(self.motor_cfg.get("psi_f", 0.095))
        self.Imax = float(self.motor_cfg.get("Imax", 15.0))
        self.Ts_current = float(self.motor_cfg.get("Ts_current", 1e-4))
        self.Ts_speed = float(self.motor_cfg.get("Ts_speed", 5e-4))
        self.dt = float(self.motor_cfg.get("Ts_position", self.motor_cfg.get("Ts", 0.002)))
        self.current_time_constant = max(
            float(self.motor_cfg.get("current_time_constant", 5.0 * self.Ts_current)),
            self.Ts_current,
        )
        self.torque_constant = float(self.motor_cfg.get("Kt", 1.5 * self.p * self.psi_f))

        self.episode_steps = int(self.environment_cfg.get("episode_steps", 1500))
        self.init_theta_range = self._deg_range(
            self.environment_cfg.get("init_theta_range_deg", (0.0, 0.0)),
            default=(0.0, 0.0),
        )
        self.init_omega_range = self._deg_range(
            self.environment_cfg.get("init_omega_range_deg_s", (0.0, 0.0)),
            default=(0.0, 0.0),
        )
        self.max_abs_theta = float(np.deg2rad(float(self.environment_cfg.get("max_abs_theta_deg", 85.0))))

        self.theta_scale = max(abs(float(np.deg2rad(float(self.reference_cfg.get("theta_scale_deg", 30.0))))), 1e-6)
        self.omega_scale = max(
            abs(float(np.deg2rad(float(self.reference_cfg.get("omega_scale_deg_s", 90.0))))),
            1e-6,
        )
        self.torque_scale = max(float(self.load_cfg.get("torque_scale", 100.0)), 1e-6)

        self.max_delta_omega = abs(
            float(np.deg2rad(float(self.rl_action_cfg.get("max_delta_omega_deg_s", 50.0))))
        )
        self.max_omega_cmd = abs(
            float(np.deg2rad(float(self.rl_action_cfg.get("max_omega_cmd_deg_s", 90.0))))
        )
        self.max_delta_omega_rate = abs(
            float(np.deg2rad(float(self.rl_action_cfg.get("max_delta_omega_rate_deg_s2", 300.0))))
        )
        self.max_omega_cmd_accel = abs(
            float(
                np.deg2rad(
                    float(
                        self.rl_action_cfg.get(
                            "max_omega_cmd_accel_deg_s2",
                            self.reference_cfg.get("max_accel_deg_s2", 300.0),
                        )
                    )
                )
            )
        )

        speed_defaults = {
            "kp": 18.0,
            "ki": 80.0,
            "integrator_limit": 0.75 * self.Imax,
            "iq_limit": self.Imax,
        }
        speed_cfg = {**speed_defaults, **dict(speed_controller or {})}
        self.speed_pi = PISpeedController(
            PISpeedControllerConfig(
                kp=float(speed_cfg["kp"]),
                ki=float(speed_cfg["ki"]),
                integrator_limit=float(speed_cfg["integrator_limit"]),
                iq_limit=float(speed_cfg["iq_limit"]),
            )
        )

        self.trajectory = GunServoTrajectoryGenerator.from_config(self.reference_cfg)
        self.nominal_load_params = self._load_params_from_config(self.load_cfg)
        self.load_model = SingleInertiaGunLoad(self.nominal_load_params)

        self.reward_mode = str(self.reward_cfg.get("mode", "quadratic_tracking"))
        self.w_theta = float(self.reward_cfg.get("w_theta", 8.0))
        self.w_omega = float(self.reward_cfg.get("w_omega", 0.15))
        self.w_iq = float(self.reward_cfg.get("w_iq", 0.01))
        self.w_action_smoothness = float(self.reward_cfg.get("w_action_smoothness", 0.02))
        self.w_saturation = float(self.reward_cfg.get("w_saturation", 0.25))
        self.w_terminal = float(self.reward_cfg.get("w_terminal", 2.0))

        self.observation_names = list(self.OBSERVATION_NAMES)
        self.action_names = list(self.ACTION_NAMES)
        self.reward_term_names = list(self.REWARD_TERM_NAMES)
        self.observation_is_normalized = True
        self.action_space = gym.spaces.Box(
            low=float(ACTION_LOW),
            high=float(ACTION_HIGH),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.full((len(self.OBSERVATION_NAMES),), -np.inf, dtype=np.float32),
            high=np.full((len(self.OBSERVATION_NAMES),), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = GunLoadState()
        self.elapsed_steps = 0
        self.iq = 0.0
        self.iq_cmd = 0.0
        self.Te = 0.0
        self.T_out = 0.0
        self.T_L_hat = 0.0
        self.disturbance_torque = 0.0
        self.encoder_noise_std = 0.0
        self.prev_action = 0.0
        self.prev_delta_omega = 0.0
        self.prev_omega_cmd = 0.0
        self.last_trajectory = TrajectoryPoint(0.0, 0.0, 0.0)
        self.last_saturation = 0
        self.last_speed_saturation = 0
        self.last_current_saturation = 0
        self.last_action_rate_saturation = 0
        self.last_accel_saturation = 0
        self.last_reward_terms: OrderedDict[str, float] = OrderedDict()
        self.last_randomization_sample: dict[str, float | bool] = {}

    @staticmethod
    def _deg_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
        raw = default if value is None else value
        lo, hi = float(raw[0]), float(raw[1])
        if lo > hi:
            lo, hi = hi, lo
        return float(np.deg2rad(lo)), float(np.deg2rad(hi))

    @staticmethod
    def _float_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
        raw = default if value is None else value
        lo, hi = float(raw[0]), float(raw[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    @staticmethod
    def _load_params_from_config(config: dict[str, Any]) -> GunLoadParams:
        defaults = GunLoadParams()
        return GunLoadParams(
            gear_ratio=float(config.get("gear_ratio", defaults.gear_ratio)),
            efficiency=float(config.get("efficiency", defaults.efficiency)),
            J_load=float(config.get("J_load", defaults.J_load)),
            B_load=float(config.get("B_load", defaults.B_load)),
            coulomb_friction=float(config.get("coulomb_friction", defaults.coulomb_friction)),
            gravity_torque_coeff=float(config.get("gravity_torque_coeff", defaults.gravity_torque_coeff)),
            gravity_phase=float(config.get("gravity_phase", defaults.gravity_phase)),
            backlash_rad=float(config.get("backlash_rad", defaults.backlash_rad)),
            torsional_stiffness=float(config.get("torsional_stiffness", defaults.torsional_stiffness)),
            torsional_damping=float(config.get("torsional_damping", defaults.torsional_damping)),
        )

    def _normalize_domain_randomization(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(config.get("enabled", False)),
            "J_load_scale_range": self._float_range(config.get("J_load_scale_range"), default=(1.0, 1.0)),
            "B_load_scale_range": self._float_range(config.get("B_load_scale_range"), default=(1.0, 1.0)),
            "friction_scale_range": self._float_range(config.get("friction_scale_range"), default=(1.0, 1.0)),
            "gravity_torque_scale_range": self._float_range(
                config.get("gravity_torque_scale_range"),
                default=(1.0, 1.0),
            ),
            "disturbance_torque_range": self._float_range(
                config.get("disturbance_torque_range"),
                default=(0.0, 0.0),
            ),
            "encoder_noise_deg_range": self._float_range(config.get("encoder_noise_deg_range"), default=(0.0, 0.0)),
        }

    @property
    def observation_normalization_scales(self) -> dict[str, float]:
        return {
            "theta": float(self.theta_scale),
            "omega": float(self.omega_scale),
            "current": max(float(self.Imax), 1e-6),
            "torque": float(self.torque_scale),
            "speed_command": max(float(self.max_omega_cmd), 1e-6),
        }

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        lo, hi = value_range
        if lo == hi:
            return float(lo)
        return float(self.np_random.uniform(lo, hi))

    def _sample_domain_randomization(self) -> None:
        params = replace(self.nominal_load_params)
        active = bool(self.apply_domain_randomization and self.domain_randomization["enabled"])
        if active:
            j_scale = self._sample_uniform(self.domain_randomization["J_load_scale_range"])
            b_scale = self._sample_uniform(self.domain_randomization["B_load_scale_range"])
            f_scale = self._sample_uniform(self.domain_randomization["friction_scale_range"])
            g_scale = self._sample_uniform(self.domain_randomization["gravity_torque_scale_range"])
            params.J_load *= j_scale
            params.B_load *= b_scale
            params.coulomb_friction *= f_scale
            params.gravity_torque_coeff *= g_scale
            self.disturbance_torque = self._sample_uniform(self.domain_randomization["disturbance_torque_range"])
            noise_deg = self._sample_uniform(self.domain_randomization["encoder_noise_deg_range"])
            self.encoder_noise_std = float(np.deg2rad(noise_deg))
        else:
            j_scale = b_scale = f_scale = g_scale = 1.0
            self.disturbance_torque = float(self.load_cfg.get("disturbance_torque", 0.0))
            self.encoder_noise_std = float(np.deg2rad(float(self.reference_cfg.get("encoder_noise_deg", 0.0))))
        self.load_model = SingleInertiaGunLoad(params)
        self.last_randomization_sample = {
            "domain_randomization_active": active,
            "J_load_scale": float(j_scale),
            "B_load_scale": float(b_scale),
            "friction_scale": float(f_scale),
            "gravity_torque_scale": float(g_scale),
            "disturbance_torque": float(self.disturbance_torque),
            "encoder_noise_std": float(self.encoder_noise_std),
        }

    def _measure_state(self) -> tuple[float, float]:
        theta_noise = (
            0.0
            if self.encoder_noise_std <= 0.0
            else float(self.np_random.normal(0.0, self.encoder_noise_std))
        )
        return float(self.state.theta + theta_noise), float(self.state.omega)

    def _current_trajectory(self) -> TrajectoryPoint:
        return self.trajectory.sample(float(self.elapsed_steps) * float(self.dt))

    def _build_observation(self) -> np.ndarray:
        theta_meas, omega_meas = self._measure_state()
        traj = self._current_trajectory()
        e_theta = float(traj.theta_ref - theta_meas)
        e_omega = float(traj.omega_ff - omega_meas)
        obs = np.asarray(
            [
                e_theta / self.theta_scale,
                e_omega / self.omega_scale,
                traj.theta_ref / self.theta_scale,
                theta_meas / self.theta_scale,
                omega_meas / self.omega_scale,
                self.prev_action,
                self.iq / max(float(self.Imax), 1e-6),
                self.T_L_hat / self.torque_scale,
                self.prev_omega_cmd / max(float(self.max_omega_cmd), 1e-6),
                float(self.last_saturation),
            ],
            dtype=np.float32,
        )
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
        return obs

    def _reward_terms(
        self,
        *,
        e_theta: float,
        e_omega: float,
        iq: float,
        delta_action: float,
        saturation_flag: int,
        terminal: bool,
    ) -> tuple[float, OrderedDict[str, float]]:
        e_theta_n = float(e_theta) / self.theta_scale
        e_omega_n = float(e_omega) / self.omega_scale
        iq_n = float(iq) / max(float(self.Imax), 1e-6)
        reward_theta = -self.w_theta * e_theta_n**2
        reward_omega = -self.w_omega * e_omega_n**2
        reward_iq = -self.w_iq * iq_n**2
        reward_action = -self.w_action_smoothness * float(delta_action) ** 2
        reward_sat = -self.w_saturation * float(saturation_flag)
        reward_terminal = -self.w_terminal * e_theta_n**2 if terminal else 0.0
        total = float(reward_theta + reward_omega + reward_iq + reward_action + reward_sat + reward_terminal)
        terms = OrderedDict(
            reward_theta=float(reward_theta),
            reward_omega=float(reward_omega),
            reward_iq=float(reward_iq),
            reward_action_smoothness=float(reward_action),
            reward_saturation=float(reward_sat),
            reward_terminal=float(reward_terminal),
            reward_total=total,
        )
        return total, terms

    def _build_info(self, *, reward: float = 0.0, done_reason: str = "") -> dict[str, Any]:
        traj = self.last_trajectory
        e_theta = float(traj.theta_ref - self.state.theta)
        e_omega = float(traj.omega_ff - self.state.omega)
        return {
            "env_id": self.env_id,
            "done_reason": done_reason,
            "reward_mode": self.reward_mode,
            "elapsed_steps": int(self.elapsed_steps),
            "time_s": float(self.elapsed_steps) * float(self.dt),
            "theta_ref": float(traj.theta_ref),
            "theta_L": float(self.state.theta),
            "e_theta": e_theta,
            "omega_ref": float(traj.omega_ff),
            "omega_ff": float(traj.omega_ff),
            "alpha_ff": float(traj.alpha_ff),
            "omega_L": float(self.state.omega),
            "omega_cmd": float(self.prev_omega_cmd),
            "iq": float(self.iq),
            "iq_cmd": float(self.iq_cmd),
            "Te": float(self.Te),
            "T_out": float(self.T_out),
            "TL_hat": float(self.T_L_hat),
            "disturbance_torque": float(self.disturbance_torque),
            "saturation_flag": int(self.last_saturation),
            "speed_saturation_flag": int(self.last_speed_saturation),
            "current_saturation_flag": int(self.last_current_saturation),
            "action_rate_saturation_flag": int(self.last_action_rate_saturation),
            "accel_saturation_flag": int(self.last_accel_saturation),
            "theta_ref_deg": float(traj.theta_ref * RAD_TO_DEG),
            "theta_L_deg": float(self.state.theta * RAD_TO_DEG),
            "e_theta_deg": float(e_theta * RAD_TO_DEG),
            "omega_ref_deg_s": float(traj.omega_ff * RAD_TO_DEG),
            "omega_L_deg_s": float(self.state.omega * RAD_TO_DEG),
            "omega_cmd_deg_s": float(self.prev_omega_cmd * RAD_TO_DEG),
            "iq_A": float(self.iq),
            "Te_Nm": float(self.Te),
            "TL_Nm": float(self.T_L_hat),
            "disturbance_torque_Nm": float(self.disturbance_torque),
            "reward": float(reward),
            "reward_terms": dict(self.last_reward_terms),
            "domain_randomization_active": bool(self.last_randomization_sample.get("domain_randomization_active", False)),
            "encoder_noise_std_rad": float(self.encoder_noise_std),
            "active_J_load": float(self.load_model.params.J_load),
            "active_B_load": float(self.load_model.params.B_load),
            "active_coulomb_friction": float(self.load_model.params.coulomb_friction),
            "active_gravity_torque_coeff": float(self.load_model.params.gravity_torque_coeff),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        self._sample_domain_randomization()
        self.trajectory.reset(rng=self.np_random)
        self.state = GunLoadState(
            theta=self._sample_uniform(self.init_theta_range),
            omega=self._sample_uniform(self.init_omega_range),
        )
        self.elapsed_steps = 0
        self.iq = 0.0
        self.iq_cmd = 0.0
        self.Te = 0.0
        self.T_out = 0.0
        self.T_L_hat = 0.0
        self.prev_action = 0.0
        self.prev_delta_omega = 0.0
        self.prev_omega_cmd = 0.0
        self.last_trajectory = self._current_trajectory()
        self.last_saturation = 0
        self.last_speed_saturation = 0
        self.last_current_saturation = 0
        self.last_action_rate_saturation = 0
        self.last_accel_saturation = 0
        self.last_reward_terms = OrderedDict()
        self.speed_pi.reset()
        return self._build_observation(), self._build_info(done_reason="")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw = np.asarray(action, dtype=np.float64).reshape(1)
        if not np.isfinite(raw).all():
            return self._build_observation(), -1.0, True, False, self._build_info(done_reason="non_finite_action")

        clipped_action = float(np.clip(raw[0], ACTION_LOW, ACTION_HIGH))
        self.last_trajectory = self._current_trajectory()
        delta_action = clipped_action - self.prev_action

        raw_delta_omega = clipped_action * self.max_delta_omega
        delta_step = self.max_delta_omega_rate * self.dt
        limited_delta_omega = float(
            np.clip(raw_delta_omega, self.prev_delta_omega - delta_step, self.prev_delta_omega + delta_step)
        )
        self.last_action_rate_saturation = int(abs(raw_delta_omega - limited_delta_omega) > 1e-10)

        raw_omega_cmd = float(self.last_trajectory.omega_ff + limited_delta_omega)
        speed_limited_cmd = float(np.clip(raw_omega_cmd, -self.max_omega_cmd, self.max_omega_cmd))
        self.last_speed_saturation = int(abs(raw_omega_cmd - speed_limited_cmd) > 1e-10)

        accel_step = self.max_omega_cmd_accel * self.dt
        omega_cmd = float(np.clip(speed_limited_cmd, self.prev_omega_cmd - accel_step, self.prev_omega_cmd + accel_step))
        self.last_accel_saturation = int(abs(speed_limited_cmd - omega_cmd) > 1e-10)

        iq_cmd, current_saturated = self.speed_pi.compute_iq_command(
            omega_cmd=omega_cmd,
            omega_meas=float(self.state.omega),
            dt=float(self.dt),
        )
        self.iq_cmd = iq_cmd
        self.last_current_saturation = int(current_saturated)

        self.iq += (self.iq_cmd - self.iq) * min(1.0, self.dt / self.current_time_constant)
        self.iq = float(np.clip(self.iq, -self.Imax, self.Imax))
        self.Te = float(self.torque_constant * self.iq)
        self.T_out = float(self.load_model.motor_to_load_torque(self.Te))
        self.T_L_hat = float(
            self.load_model.params.B_load * self.state.omega
            + self.load_model.coulomb_torque(self.state.omega)
            + self.load_model.gravity_torque(self.state.theta)
            + self.disturbance_torque
        )

        terminated = False
        truncated = False
        done_reason = ""
        self.state = self.load_model.step(
            self.state,
            motor_torque=self.Te,
            disturbance_torque=self.disturbance_torque,
            dt=float(self.dt),
        )
        self.elapsed_steps += 1

        self.last_saturation = int(
            bool(
                self.last_speed_saturation
                or self.last_current_saturation
                or self.last_action_rate_saturation
                or self.last_accel_saturation
                or abs(raw[0] - clipped_action) > 1e-10
            )
        )

        e_theta = float(self.last_trajectory.theta_ref - self.state.theta)
        e_omega = float(self.last_trajectory.omega_ff - self.state.omega)
        if abs(self.state.theta) > self.max_abs_theta:
            terminated = True
            done_reason = "angle_limit"
        if self.elapsed_steps >= self.episode_steps:
            truncated = not terminated
            done_reason = done_reason or "episode_limit"

        reward, terms = self._reward_terms(
            e_theta=e_theta,
            e_omega=e_omega,
            iq=self.iq,
            delta_action=delta_action,
            saturation_flag=self.last_saturation,
            terminal=bool(terminated),
        )
        self.last_reward_terms = terms
        self.prev_action = clipped_action
        self.prev_delta_omega = limited_delta_omega
        self.prev_omega_cmd = omega_cmd

        obs = self._build_observation()
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
            terminated = True
            done_reason = done_reason or "non_finite_observation"
            reward = -1.0

        info = self._build_info(reward=reward, done_reason=done_reason)
        info["reward_components"] = dict(terms)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

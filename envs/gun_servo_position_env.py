"""Custom gun/fire-control servo position environment.

The RL policy controls the position outer loop. Its one-dimensional action is a
load-side speed correction, while the environment simulates the shared safety
post-processing, speed PI inner loop, motor torque/current limits, reducer
torque limits, load dynamics, and encoder measurement path.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from baselines.pi_speed_controller import PISpeedController, PISpeedControllerConfig
from envs.gun_load_model import GunLoadParams, GunLoadState, SingleInertiaGunLoad
from envs.gun_servo_components import (
    DEG_TO_RAD,
    RAD_TO_DEG,
    EncoderSpec,
    GearboxSpec,
    LoadEncoder,
    LoadSpec,
    MotorSpec,
    ServoDriveSpec,
    deg_to_rad,
)
from envs.obs_parser import GUN_SERVO_ACTION_NAMES, GUN_SERVO_OBSERVATION_NAMES
from envs.trajectory_generator import GunServoTrajectoryGenerator, TrajectoryPoint

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
        "reward_action",
        "reward_torque",
        "reward_safety",
        "reward_progress",
        "reward_speed_limit",
        "reward_command_speed_limit",
        "reward_flag_u",
        "reward_flag_e",
        "reward_action_magnitude",
        "reward_action_smoothness",
        "reward_saturation",
        "reward_terminal",
        "reward_total",
    )

    def __init__(
        self,
        *,
        env_id: str = "Custom-GunServo-Position-v0",
        gun_servo_env: dict[str, Any] | None = None,
        motor: dict[str, Any] | None = None,
        servo_drive: dict[str, Any] | None = None,
        gearbox: dict[str, Any] | None = None,
        load: dict[str, Any] | None = None,
        load_encoder: dict[str, Any] | None = None,
        reference: dict[str, Any] | None = None,
        rl_controller: dict[str, Any] | None = None,
        rl_action: dict[str, Any] | None = None,
        reward: dict[str, Any] | None = None,
        safety: dict[str, Any] | None = None,
        normalization: dict[str, Any] | None = None,
        domain_randomization: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        speed_controller: dict[str, Any] | None = None,
        apply_domain_randomization: bool = False,
    ) -> None:
        super().__init__()
        self.env_id = str(env_id)
        self.is_gun_servo_position_env = True

        self.gun_servo_env_cfg = dict(gun_servo_env or {})
        self.motor_cfg = dict(motor or {})
        self.servo_drive_cfg = dict(servo_drive or {})
        self.gearbox_cfg = dict(gearbox or {})
        self.load_cfg = dict(load or {})
        self.load_encoder_cfg = dict(load_encoder or {})
        self.reference_cfg = dict(reference or {})
        self.rl_controller_cfg = dict(rl_controller or {})
        self.rl_action_cfg = {**dict(rl_action or {}), **dict(rl_controller or {})}
        self.reward_cfg = dict(reward or {})
        self.safety_cfg = dict(safety or {})
        self.normalization_cfg = dict(normalization or {})
        self.domain_randomization = self._normalize_domain_randomization(domain_randomization or {})
        self.environment_cfg = dict(environment or {})
        self.apply_domain_randomization = bool(apply_domain_randomization)

        self.motor_spec = MotorSpec.from_config(self.motor_cfg)
        gearbox_source = {**self.load_cfg, **self.gearbox_cfg}
        self.gearbox_spec = GearboxSpec.from_config(gearbox_source)
        self.active_gearbox_spec = self.gearbox_spec
        self.nominal_load_spec = LoadSpec.from_config(self.load_cfg)
        self.active_load_spec = self.nominal_load_spec
        self.nominal_encoder_spec = EncoderSpec.from_config(self.load_encoder_cfg)
        self.encoder = LoadEncoder(self.nominal_encoder_spec)
        self.servo_drive_spec = ServoDriveSpec.from_config(self.servo_drive_cfg, motor=self.motor_spec)

        self.p = float(self.motor_spec.pole_pairs)
        self.psi_f = float(self.motor_cfg.get("psi_f", self.motor_spec.kt_nm_per_a / max(1.5 * self.p, 1e-9)))
        drive_current_limit = float(
            self.servo_drive_cfg.get(
                "max_output_current_a",
                self.servo_drive_cfg.get("output_current_limit_a", self.servo_drive_spec.current_limit_a),
            )
        )
        self.Imax = float(min(self.motor_spec.max_current_a, self.servo_drive_spec.current_limit_a, drive_current_limit))
        self.Vdc_nominal = float(
            self.normalization_cfg.get(
                "vdc_nominal_v",
                self.servo_drive_cfg.get("vdc_nominal_v", self.motor_cfg.get("Vdc", 540.0)),
            )
        )
        self.Vdc = float(self.Vdc_nominal)
        self.Ts_current = float(self.motor_cfg.get("Ts_current", self.gun_servo_env_cfg.get("integration_dt", 1e-4)))
        self.Ts_speed = float(self.motor_cfg.get("Ts_speed", self.gun_servo_env_cfg.get("dt", 1e-3)))
        self.dt = float(
            self.gun_servo_env_cfg.get(
                "dt",
                self.motor_cfg.get("Ts_position", self.motor_cfg.get("Ts", 1e-3)),
            )
        )
        self.integration_dt = max(
            min(float(self.gun_servo_env_cfg.get("integration_dt", self.Ts_current)), self.dt),
            1e-7,
        )
        self.current_time_constant = max(
            float(self.motor_cfg.get("current_time_constant", self.servo_drive_spec.torque_filter_time_s)),
            0.0,
        )
        self.torque_filter_time_s = max(float(self.servo_drive_spec.torque_filter_time_s), 0.0)
        self.torque_constant = float(self.motor_spec.kt_nm_per_a)
        self.max_motor_torque = float(self.motor_spec.max_torque_nm)
        self.max_motor_speed = float(self.motor_spec.max_speed_rad_s)

        self.episode_steps = int(
            self.gun_servo_env_cfg.get("episode_steps", self.environment_cfg.get("episode_steps", 1500))
        )
        self.init_theta_range = self._deg_range(
            self.environment_cfg.get("init_theta_range_deg", (0.0, 0.0)),
            default=(0.0, 0.0),
        )
        self.init_omega_range = self._deg_range(
            self.environment_cfg.get("init_omega_range_deg_s", (0.0, 0.0)),
            default=(0.0, 0.0),
        )
        theta_lo, theta_hi = self.active_load_spec.theta_range_rad
        self.max_abs_theta = float(
            deg_to_rad(
                float(
                    self.environment_cfg.get(
                        "max_abs_theta_deg",
                        max(abs(theta_lo), abs(theta_hi)) * RAD_TO_DEG,
                    )
                )
            )
        )

        self.theta_scale = max(
            abs(
                float(
                    deg_to_rad(
                        float(
                            self.normalization_cfg.get(
                                "theta_scale_deg",
                                self.reference_cfg.get("theta_scale_deg", 30.0),
                            )
                        )
                    )
                )
            ),
            1e-6,
        )
        self.omega_scale = max(
            abs(
                float(
                    deg_to_rad(
                        float(
                            self.normalization_cfg.get(
                                "omega_scale_deg_s",
                                self.reference_cfg.get(
                                    "omega_scale_deg_s",
                                    max(self.active_load_spec.omega_limit_rad_s * RAD_TO_DEG, 30.0),
                                ),
                            )
                        )
                    )
                )
            ),
            1e-6,
        )
        self.torque_scale = max(
            float(
                self.normalization_cfg.get(
                    "torque_scale_nm",
                    self.load_cfg.get("torque_scale", self.gearbox_spec.output_torque_limit_nm),
                )
            ),
            1e-6,
        )
        self.current_scale = max(
            float(self.normalization_cfg.get("current_scale_a", self.Imax)),
            1e-6,
        )
        self.vdc_scale = max(float(self.normalization_cfg.get("vdc_nominal_v", self.Vdc_nominal)), 1e-6)

        self.max_delta_omega = abs(
            float(
                deg_to_rad(
                    float(
                        self.rl_action_cfg.get(
                            "delta_omega_limit_deg_s",
                            self.rl_action_cfg.get("max_delta_omega_deg_s", 30.0),
                        )
                    )
                )
            )
        )
        self.max_omega_cmd = abs(
            float(
                deg_to_rad(
                    float(
                        self.rl_action_cfg.get(
                            "omega_limit_deg_s",
                            self.rl_action_cfg.get(
                                "max_omega_cmd_deg_s",
                                self.active_load_spec.omega_limit_rad_s * RAD_TO_DEG,
                            ),
                        )
                    )
                )
            )
        )
        self.max_delta_omega_rate = abs(
            float(
                deg_to_rad(
                    float(
                        self.rl_action_cfg.get(
                            "delta_omega_rate_limit_deg_s2",
                            self.rl_action_cfg.get("max_delta_omega_rate_deg_s2", 600.0),
                        )
                    )
                )
            )
        )
        self.max_omega_cmd_accel = abs(
            float(
                deg_to_rad(
                    float(
                        self.rl_action_cfg.get(
                            "omega_cmd_accel_limit_deg_s2",
                            self.rl_action_cfg.get(
                                "max_omega_cmd_accel_deg_s2",
                                self.active_load_spec.alpha_limit_rad_s2 * RAD_TO_DEG,
                            ),
                        )
                    )
                )
            )
        )
        self.command_smoothing_time_s = max(float(self.rl_action_cfg.get("command_smoothing_time_s", 0.0)), 0.0)
        self.action_type = str(self.rl_action_cfg.get("action_type", self.rl_action_cfg.get("type", "load_speed_delta")))
        self.enable_u_safe = bool(self.safety_cfg.get("enable_u_safe", True))
        self.enable_e_safe = bool(self.safety_cfg.get("enable_e_safe", True))
        self.enable_x_safe = bool(self.safety_cfg.get("enable_x_safe", True))
        self.lambda_smooth = float(self.safety_cfg.get("lambda_smooth", 0.8))
        self.lambda_smooth = min(max(self.lambda_smooth, 0.0), 1.0)
        self.safety_penalty = float(self.safety_cfg.get("safety_penalty", self.reward_cfg.get("w_safety", 10.0)))

        self.speed_pi = PISpeedController(self._make_speed_pi_config(speed_controller or {}))
        self.trajectory = GunServoTrajectoryGenerator.from_config(self.reference_cfg)
        self.nominal_load_params = self._load_params_from_specs(self.gearbox_spec, self.nominal_load_spec)
        self.load_model = SingleInertiaGunLoad(self.nominal_load_params)

        self.reward_mode = str(self.reward_cfg.get("mode", "quadratic_tracking"))
        self.w_theta = float(self.reward_cfg.get("w_theta", 1.0))
        self.w_omega = float(self.reward_cfg.get("w_omega", 0.1))
        self.w_action = float(
            self.reward_cfg.get(
                "action_rate_weight",
                self.reward_cfg.get("w_action", self.reward_cfg.get("w_action_smoothness", 0.02)),
            )
        )
        self.w_torque = float(self.reward_cfg.get("w_torque", 0.02))
        self.w_safety = float(self.reward_cfg.get("w_safety", self.safety_penalty))
        self.w_iq = float(self.reward_cfg.get("w_iq", 0.0))
        self.w_progress = float(self.reward_cfg.get("w_progress", 0.0))
        self.w_speed_limit = float(self.reward_cfg.get("speed_margin_weight", self.reward_cfg.get("w_speed_limit", 0.0)))
        self.w_command_speed_limit = float(
            self.reward_cfg.get("command_speed_margin_weight", self.reward_cfg.get("command_speed_limit_weight", 0.0))
        )
        self.w_flag_u = float(self.reward_cfg.get("safety_flag_u_weight", self.reward_cfg.get("w_flag_u", 0.0)))
        self.w_flag_e = float(self.reward_cfg.get("safety_flag_e_weight", self.reward_cfg.get("w_flag_e", 0.0)))
        self.w_action_magnitude = float(
            self.reward_cfg.get("residual_action_weight", self.reward_cfg.get("action_magnitude_weight", 0.0))
        )
        self.w_saturation = float(self.reward_cfg.get("command_saturation_weight", self.reward_cfg.get("w_saturation", 0.0)))
        self.w_terminal = float(self.reward_cfg.get("unsafe_done_weight", self.reward_cfg.get("w_terminal", 0.0)))
        self.speed_limit_soft_ratio = float(np.clip(float(self.reward_cfg.get("speed_limit_soft_ratio", 0.85)), 0.0, 1.0))
        self.w_action_smoothness = float(self.w_action)
        self.action_smoothness_enabled = True
        self.action_smoothness_weight = float(self.w_action)

        baseline_pid = dict(self.rl_controller_cfg.get("baseline_pid", {}) or {})
        self.baseline_position_kp = float(baseline_pid.get("kp", self.rl_controller_cfg.get("pid_kp", 8.0)))
        self.baseline_position_ki = float(baseline_pid.get("ki", self.rl_controller_cfg.get("pid_ki", 0.0)))
        self.baseline_position_kd = float(baseline_pid.get("kd", self.rl_controller_cfg.get("pid_kd", 1.2)))

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
        self.encoder_noise_std = float(self.nominal_encoder_spec.noise_std_rad)
        self.theta_meas = 0.0
        self.omega_meas = 0.0
        self.prev_action = 0.0
        self.prev_a_safe = 0.0
        self.prev_delta_omega = 0.0
        self.prev_abs_e_theta = 0.0
        self.prev_omega_cmd = 0.0
        self.last_trajectory = TrajectoryPoint(0.0, 0.0, 0.0)
        self.last_action_raw = 0.0
        self.last_action_clip = 0.0
        self.last_action_safe = 0.0
        self.last_delta_a_safe = 0.0
        self.last_omega_cmd_raw = 0.0
        self.last_omega_cmd_safe = 0.0
        self.last_omega_m_cmd = 0.0
        self.last_iq_ref_raw = 0.0
        self.last_iq_ref = 0.0
        self.last_torque_raw = 0.0
        self.last_t_out_raw = 0.0
        self.last_flag_u_safe = 0
        self.last_flag_e_safe = 0
        self.last_flag_x_safe = 0
        self.last_sigma_safe = 0
        self.last_saturation = 0
        self.last_speed_saturation = 0
        self.last_current_saturation = 0
        self.last_action_rate_saturation = 0
        self.last_accel_saturation = 0
        self.last_motor_torque_saturation = 0
        self.last_gear_torque_saturation = 0
        self.last_constraint_violation = 0
        self.last_reward_terms: OrderedDict[str, float] = OrderedDict()
        self.last_randomization_sample: dict[str, float | bool] = {}
        self.mode_code = 0.0

    @staticmethod
    def _deg_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
        raw = default if value is None else value
        lo, hi = float(raw[0]), float(raw[1])
        if lo > hi:
            lo, hi = hi, lo
        return float(deg_to_rad(lo)), float(deg_to_rad(hi))

    @staticmethod
    def _float_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
        raw = default if value is None else value
        lo, hi = float(raw[0]), float(raw[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    @staticmethod
    def _optional_float_range(value: Any) -> tuple[float, float] | None:
        if value is None:
            return None
        lo, hi = float(value[0]), float(value[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    @staticmethod
    def _load_params_from_specs(gearbox: GearboxSpec, load: LoadSpec) -> GunLoadParams:
        return GunLoadParams(
            gear_ratio=float(gearbox.ratio),
            efficiency=float(gearbox.efficiency_nominal),
            J_load=float(load.inertia_kg_m2),
            B_load=float(load.viscous_damping_nms_per_rad),
            coulomb_friction=float(load.coulomb_friction_nm),
            gravity_torque_coeff=float(load.gravity_torque_coeff_nm),
            gravity_phase=float(load.gravity_phase_rad),
            backlash_rad=float(gearbox.backlash_rad),
            torsional_stiffness=float(gearbox.torsional_stiffness_nm_per_rad),
            torsional_damping=float(gearbox.torsional_damping_nms_per_rad),
            output_torque_limit=float(gearbox.output_torque_limit_nm),
            friction_velocity_eps=float(load.friction_eps_rad_s),
        )

    def _make_speed_pi_config(self, speed_controller: dict[str, Any]) -> PISpeedControllerConfig:
        ratio = max(float(self.gearbox_spec.ratio), 1e-9)
        reflected_inertia = float(self.motor_spec.jm_kg_m2) + float(self.nominal_load_spec.inertia_kg_m2) / ratio**2
        bandwidth = 2.0 * np.pi * float(self.servo_drive_spec.speed_loop_gain_hz)
        kp_default = reflected_inertia * bandwidth / max(float(self.motor_spec.kt_nm_per_a), 1e-9)
        ki_default = kp_default / max(float(self.servo_drive_spec.speed_loop_integral_time_s), 1e-9)
        defaults = {
            "kp": kp_default,
            "ki": ki_default,
            "integrator_limit": 0.75 * self.Imax,
            "iq_limit": self.Imax,
        }
        cfg = {**defaults, **dict(speed_controller or {})}
        return PISpeedControllerConfig(
            kp=float(cfg["kp"]),
            ki=float(cfg["ki"]),
            integrator_limit=float(cfg["integrator_limit"]),
            iq_limit=float(cfg["iq_limit"]),
        )

    def _normalize_domain_randomization(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(config.get("enabled", False)),
            "J_load_scale_range": self._float_range(
                config.get("J_load_scale_range", config.get("inertia_scale_range")),
                default=(1.0, 1.0),
            ),
            "inertia_range": self._optional_float_range(config.get("inertia_range")),
            "B_load_scale_range": self._float_range(
                config.get("B_load_scale_range", config.get("damping_scale_range")),
                default=(1.0, 1.0),
            ),
            "damping_range": self._optional_float_range(config.get("damping_range")),
            "friction_scale_range": self._float_range(config.get("friction_scale_range"), default=(1.0, 1.0)),
            "coulomb_friction_range": self._optional_float_range(config.get("coulomb_friction_range")),
            "gravity_torque_scale_range": self._float_range(
                config.get("gravity_torque_scale_range"),
                default=(1.0, 1.0),
            ),
            "gravity_torque_range": self._optional_float_range(config.get("gravity_torque_range")),
            "disturbance_torque_range": self._float_range(
                config.get("disturbance_torque_range", config.get("disturbance_range")),
                default=(0.0, 0.0),
            ),
            "gear_efficiency_range": self._optional_float_range(config.get("gear_efficiency_range")),
            "vdc_scale_range": self._float_range(config.get("vdc_scale_range"), default=(1.0, 1.0)),
            "encoder_noise_deg_range": self._float_range(config.get("encoder_noise_deg_range"), default=(0.0, 0.0)),
            "encoder_noise_rad_range": self._float_range(
                config.get("encoder_noise_rad_range", config.get("encoder_noise_std_range")),
                default=(0.0, 0.0),
            ),
        }

    @property
    def observation_normalization_scales(self) -> dict[str, float]:
        return {
            "theta": float(self.theta_scale),
            "omega": float(self.omega_scale),
            "current": max(float(self.current_scale), 1e-6),
            "torque": float(self.torque_scale),
            "vdc": float(self.vdc_scale),
            "speed_command": max(float(self.max_omega_cmd), 1e-6),
        }

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        lo, hi = value_range
        if lo == hi:
            return float(lo)
        return float(self.np_random.uniform(lo, hi))

    def _sample_optional_config_range(self, key: str) -> float | None:
        value = self.load_cfg.get(key)
        if value is None:
            return None
        return self._sample_uniform(self._float_range(value, default=(0.0, 0.0)))

    def _sample_domain_randomization(self) -> None:
        load_spec = replace(self.nominal_load_spec)
        gearbox_spec = replace(self.gearbox_spec)
        vdc_scale = 1.0
        active = bool(self.apply_domain_randomization and self.domain_randomization["enabled"])
        if active:
            j_scale = self._sample_uniform(self.domain_randomization["J_load_scale_range"])
            b_scale = self._sample_uniform(self.domain_randomization["B_load_scale_range"])
            f_scale = self._sample_uniform(self.domain_randomization["friction_scale_range"])
            g_scale = self._sample_uniform(self.domain_randomization["gravity_torque_scale_range"])
            inertia = (
                self._sample_uniform(self.domain_randomization["inertia_range"])
                if self.domain_randomization["inertia_range"] is not None
                else float(load_spec.inertia_kg_m2) * j_scale
            )
            damping = (
                self._sample_uniform(self.domain_randomization["damping_range"])
                if self.domain_randomization["damping_range"] is not None
                else float(load_spec.viscous_damping_nms_per_rad) * b_scale
            )
            friction = (
                self._sample_uniform(self.domain_randomization["coulomb_friction_range"])
                if self.domain_randomization["coulomb_friction_range"] is not None
                else float(load_spec.coulomb_friction_nm) * f_scale
            )
            gravity = (
                self._sample_uniform(self.domain_randomization["gravity_torque_range"])
                if self.domain_randomization["gravity_torque_range"] is not None
                else float(load_spec.gravity_torque_coeff_nm) * g_scale
            )
            load_spec = replace(
                load_spec,
                inertia_kg_m2=float(inertia),
                viscous_damping_nms_per_rad=float(damping),
                coulomb_friction_nm=float(friction),
                gravity_torque_coeff_nm=float(gravity),
                disturbance_torque_nm=self._sample_uniform(self.domain_randomization["disturbance_torque_range"]),
            )
            if self.domain_randomization["gear_efficiency_range"] is not None:
                gearbox_spec = replace(
                    gearbox_spec,
                    efficiency_nominal=self._sample_uniform(self.domain_randomization["gear_efficiency_range"]),
                )
            vdc_scale = self._sample_uniform(self.domain_randomization["vdc_scale_range"])
            if self.domain_randomization["encoder_noise_rad_range"] != (0.0, 0.0):
                noise_rad = self._sample_uniform(self.domain_randomization["encoder_noise_rad_range"])
            else:
                noise_deg = self._sample_uniform(self.domain_randomization["encoder_noise_deg_range"])
                noise_rad = deg_to_rad(noise_deg)
        else:
            j_scale = b_scale = f_scale = g_scale = 1.0
            noise_rad = float(self.nominal_encoder_spec.noise_std_rad)

        disturbance_step = self._sample_optional_config_range("disturbance_step_nm_range")
        disturbance_step_time = self._sample_optional_config_range("disturbance_step_time_s_range")
        if disturbance_step is not None or disturbance_step_time is not None:
            load_spec = replace(
                load_spec,
                disturbance_step_nm=(
                    float(load_spec.disturbance_step_nm)
                    if disturbance_step is None
                    else float(disturbance_step)
                ),
                disturbance_step_time_s=(
                    float(load_spec.disturbance_step_time_s)
                    if disturbance_step_time is None
                    else float(disturbance_step_time)
                ),
            )

        self.active_load_spec = load_spec
        self.active_gearbox_spec = gearbox_spec
        self.Vdc = float(self.Vdc_nominal) * float(vdc_scale)
        self.mode_code = 1.0 if active else 0.0
        self.load_model = SingleInertiaGunLoad(self._load_params_from_specs(self.active_gearbox_spec, load_spec))
        active_encoder_spec = replace(self.nominal_encoder_spec, noise_std_rad=float(noise_rad))
        self.encoder = LoadEncoder(active_encoder_spec)
        self.encoder_noise_std = float(noise_rad)
        self.last_randomization_sample = {
            "domain_randomization_active": active,
            "J_load_scale": float(j_scale),
            "B_load_scale": float(b_scale),
            "friction_scale": float(f_scale),
            "gravity_torque_scale": float(g_scale),
            "disturbance_torque": float(load_spec.disturbance_torque_nm),
            "disturbance_step": float(load_spec.disturbance_step_nm),
            "disturbance_step_time": float(load_spec.disturbance_step_time_s),
            "encoder_noise_std": float(self.encoder_noise_std),
            "gear_efficiency": float(self.active_gearbox_spec.efficiency_nominal),
            "vdc_scale": float(vdc_scale),
        }

    def _update_measurement(self) -> None:
        self.theta_meas, self.omega_meas = self.encoder.measure(
            theta=float(self.state.theta),
            omega=float(self.state.omega),
            rng=self.np_random,
        )

    def _current_trajectory(self) -> TrajectoryPoint:
        return self.trajectory.sample(float(self.elapsed_steps) * float(self.dt))

    def _build_observation(self) -> np.ndarray:
        traj = self._current_trajectory()
        e_theta = float(traj.theta_ref - self.state.theta)
        e_omega = float(traj.omega_ff - self.state.omega)
        obs = np.asarray(
            [
                e_theta / self.theta_scale,
                e_omega / self.omega_scale,
                traj.theta_ref / self.theta_scale,
                self.state.theta / self.theta_scale,
                self.state.omega / self.omega_scale,
                self.T_L_hat / self.torque_scale,
                self.iq / max(float(self.current_scale), 1e-6),
                self.Vdc / self.vdc_scale,
                self.prev_a_safe,
                self.mode_code,
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
        delta_a_safe: float,
        t_out: float,
        sigma_safe: int,
        omega_l: float,
        unsafe_terminal: bool,
    ) -> tuple[float, OrderedDict[str, float]]:
        e_theta_n = float(e_theta) / self.theta_scale
        e_omega_n = float(e_omega) / self.omega_scale
        t_out_n = float(t_out) / max(float(self.active_gearbox_spec.output_torque_limit_nm), 1e-6)
        progress_n = (float(self.prev_abs_e_theta) - abs(float(e_theta))) / self.theta_scale
        speed_limit = max(float(self.active_load_spec.omega_limit_rad_s), 1e-9)
        speed_ratio = abs(float(omega_l)) / speed_limit
        speed_margin = max(0.0, (speed_ratio - self.speed_limit_soft_ratio) / max(1.0 - self.speed_limit_soft_ratio, 1e-6))
        command_ratio = abs(float(self.last_omega_cmd_safe)) / speed_limit
        command_margin = max(
            0.0,
            (command_ratio - self.speed_limit_soft_ratio) / max(1.0 - self.speed_limit_soft_ratio, 1e-6),
        )
        saturation_active = float(
            bool(
                self.last_speed_saturation
                or self.last_accel_saturation
                or self.last_current_saturation
                or self.last_motor_torque_saturation
                or self.last_gear_torque_saturation
            )
        )
        reward_theta = -self.w_theta * e_theta_n**2
        reward_omega = -self.w_omega * e_omega_n**2
        reward_action = -self.w_action * float(delta_a_safe) ** 2
        reward_torque = -self.w_torque * t_out_n**2
        reward_safety = -self.w_safety * float(sigma_safe)
        reward_progress = self.w_progress * progress_n
        reward_speed_limit = -self.w_speed_limit * speed_margin**2
        reward_command_speed_limit = -self.w_command_speed_limit * command_margin**2
        reward_flag_u = -self.w_flag_u * float(self.last_flag_u_safe)
        reward_flag_e = -self.w_flag_e * float(self.last_flag_e_safe)
        reward_action_magnitude = -self.w_action_magnitude * float(self.last_action_safe) ** 2
        reward_saturation = -self.w_saturation * saturation_active
        reward_terminal = -self.w_terminal * float(bool(unsafe_terminal))
        total = float(
            reward_theta
            + reward_omega
            + reward_action
            + reward_torque
            + reward_safety
            + reward_progress
            + reward_speed_limit
            + reward_command_speed_limit
            + reward_flag_u
            + reward_flag_e
            + reward_action_magnitude
            + reward_saturation
            + reward_terminal
        )
        terms = OrderedDict(
            reward_theta=float(reward_theta),
            reward_omega=float(reward_omega),
            reward_action=float(reward_action),
            reward_torque=float(reward_torque),
            reward_safety=float(reward_safety),
            reward_progress=float(reward_progress),
            reward_speed_limit=float(reward_speed_limit),
            reward_command_speed_limit=float(reward_command_speed_limit),
            reward_flag_u=float(reward_flag_u),
            reward_flag_e=float(reward_flag_e),
            reward_action_magnitude=float(reward_action_magnitude),
            reward_action_smoothness=float(reward_action),
            reward_saturation=float(reward_saturation),
            reward_terminal=float(reward_terminal),
            reward_total=total,
        )
        return total, terms

    def _disturbance_for_step(self) -> float:
        spec = self.active_load_spec
        time_s = float(self.elapsed_steps) * float(self.dt)
        disturbance = float(spec.disturbance_torque_nm)
        if time_s >= float(spec.disturbance_step_time_s):
            disturbance += float(spec.disturbance_step_nm)
        if float(spec.disturbance_noise_std_nm) > 0.0:
            disturbance += float(self.np_random.normal(0.0, float(spec.disturbance_noise_std_nm)))
        return float(disturbance)

    def _estimate_load_torque(self, *, disturbance_torque: float) -> float:
        params = self.load_model.params
        return float(
            params.B_load * self.state.omega
            + self.load_model.coulomb_torque(self.state.omega)
            + self.load_model.gravity_torque(self.state.theta)
            + float(disturbance_torque)
        )

    def _build_info(self, *, reward: float = 0.0, done_reason: str = "") -> dict[str, Any]:
        traj = self.last_trajectory
        e_theta = float(traj.theta_ref - self.state.theta)
        e_omega = float(traj.omega_ff - self.state.omega)
        theta_meas_error = float(traj.theta_ref - self.theta_meas)
        raw_state = [
            e_theta,
            e_omega,
            float(traj.theta_ref),
            float(self.state.theta),
            float(self.state.omega),
            float(self.T_L_hat),
            float(self.iq),
            float(self.Vdc),
            float(self.prev_a_safe),
            float(self.mode_code),
        ]
        normalized_state = self._build_observation().astype(float).tolist()
        return {
            "env_id": self.env_id,
            "action_type": self.action_type,
            "done_reason": done_reason,
            "reward_mode": self.reward_mode,
            "elapsed_steps": int(self.elapsed_steps),
            "time_s": float(self.elapsed_steps) * float(self.dt),
            "t_s": float(self.elapsed_steps) * float(self.dt),
            "raw_state": raw_state,
            "normalized_state": normalized_state,
            "theta_ref": float(traj.theta_ref),
            "theta_ref_rad": float(traj.theta_ref),
            "theta_L": float(self.state.theta),
            "theta_L_rad": float(self.state.theta),
            "theta_load": float(self.state.theta),
            "theta_meas": float(self.theta_meas),
            "e_theta": e_theta,
            "e_theta_rad": e_theta,
            "e_theta_meas": theta_meas_error,
            "omega_ref": float(traj.omega_ff),
            "omega_L_ref": float(traj.omega_ff),
            "omega_ff": float(traj.omega_ff),
            "alpha_ff": float(traj.alpha_ff),
            "omega_L": float(self.state.omega),
            "omega_L_rad_s": float(self.state.omega),
            "omega_load": float(self.state.omega),
            "omega_meas": float(self.omega_meas),
            "omega_cmd": float(self.prev_omega_cmd),
            "omega_L_cmd_raw": float(self.last_omega_cmd_raw),
            "omega_L_cmd_safe": float(self.last_omega_cmd_safe),
            "omega_m_cmd": float(self.last_omega_m_cmd),
            "omega_m_ref": float(self.last_omega_m_cmd),
            "omega_m_ref_rad_s": float(self.last_omega_m_cmd),
            "omega_m_rad_s": float(self.active_gearbox_spec.ratio * self.state.omega),
            "e_omega_load": e_omega,
            "iq": float(self.iq),
            "i_q": float(self.iq),
            "iq_cmd": float(self.iq_cmd),
            "i_q_ref_raw": float(self.last_iq_ref_raw),
            "i_q_ref": float(self.last_iq_ref),
            "Te": float(self.Te),
            "T_e": float(self.Te),
            "T_e_raw": float(self.last_torque_raw),
            "T_out": float(self.T_out),
            "T_out_raw": float(self.last_t_out_raw),
            "T_L": float(self.disturbance_torque),
            "T_L_hat": float(self.T_L_hat),
            "T_hat_L": float(self.T_L_hat),
            "TL_hat": float(self.T_L_hat),
            "disturbance_torque": float(self.disturbance_torque),
            "V_dc": float(self.Vdc),
            "action_raw": float(self.last_action_raw),
            "a_raw": float(self.last_action_raw),
            "a_clip": float(self.last_action_clip),
            "action_safe": float(self.last_action_safe),
            "a_safe": float(self.last_action_safe),
            "a_safe_prev": float(self.prev_a_safe),
            "delta_a_safe": float(self.last_delta_a_safe),
            "m_k": float(self.mode_code),
            "flag_U_safe": int(self.last_flag_u_safe),
            "flag_E_safe": int(self.last_flag_e_safe),
            "flag_X_safe": int(self.last_flag_x_safe),
            "sigma_safe": int(self.last_sigma_safe),
            "saturation_flag": int(self.last_saturation),
            "speed_saturation_flag": int(self.last_speed_saturation),
            "current_saturation_flag": int(self.last_current_saturation),
            "action_rate_saturation_flag": int(self.last_action_rate_saturation),
            "accel_saturation_flag": int(self.last_accel_saturation),
            "motor_torque_saturation_flag": int(self.last_motor_torque_saturation),
            "gear_torque_saturation_flag": int(self.last_gear_torque_saturation),
            "constraint_violation": int(self.last_constraint_violation),
            "theta_ref_deg": float(traj.theta_ref * RAD_TO_DEG),
            "theta_L_deg": float(self.state.theta * RAD_TO_DEG),
            "theta_load_deg": float(self.state.theta * RAD_TO_DEG),
            "theta_meas_deg": float(self.theta_meas * RAD_TO_DEG),
            "e_theta_deg": float(e_theta * RAD_TO_DEG),
            "e_theta_meas_deg": float(theta_meas_error * RAD_TO_DEG),
            "omega_ref_deg_s": float(traj.omega_ff * RAD_TO_DEG),
            "omega_L_deg_s": float(self.state.omega * RAD_TO_DEG),
            "omega_load_deg_s": float(self.state.omega * RAD_TO_DEG),
            "omega_meas_deg_s": float(self.omega_meas * RAD_TO_DEG),
            "omega_cmd_deg_s": float(self.prev_omega_cmd * RAD_TO_DEG),
            "iq_A": float(self.iq),
            "torque_motor_nm": float(self.Te),
            "Te_Nm": float(self.Te),
            "T_e_Nm": float(self.Te),
            "torque_out_nm": float(self.T_out),
            "T_out_Nm": float(self.T_out),
            "T_out_max_Nm": float(self.active_gearbox_spec.output_torque_limit_nm),
            "TL_Nm": float(self.T_L_hat),
            "disturbance_nm": float(self.disturbance_torque),
            "disturbance_torque_Nm": float(self.disturbance_torque),
            "disturbance_step_Nm": float(self.active_load_spec.disturbance_step_nm),
            "disturbance_step_time_s": float(self.active_load_spec.disturbance_step_time_s),
            "reward": float(reward),
            "reward_terms": dict(self.last_reward_terms),
            "domain_randomization_active": bool(self.last_randomization_sample.get("domain_randomization_active", False)),
            "encoder_noise_std_rad": float(self.encoder_noise_std),
            "active_J_load": float(self.load_model.params.J_load),
            "active_B_load": float(self.load_model.params.B_load),
            "active_coulomb_friction": float(self.load_model.params.coulomb_friction),
            "active_gravity_torque_coeff": float(self.load_model.params.gravity_torque_coeff),
            "gear_ratio": float(self.active_gearbox_spec.ratio),
            "gear_efficiency": float(self.active_gearbox_spec.efficiency_nominal),
            "gear_output_torque_limit_nm": float(self.active_gearbox_spec.output_torque_limit_nm),
            "sampled_theta_initial_deg": float(self.trajectory.theta_initial * RAD_TO_DEG),
            "sampled_theta_final_deg": float(self.trajectory.theta_final * RAD_TO_DEG),
            "sampled_step_time_s": float(getattr(self.trajectory, "active_step_time", 0.0)),
            "sampled_max_speed_deg_s": float(getattr(self.trajectory, "active_max_speed", 0.0) * RAD_TO_DEG),
            "sampled_max_accel_deg_s2": float(getattr(self.trajectory, "active_max_accel", 0.0) * RAD_TO_DEG),
            "sampled_sine_amplitude_deg": float(getattr(self.trajectory, "active_sine_amplitude", 0.0) * RAD_TO_DEG),
            "sampled_sine_frequency_hz": float(getattr(self.trajectory, "active_sine_frequency_hz", 0.0)),
            "sampled_sine_phase_rad": float(getattr(self.trajectory, "active_sine_phase", 0.0)),
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
        self.disturbance_torque = float(self.active_load_spec.disturbance_torque_nm)
        self.T_L_hat = self._estimate_load_torque(disturbance_torque=self.disturbance_torque)
        self.prev_action = 0.0
        self.prev_a_safe = 0.0
        self.prev_delta_omega = 0.0
        self.prev_omega_cmd = 0.0
        self.last_trajectory = self._current_trajectory()
        self.prev_abs_e_theta = abs(float(self.last_trajectory.theta_ref - self.state.theta))
        self.last_action_raw = 0.0
        self.last_action_clip = 0.0
        self.last_action_safe = 0.0
        self.last_delta_a_safe = 0.0
        self.last_omega_cmd_raw = 0.0
        self.last_omega_cmd_safe = 0.0
        self.last_omega_m_cmd = 0.0
        self.last_iq_ref_raw = 0.0
        self.last_iq_ref = 0.0
        self.last_torque_raw = 0.0
        self.last_t_out_raw = 0.0
        self.last_flag_u_safe = 0
        self.last_flag_e_safe = 0
        self.last_flag_x_safe = 0
        self.last_sigma_safe = 0
        self.last_saturation = 0
        self.last_speed_saturation = 0
        self.last_current_saturation = 0
        self.last_action_rate_saturation = 0
        self.last_accel_saturation = 0
        self.last_motor_torque_saturation = 0
        self.last_gear_torque_saturation = 0
        self.last_constraint_violation = 0
        self.last_reward_terms = OrderedDict()
        self.speed_pi.reset()
        self.encoder.reset(theta=float(self.state.theta), omega=float(self.state.omega))
        self._update_measurement()
        return self._build_observation(), self._build_info(done_reason="")

    def _safe_speed_command(self, action_clip: float, raw_action: float) -> tuple[float, float, float]:
        action_clip_saturated = int(abs(float(raw_action) - float(action_clip)) > 1e-10)
        self.last_action_clip = float(action_clip)
        raw_delta_omega = float(action_clip) * float(self.max_delta_omega)
        raw_omega_cmd = float(self.last_trajectory.omega_ff + raw_delta_omega)
        self.last_omega_cmd_raw = raw_omega_cmd
        speed_limit = max(
            1e-9,
            min(
                float(self.max_omega_cmd),
                float(self.active_load_spec.omega_limit_rad_s),
                self.max_motor_speed / self.active_gearbox_spec.ratio,
            ),
        )
        if self.enable_u_safe:
            speed_limited_cmd = float(np.clip(raw_omega_cmd, -speed_limit, speed_limit))
        else:
            speed_limited_cmd = raw_omega_cmd
        self.last_speed_saturation = int(abs(raw_omega_cmd - speed_limited_cmd) > 1e-10)

        if self.enable_u_safe and self.max_omega_cmd_accel > 0.0:
            accel_step = self.max_omega_cmd_accel * self.dt
            accel_limited_cmd = float(
                np.clip(speed_limited_cmd, self.prev_omega_cmd - accel_step, self.prev_omega_cmd + accel_step)
            )
        else:
            accel_limited_cmd = speed_limited_cmd
        self.last_accel_saturation = int(abs(speed_limited_cmd - accel_limited_cmd) > 1e-10)
        self.last_action_rate_saturation = self.last_accel_saturation

        omega_cmd = accel_limited_cmd
        if self.enable_u_safe:
            omega_cmd = float(
                self.lambda_smooth * accel_limited_cmd + (1.0 - self.lambda_smooth) * self.prev_omega_cmd
            )
        elif self.command_smoothing_time_s > 0.0:
            alpha = float(self.dt / (self.command_smoothing_time_s + self.dt))
            omega_cmd = float(self.prev_omega_cmd + alpha * (accel_limited_cmd - self.prev_omega_cmd))
        self.last_omega_cmd_safe = float(omega_cmd)
        self.last_omega_m_cmd = float(self.active_gearbox_spec.ratio * omega_cmd)
        self.last_action_safe = float(np.clip(omega_cmd / speed_limit, ACTION_LOW, ACTION_HIGH))
        self.last_delta_a_safe = float(self.last_action_safe - self.prev_a_safe)
        self.last_flag_u_safe = int(
            bool(
                action_clip_saturated
                or self.last_speed_saturation
                or self.last_accel_saturation
            )
        )
        return raw_delta_omega, omega_cmd, self.last_action_safe

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw = np.asarray(action, dtype=np.float64).reshape(1)
        if not np.isfinite(raw).all():
            return self._build_observation(), -1.0, True, False, self._build_info(done_reason="non_finite_action")

        clipped_action = float(np.clip(raw[0], ACTION_LOW, ACTION_HIGH))
        self.last_action_raw = float(raw[0])
        self.last_trajectory = self._current_trajectory()
        _raw_delta_omega, omega_cmd, _a_safe = self._safe_speed_command(clipped_action, raw_action=float(raw[0]))
        omega_m_cmd = float(self.active_gearbox_spec.ratio * omega_cmd)
        self.disturbance_torque = self._disturbance_for_step()

        self.last_current_saturation = 0
        self.last_motor_torque_saturation = 0
        self.last_gear_torque_saturation = 0
        self.last_flag_e_safe = 0
        substeps = max(1, int(np.ceil(float(self.dt) / float(self.integration_dt))))
        sub_dt = float(self.dt) / float(substeps)
        for _ in range(substeps):
            omega_m_meas = float(self.active_gearbox_spec.ratio * self.state.omega)
            iq_raw, iq_cmd, current_saturated = self.speed_pi.compute_iq_command_raw(
                omega_cmd=omega_m_cmd,
                omega_meas=omega_m_meas,
                dt=sub_dt,
            )
            self.last_iq_ref_raw = float(iq_raw)
            self.iq_cmd = float(np.clip(iq_cmd, -self.Imax, self.Imax))
            self.last_iq_ref = float(self.iq_cmd)
            self.last_current_saturation = int(
                self.last_current_saturation or current_saturated or abs(iq_cmd - self.iq_cmd) > 1e-10
            )
            self.iq += (self.iq_cmd - self.iq) * min(1.0, sub_dt / max(self.Ts_current, 1e-9))
            self.iq = float(np.clip(self.iq, -self.Imax, self.Imax))

            raw_torque = float(self.torque_constant * self.iq)
            self.last_torque_raw = raw_torque
            target_torque = float(np.clip(raw_torque, -self.max_motor_torque, self.max_motor_torque))
            self.last_motor_torque_saturation = int(
                self.last_motor_torque_saturation or abs(raw_torque - target_torque) > 1e-10
            )
            if self.torque_filter_time_s > 0.0:
                alpha = min(1.0, sub_dt / self.torque_filter_time_s)
                self.Te += (target_torque - self.Te) * alpha
            else:
                self.Te = target_torque
            self.Te = float(np.clip(self.Te, -self.max_motor_torque, self.max_motor_torque))
            self.last_t_out_raw = float(
                self.active_gearbox_spec.efficiency_nominal * self.active_gearbox_spec.ratio * self.Te
            )
            self.T_out, gear_saturated = self.active_gearbox_spec.motor_to_load_torque(self.Te)
            self.last_gear_torque_saturation = int(self.last_gear_torque_saturation or gear_saturated)
            self.last_flag_e_safe = int(
                bool(
                    self.last_current_saturation
                    or self.last_motor_torque_saturation
                    or self.last_gear_torque_saturation
                )
            )
            self.state = self.load_model.step(
                self.state,
                motor_torque=self.Te,
                disturbance_torque=self.disturbance_torque,
                dt=sub_dt,
            )

        self.elapsed_steps += 1
        self.T_L_hat = self._estimate_load_torque(disturbance_torque=self.disturbance_torque)
        self._update_measurement()

        self.last_saturation = int(
            bool(
                self.last_flag_u_safe
                or self.last_flag_e_safe
            )
        )

        terminated = False
        truncated = False
        done_reason = ""
        theta_min, theta_max = self.active_load_spec.theta_range_rad
        angle_violation = bool(
            self.state.theta < theta_min or self.state.theta > theta_max or abs(self.state.theta) > self.max_abs_theta
        )
        speed_violation = bool(abs(self.state.omega) > float(self.active_load_spec.omega_limit_rad_s))
        torque_violation = bool(abs(self.T_out) > float(self.active_gearbox_spec.output_torque_limit_nm) + 1e-9)
        self.last_sigma_safe = int(bool(angle_violation or speed_violation or torque_violation))
        self.last_flag_x_safe = int(bool(self.enable_x_safe and self.last_sigma_safe))
        if self.enable_x_safe and angle_violation:
            terminated = True
            done_reason = "theta_limit_violation"
        elif self.enable_x_safe and speed_violation:
            terminated = True
            done_reason = "omega_limit_violation"
        elif self.enable_x_safe and torque_violation:
            terminated = True
            done_reason = "torque_limit_violation"
        if self.elapsed_steps >= self.episode_steps:
            truncated = not terminated
            done_reason = done_reason or "episode_limit"

        self.last_constraint_violation = int(
            bool(
                angle_violation
                or speed_violation
                or torque_violation
                or self.last_flag_e_safe
            )
        )

        e_theta = float(self.last_trajectory.theta_ref - self.state.theta)
        e_omega = float(self.last_trajectory.omega_ff - self.state.omega)
        reward, terms = self._reward_terms(
            e_theta=e_theta,
            e_omega=e_omega,
            delta_a_safe=self.last_delta_a_safe,
            t_out=self.T_out,
            sigma_safe=self.last_sigma_safe if self.enable_x_safe else 0,
            omega_l=self.state.omega,
            unsafe_terminal=terminated and "violation" in done_reason,
        )
        self.last_reward_terms = terms
        self.prev_action = self.last_action_safe
        self.prev_a_safe = self.last_action_safe
        self.prev_abs_e_theta = abs(e_theta)
        self.prev_delta_omega = _raw_delta_omega
        self.prev_omega_cmd = omega_cmd

        obs = self._build_observation()
        if not np.isfinite(obs).all() or not np.isfinite(reward):
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

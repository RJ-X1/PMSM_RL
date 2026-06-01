"""Canonical evaluation entrypoint for RL and PI baselines."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
from dataclasses import replace
from typing import Any

import numpy as np

from baselines.cascade_servo_controller import CascadeServoController, CascadeServoControllerConfig
from baselines.pid_position_controller import PIDPositionControllerConfig
from baselines.pi_speed_controller import PISpeedControllerConfig
from baselines.pi_current_controller import PICurrentController
from envs.make_env import make_eval_env
from envs.obs_parser import build_observation_spec, parse_flat_observation, required_eval_trace_columns
from utils.config import (
    apply_train_action_smoothness_override,
    apply_train_reward_override,
    load_yaml,
    parse_env_config,
    parse_eval_config,
    parse_pi_config,
    parse_train_config,
)
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_PI_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    build_rl_agent,
    make_env_build_config,
    resolve_agent_name,
)
from utils.residual_control import (
    apply_residual_action_scale_from_env,
    compose_residual_action,
    gun_servo_controller_overrides,
    is_pid_pi_controller,
    is_residual_controller,
    is_rl_actor_controller,
    normalize_controller_name,
    resolve_residual_settings,
)
from utils.run_layout import ensure_run_layout, make_run_layout
from utils.seed import set_seed

RAD_PER_SEC_TO_RPM = 60.0 / (2.0 * np.pi)
DEFAULT_SCENARIO_NAME = "default"
TEST1_SCENARIO_KEYS = {"test1", "test1nominalsteptracking"}
TEST1_REFERENCE_STEP_TIME_S = 0.02
GUN_SERVO_ENV_ID = "Custom-GunServo-Position-v0"


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DDPG or PI baseline and export trajectory")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--pi-config", type=Path, default=DEFAULT_PI_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-tag",
        choices=("best", "latest"),
        default="best",
        help="Preferred RL checkpoint alias when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--controller",
        choices=(
            "rl",
            "pi",
            "pid",
            "pid_pi_foc",
            "td3_pi",
            "sc_td3_pi",
            "mc_sc_td3_pi",
            "smc_pi_foc",
            "residual",
            "pi_rl_residual",
        ),
        default="rl",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed override")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--residual-action-clip", type=float, default=None)
    parser.add_argument("--residual-zero-test", action="store_true")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario label stored in the exported CSV; Test-1 applies nominal eval overrides.",
    )
    return parser


def _extract_done_reason(info: Any, *, terminated: bool, truncated: bool) -> str:
    """Best-effort termination reason from env info with clear fallbacks."""
    if isinstance(info, dict):
        for key in (
            "done_reason",
            "termination_reason",
            "terminated_reason",
            "truncation_reason",
            "reason",
        ):
            value = info.get(key)
            if value not in (None, ""):
                return str(value)
        if bool(info.get("TimeLimit.truncated", False)):
            return "time_limit"
        for key, value in info.items():
            if "reason" in str(key).lower() and value not in (None, ""):
                return f"{key}={value}"
    if terminated and truncated:
        return "terminated+truncated"
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return ""


def _default_output_csv_path(train_cfg: Any, controller_name: str) -> Path:
    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    return layout.eval_dir / f"eval_{controller_name}.csv"


def _default_checkpoint_path(train_cfg: Any, *, tag: str = "best") -> Path:
    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    if str(tag) == "best":
        best_path = layout.checkpoints_dir / "checkpoint_best.pt"
        if best_path.exists():
            return best_path
    return layout.checkpoints_dir / "checkpoint_latest.pt"


def _apply_checkpoint_hparams(train_cfg: Any, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return
    hparams = payload.get("hparams") if isinstance(payload, dict) else None
    if not isinstance(hparams, dict):
        return
    for name in (
        "hidden_dim",
        "gamma",
        "tau",
        "lr_actor",
        "lr_critic",
        "target_policy_noise",
        "target_noise_clip",
        "policy_delay",
    ):
        source_name = {
            "lr_actor": "actor_lr",
            "lr_critic": "critic_lr",
        }.get(name, name)
        if source_name in hparams and hasattr(train_cfg, name):
            setattr(train_cfg, name, hparams[source_name])


def _apply_controller_env_overrides(env_cfg: Any, controller_name: str) -> tuple[Any, bool | None]:
    overrides = gun_servo_controller_overrides(controller_name)
    if not overrides:
        return env_cfg, None
    safety = {**dict(getattr(env_cfg, "gun_safety", {}) or {}), **dict(overrides.get("safety", {}) or {})}
    domain_randomization = {
        **dict(getattr(env_cfg, "gun_domain_randomization", {}) or {}),
        **dict(overrides.get("domain_randomization", {}) or {}),
    }
    apply_dr = overrides.get("apply_domain_randomization")
    return (
        replace(env_cfg, gun_safety=safety, gun_domain_randomization=domain_randomization),
        None if apply_dr is None else bool(apply_dr),
    )


def _make_pi_controller(
    *,
    act_dim: int,
    spec: Any,
    pi_cfg: Any,
    env_cfg: Any,
) -> PICurrentController:
    motor_params = env_cfg.motor.to_motor_params() if bool(getattr(env_cfg, "use_custom_env", False)) else None
    pi = PICurrentController(
        action_dim=act_dim,
        signal_names=spec.signal_names,
        config=pi_cfg.to_controller_config(),
        motor_params=motor_params,
    )
    pi.reset()
    return pi


def _make_cascade_servo_controller(*, spec: Any, env: Any) -> CascadeServoController:
    base_env = getattr(env, "unwrapped", env)
    scales = getattr(base_env, "observation_normalization_scales", {})
    max_delta_omega = float(getattr(base_env, "max_delta_omega", np.deg2rad(50.0)))
    iq_limit = float(getattr(base_env, "Imax", 12.0))
    speed_pi = getattr(base_env, "speed_pi", None)
    speed_cfg = getattr(speed_pi, "config", None)
    config = CascadeServoControllerConfig(
        position=PIDPositionControllerConfig(
            kp=float(getattr(base_env, "baseline_position_kp", 8.0)),
            ki=float(getattr(base_env, "baseline_position_ki", 0.0)),
            kd=float(getattr(base_env, "baseline_position_kd", 1.2)),
            integrator_limit=float(np.deg2rad(30.0)),
            max_delta_omega=max_delta_omega,
        ),
        speed=PISpeedControllerConfig(
            kp=float(getattr(speed_cfg, "kp", 18.0)),
            ki=float(getattr(speed_cfg, "ki", 80.0)),
            integrator_limit=float(getattr(speed_cfg, "integrator_limit", 0.75 * iq_limit)),
            iq_limit=float(getattr(speed_cfg, "iq_limit", iq_limit)),
        ),
        theta_scale=float(scales.get("theta", np.deg2rad(30.0))),
        omega_scale=float(scales.get("omega", np.deg2rad(90.0))),
        dt=float(getattr(base_env, "dt", 0.002)),
    )
    controller = CascadeServoController(signal_names=spec.signal_names, config=config)
    controller.reset()
    return controller


def _residual_csv_fieldnames() -> list[str]:
    return [
        "action_pi_d",
        "action_pi_q",
        "action_residual_raw_d",
        "action_residual_raw_q",
        "action_residual_d",
        "action_residual_q",
        "action_total_d",
        "action_total_q",
        "action_total_clipped_d",
        "action_total_clipped_q",
        "residual_scale",
        "residual_action_clip",
        "residual_zero_test",
        "residual_clip_applied",
        "action_total_clip_applied",
    ]


def _add_residual_row_fields(row: dict[str, Any], residual_trace: dict[str, Any], *, settings: Any) -> None:
    labels = (
        ("action_pi", "action_pi"),
        ("action_residual_raw", "action_residual_raw"),
        ("action_residual", "action_residual"),
        ("action_total", "action_total"),
        ("action_total_clipped", "action_total_clipped"),
    )
    for trace_key, column_prefix in labels:
        values = np.asarray(residual_trace[trace_key], dtype=np.float64).reshape(-1)
        if values.size >= 1:
            row[f"{column_prefix}_d"] = float(values[0])
        if values.size >= 2:
            row[f"{column_prefix}_q"] = float(values[1])
    row["residual_scale"] = float(settings.residual_scale)
    row["residual_action_clip"] = (
        "" if settings.residual_action_clip is None else float(settings.residual_action_clip)
    )
    row["residual_zero_test"] = int(bool(settings.residual_zero_test))
    row["residual_clip_applied"] = int(bool(residual_trace["residual_clip_applied"]))
    row["action_total_clip_applied"] = int(bool(residual_trace["action_total_clip_applied"]))


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalize_scenario_key(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _resolve_test_conditions_path(eval_config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = eval_config_path.parent / path
    if candidate.exists():
        return candidate
    return ROOT / path


def _scenario_matches(condition: dict[str, Any], scenario_name: str) -> bool:
    scenario_key = _normalize_scenario_key(scenario_name)
    candidates = (
        condition.get("id"),
        condition.get("name"),
    )
    return any(_normalize_scenario_key(candidate) == scenario_key for candidate in candidates)


def _is_test1_condition(condition: dict[str, Any]) -> bool:
    candidates = (
        condition.get("id"),
        condition.get("name"),
    )
    return any(_normalize_scenario_key(candidate) in TEST1_SCENARIO_KEYS for candidate in candidates)


def _find_eval_condition(eval_config_path: Path, scenario_name: str) -> dict[str, Any] | None:
    eval_cfg = parse_eval_config(eval_config_path)
    raw_eval_cfg = load_yaml(eval_config_path)
    scenario_map = raw_eval_cfg.get("scenario_condition_map", {})
    mapped_scenario = scenario_name
    if isinstance(scenario_map, dict):
        mapped_scenario = str(scenario_map.get(scenario_name, scenario_name))
    conditions_path = _resolve_test_conditions_path(
        eval_config_path,
        str(eval_cfg.test_conditions_path),
    )
    data = load_yaml(conditions_path)
    conditions = data.get("test_scenarios", [])
    if not isinstance(conditions, list):
        raise TypeError(f"Expected 'test_scenarios' to be a list in {conditions_path}")
    for condition in conditions:
        if isinstance(condition, dict) and _scenario_matches(condition, mapped_scenario):
            return condition
    return None


def _apply_test1_env_overrides(env_cfg: Any, condition: dict[str, Any]) -> Any:
    reference = condition.get("reference", {})
    if not isinstance(reference, dict):
        reference = {}
    omega_m_rpm = float(condition.get("omega_m_rpm", 1000.0))
    omega_m = omega_m_rpm / RAD_PER_SEC_TO_RPM
    load_torque = float(condition.get("load_torque_nm", 1.0))
    init_i_d = float(reference.get("i_d_initial", 0.0))
    init_i_q = float(reference.get("i_q_initial", 0.0))
    ref_i_d = float(reference.get("i_d_final", -3.0))
    ref_i_q = float(reference.get("i_q_final", 10.0))

    return replace(
        env_cfg,
        environment=replace(
            env_cfg.environment,
            init_i_d_range=(init_i_d, init_i_d),
            init_i_q_range=(init_i_q, init_i_q),
            init_omega_m_range=(omega_m, omega_m),
            speed_mode="fixed",
            episode_steps=2000,
            load_torque=load_torque,
            load_torque_range=(load_torque, load_torque),
        ),
        reference=replace(
            env_cfg.reference,
            ref_i_d=ref_i_d,
            ref_i_q=ref_i_q,
            randomize_on_reset=False,
            ref_i_d_range=(init_i_d, ref_i_d),
            ref_i_q_range=(init_i_q, ref_i_q),
            reference_profile="step",
            ref_i_d_initial=init_i_d,
            ref_i_q_initial=init_i_q,
            ref_i_d_final=ref_i_d,
            ref_i_q_final=ref_i_q,
            reference_step_time_s=TEST1_REFERENCE_STEP_TIME_S,
            reference_step_step=None,
        ),
        noise=replace(
            env_cfg.noise,
            observation_noise_std=0.0,
            process_noise_std=0.0,
        ),
        domain_randomization=replace(
            env_cfg.domain_randomization,
            enabled=False,
            sigma_i_range=(0.0, 0.0),
            sigma_omega_rpm_range=(0.0, 0.0),
            load_torque_range=(load_torque, load_torque),
            load_change_interval_steps_range=(0, 0),
        ),
    )


def _apply_gun_servo_env_overrides(env_cfg: Any, condition: dict[str, Any]) -> Any:
    def merged_dict(field_name: str, condition_key: str) -> dict[str, Any]:
        base = dict(getattr(env_cfg, field_name, {}) or {})
        override = condition.get(condition_key)
        if isinstance(override, dict):
            base.update(override)
        return base

    reference = merged_dict("gun_reference", "reference")
    load = merged_dict("gun_load", "load")
    if "disturbance_torque_Nm" in condition:
        load["disturbance_torque"] = float(condition["disturbance_torque_Nm"])
    if "disturbance_step_Nm" in condition:
        load["disturbance_step_nm"] = float(condition["disturbance_step_Nm"])
    domain_randomization = merged_dict("gun_domain_randomization", "domain_randomization")
    if "disturbance_torque_Nm" in condition and "disturbance_torque_range" not in domain_randomization:
        value = float(condition["disturbance_torque_Nm"])
        domain_randomization["disturbance_torque_range"] = [value, value]
    return replace(
        env_cfg,
        gun_servo_env=merged_dict("gun_servo_env", "gun_servo_env"),
        gun_servo_drive=merged_dict("gun_servo_drive", "servo_drive"),
        gun_gearbox=merged_dict("gun_gearbox", "gearbox"),
        gun_reference=reference,
        gun_load=load,
        gun_load_encoder=merged_dict("gun_load_encoder", "load_encoder"),
        gun_rl_action=merged_dict("gun_rl_action", "rl_action"),
        gun_rl_controller=merged_dict("gun_rl_controller", "rl_controller"),
        gun_reward=merged_dict("gun_reward", "reward"),
        gun_safety=merged_dict("gun_safety", "safety"),
        gun_normalization=merged_dict("gun_normalization", "normalization"),
        gun_domain_randomization=domain_randomization,
        gun_environment=merged_dict("gun_environment", "environment"),
        gun_speed_controller=merged_dict("gun_speed_controller", "speed_controller"),
    )


def _maybe_apply_eval_scenario(env_cfg: Any, eval_config_path: Path, scenario_name: str | None) -> Any:
    if scenario_name in (None, ""):
        return env_cfg
    condition = _find_eval_condition(eval_config_path, str(scenario_name))
    if condition is not None and str(getattr(env_cfg, "env_id", "")) == GUN_SERVO_ENV_ID:
        print(f"scenario={scenario_name} matched gun-servo eval condition; applying overrides.")
        return _apply_gun_servo_env_overrides(env_cfg, condition)
    if condition is not None and _is_test1_condition(condition):
        print(
            f"scenario={scenario_name} matched Test-1 nominal step tracking; "
            "applying fixed-speed step-reference eval overrides."
        )
        return _apply_test1_env_overrides(env_cfg, condition)
    print(f"scenario={scenario_name} did not match a supported eval condition; using env-config as-is.")
    return env_cfg


def run_evaluation(
    *,
    env_config_path: Path,
    train_config_path: Path,
    pi_config_path: Path,
    controller_name: str,
    checkpoint_path: Path | None,
    checkpoint_tag: str,
    max_steps_override: int | None,
    output_csv_path: Path | None,
    seed_override: int | None = None,
    scenario_name: str | None = None,
    eval_config_path: Path | None = None,
    residual_scale_override: float | None = None,
    residual_action_clip_override: float | None = None,
    residual_zero_test_override: bool | None = None,
) -> dict[str, Any]:
    """Run one evaluation episode and export the trajectory CSV."""
    env_cfg = parse_env_config(env_config_path)
    train_cfg = parse_train_config(train_config_path)
    effective_eval_config_path = Path(eval_config_path or DEFAULT_EVAL_CONFIG_PATH)
    eval_cfg = parse_eval_config(effective_eval_config_path)
    effective_max_steps_override = max_steps_override if max_steps_override is not None else eval_cfg.max_steps
    controller_name = normalize_controller_name(controller_name)
    if controller_name == "smc_pi_foc":
        raise NotImplementedError("smc_pi_foc is reserved for the second-round robust-control baseline.")
    residual_settings = resolve_residual_settings(
        train_cfg,
        eval_cfg,
        residual_scale_override=residual_scale_override,
        residual_action_clip_override=residual_action_clip_override,
        residual_zero_test_override=residual_zero_test_override,
    )
    env_cfg = apply_train_reward_override(env_cfg, train_cfg)
    env_cfg = apply_train_action_smoothness_override(env_cfg, train_cfg)
    env_cfg = _maybe_apply_eval_scenario(
        env_cfg,
        effective_eval_config_path,
        scenario_name,
    )
    env_cfg, controller_apply_dr = _apply_controller_env_overrides(env_cfg, controller_name)
    raw_eval_cfg = load_yaml(effective_eval_config_path)
    if isinstance(raw_eval_cfg, dict) and "apply_domain_randomization" in raw_eval_cfg:
        controller_apply_dr = bool(raw_eval_cfg.get("apply_domain_randomization"))
    if effective_max_steps_override is not None and str(getattr(env_cfg, "env_id", "")) == GUN_SERVO_ENV_ID:
        env_cfg = replace(
            env_cfg,
            gun_servo_env={**dict(getattr(env_cfg, "gun_servo_env", {}) or {}), "episode_steps": int(effective_max_steps_override)},
            gun_environment={**dict(getattr(env_cfg, "gun_environment", {}) or {}), "episode_steps": int(effective_max_steps_override)},
        )
    pi_cfg = parse_pi_config(pi_config_path)
    eval_seed = int(seed_override if seed_override is not None else env_cfg.seed)
    set_seed(eval_seed)

    env = make_eval_env(make_env_build_config(env_cfg, seed_override=eval_seed, apply_domain_randomization=controller_apply_dr))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low.min())
    action_high = float(env.action_space.high.max())
    default_eval_steps = getattr(train_cfg, "max_steps_per_episode", None)
    if default_eval_steps is None:
        default_eval_steps = getattr(getattr(env_cfg, "environment", None), "episode_steps", 500)
    max_steps = int(effective_max_steps_override if effective_max_steps_override is not None else default_eval_steps)
    scenario = str(scenario_name).strip() if scenario_name not in (None, "") else DEFAULT_SCENARIO_NAME

    obs, _info = env.reset(seed=eval_seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    signal_fieldnames = list(parse_flat_observation(obs, spec=spec).keys())
    base_env = getattr(env, "unwrapped", env)
    residual_mode = is_residual_controller(controller_name)
    actor_mode = is_rl_actor_controller(controller_name)
    if residual_mode:
        apply_residual_action_scale_from_env(residual_settings, base_env)
    reward_term_names = list(getattr(base_env, "reward_term_names", []))
    reward_mode_name = getattr(base_env, "reward_mode", "")
    action_smoothness_enabled = bool(getattr(base_env, "action_smoothness_enabled", False))
    action_smoothness_weight = float(getattr(base_env, "action_smoothness_weight", 0.0))
    motor_params = getattr(base_env, "motor_params", None)
    sim_dt = _safe_float(getattr(motor_params, "Ts", None))
    if sim_dt is None:
        sim_dt = _safe_float(getattr(base_env, "dt", None))
    missing_trace_columns = [
        name
        for name in required_eval_trace_columns(layout=spec.layout)
        if name not in signal_fieldnames and name not in spec.action_names
    ]
    if missing_trace_columns:
        raise KeyError(f"Evaluation export is missing required trace columns: {missing_trace_columns}")

    if actor_mode or residual_mode:
        resolved_checkpoint = checkpoint_path or _default_checkpoint_path(train_cfg, tag=checkpoint_tag)
        _apply_checkpoint_hparams(train_cfg, resolved_checkpoint)
        agent = build_rl_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            train_cfg=train_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        agent.load_checkpoint(resolved_checkpoint)

        if residual_mode:
            if spec.layout == "gun_servo_position":
                pi = _make_cascade_servo_controller(spec=spec, env=env)
            else:
                pi = _make_pi_controller(act_dim=act_dim, spec=spec, pi_cfg=pi_cfg, env_cfg=env_cfg)

            def policy_fn(policy_obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
                action_pi = pi.compute_action(policy_obs)
                action_residual_raw = agent.select_action(policy_obs, add_noise=False)
                residual_trace = compose_residual_action(
                    action_pi=action_pi,
                    action_residual_raw=action_residual_raw,
                    action_low=action_low,
                    action_high=action_high,
                    settings=residual_settings,
                )
                return residual_trace["action_total_clipped"], residual_trace

        else:

            def policy_fn(policy_obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
                return agent.select_action(policy_obs, add_noise=False), None

    elif is_pid_pi_controller(controller_name):
        resolved_checkpoint = checkpoint_path
        if spec.layout == "gun_servo_position":
            cascade = _make_cascade_servo_controller(spec=spec, env=env)

            def policy_fn(policy_obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
                return cascade.compute_action(policy_obs), None

        else:
            if controller_name == "pid":
                raise ValueError("controller=pid is only supported for layout=gun_servo_position")
            pi = _make_pi_controller(act_dim=act_dim, spec=spec, pi_cfg=pi_cfg, env_cfg=env_cfg)

            def policy_fn(policy_obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
                return pi.compute_action(policy_obs), None
    else:
        raise ValueError(f"Unsupported controller: {controller_name}")

    final_output_csv = output_csv_path or _default_output_csv_path(train_cfg, controller_name)
    final_output_csv.parent.mkdir(parents=True, exist_ok=True)
    cum_reward = 0.0
    terminated = False
    truncated = False
    final_done_reason = ""

    fieldnames = [
        "run_id",
        "algorithm",
        "seed",
        "episode",
        "controller",
        "env_id",
        "layout",
        "scenario",
        "step",
        "t_s",
        "reward",
        "cum_reward",
        "terminated",
        "truncated",
        "done",
        "done_reason",
        "termination_reason",
        "reward_mode",
        "action_smoothness_enabled",
        "action_smoothness_weight",
        "time_s",
        "i_d_phys",
        "i_q_phys",
        "ref_i_d_phys",
        "ref_i_q_phys",
        "omega_m_phys",
        "speed_rpm",
        "load_torque_nm",
        "u_d",
        "u_q",
        "active_Umax",
        "action_saturated",
    ]
    if spec.layout == "gun_servo_position":
        fieldnames += [
            "theta_ref_rad",
            "theta_ref_deg",
            "theta_L_rad",
            "theta_L_deg",
            "omega_L_rad_s",
            "omega_m_rad_s",
            "omega_m_ref_rad_s",
            "e_theta_rad",
            "e_theta_deg",
            "e_omega_load",
            "omega_ref_deg_s",
            "omega_L_deg_s",
            "omega_cmd_deg_s",
            "a_raw",
            "a_clip",
            "a_safe",
            "delta_a_safe",
            "omega_L_cmd_raw",
            "omega_L_cmd_safe",
            "omega_m_cmd",
            "iq_A",
            "i_q",
            "i_q_ref_raw",
            "i_q_ref",
            "Te_Nm",
            "T_e",
            "T_out_Nm",
            "T_out_max_Nm",
            "T_out_raw",
            "T_L",
            "T_hat_L",
            "TL_Nm",
            "V_dc",
            "theta_meas_deg",
            "action_raw",
            "action_safe",
            "disturbance_torque_Nm",
            "disturbance_step_Nm",
            "disturbance_step_time_s",
            "domain_randomization_active",
            "encoder_noise_std_rad",
            "active_J_load",
            "active_B_load",
            "active_coulomb_friction",
            "active_gravity_torque_coeff",
            "gear_efficiency",
            "sampled_theta_initial_deg",
            "sampled_theta_final_deg",
            "sampled_step_time_s",
            "sampled_max_speed_deg_s",
            "sampled_max_accel_deg_s2",
            "sampled_sine_amplitude_deg",
            "sampled_sine_frequency_hz",
            "sampled_sine_phase_rad",
            "sigma_safe",
            "flag_U_safe",
            "flag_E_safe",
            "flag_X_safe",
            "saturation_flag",
            "speed_saturation_flag",
            "current_saturation_flag",
            "action_rate_saturation_flag",
            "accel_saturation_flag",
            "motor_torque_saturation_flag",
            "gear_torque_saturation_flag",
            "constraint_violation",
        ]
    if residual_mode:
        fieldnames += _residual_csv_fieldnames()
    fieldnames += signal_fieldnames + spec.action_names + reward_term_names
    fieldnames = list(dict.fromkeys(fieldnames))

    with final_output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(max_steps):
            action, residual_trace = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(act_dim)
            action = np.clip(action, action_low, action_high)
            if not np.isfinite(action).all():
                raise FloatingPointError(
                    f"{controller_name} produced non-finite action at step {step}: {action}"
                )

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            cum_reward += float(reward)
            done_reason = _extract_done_reason(_info, terminated=bool(terminated), truncated=bool(truncated))
            final_done_reason = done_reason
            parsed_obs = parse_flat_observation(obs, spec=spec)

            row = {
                "run_id": train_cfg.run_name,
                "algorithm": controller_name,
                "seed": eval_seed,
                "episode": 0,
                "controller": controller_name,
                "env_id": env_cfg.env_id,
                "layout": spec.layout,
                "scenario": scenario,
                "step": step,
                "reward": float(reward),
                "cum_reward": float(cum_reward),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "done": int(done),
                "done_reason": done_reason,
                "termination_reason": done_reason,
                "reward_mode": str(reward_mode_name),
                "action_smoothness_enabled": int(action_smoothness_enabled),
                "action_smoothness_weight": float(action_smoothness_weight),
            }
            if residual_mode and residual_trace is not None:
                _add_residual_row_fields(row, residual_trace, settings=residual_settings)
            if sim_dt is not None:
                row["time_s"] = float(step) * sim_dt
                row["t_s"] = float(step) * sim_dt
            row.update(parsed_obs)
            row.update({name: float(value) for name, value in zip(spec.action_names, action)})
            if spec.layout == "custom_dq":
                scales = getattr(base_env, "observation_normalization_scales", {})
                current_scale = _safe_float(scales.get("current"))
                omega_scale = _safe_float(scales.get("omega_m_rad_per_sec"))
                load_scale = _safe_float(scales.get("load_torque"))
                if current_scale is not None:
                    if "i_d" in parsed_obs:
                        row["i_d_phys"] = float(parsed_obs["i_d"]) * current_scale
                    if "i_q" in parsed_obs:
                        row["i_q_phys"] = float(parsed_obs["i_q"]) * current_scale
                    if "ref_i_d" in parsed_obs:
                        row["ref_i_d_phys"] = float(parsed_obs["ref_i_d"]) * current_scale
                    if "ref_i_q" in parsed_obs:
                        row["ref_i_q_phys"] = float(parsed_obs["ref_i_q"]) * current_scale
                if omega_scale is not None and "omega_m" in parsed_obs:
                    omega_m_phys = float(parsed_obs["omega_m"]) * omega_scale
                    row["omega_m_phys"] = omega_m_phys
                    row["speed_rpm"] = omega_m_phys * RAD_PER_SEC_TO_RPM
                if load_scale is not None and "T_L" in parsed_obs:
                    row["load_torque_nm"] = float(parsed_obs["T_L"]) * load_scale
            elif spec.layout == "gun_servo_position":
                scales = getattr(base_env, "observation_normalization_scales", {})
                theta_scale = _safe_float(scales.get("theta"))
                omega_scale = _safe_float(scales.get("omega"))
                current_scale = _safe_float(scales.get("current"))
                torque_scale = _safe_float(scales.get("torque"))
                if theta_scale is not None:
                    if "e_theta" in parsed_obs:
                        row["e_theta_deg"] = float(parsed_obs["e_theta"]) * theta_scale * (180.0 / np.pi)
                    if "theta_ref" in parsed_obs:
                        row["theta_ref_deg"] = float(parsed_obs["theta_ref"]) * theta_scale * (180.0 / np.pi)
                    if "theta_L" in parsed_obs:
                        row["theta_L_deg"] = float(parsed_obs["theta_L"]) * theta_scale * (180.0 / np.pi)
                if omega_scale is not None:
                    if "e_omega" in parsed_obs:
                        row["omega_ref_deg_s"] = (
                            float(parsed_obs["e_omega"]) + float(parsed_obs.get("omega_L", 0.0))
                        ) * omega_scale * (180.0 / np.pi)
                    if "omega_L" in parsed_obs:
                        row["omega_L_deg_s"] = float(parsed_obs["omega_L"]) * omega_scale * (180.0 / np.pi)
                    if "omega_cmd" in parsed_obs:
                        row["omega_cmd_deg_s"] = float(parsed_obs["omega_cmd"]) * omega_scale * (180.0 / np.pi)
                if current_scale is not None and "iq" in parsed_obs:
                    row["iq_A"] = float(parsed_obs["iq"]) * current_scale
                if torque_scale is not None:
                    if "T_hat_L" in parsed_obs:
                        row["TL_Nm"] = float(parsed_obs["T_hat_L"]) * torque_scale
                    elif "T_L_hat" in parsed_obs:
                        row["TL_Nm"] = float(parsed_obs["T_L_hat"]) * torque_scale
                if "saturation_flag" in parsed_obs:
                    row["saturation_flag"] = int(float(parsed_obs["saturation_flag"]) > 0.5)
            if isinstance(_info, dict):
                u_d = _safe_float(_info.get("prev_u_d"))
                u_q = _safe_float(_info.get("prev_u_q"))
                active_umax = _safe_float(_info.get("active_Umax"))
                if u_d is not None:
                    row["u_d"] = u_d
                if u_q is not None:
                    row["u_q"] = u_q
                if active_umax is not None:
                    row["active_Umax"] = active_umax
                if (
                    spec.layout == "custom_dq"
                    and active_umax is not None
                    and u_d is not None
                    and u_q is not None
                    and len(spec.action_names) >= 2
                ):
                    raw_command = np.clip(action[:2], action_low, action_high) * active_umax
                    applied_command = np.asarray([u_d, u_q], dtype=np.float64)
                    saturation_error = float(np.linalg.norm(raw_command - applied_command))
                    row["action_saturated"] = int(
                        saturation_error > (1e-6 * max(1.0, abs(active_umax)))
                    )
                if spec.layout == "gun_servo_position":
                    for key in (
                        "theta_ref_rad",
                        "theta_ref_deg",
                        "theta_L_rad",
                        "theta_L_deg",
                        "omega_L_rad_s",
                        "omega_m_rad_s",
                        "omega_m_ref_rad_s",
                        "e_theta_rad",
                        "e_theta_deg",
                        "e_omega_load",
                        "omega_ref_deg_s",
                        "omega_L_deg_s",
                        "omega_cmd_deg_s",
                        "a_raw",
                        "a_clip",
                        "a_safe",
                        "delta_a_safe",
                        "omega_L_cmd_raw",
                        "omega_L_cmd_safe",
                        "omega_m_cmd",
                        "iq_A",
                        "i_q",
                        "i_q_ref_raw",
                        "i_q_ref",
                        "Te_Nm",
                        "T_e",
                        "T_out_Nm",
                        "T_out_max_Nm",
                        "T_out_raw",
                        "T_L",
                        "T_hat_L",
                        "TL_Nm",
                        "V_dc",
                        "theta_meas_deg",
                        "action_raw",
                        "action_safe",
                        "disturbance_torque_Nm",
                        "disturbance_step_Nm",
                        "disturbance_step_time_s",
                        "domain_randomization_active",
                        "encoder_noise_std_rad",
                        "active_J_load",
                        "active_B_load",
                        "active_coulomb_friction",
                        "active_gravity_torque_coeff",
                        "gear_efficiency",
                        "sampled_theta_initial_deg",
                        "sampled_theta_final_deg",
                        "sampled_step_time_s",
                        "sampled_max_speed_deg_s",
                        "sampled_max_accel_deg_s2",
                        "sampled_sine_amplitude_deg",
                        "sampled_sine_frequency_hz",
                        "sampled_sine_phase_rad",
                    ):
                        value = _safe_float(_info.get(key))
                        if value is not None:
                            row[key] = value
                    for key in (
                        "saturation_flag",
                        "speed_saturation_flag",
                        "current_saturation_flag",
                        "action_rate_saturation_flag",
                        "accel_saturation_flag",
                        "motor_torque_saturation_flag",
                        "gear_torque_saturation_flag",
                        "constraint_violation",
                        "sigma_safe",
                        "flag_U_safe",
                        "flag_E_safe",
                        "flag_X_safe",
                    ):
                        value = _safe_float(_info.get(key))
                        if value is not None:
                            row[key] = int(value > 0.5)
            if reward_term_names:
                reward_terms = _info.get("reward_terms", {}) if isinstance(_info, dict) else {}
                row.update({name: float(reward_terms.get(name, 0.0)) for name in reward_term_names})
            writer.writerow(row)

            obs = next_obs
            if done:
                break

    return {
        "controller": controller_name,
        "agent_name": (
            f"pi+{resolve_agent_name(train_cfg)}_residual"
            if residual_mode
            else resolve_agent_name(train_cfg) if actor_mode else "cascade_pid" if spec.layout == "gun_servo_position" else "pi"
        ),
        "env_id": env_cfg.env_id,
        "layout": spec.layout,
        "scenario": scenario,
        "episode_return": float(cum_reward),
        "steps": int(step + 1),
        "done_reason": final_done_reason or "not_done",
        "output_csv": final_output_csv,
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_tag": str(checkpoint_tag) if actor_mode or residual_mode else None,
        "seed": eval_seed,
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "done": int(bool(terminated or truncated)),
        "residual_scale": float(residual_settings.residual_scale),
        "residual_action_clip": residual_settings.residual_action_clip,
        "residual_zero_test": bool(residual_settings.residual_zero_test),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_eval_cfg = load_yaml(args.eval_config)
    configured_scenarios = raw_eval_cfg.get("scenarios", [])
    if args.scenario in (None, "") and args.output_csv is None and isinstance(configured_scenarios, list) and configured_scenarios:
        run_name = str(raw_eval_cfg.get("run_name", "eval"))
        train_config_path = Path(raw_eval_cfg.get("train_config", args.train_config))
        eval_dir = ROOT / "outputs" / "runs" / run_name / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for scenario in configured_scenarios:
            scenario_name = str(scenario.get("id", scenario.get("name"))) if isinstance(scenario, dict) else str(scenario)
            output_csv = eval_dir / f"{scenario_name}_{args.controller}.csv"
            result = run_evaluation(
                env_config_path=args.env_config,
                eval_config_path=args.eval_config,
                train_config_path=train_config_path,
                pi_config_path=args.pi_config,
                controller_name=str(args.controller),
                checkpoint_path=args.checkpoint,
                checkpoint_tag=str(args.checkpoint_tag),
                max_steps_override=args.max_steps,
                output_csv_path=output_csv,
                seed_override=args.seed,
                scenario_name=scenario_name,
                residual_scale_override=args.residual_scale,
                residual_action_clip_override=args.residual_action_clip,
                residual_zero_test_override=True if args.residual_zero_test else None,
            )
            print(
                f"controller={result['controller']} env_id={result['env_id']} "
                f"scenario={result['scenario']} episode_return={result['episode_return']:.3f} "
                f"steps={result['steps']} done_reason={result['done_reason']} csv={result['output_csv']}"
            )
        return
    result = run_evaluation(
        env_config_path=args.env_config,
        eval_config_path=args.eval_config,
        train_config_path=args.train_config,
        pi_config_path=args.pi_config,
        controller_name=str(args.controller),
        checkpoint_path=args.checkpoint,
        checkpoint_tag=str(args.checkpoint_tag),
        max_steps_override=args.max_steps,
        output_csv_path=args.output_csv,
        seed_override=args.seed,
        scenario_name=args.scenario,
        residual_scale_override=args.residual_scale,
        residual_action_clip_override=args.residual_action_clip,
        residual_zero_test_override=True if args.residual_zero_test else None,
    )
    print(
        f"controller={result['controller']} agent={result['agent_name']} "
        f"env_id={result['env_id']} layout={result['layout']} scenario={result['scenario']} "
        f"episode_return={result['episode_return']:.3f} steps={result['steps']} "
        f"done_reason={result['done_reason']} checkpoint={result['checkpoint_path']} "
        f"csv={result['output_csv']}"
    )


if __name__ == "__main__":
    main()

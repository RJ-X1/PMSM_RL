"""Helpers for PI-RL residual action composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


RESIDUAL_CONTROLLER_ALIASES = {"residual", "pi_rl_residual", "pi-rl-residual"}
RL_ACTOR_CONTROLLERS = {"rl", "td3_pi", "sc_td3_pi", "mc_sc_td3_pi"}
PID_PI_CONTROLLERS = {"pi", "pid", "pid_pi_foc"}
CONTROLLER_ALIASES = {
    "td3-pi": "td3_pi",
    "sc-td3-pi": "sc_td3_pi",
    "mc-sc-td3-pi": "mc_sc_td3_pi",
    "pid-pi-foc": "pid_pi_foc",
    "smc-pi-foc": "smc_pi_foc",
}


@dataclass(slots=True)
class ResidualControlSettings:
    """Runtime settings for normalized residual-action composition."""

    residual_scale: float = 0.2
    residual_action_clip: float | None = 0.3
    residual_zero_test: bool = False
    baseline_controller: str = "pi"
    baseline_weight: float = 1.0
    residual_weight: float = 1.0
    residual_action_scale_deg_s: float | None = None


def normalize_controller_name(value: Any) -> str:
    """Normalize public controller labels while preserving existing modes."""
    name = str(value).strip().lower()
    name = CONTROLLER_ALIASES.get(name, name)
    if name in RESIDUAL_CONTROLLER_ALIASES:
        return "residual"
    return name


def is_residual_controller(value: Any) -> bool:
    """Return whether a controller label requests PI-RL residual control."""
    return normalize_controller_name(value) == "residual"


def is_rl_actor_controller(value: Any) -> bool:
    """Return whether a controller label uses a learned actor as the position loop."""
    return normalize_controller_name(value) in RL_ACTOR_CONTROLLERS


def is_pid_pi_controller(value: Any) -> bool:
    """Return whether a controller label uses the classical position PID baseline."""
    return normalize_controller_name(value) in PID_PI_CONTROLLERS


def gun_servo_controller_overrides(value: Any) -> dict[str, Any]:
    """Return gun-servo env overrides implied by a public controller label."""
    name = normalize_controller_name(value)
    if name == "td3_pi":
        return {
            "safety": {
                "enable_u_safe": False,
                "enable_e_safe": True,
                "enable_x_safe": False,
            },
            "apply_domain_randomization": False,
        }
    if name in {"sc_td3_pi", "pid_pi_foc", "pid", "pi", "rl", "residual"}:
        return {
            "safety": {
                "enable_u_safe": True,
                "enable_e_safe": True,
                "enable_x_safe": True,
            },
            "apply_domain_randomization": False,
        }
    if name == "mc_sc_td3_pi":
        return {
            "safety": {
                "enable_u_safe": True,
                "enable_e_safe": True,
                "enable_x_safe": True,
            },
            "domain_randomization": {"enabled": True},
            "apply_domain_randomization": True,
        }
    return {}


def _first_config_value(configs: tuple[Any, ...], name: str, default: Any) -> Any:
    for cfg in configs:
        if cfg is None or not hasattr(cfg, name):
            continue
        value = getattr(cfg, name)
        if value is not None:
            return value
    return default


def resolve_residual_settings(
    *configs: Any,
    residual_scale_override: float | None = None,
    residual_action_clip_override: float | None = None,
    residual_zero_test_override: bool | None = None,
) -> ResidualControlSettings:
    """Resolve residual settings from config objects plus optional CLI overrides."""
    defaults = ResidualControlSettings()
    scale = _first_config_value(configs, "residual_scale", defaults.residual_scale)
    clip = _first_config_value(configs, "residual_action_clip", defaults.residual_action_clip)
    zero_test = _first_config_value(configs, "residual_zero_test", defaults.residual_zero_test)
    baseline_controller = _first_config_value(configs, "residual_baseline_controller", defaults.baseline_controller)
    baseline_weight = _first_config_value(configs, "residual_baseline_weight", defaults.baseline_weight)
    residual_weight = _first_config_value(configs, "residual_weight", defaults.residual_weight)
    action_scale_deg_s = _first_config_value(
        configs,
        "residual_action_scale_deg_s",
        defaults.residual_action_scale_deg_s,
    )

    if residual_scale_override is not None:
        scale = residual_scale_override
    if residual_action_clip_override is not None:
        clip = residual_action_clip_override
    if residual_zero_test_override is not None:
        zero_test = residual_zero_test_override

    clip_value = None if clip in (None, "") else abs(float(clip))
    return ResidualControlSettings(
        residual_scale=float(scale),
        residual_action_clip=clip_value,
        residual_zero_test=bool(zero_test),
        baseline_controller=str(baseline_controller),
        baseline_weight=float(baseline_weight),
        residual_weight=float(residual_weight),
        residual_action_scale_deg_s=None if action_scale_deg_s in (None, "") else abs(float(action_scale_deg_s)),
    )


def apply_residual_action_scale_from_env(settings: ResidualControlSettings, base_env: Any) -> None:
    """Resolve a degree-per-second residual scale against a gun-servo env."""
    scale_deg_s = settings.residual_action_scale_deg_s
    if scale_deg_s in (None, ""):
        return
    max_delta = abs(float(getattr(base_env, "max_delta_omega", 0.0)))
    if max_delta <= 0.0:
        return
    settings.residual_scale = float(np.deg2rad(float(scale_deg_s)) / max_delta)


def residual_agent_action_bounds(
    *,
    action_low: float,
    action_high: float,
    settings: ResidualControlSettings,
) -> tuple[float, float]:
    """Return action bounds for the RL residual policy output."""
    if settings.residual_action_clip is None:
        return float(action_low), float(action_high)
    clip = min(abs(float(settings.residual_action_clip)), abs(float(action_low)), abs(float(action_high)))
    return -clip, clip


def compose_residual_action(
    *,
    action_pi: np.ndarray,
    action_residual_raw: np.ndarray,
    action_low: float,
    action_high: float,
    settings: ResidualControlSettings,
) -> dict[str, Any]:
    """Compose normalized PI and RL residual actions into an env-facing action."""
    pi = np.asarray(action_pi, dtype=np.float32).reshape(-1)
    raw = np.asarray(action_residual_raw, dtype=np.float32).reshape(pi.shape)
    if bool(settings.residual_zero_test):
        raw = np.zeros_like(raw, dtype=np.float32)

    if settings.residual_action_clip is None:
        raw_clipped = raw.astype(np.float32, copy=True)
    else:
        limit = abs(float(settings.residual_action_clip))
        raw_clipped = np.clip(raw, -limit, limit).astype(np.float32, copy=False)

    residual = (
        float(settings.residual_weight) * float(settings.residual_scale) * raw_clipped
    ).astype(np.float32, copy=False)
    total = (float(settings.baseline_weight) * pi + residual).astype(np.float32, copy=False)
    total_clipped = np.clip(total, float(action_low), float(action_high)).astype(np.float32, copy=False)
    return {
        "action_pi": pi.astype(np.float32, copy=False),
        "action_residual_raw": raw.astype(np.float32, copy=False),
        "action_residual_raw_clipped": raw_clipped.astype(np.float32, copy=False),
        "action_residual": residual,
        "action_total": total,
        "action_total_clipped": total_clipped,
        "residual_clip_applied": bool(np.any(np.abs(raw - raw_clipped) > 1e-7)),
        "action_total_clip_applied": bool(np.any(np.abs(total - total_clipped) > 1e-7)),
    }

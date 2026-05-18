"""Helpers for PI-RL residual action composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


RESIDUAL_CONTROLLER_ALIASES = {"residual", "pi_rl_residual", "pi-rl-residual"}


@dataclass(slots=True)
class ResidualControlSettings:
    """Runtime settings for normalized residual-action composition."""

    residual_scale: float = 0.2
    residual_action_clip: float | None = 0.3
    residual_zero_test: bool = False


def normalize_controller_name(value: Any) -> str:
    """Normalize public controller labels while preserving existing modes."""
    name = str(value).strip().lower()
    if name in RESIDUAL_CONTROLLER_ALIASES:
        return "residual"
    return name


def is_residual_controller(value: Any) -> bool:
    """Return whether a controller label requests PI-RL residual control."""
    return normalize_controller_name(value) == "residual"


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
    )


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

    residual = (float(settings.residual_scale) * raw_clipped).astype(np.float32, copy=False)
    total = (pi + residual).astype(np.float32, copy=False)
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

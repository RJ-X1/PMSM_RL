"""Run PID baseline smoke checks for the gun-servo position environment."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.make_env import make_eval_env
from envs.obs_parser import build_observation_spec
from scripts.core.evaluate import (
    _apply_gun_servo_env_overrides,
    _extract_done_reason,
    _make_cascade_servo_controller,
)
from utils.config import load_yaml, parse_env_config
from utils.experiment_factory import make_env_build_config
from utils.seed import set_seed


CSV_FIELDS = [
    "run_id",
    "algorithm",
    "scenario",
    "seed",
    "episode",
    "step",
    "t_s",
    "theta_ref_rad",
    "theta_ref_deg",
    "theta_L_rad",
    "theta_L_deg",
    "omega_L_rad_s",
    "omega_m_rad_s",
    "omega_m_ref_rad_s",
    "omega_L_cmd_safe",
    "omega_L_cmd_safe_deg_s",
    "omega_m_cmd",
    "omega_cmd_deg_s",
    "e_theta_rad",
    "e_theta_deg",
    "e_omega_load",
    "omega_L_deg_s",
    "a_raw",
    "a_safe",
    "delta_a_safe",
    "i_q",
    "T_e",
    "T_out",
    "T_L",
    "T_hat_L",
    "V_dc",
    "flag_U_safe",
    "flag_E_safe",
    "flag_X_safe",
    "sigma_safe",
    "reward",
    "done",
    "done_reason",
    "controller",
    "env_id",
    "layout",
    "terminated",
    "truncated",
    "cum_reward",
    "gear_output_torque_limit_nm",
    "constraint_violation",
    "act_delta_omega",
]

SUMMARY_FIELDS = [
    "algorithm",
    "scenario",
    "scenario_group",
    "include_in_baseline",
    "seed",
    "final_theta_error_deg",
    "final_theta_L_deg",
    "final_theta_ref_deg",
    "final_omega_L_deg_s",
    "max_abs_omega_L_deg_s",
    "max_abs_omega_L_cmd_deg_s",
    "first_violation_time_s",
    "first_violation_type",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "overshoot_percent",
    "settling_time_s",
    "settling_time_s_0p1deg",
    "recovery_time_s",
    "recovery_time_s_0p1deg",
    "max_abs_iq_a",
    "max_abs_Te_nm",
    "max_abs_Tout_nm",
    "torque_usage_peak",
    "action_variation_sum",
    "constraint_violation_count",
    "episode_return",
    "done_reason",
    "episode_steps",
    "csv_path",
]

DEFAULT_SCENARIO_MAP = {
    "C1_10deg_step": "step_10deg",
    "C2_20deg_step": "step_20deg",
    "C3_trapezoid_tracking": "trapezoid_15deg",
    "C4_sine_tracking": "sine_tracking",
    "C5_disturbance_rejection": "disturbance_step",
    "C2a_20deg_step_safe": "step_20deg_speed_limited_pid_v2",
    "C2b_20deg_step_aggressive": "step_20deg_aggressive",
    "C4a_sine_tracking_safe": "sine_tracking_safe_pid_v2",
    "C4_sine_tracking_aggressive": "sine_tracking_aggressive",
    "C5a_disturbance_after_settling": "disturbance_after_settling_pid_v2",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    return p


def _normalize_key(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any) -> int:
    return int(_safe_float(value, default=0.0) > 0.5)


def _as_scenarios(raw: Any) -> list[str]:
    if raw is None:
        return list(DEFAULT_SCENARIO_MAP)
    if not isinstance(raw, list):
        raise TypeError("Expected eval config 'scenarios' to be a list")
    scenarios: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = item.get("id", item.get("name"))
        else:
            value = item
        if value not in (None, ""):
            scenarios.append(str(value))
    if not scenarios:
        raise ValueError("No scenarios configured for PID smoke evaluation")
    return scenarios


def _optional_scenarios(raw: Any) -> list[str]:
    if raw is None:
        return []
    return _as_scenarios(raw)


def _load_test_conditions(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    conditions = data.get("test_scenarios", [])
    if not isinstance(conditions, list):
        raise TypeError(f"Expected 'test_scenarios' to be a list in {path}")
    return [dict(item) for item in conditions if isinstance(item, dict)]


def _find_condition(
    scenario: str,
    *,
    conditions: list[dict[str, Any]],
    scenario_map: dict[str, Any],
) -> dict[str, Any]:
    target = scenario_map.get(scenario, DEFAULT_SCENARIO_MAP.get(scenario, scenario))
    target_key = _normalize_key(target)
    for condition in conditions:
        candidates = (condition.get("id"), condition.get("name"))
        if any(_normalize_key(candidate) == target_key for candidate in candidates):
            return condition
    available = ", ".join(str(item.get("id", item.get("name", ""))) for item in conditions)
    raise KeyError(f"Scenario '{scenario}' maps to '{target}', but no test condition matched. Available: {available}")


def _episode_steps_from_config(
    env_cfg: Any,
    eval_data: dict[str, Any],
    *,
    scenario: str | None = None,
    condition: dict[str, Any] | None = None,
) -> int | None:
    scenario_max_steps = eval_data.get("scenario_max_steps", {})
    if isinstance(scenario_max_steps, dict) and scenario in scenario_max_steps:
        return int(scenario_max_steps[str(scenario)])
    max_steps = (condition or {}).get("max_steps", eval_data.get("max_steps"))
    if max_steps not in (None, ""):
        return int(max_steps)
    scenario_episode_time_s = eval_data.get("scenario_episode_time_s", {})
    if isinstance(scenario_episode_time_s, dict) and scenario in scenario_episode_time_s:
        episode_time_s = scenario_episode_time_s[str(scenario)]
    else:
        episode_time_s = (condition or {}).get("episode_time_s", eval_data.get("episode_time_s"))
    if episode_time_s in (None, ""):
        return None
    dt = float(dict(getattr(env_cfg, "gun_servo_env", {}) or {}).get("dt", 1.0e-3))
    return max(1, int(round(float(episode_time_s) / max(dt, 1e-12))))


def _with_episode_steps(env_cfg: Any, steps: int | None) -> Any:
    if steps is None:
        return env_cfg
    gun_servo_env = dict(getattr(env_cfg, "gun_servo_env", {}) or {})
    gun_environment = dict(getattr(env_cfg, "gun_environment", {}) or {})
    gun_servo_env["episode_steps"] = int(steps)
    gun_environment["episode_steps"] = int(steps)
    return replace(env_cfg, gun_servo_env=gun_servo_env, gun_environment=gun_environment)


def _make_csv_row(
    *,
    run_id: str,
    scenario: str,
    seed: int,
    episode: int,
    step: int,
    action: np.ndarray,
    reward: float,
    cum_reward: float,
    terminated: bool,
    truncated: bool,
    done_reason: str,
    info: dict[str, Any],
    env_id: str,
    layout: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: float("nan") for field in CSV_FIELDS}
    row.update(
        {
            "run_id": run_id,
            "algorithm": "pid",
            "scenario": scenario,
            "seed": int(seed),
            "episode": int(episode),
            "step": int(step),
            "reward": float(reward),
            "done": int(bool(terminated or truncated)),
            "done_reason": done_reason,
            "controller": "pid",
            "env_id": env_id,
            "layout": layout,
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
            "cum_reward": float(cum_reward),
            "act_delta_omega": float(np.asarray(action, dtype=float).reshape(-1)[0]),
        }
    )
    float_keys = {
        "t_s": ("t_s", "time_s"),
        "theta_ref_rad": ("theta_ref_rad", "theta_ref"),
        "theta_ref_deg": ("theta_ref_deg",),
        "theta_L_rad": ("theta_L_rad", "theta_L"),
        "theta_L_deg": ("theta_L_deg",),
        "omega_L_rad_s": ("omega_L_rad_s", "omega_L"),
        "omega_m_rad_s": ("omega_m_rad_s",),
        "omega_m_ref_rad_s": ("omega_m_ref_rad_s", "omega_m_ref"),
        "omega_L_cmd_safe": ("omega_L_cmd_safe",),
        "omega_L_cmd_safe_deg_s": ("omega_L_cmd_safe_deg_s", "omega_cmd_deg_s"),
        "omega_m_cmd": ("omega_m_cmd",),
        "omega_cmd_deg_s": ("omega_cmd_deg_s",),
        "e_theta_rad": ("e_theta_rad", "e_theta"),
        "e_theta_deg": ("e_theta_deg",),
        "e_omega_load": ("e_omega_load",),
        "omega_L_deg_s": ("omega_L_deg_s",),
        "a_raw": ("a_raw", "action_raw"),
        "a_safe": ("a_safe", "action_safe"),
        "delta_a_safe": ("delta_a_safe",),
        "i_q": ("i_q", "iq", "iq_A"),
        "T_e": ("T_e", "Te", "Te_Nm"),
        "T_out": ("T_out", "T_out_Nm", "torque_out_nm"),
        "T_L": ("T_L", "disturbance_torque", "disturbance_torque_Nm"),
        "T_hat_L": ("T_hat_L", "T_L_hat", "TL_hat", "TL_Nm"),
        "V_dc": ("V_dc",),
        "gear_output_torque_limit_nm": ("gear_output_torque_limit_nm", "T_out_max_Nm"),
    }
    for out_key, candidates in float_keys.items():
        for candidate in candidates:
            if candidate in info:
                row[out_key] = _safe_float(info.get(candidate))
                break
    for flag in ("flag_U_safe", "flag_E_safe", "flag_X_safe", "sigma_safe", "constraint_violation"):
        row[flag] = _safe_int(info.get(flag))
    if not math.isfinite(_safe_float(row.get("omega_L_cmd_safe_deg_s"))):
        omega_cmd_rad = _safe_float(row.get("omega_L_cmd_safe"))
        if math.isfinite(omega_cmd_rad):
            row["omega_L_cmd_safe_deg_s"] = float(np.rad2deg(omega_cmd_rad))
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(column)) for row in rows], dtype=np.float64)


def _first_reference_change(reference: np.ndarray) -> int:
    if reference.size < 2:
        return 0
    finite = reference[np.isfinite(reference)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    changes = np.flatnonzero(np.abs(np.diff(reference)) > 1e-6 * scale)
    return int(changes[0] + 1) if changes.size else 0


def _stays_within_time(
    *,
    time_s: np.ndarray,
    error: np.ndarray,
    threshold: float,
    start_idx: int,
) -> float:
    if time_s.size == 0 or error.size == 0 or time_s.size != error.size:
        return float("nan")
    start_idx = min(max(int(start_idx), 0), len(error) - 1)
    within = np.abs(error[start_idx:]) <= float(threshold)
    stable_from_here = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(stable_from_here)
    if candidates.size == 0:
        return float("nan")
    idx = int(candidates[0] + start_idx)
    return float(time_s[idx] - time_s[start_idx])


def _overshoot_percent(reference: np.ndarray, response: np.ndarray) -> float:
    if reference.size == 0 or response.size != reference.size:
        return float("nan")
    step_idx = _first_reference_change(reference)
    initial_ref = float(reference[max(step_idx - 1, 0)])
    final_ref = float(reference[-1])
    amplitude = final_ref - initial_ref
    if not math.isfinite(amplitude) or abs(amplitude) <= 1e-12:
        return float("nan")
    direction = 1.0 if amplitude > 0.0 else -1.0
    overshoot = max(0.0, float(np.nanmax(direction * (response[step_idx:] - final_ref))))
    return float(100.0 * overshoot / abs(amplitude))


def _recovery_time(
    *,
    time_s: np.ndarray,
    error: np.ndarray,
    disturbance: np.ndarray,
    threshold_deg: float,
) -> float:
    if disturbance.size != error.size or time_s.size != error.size or disturbance.size < 2:
        return float("nan")
    finite = disturbance[np.isfinite(disturbance)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    changes = np.flatnonzero(np.abs(np.diff(disturbance)) > 1e-6 * scale)
    if changes.size == 0:
        return float("nan")
    start_idx = int(changes[0] + 1)
    return _stays_within_time(
        time_s=time_s,
        error=error,
        threshold=float(threshold_deg),
        start_idx=start_idx,
    )


def _max_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _flag_count(rows: list[dict[str, Any]], field: str) -> int:
    return int(np.sum(_series(rows, field) > 0.5))


def _first_violation(rows: list[dict[str, Any]]) -> tuple[float, str]:
    if not rows:
        return float("nan"), ""
    done_reason = str(rows[-1].get("done_reason", ""))
    hard_violation = any(token in done_reason for token in ("theta_limit", "omega_limit", "torque_limit"))
    candidates = ("flag_X_safe", "sigma_safe") if hard_violation else (
        "flag_U_safe",
        "flag_E_safe",
        "flag_X_safe",
        "sigma_safe",
    )
    for row in rows:
        active = [name for name in candidates if _safe_float(row.get(name), default=0.0) > 0.5]
        if active:
            label = "+".join(active)
            if hard_violation:
                label = f"{label}:{done_reason}"
            return _safe_float(row.get("t_s")), label
    return float("nan"), ""


def _summarize_rows(
    *,
    rows: list[dict[str, Any]],
    scenario: str,
    scenario_group: str,
    seed: int,
    threshold_deg: float,
    aux_threshold_deg: float,
    csv_path: Path,
) -> dict[str, Any]:
    time_s = _series(rows, "t_s")
    theta_ref = _series(rows, "theta_ref_deg")
    theta_l = _series(rows, "theta_L_deg")
    error = theta_ref - theta_l
    step_idx = _first_reference_change(theta_ref)
    is_plain_step = "step" in scenario.lower() and "disturbance" not in scenario.lower()
    is_disturbance = "disturbance" in scenario.lower()
    max_abs_tout = _max_abs(_series(rows, "T_out"))
    torque_limit = _max_abs(_series(rows, "gear_output_torque_limit_nm"))
    torque_usage_peak = (
        max_abs_tout / torque_limit
        if math.isfinite(max_abs_tout) and math.isfinite(torque_limit) and torque_limit > 0.0
        else float("nan")
    )
    flag_matrix = np.vstack(
        [
            _series(rows, "flag_U_safe") > 0.5,
            _series(rows, "flag_E_safe") > 0.5,
            _series(rows, "flag_X_safe") > 0.5,
            _series(rows, "sigma_safe") > 0.5,
        ]
    )
    episode_return = float(np.nansum(_series(rows, "reward")))
    first_violation_time, first_violation_type = _first_violation(rows)
    recovery_time = (
        _recovery_time(
            time_s=time_s,
            error=error,
            disturbance=_series(rows, "T_L"),
            threshold_deg=float(threshold_deg),
        )
        if is_disturbance
        else float("nan")
    )
    recovery_time_aux = (
        _recovery_time(
            time_s=time_s,
            error=error,
            disturbance=_series(rows, "T_L"),
            threshold_deg=float(aux_threshold_deg),
        )
        if is_disturbance
        else float("nan")
    )
    return {
        "algorithm": "pid",
        "scenario": scenario,
        "scenario_group": scenario_group,
        "include_in_baseline": int(str(scenario_group) == "primary"),
        "seed": int(seed),
        "final_theta_error_deg": float(error[-1]) if error.size else float("nan"),
        "final_theta_L_deg": float(theta_l[-1]) if theta_l.size else float("nan"),
        "final_theta_ref_deg": float(theta_ref[-1]) if theta_ref.size else float("nan"),
        "final_omega_L_deg_s": float(_series(rows, "omega_L_deg_s")[-1]) if rows else float("nan"),
        "max_abs_omega_L_deg_s": _max_abs(_series(rows, "omega_L_deg_s")),
        "max_abs_omega_L_cmd_deg_s": _max_abs(_series(rows, "omega_L_cmd_safe_deg_s")),
        "first_violation_time_s": first_violation_time,
        "first_violation_type": first_violation_type,
        "flag_U_safe_count": _flag_count(rows, "flag_U_safe"),
        "flag_E_safe_count": _flag_count(rows, "flag_E_safe"),
        "flag_X_safe_count": _flag_count(rows, "flag_X_safe"),
        "sigma_safe_count": _flag_count(rows, "sigma_safe"),
        "rmse_theta_deg": float(math.sqrt(float(np.nanmean(error**2)))) if error.size else float("nan"),
        "mae_theta_deg": float(np.nanmean(np.abs(error))) if error.size else float("nan"),
        "max_abs_theta_error_deg": _max_abs(error),
        "overshoot_percent": _overshoot_percent(theta_ref, theta_l) if is_plain_step else float("nan"),
        "settling_time_s": _stays_within_time(
            time_s=time_s,
            error=error,
            threshold=float(threshold_deg),
            start_idx=step_idx,
        ),
        "settling_time_s_0p1deg": _stays_within_time(
            time_s=time_s,
            error=error,
            threshold=float(aux_threshold_deg),
            start_idx=step_idx,
        ),
        "recovery_time_s": recovery_time,
        "recovery_time_s_0p1deg": recovery_time_aux,
        "max_abs_iq_a": _max_abs(_series(rows, "i_q")),
        "max_abs_Te_nm": _max_abs(_series(rows, "T_e")),
        "max_abs_Tout_nm": max_abs_tout,
        "torque_usage_peak": torque_usage_peak,
        "action_variation_sum": float(np.nansum(np.abs(_series(rows, "delta_a_safe")))),
        "constraint_violation_count": int(np.sum(np.any(flag_matrix, axis=0))),
        "episode_return": episode_return,
        "done_reason": str(rows[-1].get("done_reason", "")) if rows else "",
        "episode_steps": len(rows),
        "csv_path": str(csv_path),
    }


def _plot_position(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _series(rows, "theta_ref_deg"), "k--", linewidth=1.3, label="theta_ref_deg")
    ax.plot(time_s, _series(rows, "theta_L_deg"), linewidth=1.3, label="theta_L_deg")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [deg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_error(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _series(rows, "e_theta_deg"), linewidth=1.3, label="e_theta_deg")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [deg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_speed_command(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _series(rows, "omega_L_cmd_safe_deg_s"), "k--", linewidth=1.2, label="omega_L_cmd_safe")
    ax.plot(time_s, _series(rows, "omega_L_deg_s"), linewidth=1.2, label="omega_L")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [deg/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_torque(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s")
    fig, ax_left = plt.subplots(figsize=(9, 4.8))
    ax_right = ax_left.twinx()
    line_te = ax_left.plot(time_s, _series(rows, "T_e"), linewidth=1.2, label="T_e [Nm]", color="tab:blue")
    line_tout = ax_right.plot(time_s, _series(rows, "T_out"), linewidth=1.2, label="T_out [Nm]", color="tab:orange")
    ax_left.set_xlabel("time [s]")
    ax_left.set_ylabel("T_e [Nm]")
    ax_right.set_ylabel("T_out [Nm]")
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.3)
    lines = line_te + line_tout
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_safety_flags(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, field in enumerate(("flag_U_safe", "flag_E_safe", "flag_X_safe", "sigma_safe")):
        ax.step(time_s, _series(rows, field) + 0.05 * idx, where="post", linewidth=1.1, label=field)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("flag [-]")
    ax.set_ylim(-0.1, 1.35)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_disturbance_zoom(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s")
    torque = _series(rows, "T_L")
    if time_s.size < 2 or torque.size != time_s.size:
        return
    changes = np.flatnonzero(np.abs(np.diff(torque)) > 1e-6 * max(1.0, _max_abs(torque)))
    if changes.size == 0:
        return
    center = float(time_s[int(changes[0] + 1)])
    mask = (time_s >= center - 1.0) & (time_s <= center + 1.0)
    fig, ax_left = plt.subplots(figsize=(9, 4.8))
    ax_right = ax_left.twinx()
    line_err = ax_left.plot(time_s[mask], _series(rows, "e_theta_deg")[mask], linewidth=1.2, label="e_theta_deg")
    line_tl = ax_right.plot(time_s[mask], torque[mask], "k--", linewidth=1.1, label="T_L [Nm]")
    ax_left.axvline(center, color="tab:red", linewidth=0.9, alpha=0.8)
    ax_left.set_xlabel("time [s]")
    ax_left.set_ylabel("position error [deg]")
    ax_right.set_ylabel("disturbance torque [Nm]")
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.3)
    lines = line_err + line_tl
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _copy_configs(*, env_config: Path, eval_config: Path, test_conditions: Path, output_dir: Path) -> dict[str, str]:
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, str] = {}
    for label, src in (
        ("env", env_config),
        ("eval", eval_config),
        ("test_conditions", test_conditions),
    ):
        dst = configs_dir / src.name
        shutil.copy2(src, dst)
        snapshots[label] = str(dst)
    return snapshots


def _write_metadata(
    *,
    output_dir: Path,
    run_name: str,
    env_config: Path,
    eval_config: Path,
    test_conditions: Path,
    snapshots: dict[str, str],
    summaries: list[dict[str, Any]],
) -> None:
    payload = {
        "run_name": run_name,
        "algorithm": "pid",
        "task_name": "gun_servo_position",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config_paths": {
            "env": str(env_config),
            "eval": str(eval_config),
            "test_conditions": str(test_conditions),
        },
        "snapshot_config_paths": snapshots,
        "scenarios": [row["scenario"] for row in summaries],
        "summary_csv": str(output_dir / "summaries" / "pid_smoke_summary.csv"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_one_scenario(
    *,
    base_env_cfg: Any,
    condition: dict[str, Any],
    scenario: str,
    seed: int,
    run_name: str,
    episode_steps: int | None,
    eval_dir: Path,
    figures_dir: Path,
    save_csv: bool,
    save_figures: bool,
) -> tuple[list[dict[str, Any]], Path]:
    env_cfg = _apply_gun_servo_env_overrides(base_env_cfg, condition)
    env_cfg = _with_episode_steps(env_cfg, episode_steps)
    set_seed(seed)
    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed))
    obs, info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    controller = _make_cascade_servo_controller(spec=spec, env=env)
    controller.reset()

    max_steps = int(episode_steps or dict(getattr(env_cfg, "gun_servo_env", {}) or {}).get("episode_steps", 3000))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    cum_reward = 0.0
    csv_path = eval_dir / f"{scenario}_pid.csv"

    for step in range(max_steps):
        action = np.asarray(controller.compute_action(obs), dtype=np.float32).reshape(env.action_space.shape)
        action = np.clip(action, action_low, action_high)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"PID produced non-finite action for {scenario} at step {step}: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(obs).all() or not math.isfinite(float(reward)):
            raise FloatingPointError(f"Env produced non-finite data for {scenario} at step {step}")
        cum_reward += float(reward)
        done_reason = _extract_done_reason(info, terminated=bool(terminated), truncated=bool(truncated))
        rows.append(
            _make_csv_row(
                run_id=run_name,
                scenario=scenario,
                seed=seed,
                episode=0,
                step=step,
                action=action,
                reward=float(reward),
                cum_reward=cum_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
                done_reason=done_reason,
                info=dict(info),
                env_id=str(env_cfg.env_id),
                layout=spec.layout,
            )
        )
        if bool(terminated or truncated):
            break

    if save_csv:
        _write_rows(csv_path, rows, CSV_FIELDS)
    if save_figures:
        _plot_position(rows, figures_dir / f"{scenario}_position.png", title=f"{scenario} position tracking")
        _plot_error(rows, figures_dir / f"{scenario}_error.png", title=f"{scenario} position error")
        _plot_speed_command(rows, figures_dir / f"{scenario}_speed_command.png", title=f"{scenario} speed command")
        _plot_torque(rows, figures_dir / f"{scenario}_torque.png", title=f"{scenario} torque")
        _plot_safety_flags(rows, figures_dir / f"{scenario}_safety_flags.png", title=f"{scenario} safety flags")
        if "C5a" in scenario or "disturbance_after_settling" in scenario:
            _plot_disturbance_zoom(
                rows,
                figures_dir / f"{scenario}_disturbance_zoom.png",
                title=f"{scenario} disturbance zoom",
            )
    env.close()
    return rows, csv_path


def main() -> None:
    args = build_arg_parser().parse_args()
    env_config_path = _resolve_path(args.env_config)
    eval_config_path = _resolve_path(args.eval_config)
    eval_data = load_yaml(eval_config_path)
    test_conditions_path = _resolve_path(
        eval_data.get("test_conditions", eval_data.get("test_conditions_path", "configs/eval/gun_servo_test_conditions.yaml"))
    )
    conditions = _load_test_conditions(test_conditions_path)
    primary_scenarios = _as_scenarios(eval_data.get("scenarios"))
    aggressive_scenarios = _optional_scenarios(eval_data.get("aggressive_scenarios"))
    scenarios = list(dict.fromkeys([*primary_scenarios, *aggressive_scenarios]))
    scenario_groups = {
        **{scenario: "primary" for scenario in primary_scenarios},
        **{scenario: "aggressive" for scenario in aggressive_scenarios},
    }
    scenario_map = {
        **DEFAULT_SCENARIO_MAP,
        **dict(eval_data.get("scenario_condition_map", {}) or {}),
    }
    run_name = str(eval_data.get("run_name", "gun_servo_pid_smoke"))
    output_dir = Path(args.output_dir) if args.output_dir is not None else ROOT / "outputs" / "runs" / run_name
    output_dir = _resolve_path(output_dir)
    eval_dir = output_dir / "eval"
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    for path in (eval_dir, summaries_dir, figures_dir, output_dir / "configs"):
        path.mkdir(parents=True, exist_ok=True)

    base_env_cfg = parse_env_config(env_config_path)
    seed = int(eval_data.get("seed", base_env_cfg.seed))
    save_csv = bool(eval_data.get("save_csv", True))
    save_summary = bool(eval_data.get("save_summary", True))
    save_figures = bool(eval_data.get("save_figures", True))
    threshold_deg = float(eval_data.get("settling_threshold_deg", 0.02))
    aux_threshold_deg = float(eval_data.get("aux_settling_threshold_deg", 0.1))

    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        condition = _find_condition(scenario, conditions=conditions, scenario_map=scenario_map)
        episode_steps = _episode_steps_from_config(
            base_env_cfg,
            eval_data,
            scenario=scenario,
            condition=condition,
        )
        rows, csv_path = _run_one_scenario(
            base_env_cfg=base_env_cfg,
            condition=condition,
            scenario=scenario,
            seed=seed,
            run_name=run_name,
            episode_steps=episode_steps,
            eval_dir=eval_dir,
            figures_dir=figures_dir,
            save_csv=save_csv,
            save_figures=save_figures,
        )
        summaries.append(
            _summarize_rows(
                rows=rows,
                scenario=scenario,
                scenario_group=scenario_groups.get(scenario, "primary"),
                seed=seed,
                threshold_deg=threshold_deg,
                aux_threshold_deg=aux_threshold_deg,
                csv_path=csv_path,
            )
        )
        print(
            f"scenario={scenario} steps={len(rows)} return={summaries[-1]['episode_return']:.3f} "
            f"done_reason={summaries[-1]['done_reason']} csv={csv_path}"
        )

    summary_csv = summaries_dir / "pid_smoke_summary.csv"
    if save_summary:
        _write_rows(summary_csv, summaries, SUMMARY_FIELDS)
    snapshots = _copy_configs(
        env_config=env_config_path,
        eval_config=eval_config_path,
        test_conditions=test_conditions_path,
        output_dir=output_dir,
    )
    _write_metadata(
        output_dir=output_dir,
        run_name=run_name,
        env_config=env_config_path,
        eval_config=eval_config_path,
        test_conditions=test_conditions_path,
        snapshots=snapshots,
        summaries=summaries,
    )
    print(f"saved PID smoke summary: {summary_csv}")
    print(f"saved PID smoke outputs: {output_dir}")


if __name__ == "__main__":
    main()

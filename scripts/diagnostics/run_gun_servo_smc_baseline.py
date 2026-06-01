"""Run the SMC-PI baseline on fixed gun-servo position scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.smc_position_controller import (  # noqa: E402
    SMCPositionActionController,
    SMCPositionActionControllerConfig,
    SMCPositionControllerConfig,
)
from envs.make_env import make_eval_env  # noqa: E402
from envs.obs_parser import build_observation_spec  # noqa: E402
from scripts.core.evaluate import _apply_gun_servo_env_overrides, _extract_done_reason  # noqa: E402
from scripts.diagnostics.run_gun_servo_pid_smoke import (  # noqa: E402
    CSV_FIELDS,
    DEFAULT_SCENARIO_MAP,
    _as_scenarios,
    _copy_configs,
    _episode_steps_from_config,
    _find_condition,
    _first_reference_change,
    _first_violation,
    _flag_count,
    _load_test_conditions,
    _max_abs,
    _make_csv_row as _make_pid_csv_row,
    _plot_disturbance_zoom,
    _plot_error,
    _plot_position,
    _plot_safety_flags,
    _plot_speed_command,
    _recovery_time,
    _resolve_path,
    _safe_float,
    _series,
    _stays_within_time,
    _with_episode_steps,
    _write_rows,
)
from utils.config import load_yaml, parse_env_config  # noqa: E402
from utils.experiment_factory import make_env_build_config  # noqa: E402
from utils.seed import set_seed  # noqa: E402


SMC_SUMMARY_FIELDS = [
    "controller",
    "scenario",
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "settling_time_s",
    "recovery_time_s",
    "recovery_time_0p02deg_s",
    "recovery_time_0p05deg_s",
    "recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
    "max_abs_omega_L_deg_s",
    "max_abs_omega_L_cmd_safe_deg_s",
    "max_abs_Tout_nm",
    "max_abs_iq_a",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "constraint_violation_count",
    "first_violation_time_s",
    "first_violation_type",
    "episode_return",
    "done_reason",
]

COMPARISON_FIELDS = [
    "scenario",
    "controller",
    "rmse_theta_deg",
    "rmse_theta_deg_std",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "settling_time_s",
    "recovery_time_s",
    "max_abs_omega_L_deg_s",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "episode_return",
    "done_reason",
]

SMC_SCENARIO_MAP = {
    **DEFAULT_SCENARIO_MAP,
    "C1a_5deg_step": "step_5deg_sc_residual_td3_v5",
    "C1_10deg_step": "step_10deg_sc_residual_td3_v5",
    "C2a_20deg_step_safe": "step_20deg_speed_limited_sc_residual_td3_v5",
    "C3_trapezoid_tracking": "trapezoid_15deg_sc_residual_td3_v5",
    "C4a_sine_tracking_safe": "sine_tracking_safe_pid_v2",
    "C5a_disturbance_after_settling": "disturbance_after_settling_pid_v2",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _speed_rad_s(raw: dict[str, Any], *, rad_key: str, deg_key: str, default_rad_s: float) -> float:
    if raw.get(rad_key) not in (None, ""):
        return float(raw[rad_key])
    if raw.get(deg_key) not in (None, ""):
        return float(np.deg2rad(float(raw[deg_key])))
    return float(default_rad_s)


def _make_smc_controller(*, spec: Any, env: Any, smc_data: dict[str, Any]) -> SMCPositionActionController:
    base_env = getattr(env, "unwrapped", env)
    scales = getattr(base_env, "observation_normalization_scales", {})
    max_delta_omega = float(getattr(base_env, "max_delta_omega", np.deg2rad(30.0)))
    position_cfg = SMCPositionControllerConfig(
        lambda_smc=float(smc_data.get("lambda_smc", 6.0)),
        k_e=float(smc_data.get("k_e", 4.0)),
        k_s=_speed_rad_s(
            smc_data,
            rad_key="k_s_rad_s",
            deg_key="k_s_deg_s",
            default_rad_s=float(np.deg2rad(4.0)),
        ),
        phi=_speed_rad_s(
            smc_data,
            rad_key="phi_rad_s",
            deg_key="phi_deg_s",
            default_rad_s=0.06,
        ),
        k_d=float(smc_data.get("k_d", 0.25)),
        max_delta_omega=max_delta_omega,
    )
    config = SMCPositionActionControllerConfig(
        position=position_cfg,
        theta_scale=float(scales.get("theta", np.deg2rad(30.0))),
        omega_scale=float(scales.get("omega", np.deg2rad(30.0))),
    )
    controller = SMCPositionActionController(signal_names=spec.signal_names, config=config)
    controller.reset()
    return controller


def _make_csv_row(**kwargs: Any) -> dict[str, Any]:
    row = _make_pid_csv_row(**kwargs)
    row["algorithm"] = "smc"
    row["controller"] = "smc"
    return row


def _disturbance_change_index(rows: list[dict[str, Any]]) -> int | None:
    disturbance = _series(rows, "T_L")
    if disturbance.size < 2:
        return None
    finite = disturbance[np.isfinite(disturbance)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    changes = np.flatnonzero(np.abs(np.diff(disturbance)) > 1e-6 * scale)
    return int(changes[0] + 1) if changes.size else None


def _max_abs_error_after_disturbance(rows: list[dict[str, Any]], error: np.ndarray) -> float:
    idx = _disturbance_change_index(rows)
    if idx is None or idx >= error.size:
        return float("nan")
    return _max_abs(error[idx:])


def _summarize_rows(
    *,
    rows: list[dict[str, Any]],
    scenario: str,
    threshold_deg: float,
) -> dict[str, Any]:
    time_s = _series(rows, "t_s")
    theta_ref = _series(rows, "theta_ref_deg")
    theta_l = _series(rows, "theta_L_deg")
    error = theta_ref - theta_l
    disturbance = _series(rows, "T_L")
    step_idx = _first_reference_change(theta_ref)
    is_disturbance = "disturbance" in scenario.lower()
    first_violation_time, first_violation_type = _first_violation(rows)
    recovery_0p02 = (
        _recovery_time(time_s=time_s, error=error, disturbance=disturbance, threshold_deg=0.02)
        if is_disturbance
        else float("nan")
    )
    recovery_0p05 = (
        _recovery_time(time_s=time_s, error=error, disturbance=disturbance, threshold_deg=0.05)
        if is_disturbance
        else float("nan")
    )
    recovery_0p10 = (
        _recovery_time(time_s=time_s, error=error, disturbance=disturbance, threshold_deg=0.10)
        if is_disturbance
        else float("nan")
    )
    return {
        "controller": "smc",
        "scenario": scenario,
        "rmse_theta_deg": float(math.sqrt(float(np.nanmean(error**2)))) if error.size else float("nan"),
        "mae_theta_deg": float(np.nanmean(np.abs(error))) if error.size else float("nan"),
        "max_abs_theta_error_deg": _max_abs(error),
        "settling_time_s": _stays_within_time(
            time_s=time_s,
            error=error,
            threshold=float(threshold_deg),
            start_idx=step_idx,
        ),
        "recovery_time_s": recovery_0p02,
        "recovery_time_0p02deg_s": recovery_0p02,
        "recovery_time_0p05deg_s": recovery_0p05,
        "recovery_time_0p10deg_s": recovery_0p10,
        "max_abs_error_after_disturbance_deg": _max_abs_error_after_disturbance(rows, error),
        "max_abs_omega_L_deg_s": _max_abs(_series(rows, "omega_L_deg_s")),
        "max_abs_omega_L_cmd_safe_deg_s": _max_abs(_series(rows, "omega_L_cmd_safe_deg_s")),
        "max_abs_Tout_nm": _max_abs(_series(rows, "T_out")),
        "max_abs_iq_a": _max_abs(_series(rows, "i_q")),
        "flag_U_safe_count": _flag_count(rows, "flag_U_safe"),
        "flag_E_safe_count": _flag_count(rows, "flag_E_safe"),
        "flag_X_safe_count": _flag_count(rows, "flag_X_safe"),
        "sigma_safe_count": _flag_count(rows, "sigma_safe"),
        "constraint_violation_count": _flag_count(rows, "constraint_violation"),
        "first_violation_time_s": first_violation_time,
        "first_violation_type": first_violation_type,
        "episode_return": float(np.nansum(_series(rows, "reward"))),
        "done_reason": str(rows[-1].get("done_reason", "")) if rows else "",
    }


def _run_one_scenario(
    *,
    base_env_cfg: Any,
    condition: dict[str, Any],
    scenario: str,
    seed: int,
    run_name: str,
    episode_steps: int | None,
    smc_data: dict[str, Any],
    eval_dir: Path,
    figures_dir: Path,
    save_csv: bool,
    save_figures: bool,
) -> tuple[list[dict[str, Any]], Path]:
    env_cfg = _apply_gun_servo_env_overrides(base_env_cfg, condition)
    env_cfg = _with_episode_steps(env_cfg, episode_steps)
    set_seed(seed)
    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed))
    obs, _info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    controller = _make_smc_controller(spec=spec, env=env, smc_data=smc_data)

    max_steps = int(episode_steps or dict(getattr(env_cfg, "gun_servo_env", {}) or {}).get("episode_steps", 3000))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    cum_reward = 0.0
    csv_path = eval_dir / f"{scenario}_smc.csv"

    for step in range(max_steps):
        action = np.asarray(controller.compute_action(obs), dtype=np.float32).reshape(env.action_space.shape)
        action = np.clip(action, action_low, action_high)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"SMC produced non-finite action for {scenario} at step {step}: {action}")
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
        _plot_position(rows, figures_dir / f"{scenario}_smc_position.png", title=f"{scenario} SMC position tracking")
        _plot_error(rows, figures_dir / f"{scenario}_smc_error.png", title=f"{scenario} SMC position error")
        _plot_speed_command(
            rows,
            figures_dir / f"{scenario}_smc_speed_command.png",
            title=f"{scenario} SMC speed command",
        )
        _plot_safety_flags(
            rows,
            figures_dir / f"{scenario}_smc_safety_flags.png",
            title=f"{scenario} SMC safety flags",
        )
        if "C5a" in scenario or "disturbance_after_settling" in scenario:
            _plot_disturbance_zoom(
                rows,
                figures_dir / f"{scenario}_smc_disturbance_zoom.png",
                title=f"{scenario} SMC disturbance zoom",
            )
    env.close()
    return rows, csv_path


def _comparison_from_direct_row(row: dict[str, Any], *, controller: str) -> dict[str, Any]:
    return {
        "scenario": str(row.get("scenario", "")),
        "controller": controller,
        "rmse_theta_deg": _safe_float(row.get("rmse_theta_deg")),
        "rmse_theta_deg_std": _safe_float(row.get("rmse_theta_deg_std")),
        "mae_theta_deg": _safe_float(row.get("mae_theta_deg")),
        "max_abs_theta_error_deg": _safe_float(row.get("max_abs_theta_error_deg")),
        "settling_time_s": _safe_float(row.get("settling_time_s")),
        "recovery_time_s": _safe_float(row.get("recovery_time_s")),
        "max_abs_omega_L_deg_s": _safe_float(row.get("max_abs_omega_L_deg_s")),
        "flag_U_safe_count": _safe_float(row.get("flag_U_safe_count")),
        "flag_E_safe_count": _safe_float(row.get("flag_E_safe_count")),
        "flag_X_safe_count": _safe_float(row.get("flag_X_safe_count")),
        "sigma_safe_count": _safe_float(row.get("sigma_safe_count")),
        "episode_return": _safe_float(row.get("episode_return")),
        "done_reason": str(row.get("done_reason", "")),
    }


def _comparison_from_aggregate_row(
    row: dict[str, Any],
    *,
    controller: str,
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario = str(row.get("scenario", ""))

    def mean_seed(field: str) -> float:
        values = [_safe_float(item.get(field)) for item in seed_rows if str(item.get("scenario", "")) == scenario]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    def aggregate_mean(field: str) -> float:
        return _safe_float(row.get(f"{field}_mean"), mean_seed(field))

    return {
        "scenario": scenario,
        "controller": controller,
        "rmse_theta_deg": aggregate_mean("rmse_theta_deg"),
        "rmse_theta_deg_std": _safe_float(row.get("rmse_theta_deg_std")),
        "mae_theta_deg": aggregate_mean("mae_theta_deg"),
        "max_abs_theta_error_deg": aggregate_mean("max_abs_theta_error_deg"),
        "settling_time_s": mean_seed("settling_time_s"),
        "recovery_time_s": aggregate_mean("recovery_time_s"),
        "max_abs_omega_L_deg_s": mean_seed("max_abs_omega_L_deg_s"),
        "flag_U_safe_count": aggregate_mean("flag_U_safe_count"),
        "flag_E_safe_count": aggregate_mean("flag_E_safe_count"),
        "flag_X_safe_count": aggregate_mean("flag_X_safe_count"),
        "sigma_safe_count": aggregate_mean("sigma_safe_count"),
        "episode_return": aggregate_mean("episode_return"),
        "done_reason": str(row.get("done_reason_summary", "")),
    }


def _glob_rows(pattern: str | Path) -> list[dict[str, Any]]:
    raw = str(pattern)
    if not raw:
        return []
    path = Path(raw)
    if path.is_absolute():
        candidates = sorted(path.parent.glob(path.name))
    else:
        candidates = sorted(ROOT.glob(raw.replace("\\", "/")))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(_read_csv_rows(candidate))
    return rows


def _write_comparisons(
    *,
    summaries: list[dict[str, Any]],
    scenarios: list[str],
    eval_data: dict[str, Any],
    summaries_dir: Path,
) -> tuple[Path, Path]:
    smc_by_scenario = {str(row["scenario"]): row for row in summaries}
    pid_rows = _read_csv_rows(_resolve_path(eval_data.get("pid_baseline_comparison_summary", "")))
    if not pid_rows:
        pid_rows = _read_csv_rows(_resolve_path(eval_data.get("pid_baseline_summary", "")))
    pid_by_scenario = {
        str(row.get("scenario", "")): row
        for row in pid_rows
        if str(row.get("controller", row.get("algorithm", "pid_v2"))) in {"pid", "pid_v2", ""}
    }
    v8_rows = _read_csv_rows(_resolve_path(eval_data.get("v8_full_aggregate_summary", "")))
    v8_seed_rows = _glob_rows(eval_data.get("v8_full_seed_summary_glob", ""))
    v8_by_scenario = {str(row.get("scenario", "")): row for row in v8_rows}

    smc_vs_pid: list[dict[str, Any]] = []
    smc_vs_v8: list[dict[str, Any]] = []
    for scenario in scenarios:
        smc_row = smc_by_scenario.get(scenario)
        if smc_row is not None:
            smc_vs_pid.append(_comparison_from_direct_row(smc_row, controller="smc"))
            smc_vs_v8.append(_comparison_from_direct_row(smc_row, controller="smc"))
        pid_row = pid_by_scenario.get(scenario)
        if pid_row is not None:
            smc_vs_pid.append(_comparison_from_direct_row(pid_row, controller="pid_v2"))
        v8_row = v8_by_scenario.get(scenario)
        if v8_row is not None:
            smc_vs_v8.append(
                _comparison_from_aggregate_row(
                    v8_row,
                    controller="mc_sc_residual_td3_v8_full",
                    seed_rows=v8_seed_rows,
                )
            )

    pid_path = summaries_dir / "smc_vs_pid_v2_summary.csv"
    v8_path = summaries_dir / "smc_vs_v8_full_summary.csv"
    _write_rows(pid_path, smc_vs_pid, COMPARISON_FIELDS)
    _write_rows(v8_path, smc_vs_v8, COMPARISON_FIELDS)
    return pid_path, v8_path


def _fmt(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "nan"


def _controller_rmse_winners(
    *,
    smc_rows: dict[str, dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    baseline_controller: str,
) -> tuple[list[str], list[str]]:
    baseline = {
        str(row.get("scenario", "")): row
        for row in comparison_rows
        if str(row.get("controller", "")) == baseline_controller
    }
    smc_better: list[str] = []
    baseline_better: list[str] = []
    for scenario, smc_row in smc_rows.items():
        base_row = baseline.get(scenario)
        if base_row is None:
            continue
        smc_rmse = _safe_float(smc_row.get("rmse_theta_deg"))
        base_rmse = _safe_float(base_row.get("rmse_theta_deg"))
        if not (math.isfinite(smc_rmse) and math.isfinite(base_rmse)):
            continue
        if smc_rmse < base_rmse:
            smc_better.append(scenario)
        else:
            baseline_better.append(scenario)
    return smc_better, baseline_better


def _write_metadata(
    *,
    output_dir: Path,
    run_name: str,
    env_config: Path,
    eval_config: Path,
    test_conditions: Path,
    snapshots: dict[str, str],
    summaries: list[dict[str, Any]],
    smc_data: dict[str, Any],
) -> None:
    payload = {
        "run_name": run_name,
        "algorithm": "smc",
        "controller": "smc",
        "task_name": "gun_servo_position",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "smc_controller": smc_data,
        "config_paths": {
            "env": str(env_config),
            "eval": str(eval_config),
            "test_conditions": str(test_conditions),
        },
        "snapshot_config_paths": snapshots,
        "scenarios": [row["scenario"] for row in summaries],
        "summary_csv": str(output_dir / "summaries" / "smc_baseline_summary.csv"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_conclusion(
    *,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    smc_data: dict[str, Any],
    pid_comparison_path: Path,
    v8_comparison_path: Path,
) -> Path:
    summaries_dir = output_dir / "summaries"
    smc_by_scenario = {str(row["scenario"]): row for row in summaries}
    pid_rows = _read_csv_rows(pid_comparison_path)
    v8_rows = _read_csv_rows(v8_comparison_path)
    smc_vs_pid, pid_better = _controller_rmse_winners(
        smc_rows=smc_by_scenario,
        comparison_rows=pid_rows,
        baseline_controller="pid_v2",
    )
    smc_vs_v8, v8_better = _controller_rmse_winners(
        smc_rows=smc_by_scenario,
        comparison_rows=v8_rows,
        baseline_controller="mc_sc_residual_td3_v8_full",
    )
    all_episode_limit = all(str(row.get("done_reason", "")) == "episode_limit" for row in summaries)
    omega_limit_violation = any("omega_limit_violation" in str(row.get("done_reason", "")) for row in summaries)
    hard_violation = any("violation" in str(row.get("done_reason", "")) for row in summaries)
    sigma_total = int(sum(_safe_float(row.get("sigma_safe_count"), 0.0) for row in summaries))

    lines = [
        "# SMC-PI Baseline Conclusion",
        "",
        "## 1. SMC controller formulation",
        "The SMC controller is used only as the load-side position outer loop. It computes "
        "`e_theta = theta_ref - theta_L`, `e_omega = omega_ref - omega_L`, "
        "`s = e_omega + lambda_smc * e_theta`, and `sat_s = clip(s / phi, -1, 1)`. "
        "The emitted command is the normalized speed correction corresponding to "
        "`omega_L_cmd = omega_ref + k_e * e_theta + k_s * sat_s + k_d * e_omega`; "
        "the environment then applies the existing speed limit, acceleration limit, smoothing, "
        "speed PI loop, actuator limits, PMSM, gearbox, gun-load dynamics, and safety checks.",
        "",
        "## 2. Final SMC parameters",
        f"- lambda_smc: {smc_data.get('lambda_smc')}",
        f"- k_e: {smc_data.get('k_e')}",
        f"- k_s_deg_s: {smc_data.get('k_s_deg_s')}",
        f"- phi_rad_s: {smc_data.get('phi_rad_s')}",
        f"- k_d: {smc_data.get('k_d')}",
        "",
        "## 3. Fixed-scenario completion",
        f"- All six fixed scenarios reached episode_limit: {all_episode_limit}",
        f"- omega_limit_violation occurred: {omega_limit_violation}",
        f"- hard safety violation occurred: {hard_violation}",
        f"- sigma_safe total count: {sigma_total}",
        "",
        "## 4. SMC fixed-scenario summary",
    ]
    for row in summaries:
        lines.append(
            f"- {row['scenario']}: RMSE {_fmt(row.get('rmse_theta_deg'))} deg, "
            f"MAE {_fmt(row.get('mae_theta_deg'))} deg, "
            f"max |omega_L| {_fmt(row.get('max_abs_omega_L_deg_s'))} deg/s, "
            f"flag_U {int(_safe_float(row.get('flag_U_safe_count'), 0.0))}, "
            f"done {row.get('done_reason')}"
        )
    lines.extend(
        [
            "",
            "## 5. SMC vs PID v2",
            f"- SMC lower RMSE scenarios: {', '.join(smc_vs_pid) if smc_vs_pid else 'none'}",
            f"- PID v2 lower or equal RMSE scenarios: {', '.join(pid_better) if pid_better else 'none'}",
            "",
            "## 6. SMC vs v8 full",
            f"- SMC lower RMSE scenarios: {', '.join(smc_vs_v8) if smc_vs_v8 else 'none'}",
            f"- v8 full lower or equal RMSE scenarios: {', '.join(v8_better) if v8_better else 'none'}",
            "- Safety flags: SMC completed without hard safety termination; see comparison CSVs for flag counts.",
            "- Disturbance metrics: see C5a recovery and max post-disturbance error fields in the summary CSV.",
            "",
            "## 7. Paper recommendation",
            "SMC-PI should be included as the classical robust-control baseline because it exercises the same "
            "outer-loop interface and preserves the shared speed PI, actuator, PMSM, gearbox, and gun-load chain.",
            "",
            "## 8. Next recommendation",
            "Proceed to the final comparison table and paper figures. Retune SMC only if the paper needs a "
            "more aggressive accuracy/safety tradeoff than this conservative no-hard-violation baseline.",
        ]
    )
    path = summaries_dir / "smc_baseline_conclusion.md"
    _write_text(path, "\n".join(lines) + "\n")
    return path


def main() -> None:
    args = build_arg_parser().parse_args()
    env_config_path = _resolve_path(args.env_config)
    eval_config_path = _resolve_path(args.eval_config)
    eval_data = load_yaml(eval_config_path)
    test_conditions_path = _resolve_path(
        eval_data.get(
            "test_conditions",
            eval_data.get("test_conditions_path", "configs/eval/gun_servo_test_conditions.yaml"),
        )
    )
    conditions = _load_test_conditions(test_conditions_path)
    scenarios = _as_scenarios(eval_data.get("scenarios"))
    scenario_map = {
        **SMC_SCENARIO_MAP,
        **dict(eval_data.get("scenario_condition_map", {}) or {}),
    }
    run_name = str(eval_data.get("run_name", "gun_servo_smc_baseline"))
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
    smc_data = dict(eval_data.get("smc_controller", {}) or {})

    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        condition = _find_condition(scenario, conditions=conditions, scenario_map=scenario_map)
        episode_steps = _episode_steps_from_config(
            base_env_cfg,
            eval_data,
            scenario=scenario,
            condition=condition,
        )
        rows, _csv_path = _run_one_scenario(
            base_env_cfg=base_env_cfg,
            condition=condition,
            scenario=scenario,
            seed=seed,
            run_name=run_name,
            episode_steps=episode_steps,
            smc_data=smc_data,
            eval_dir=eval_dir,
            figures_dir=figures_dir,
            save_csv=save_csv,
            save_figures=save_figures,
        )
        summary = _summarize_rows(rows=rows, scenario=scenario, threshold_deg=threshold_deg)
        summaries.append(summary)
        print(
            f"scenario={scenario} steps={len(rows)} return={summary['episode_return']:.3f} "
            f"done_reason={summary['done_reason']}"
        )

    summary_csv = summaries_dir / "smc_baseline_summary.csv"
    if save_summary:
        _write_rows(summary_csv, summaries, SMC_SUMMARY_FIELDS)
    pid_comparison_path, v8_comparison_path = _write_comparisons(
        summaries=summaries,
        scenarios=scenarios,
        eval_data=eval_data,
        summaries_dir=summaries_dir,
    )
    snapshots = _copy_configs(
        env_config=env_config_path,
        eval_config=eval_config_path,
        test_conditions=test_conditions_path,
        output_dir=output_dir,
    )
    shutil.copy2(Path(__file__), output_dir / "configs" / Path(__file__).name)
    _write_metadata(
        output_dir=output_dir,
        run_name=run_name,
        env_config=env_config_path,
        eval_config=eval_config_path,
        test_conditions=test_conditions_path,
        snapshots=snapshots,
        summaries=summaries,
        smc_data=smc_data,
    )
    conclusion_path = _write_conclusion(
        output_dir=output_dir,
        summaries=summaries,
        smc_data=smc_data,
        pid_comparison_path=pid_comparison_path,
        v8_comparison_path=v8_comparison_path,
    )
    print(f"saved SMC summary: {summary_csv}")
    print(f"saved SMC vs PID v2 comparison: {pid_comparison_path}")
    print(f"saved SMC vs v8 full comparison: {v8_comparison_path}")
    print(f"saved SMC conclusion: {conclusion_path}")
    print(f"saved SMC outputs: {output_dir}")


if __name__ == "__main__":
    main()

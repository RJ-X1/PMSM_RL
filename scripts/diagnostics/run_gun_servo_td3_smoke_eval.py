"""Batch-evaluate a TD3 gun-servo smoke checkpoint and compare it with PID v2."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.evaluate import run_evaluation
from utils.config import load_yaml


SUMMARY_FIELDS = [
    "scenario",
    "controller",
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "settling_time_s",
    "recovery_time_s",
    "error_at_disturbance_deg",
    "max_abs_error_after_disturbance_deg",
    "peak_error_in_1s_after_disturbance_deg",
    "recovery_time_0p02deg_s",
    "recovery_time_0p05deg_s",
    "recovery_time_0p10deg_s",
    "mean_a_raw",
    "max_abs_a_raw",
    "mean_a_safe",
    "max_abs_a_safe",
    "mean_omega_L_cmd_safe_deg_s",
    "max_abs_omega_L_cmd_safe_deg_s",
    "max_abs_omega_L_deg_s",
    "max_abs_Tout_nm",
    "constraint_violation_count",
    "first_violation_time_s",
    "first_violation_type",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "episode_return",
    "done_reason",
    "checkpoint_path",
    "csv_path",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series(rows: list[dict[str, str]], *columns: str) -> np.ndarray:
    for column in columns:
        if rows and column in rows[0]:
            return np.asarray([_safe_float(row.get(column)) for row in rows], dtype=np.float64)
    return np.full(len(rows), np.nan, dtype=np.float64)


def _max_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _settling_time(time_s: np.ndarray, error: np.ndarray, *, threshold_deg: float = 0.02) -> float:
    if time_s.size == 0 or time_s.size != error.size:
        return float("nan")
    within = np.abs(error) <= float(threshold_deg)
    stable = np.logical_and.accumulate(within[::-1])[::-1]
    idx = np.flatnonzero(stable)
    return float(time_s[int(idx[0])]) if idx.size else float("nan")


def _recovery_time(
    time_s: np.ndarray,
    error: np.ndarray,
    disturbance: np.ndarray,
    *,
    threshold_deg: float = 0.1,
) -> float:
    if time_s.size == 0 or time_s.size != error.size or disturbance.size != error.size:
        return float("nan")
    finite_disturbance = disturbance[np.isfinite(disturbance)]
    if finite_disturbance.size == 0:
        return float("nan")
    threshold = max(1e-9, 0.05 * max(1.0, float(np.nanmax(np.abs(finite_disturbance)))))
    changes = np.flatnonzero(np.abs(np.diff(disturbance)) > threshold)
    if changes.size == 0:
        active = np.flatnonzero(np.abs(disturbance) > threshold)
        if active.size == 0:
            return float("nan")
        start = int(active[0])
    else:
        start = int(changes[0] + 1)
    post_error = np.abs(error[start:])
    finite_error = post_error[np.isfinite(post_error)]
    if finite_error.size == 0:
        return float("nan")
    threshold_error = max(float(threshold_deg), 0.02 * max(1.0, float(np.nanmax(finite_error))))
    recovered = np.flatnonzero(post_error <= threshold_error)
    if recovered.size == 0:
        return float("nan")
    return float(time_s[start + int(recovered[0])] - time_s[start])


def _disturbance_start_index(disturbance: np.ndarray) -> int | None:
    finite_disturbance = disturbance[np.isfinite(disturbance)]
    if finite_disturbance.size == 0:
        return None
    threshold = max(1e-9, 0.05 * max(1.0, float(np.nanmax(np.abs(finite_disturbance)))))
    changes = np.flatnonzero(np.abs(np.diff(disturbance)) > threshold)
    if changes.size:
        return int(changes[0] + 1)
    active = np.flatnonzero(np.abs(disturbance) > threshold)
    return int(active[0]) if active.size else None


def _first_recovery_time(
    time_s: np.ndarray,
    error: np.ndarray,
    start: int,
    threshold_deg: float,
) -> float:
    post_error = np.abs(error[start:])
    recovered = np.flatnonzero(post_error <= float(threshold_deg))
    if recovered.size == 0:
        return float("nan")
    return float(time_s[start + int(recovered[0])] - time_s[start])


def _disturbance_metrics(time_s: np.ndarray, error: np.ndarray, disturbance: np.ndarray) -> dict[str, float]:
    empty = {
        "error_at_disturbance_deg": float("nan"),
        "max_abs_error_after_disturbance_deg": float("nan"),
        "peak_error_in_1s_after_disturbance_deg": float("nan"),
        "recovery_time_0p02deg_s": float("nan"),
        "recovery_time_0p05deg_s": float("nan"),
        "recovery_time_0p10deg_s": float("nan"),
    }
    if time_s.size == 0 or time_s.size != error.size or disturbance.size != error.size:
        return empty
    start = _disturbance_start_index(disturbance)
    if start is None or start >= error.size or not math.isfinite(float(time_s[start])):
        return empty
    post_error = error[start:]
    finite_post_error = post_error[np.isfinite(post_error)]
    if finite_post_error.size == 0:
        return empty
    one_second = np.logical_and(time_s[start:] >= time_s[start], time_s[start:] <= time_s[start] + 1.0)
    one_second_error = post_error[one_second]
    finite_one_second_error = one_second_error[np.isfinite(one_second_error)]
    return {
        "error_at_disturbance_deg": float(error[start]),
        "max_abs_error_after_disturbance_deg": float(np.max(np.abs(finite_post_error))),
        "peak_error_in_1s_after_disturbance_deg": (
            float(np.max(np.abs(finite_one_second_error))) if finite_one_second_error.size else float("nan")
        ),
        "recovery_time_0p02deg_s": _first_recovery_time(time_s, error, start, 0.02),
        "recovery_time_0p05deg_s": _first_recovery_time(time_s, error, start, 0.05),
        "recovery_time_0p10deg_s": _first_recovery_time(time_s, error, start, 0.10),
    }


def _constraint_count(rows: list[dict[str, str]]) -> int:
    flags = [
        _series(rows, "flag_U_safe"),
        _series(rows, "flag_E_safe"),
        _series(rows, "flag_X_safe"),
        _series(rows, "sigma_safe"),
        _series(rows, "constraint_violation"),
    ]
    matrix = np.vstack([flag > 0.5 for flag in flags])
    return int(np.sum(np.any(matrix, axis=0)))


def _flag_count(rows: list[dict[str, str]], column: str) -> int:
    return int(np.sum(_series(rows, column) > 0.5))


def _omega_cmd_deg_s(rows: list[dict[str, str]]) -> np.ndarray:
    values = _series(rows, "omega_L_cmd_safe_deg_s", "omega_cmd_deg_s")
    if np.any(np.isfinite(values)):
        return values
    rad_values = _series(rows, "omega_L_cmd_safe")
    return np.rad2deg(rad_values)


def _first_violation(rows: list[dict[str, str]]) -> tuple[float, str]:
    if not rows:
        return float("nan"), ""
    done_reason = str(rows[-1].get("done_reason", ""))
    hard_violation = "violation" in done_reason
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
            return _safe_float(row.get("t_s", row.get("time_s"))), label
    return float("nan"), ""


def summarize_trace(path: Path, *, scenario: str, controller: str, checkpoint_path: Path | None = None) -> dict[str, Any]:
    rows = _read_rows(path)
    theta_ref = _series(rows, "theta_ref_deg")
    theta_l = _series(rows, "theta_L_deg")
    error = theta_ref - theta_l
    time_s = _series(rows, "t_s", "time_s")
    a_raw = _series(rows, "a_raw", "action_raw", "act_delta_omega")
    a_safe = _series(rows, "a_safe", "action_safe")
    omega_cmd = _omega_cmd_deg_s(rows)
    disturbance = _series(rows, "disturbance_torque_Nm", "disturbance_torque", "disturbance_nm")
    disturbance_metrics = _disturbance_metrics(time_s, error, disturbance)
    first_violation_time, first_violation_type = _first_violation(rows)
    return {
        "scenario": scenario,
        "controller": controller,
        "rmse_theta_deg": float(np.sqrt(np.nanmean(error**2))) if error.size else float("nan"),
        "mae_theta_deg": float(np.nanmean(np.abs(error))) if error.size else float("nan"),
        "max_abs_theta_error_deg": _max_abs(error),
        "settling_time_s": _settling_time(time_s, error),
        "recovery_time_s": _recovery_time(time_s, error, disturbance),
        **disturbance_metrics,
        "mean_a_raw": float(np.nanmean(a_raw)) if np.any(np.isfinite(a_raw)) else float("nan"),
        "max_abs_a_raw": _max_abs(a_raw),
        "mean_a_safe": float(np.nanmean(a_safe)) if np.any(np.isfinite(a_safe)) else float("nan"),
        "max_abs_a_safe": _max_abs(a_safe),
        "mean_omega_L_cmd_safe_deg_s": float(np.nanmean(omega_cmd)) if np.any(np.isfinite(omega_cmd)) else float("nan"),
        "max_abs_omega_L_cmd_safe_deg_s": _max_abs(omega_cmd),
        "max_abs_omega_L_deg_s": _max_abs(_series(rows, "omega_L_deg_s")),
        "max_abs_Tout_nm": _max_abs(_series(rows, "T_out_Nm", "T_out")),
        "constraint_violation_count": _constraint_count(rows),
        "first_violation_time_s": first_violation_time,
        "first_violation_type": first_violation_type,
        "flag_U_safe_count": _flag_count(rows, "flag_U_safe"),
        "flag_E_safe_count": _flag_count(rows, "flag_E_safe"),
        "flag_X_safe_count": _flag_count(rows, "flag_X_safe"),
        "sigma_safe_count": _flag_count(rows, "sigma_safe"),
        "episode_return": _safe_float(rows[-1].get("cum_reward", rows[-1].get("episode_return"))) if rows else float("nan"),
        "done_reason": str(rows[-1].get("done_reason", "")) if rows else "",
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "csv_path": str(path),
    }


def _plot_position(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s", "time_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _series(rows, "theta_ref_deg"), "k--", linewidth=1.2, label="theta_ref_deg")
    ax.plot(time_s, _series(rows, "theta_L_deg"), linewidth=1.2, label="theta_L_deg")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [deg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_error(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s", "time_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _series(rows, "e_theta_deg"), linewidth=1.2, label="e_theta_deg")
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


def _plot_action_trace(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s", "time_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _series(rows, "a_raw", "action_raw", "act_delta_omega"), linewidth=1.1, label="a_raw")
    ax.plot(time_s, _series(rows, "a_safe", "action_safe"), linewidth=1.1, label="a_safe")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("action [-]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_speed_command(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s", "time_s")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, _omega_cmd_deg_s(rows), "k--", linewidth=1.1, label="omega_L_cmd_safe")
    ax.plot(time_s, _series(rows, "omega_L_deg_s"), linewidth=1.1, label="omega_L")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [deg/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_safety_flags(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s", "time_s")
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


def _plot_disturbance_zoom(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    time_s = _series(rows, "t_s", "time_s")
    error = _series(rows, "e_theta_deg")
    disturbance = _series(rows, "disturbance_torque_Nm", "disturbance_torque", "disturbance_nm")
    if time_s.size == 0 or disturbance.size != time_s.size:
        return
    finite_disturbance = disturbance[np.isfinite(disturbance)]
    if finite_disturbance.size == 0:
        return
    threshold = max(1e-9, 0.05 * max(1.0, float(np.nanmax(np.abs(finite_disturbance)))))
    changes = np.flatnonzero(np.abs(np.diff(disturbance)) > threshold)
    center = int(changes[0] + 1) if changes.size else int(np.argmax(np.abs(disturbance)))
    center_t = float(time_s[center])
    mask = (time_s >= center_t - 0.75) & (time_s <= center_t + 1.5)
    if not np.any(mask):
        mask = np.ones_like(time_s, dtype=bool)
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(time_s[mask], error[mask], linewidth=1.2, label="e_theta_deg")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("error [deg]")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(time_s[mask], disturbance[mask], color="tab:red", linewidth=1.0, alpha=0.8, label="disturbance")
    ax2.set_ylabel("disturbance [Nm]")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_return_curve(train_log: Path, output_path: Path) -> None:
    if not train_log.exists():
        return
    rows = _read_rows(train_log)
    if not rows:
        return
    steps = _series(rows, "global_steps", "step")
    returns = _series(rows, "episode_return")
    eval_returns = _series(rows, "latest_eval_return", "eval_return")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, returns, linewidth=1.1, label="episode_return")
    if np.any(np.isfinite(eval_returns)):
        ax.plot(steps, eval_returns, "o-", linewidth=1.0, markersize=3, label="eval_return")
    ax.set_xlabel("step")
    ax.set_ylabel("return")
    ax.set_title("TD3 smoke returns")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _pid_rows(path: Path, scenarios: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_rows(path)
    wanted = set(scenarios)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("scenario") not in wanted:
            continue
        csv_path = Path(str(row.get("csv_path", "")))
        if csv_path.exists():
            pid_summary = summarize_trace(csv_path, scenario=str(row.get("scenario", "")), controller="pid_v2")
            pid_summary.update(
                {
                    "rmse_theta_deg": row.get("rmse_theta_deg", pid_summary["rmse_theta_deg"]),
                    "mae_theta_deg": row.get("mae_theta_deg", pid_summary["mae_theta_deg"]),
                    "max_abs_theta_error_deg": row.get(
                        "max_abs_theta_error_deg",
                        pid_summary["max_abs_theta_error_deg"],
                    ),
                    "settling_time_s": row.get("settling_time_s", pid_summary["settling_time_s"]),
                    "recovery_time_s": row.get("recovery_time_s", pid_summary["recovery_time_s"]),
                    "constraint_violation_count": row.get(
                        "constraint_violation_count",
                        pid_summary["constraint_violation_count"],
                    ),
                    "episode_return": row.get("episode_return", pid_summary["episode_return"]),
                    "done_reason": row.get("done_reason", pid_summary["done_reason"]),
                }
            )
            out.append(pid_summary)
        else:
            out.append(
                {
                    field: row.get(field, "")
                    for field in SUMMARY_FIELDS
                }
                | {"controller": "pid_v2", "checkpoint_path": ""}
            )
    return out


def _evaluate_missing_pid_rows(
    *,
    existing_rows: list[dict[str, Any]],
    scenarios: list[str],
    env_config: Path,
    eval_config: Path,
    train_config: Path,
    eval_dir: Path,
    max_steps: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    existing_scenarios = {str(row.get("scenario", "")) for row in existing_rows}
    rows = list(existing_rows)
    for scenario in scenarios:
        if scenario in existing_scenarios:
            continue
        output_csv = eval_dir / f"{scenario}_pid.csv"
        result = run_evaluation(
            env_config_path=env_config,
            eval_config_path=eval_config,
            train_config_path=train_config,
            pi_config_path=Path("configs/pi/pi_default.yaml"),
            controller_name="pid",
            checkpoint_path=None,
            checkpoint_tag="best",
            max_steps_override=max_steps,
            output_csv_path=output_csv,
            seed_override=seed,
            scenario_name=scenario,
        )
        summary = summarize_trace(output_csv, scenario=scenario, controller="pid_v2", checkpoint_path=None)
        summary["episode_return"] = float(result.get("episode_return", summary["episode_return"]))
        summary["done_reason"] = str(result.get("done_reason", summary["done_reason"]))
        rows.append(summary)
        existing_scenarios.add(scenario)
        print(
            f"pid fallback scenario={scenario} return={summary['episode_return']:.3f} "
            f"rmse={summary['rmse_theta_deg']:.4f} done_reason={summary['done_reason']} csv={output_csv}"
        )
    return rows


def _existing_summary_rows(path: Path, scenarios: list[str], *, controller: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    wanted = set(scenarios)
    out: list[dict[str, Any]] = []
    for row in _read_rows(path):
        if row.get("scenario") not in wanted:
            continue
        normalized = {field: row.get(field, "") for field in SUMMARY_FIELDS}
        if controller not in (None, ""):
            normalized["controller"] = str(controller)
        out.append(normalized)
    return out


def _resolve_checkpoint(path: Path) -> Path:
    p = _resolve_path(path)
    if p.exists():
        return p
    if p.name == "best.pt":
        fallback = p.with_name("checkpoint_best.pt")
        if fallback.exists():
            return fallback
    if p.name == "latest.pt":
        fallback = p.with_name("checkpoint_latest.pt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_config = _resolve_path(args.eval_config)
    env_config = _resolve_path(args.env_config)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    eval_data = load_yaml(eval_config)
    run_name = str(eval_data.get("run_name", "gun_servo_td3_smoke_c1"))
    summary_output_name = str(eval_data.get("summary_output_name", "td3_smoke_summary.csv"))
    comparison_output_name = str(eval_data.get("comparison_output_name", "td3_vs_pid_v2_summary.csv"))
    controller_label = str(eval_data.get("td3_controller_label", "td3"))
    evaluation_controller = str(eval_data.get("controller", "rl"))
    output_dir = _resolve_path(args.output_dir or (ROOT / "outputs" / "runs" / run_name))
    train_config = _resolve_path(eval_data.get("train_config", "configs/train/gun_servo_td3_smoke.yaml"))
    scenarios = [str(item) for item in eval_data.get("scenarios", [])]
    if not scenarios:
        raise ValueError("Expected eval config to define a non-empty 'scenarios' list")
    max_steps = None if eval_data.get("max_steps") in (None, "") else int(eval_data["max_steps"])

    eval_dir = output_dir / "eval"
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    for path in (eval_dir, summaries_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        output_csv = eval_dir / f"{scenario}_td3.csv"
        result = run_evaluation(
            env_config_path=env_config,
            eval_config_path=eval_config,
            train_config_path=train_config,
            pi_config_path=Path("configs/pi/pi_default.yaml"),
            controller_name=evaluation_controller,
            checkpoint_path=checkpoint,
            checkpoint_tag="best",
            max_steps_override=max_steps,
            output_csv_path=output_csv,
            seed_override=int(eval_data.get("seed", 0)),
            scenario_name=scenario,
        )
        rows = _read_rows(output_csv)
        summary = summarize_trace(output_csv, scenario=scenario, controller=controller_label, checkpoint_path=checkpoint)
        summary["episode_return"] = float(result.get("episode_return", summary["episode_return"]))
        summary["done_reason"] = str(result.get("done_reason", summary["done_reason"]))
        summary_rows.append(summary)
        _plot_position(rows, figures_dir / f"{scenario}_td3_position.png", title=f"{scenario} TD3 position")
        _plot_error(rows, figures_dir / f"{scenario}_td3_error.png", title=f"{scenario} TD3 error")
        _plot_action_trace(rows, figures_dir / f"{scenario}_td3_action_trace.png", title=f"{scenario} TD3 action")
        _plot_speed_command(rows, figures_dir / f"{scenario}_td3_speed_command.png", title=f"{scenario} TD3 speed")
        _plot_safety_flags(rows, figures_dir / f"{scenario}_td3_safety_flags.png", title=f"{scenario} TD3 safety")
        if "disturbance" in scenario.lower():
            _plot_disturbance_zoom(
                rows,
                figures_dir / f"{scenario}_td3_disturbance_zoom.png",
                title=f"{scenario} TD3 disturbance recovery",
            )
        print(
            f"scenario={scenario} return={summary['episode_return']:.3f} "
            f"rmse={summary['rmse_theta_deg']:.4f} done_reason={summary['done_reason']} csv={output_csv}"
        )

    td3_summary = summaries_dir / summary_output_name
    _write_rows(td3_summary, summary_rows, SUMMARY_FIELDS)

    pid_summary_path = _resolve_path(
        eval_data.get("pid_baseline_summary", "outputs/runs/gun_servo_pid_smoke_v2/summaries/pid_smoke_summary.csv")
    )
    pid_rows = _pid_rows(pid_summary_path, scenarios)
    if bool(eval_data.get("pid_evaluate_missing_scenarios", False)):
        pid_rows = _evaluate_missing_pid_rows(
            existing_rows=pid_rows,
            scenarios=scenarios,
            env_config=env_config,
            eval_config=eval_config,
            train_config=train_config,
            eval_dir=eval_dir,
            max_steps=max_steps,
            seed=int(eval_data.get("seed", 0)),
        )
    comparison_rows = [*pid_rows, *summary_rows]
    comparison_csv = summaries_dir / comparison_output_name
    _write_rows(comparison_csv, comparison_rows, SUMMARY_FIELDS)
    td3_baseline_summary = eval_data.get("td3_baseline_summary")
    if td3_baseline_summary not in (None, ""):
        td3_baseline_path = _resolve_path(td3_baseline_summary)
        td3_baseline_label = eval_data.get("td3_baseline_controller_label")
        td3_comparison_output_name = str(eval_data.get("td3_comparison_output_name", "td3_vs_td3_baseline_summary.csv"))
        td3_comparison_rows = [
            *_existing_summary_rows(td3_baseline_path, scenarios, controller=td3_baseline_label),
            *summary_rows,
        ]
        td3_comparison_csv = summaries_dir / td3_comparison_output_name
        _write_rows(td3_comparison_csv, td3_comparison_rows, SUMMARY_FIELDS)
        print(f"saved TD3 baseline comparison: {td3_comparison_csv}")
    for item in eval_data.get("additional_td3_comparisons", []) or []:
        if not isinstance(item, dict):
            continue
        baseline_summary = item.get("baseline_summary")
        if baseline_summary in (None, ""):
            continue
        baseline_path = _resolve_path(baseline_summary)
        baseline_label = item.get("baseline_controller_label")
        output_name = str(item.get("output_name", f"{controller_label}_vs_baseline_summary.csv"))
        rows = [
            *_existing_summary_rows(baseline_path, scenarios, controller=baseline_label),
            *summary_rows,
        ]
        output_csv = summaries_dir / output_name
        _write_rows(output_csv, rows, SUMMARY_FIELDS)
        print(f"saved additional TD3 comparison: {output_csv}")
    _plot_return_curve(output_dir / "train_log.csv", figures_dir / "td3_return_curve.png")
    print(f"saved TD3 summary: {td3_summary}")
    print(f"saved TD3 vs PID v2 comparison: {comparison_csv}")


if __name__ == "__main__":
    main()

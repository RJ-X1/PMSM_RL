"""Monte Carlo evaluation for randomized gun-servo TD3 checkpoints."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
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
from scripts.diagnostics.run_gun_servo_td3_smoke_eval import SUMMARY_FIELDS, summarize_trace
from utils.config import load_yaml


MC_EPISODE_FIELDS = [
    "episode",
    "seed",
    "scenario",
    "controller",
    "success",
    "episode_limit",
    "omega_limit_violation",
    "hard_safety_violation",
    *SUMMARY_FIELDS,
    "active_J_load",
    "active_B_load",
    "active_coulomb_friction",
    "active_gravity_torque_coeff",
    "gear_efficiency",
    "encoder_noise_std_rad",
    "vdc_scale",
    "disturbance_step_Nm",
    "disturbance_step_time_s",
    "sampled_theta_final_deg",
    "sampled_step_time_s",
    "sampled_max_speed_deg_s",
    "sampled_max_accel_deg_s2",
    "sampled_sine_amplitude_deg",
    "sampled_sine_frequency_hz",
    "sampled_sine_phase_rad",
]

MC_SUMMARY_FIELDS = [
    "num_episodes",
    "disturbance_episode_count",
    "mean_rmse_theta_deg",
    "std_rmse_theta_deg",
    "max_rmse_theta_deg",
    "mean_mae_theta_deg",
    "max_abs_theta_error_deg",
    "success_rate",
    "episode_limit_rate",
    "omega_limit_violation_count",
    "hard_safety_violation_count",
    "total_flag_U_safe_count",
    "mean_flag_U_safe_count",
    "mean_recovery_time_s",
    "mean_recovery_time_0p02deg_s",
    "mean_recovery_time_0p05deg_s",
    "mean_recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
    "max_error_after_disturbance_deg",
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite(values: list[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(value) for value in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean(values: list[Any]) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: list[Any]) -> float:
    arr = _finite(values)
    return float(np.std(arr)) if arr.size else float("nan")


def _max(values: list[Any]) -> float:
    arr = _finite(values)
    return float(np.max(arr)) if arr.size else float("nan")


def _is_episode_limit(row: dict[str, Any]) -> bool:
    return str(row.get("done_reason", "")) == "episode_limit"


def _has_omega_violation(row: dict[str, Any]) -> bool:
    return "omega_limit_violation" in str(row.get("done_reason", ""))


def _has_hard_violation(row: dict[str, Any]) -> bool:
    done_reason = str(row.get("done_reason", ""))
    if "violation" in done_reason:
        return True
    if _safe_float(row.get("max_abs_omega_L_deg_s")) > 30.000001:
        return True
    return any(
        _safe_float(row.get(name), default=0.0) > 0.5
        for name in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count")
    )


def _scenario_probabilities(scenarios: list[str], weights_cfg: Any) -> np.ndarray | None:
    if not scenarios or not isinstance(weights_cfg, dict):
        return None
    weights = np.asarray([float(weights_cfg.get(name, 0.0)) for name in scenarios], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    return weights / total


def _first_row_values(csv_path: Path) -> dict[str, Any]:
    rows = _read_rows(csv_path)
    if not rows:
        return {}
    row = rows[0]
    out: dict[str, Any] = {}
    for name in (
        "active_J_load",
        "active_B_load",
        "active_coulomb_friction",
        "active_gravity_torque_coeff",
        "gear_efficiency",
        "encoder_noise_std_rad",
        "disturbance_step_Nm",
        "disturbance_step_time_s",
        "sampled_theta_final_deg",
        "sampled_step_time_s",
        "sampled_max_speed_deg_s",
        "sampled_max_accel_deg_s2",
        "sampled_sine_amplitude_deg",
        "sampled_sine_frequency_hz",
        "sampled_sine_phase_rad",
    ):
        out[name] = row.get(name, "")
    vdc = _safe_float(row.get("V_dc"))
    out["vdc_scale"] = "" if not math.isfinite(vdc) else vdc / 540.0
    return out


def _aggregate_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    disturbance_rows = [row for row in rows if "disturbance" in str(row.get("scenario", "")).lower()]
    hard_count = sum(1 for row in rows if _has_hard_violation(row))
    omega_count = sum(1 for row in rows if _has_omega_violation(row))
    episode_limit_count = sum(1 for row in rows if _is_episode_limit(row))
    success_count = sum(1 for row in rows if int(row.get("success", 0)) == 1)
    max_disturbance_error = _max([row.get("max_abs_error_after_disturbance_deg") for row in disturbance_rows])
    return {
        "num_episodes": len(rows),
        "disturbance_episode_count": len(disturbance_rows),
        "mean_rmse_theta_deg": _mean([row.get("rmse_theta_deg") for row in rows]),
        "std_rmse_theta_deg": _std([row.get("rmse_theta_deg") for row in rows]),
        "max_rmse_theta_deg": _max([row.get("rmse_theta_deg") for row in rows]),
        "mean_mae_theta_deg": _mean([row.get("mae_theta_deg") for row in rows]),
        "max_abs_theta_error_deg": _max([row.get("max_abs_theta_error_deg") for row in rows]),
        "success_rate": float(success_count) / float(n),
        "episode_limit_rate": float(episode_limit_count) / float(n),
        "omega_limit_violation_count": omega_count,
        "hard_safety_violation_count": hard_count,
        "total_flag_U_safe_count": sum(int(_safe_float(row.get("flag_U_safe_count"), default=0.0)) for row in rows),
        "mean_flag_U_safe_count": _mean([row.get("flag_U_safe_count") for row in rows]),
        "mean_recovery_time_s": _mean([row.get("recovery_time_s") for row in disturbance_rows]),
        "mean_recovery_time_0p02deg_s": _mean([row.get("recovery_time_0p02deg_s") for row in disturbance_rows]),
        "mean_recovery_time_0p05deg_s": _mean([row.get("recovery_time_0p05deg_s") for row in disturbance_rows]),
        "mean_recovery_time_0p10deg_s": _mean([row.get("recovery_time_0p10deg_s") for row in disturbance_rows]),
        "max_abs_error_after_disturbance_deg": max_disturbance_error,
        "max_error_after_disturbance_deg": max_disturbance_error,
    }


def _plot_mc_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    rmse = _finite([row.get("rmse_theta_deg") for row in rows])
    if not rmse.size:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot([rmse], labels=["all"], showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("MC RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_scenario_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    scenarios = sorted({str(row.get("scenario", "")) for row in rows})
    series = [_finite([row.get("rmse_theta_deg") for row in rows if str(row.get("scenario", "")) == scenario]) for scenario in scenarios]
    series = [item for item in series if item.size]
    labels = [scenario for scenario in scenarios if _finite([row.get("rmse_theta_deg") for row in rows if str(row.get("scenario", "")) == scenario]).size]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.boxplot(series, labels=labels, showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("MC RMSE by scenario family")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_violation_counts(rows: list[dict[str, Any]], path: Path) -> None:
    labels = ["omega", "hard", "flag_U", "episode_not_limit"]
    values = [
        sum(1 for row in rows if _has_omega_violation(row)),
        sum(1 for row in rows if _has_hard_violation(row)),
        sum(int(_safe_float(row.get("flag_U_safe_count"), default=0.0)) for row in rows),
        sum(1 for row in rows if not _is_episode_limit(row)),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color=["tab:red", "tab:orange", "tab:blue", "tab:gray"])
    ax.set_ylabel("count")
    ax.set_title("MC violation and safety-intervention counts")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_recovery(rows: list[dict[str, Any]], path: Path) -> None:
    disturbance_rows = [row for row in rows if "disturbance" in str(row.get("scenario", "")).lower()]
    labels = ["legacy", "0.02deg", "0.05deg", "0.10deg"]
    columns = ["recovery_time_s", "recovery_time_0p02deg_s", "recovery_time_0p05deg_s", "recovery_time_0p10deg_s"]
    series = [_finite([row.get(column) for row in disturbance_rows]) for column in columns]
    plotted = [(label, item) for label, item in zip(labels, series) if item.size]
    if not plotted:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot([item for _, item in plotted], labels=[label for label, _ in plotted], showmeans=True)
    ax.set_ylabel("recovery time [s]")
    ax.set_title("MC disturbance recovery")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rows_by_controller(rows: list[dict[str, str]], controller: str) -> dict[str, dict[str, str]]:
    return {str(row.get("scenario", "")): row for row in rows if str(row.get("controller", "")) == controller}


def _comparison_lines(
    path: Path,
    *,
    baseline_controller: str,
    current_controller: str,
    max_rows: int = 8,
) -> list[str]:
    rows = _read_rows(path)
    if not rows:
        return ["- comparison file missing"]
    baseline = _rows_by_controller(rows, baseline_controller)
    current = _rows_by_controller(rows, current_controller)
    lines: list[str] = []
    for scenario in sorted(current):
        if scenario not in baseline:
            continue
        rmse_delta = _safe_float(current[scenario].get("rmse_theta_deg")) - _safe_float(
            baseline[scenario].get("rmse_theta_deg")
        )
        flag_delta = _safe_float(current[scenario].get("flag_U_safe_count"), 0.0) - _safe_float(
            baseline[scenario].get("flag_U_safe_count"), 0.0
        )
        lines.append(f"- {scenario}: d_rmse={rmse_delta:.6g} deg, d_flag_U={flag_delta:.0f}")
        if len(lines) >= max_rows:
            break
    return lines or ["- no overlapping scenarios"]


def _fixed_eval_status(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "all_episode_limit": all(str(row.get("done_reason", "")) == "episode_limit" for row in rows) if rows else False,
        "omega_limit_violation_count": sum(1 for row in rows if "omega_limit_violation" in str(row.get("done_reason", ""))),
        "hard_safety_violation_count": sum(
            1
            for row in rows
            if any(_safe_float(row.get(name), default=0.0) > 0.5 for name in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count"))
            or "violation" in str(row.get("done_reason", ""))
        ),
        "mean_rmse_theta_deg": _mean([row.get("rmse_theta_deg") for row in rows]),
        "total_flag_U_safe_count": sum(int(_safe_float(row.get("flag_U_safe_count"), default=0.0)) for row in rows),
    }


def _training_completed(run_dir: Path) -> bool:
    rows = _read_rows(run_dir / "train_log.csv")
    if not rows:
        return False
    return int(_safe_float(rows[-1].get("global_steps", rows[-1].get("step")), default=-1.0)) >= 600000


def _maybe_write_conclusion(run_dir: Path, checkpoint: Path, mc_summary: dict[str, Any]) -> None:
    summaries_dir = run_dir / "summaries"
    fixed_summary_path = summaries_dir / "mc_sc_residual_td3_v8_fixed_summary.csv"
    v7_path = summaries_dir / "mc_sc_residual_td3_v8_vs_v7_summary.csv"
    v6_path = summaries_dir / "mc_sc_residual_td3_v8_vs_v6_summary.csv"
    pid_path = summaries_dir / "mc_sc_residual_td3_v8_vs_pid_v2_summary.csv"
    best_reason_path = summaries_dir / "best_checkpoint.txt"
    fixed_rows = _read_rows(fixed_summary_path)
    fixed_status = _fixed_eval_status(fixed_rows)
    best_reason = best_reason_path.read_text(encoding="utf-8") if best_reason_path.exists() else ""
    c5_rows = [row for row in fixed_rows if "disturbance" in str(row.get("scenario", "")).lower()]
    c5 = c5_rows[0] if c5_rows else {}
    pid_rows = _read_rows(pid_path)
    pid_current = _rows_by_controller(pid_rows, "mc_sc_residual_td3_v8")
    pid_base = _rows_by_controller(pid_rows, "pid_v2")
    competitive = [
        scenario
        for scenario, row in pid_current.items()
        if scenario in pid_base and _safe_float(row.get("rmse_theta_deg")) <= _safe_float(pid_base[scenario].get("rmse_theta_deg"))
    ]
    pid_better = [
        scenario
        for scenario, row in pid_current.items()
        if scenario in pid_base and _safe_float(row.get("rmse_theta_deg")) > _safe_float(pid_base[scenario].get("rmse_theta_deg"))
    ]
    proceed = bool(
        fixed_status["all_episode_limit"]
        and fixed_status["omega_limit_violation_count"] == 0
        and fixed_status["hard_safety_violation_count"] == 0
        and float(mc_summary.get("success_rate", 0.0)) >= 0.95
        and int(mc_summary.get("omega_limit_violation_count", 1)) == 0
        and int(mc_summary.get("hard_safety_violation_count", 1)) == 0
    )
    lines = [
        "# MC-SC-Residual TD3 Randomized v8 Result Conclusion",
        "",
        "## Best checkpoint",
        f"- Path: {checkpoint}",
        f"- Reason: {best_reason.splitlines()[0] if best_reason else 'selection reason unavailable'}",
        "",
        "## Completion status",
        f"- Training completed successfully: {_training_completed(run_dir)}",
        f"- Fixed evaluation completed successfully: {bool(fixed_rows)}",
        f"- Monte Carlo evaluation completed successfully: {bool(mc_summary)}",
        f"- All fixed scenarios reached episode_limit: {fixed_status['all_episode_limit']}",
        f"- Fixed omega_limit_violation count: {fixed_status['omega_limit_violation_count']}",
        f"- Fixed hard safety violation count: {fixed_status['hard_safety_violation_count']}",
        "",
        "## v8 vs v7",
        *_comparison_lines(v7_path, baseline_controller="mc_sc_residual_td3_v7", current_controller="mc_sc_residual_td3_v8"),
        "",
        "## v8 vs v6",
        *_comparison_lines(v6_path, baseline_controller="mc_sc_residual_td3_v6", current_controller="mc_sc_residual_td3_v8"),
        "",
        "## Monte Carlo robustness",
        f"- success_rate: {mc_summary.get('success_rate', '')}",
        f"- mean/std/max RMSE deg: {mc_summary.get('mean_rmse_theta_deg', '')} / {mc_summary.get('std_rmse_theta_deg', '')} / {mc_summary.get('max_rmse_theta_deg', '')}",
        f"- omega_limit_violation_count: {mc_summary.get('omega_limit_violation_count', '')}",
        f"- hard_safety_violation_count: {mc_summary.get('hard_safety_violation_count', '')}",
        f"- mean recovery time s: {mc_summary.get('mean_recovery_time_s', '')}",
        f"- max disturbance error deg: {mc_summary.get('max_error_after_disturbance_deg', '')}",
        "",
        "## v8 vs PID v2",
        f"- Competitive or lower RMSE scenarios: {', '.join(competitive) if competitive else 'none'}",
        f"- PID lower RMSE scenarios: {', '.join(pid_better) if pid_better else 'none'}",
        "",
        "## C5 disturbance metrics",
        f"- error_at_disturbance_deg: {c5.get('error_at_disturbance_deg', '')}",
        f"- max_abs_error_after_disturbance_deg: {c5.get('max_abs_error_after_disturbance_deg', '')}",
        f"- peak_error_in_1s_after_disturbance_deg: {c5.get('peak_error_in_1s_after_disturbance_deg', '')}",
        f"- recovery_time_0p02deg_s: {c5.get('recovery_time_0p02deg_s', '')}",
        f"- recovery_time_0p05deg_s: {c5.get('recovery_time_0p05deg_s', '')}",
        f"- recovery_time_0p10deg_s: {c5.get('recovery_time_0p10deg_s', '')}",
        "",
        "## Recommendation",
        (
            "- Proceed to formal final training with a larger step count and multi-seed runs."
            if proceed
            else "- Tune v8 randomization ranges or reward weights again before final multi-seed training."
        ),
        "",
    ]
    (summaries_dir / "v8_result_conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_config = _resolve_path(args.eval_config)
    env_config = _resolve_path(args.env_config)
    checkpoint = _resolve_path(args.checkpoint)
    if not checkpoint.exists() and checkpoint.name == "best.pt":
        checkpoint = checkpoint.with_name("checkpoint_best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    eval_data = load_yaml(eval_config)
    run_name = str(eval_data.get("run_name", "gun_servo_mc_eval"))
    output_dir = _resolve_path(args.output_dir or (ROOT / "outputs" / "runs" / run_name))
    train_config = _resolve_path(eval_data.get("train_config", "configs/train/gun_servo_td3_smoke.yaml"))
    max_steps = None if eval_data.get("max_steps") in (None, "") else int(eval_data["max_steps"])
    seed = int(eval_data.get("seed", 0))
    num_episodes = int(eval_data.get("num_episodes", eval_data.get("episodes", 50)))
    controller = str(eval_data.get("controller", "residual"))
    controller_label = str(eval_data.get("td3_controller_label", "td3"))
    scenarios = [str(item) for item in eval_data.get("scenario_families", eval_data.get("scenarios", []))]
    if not scenarios:
        raise ValueError("Expected eval config to define scenario_families or scenarios")

    probabilities = _scenario_probabilities(
        scenarios,
        eval_data.get("scenario_family_weights", eval_data.get("scenario_weights")),
    )
    rng = np.random.default_rng(seed)
    eval_dir = output_dir / "eval" / "mc"
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    for path in (eval_dir, summaries_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    for episode in range(num_episodes):
        scenario = str(rng.choice(scenarios, p=probabilities))
        episode_seed = seed + episode
        output_csv = eval_dir / f"episode_{episode:04d}_{scenario}_td3.csv"
        result = run_evaluation(
            env_config_path=env_config,
            eval_config_path=eval_config,
            train_config_path=train_config,
            pi_config_path=Path("configs/pi/pi_default.yaml"),
            controller_name=controller,
            checkpoint_path=checkpoint,
            checkpoint_tag="best",
            max_steps_override=max_steps,
            output_csv_path=output_csv,
            seed_override=episode_seed,
            scenario_name=scenario,
        )
        summary = summarize_trace(output_csv, scenario=scenario, controller=controller_label, checkpoint_path=checkpoint)
        summary["episode_return"] = float(result.get("episode_return", summary["episode_return"]))
        summary["done_reason"] = str(result.get("done_reason", summary["done_reason"]))
        hard = _has_hard_violation(summary)
        omega = _has_omega_violation(summary)
        episode_limit = _is_episode_limit(summary)
        row = {
            "episode": episode,
            "seed": episode_seed,
            "scenario": scenario,
            "controller": controller_label,
            "success": int(episode_limit and not omega and not hard),
            "episode_limit": int(episode_limit),
            "omega_limit_violation": int(omega),
            "hard_safety_violation": int(hard),
            **summary,
            **_first_row_values(output_csv),
        }
        episode_rows.append(row)
        print(
            f"mc_episode={episode + 1}/{num_episodes} scenario={scenario} "
            f"rmse={float(summary['rmse_theta_deg']):.4f} done_reason={summary['done_reason']}"
        )

    episode_output = summaries_dir / str(eval_data.get("episode_output_name", "mc_eval_episodes.csv"))
    _write_rows(episode_output, episode_rows, list(dict.fromkeys(MC_EPISODE_FIELDS)))

    mc_summary = _aggregate_episode_rows(episode_rows)
    summary_output = summaries_dir / str(eval_data.get("summary_output_name", "mc_eval_summary.csv"))
    _write_rows(summary_output, [mc_summary], MC_SUMMARY_FIELDS)
    (summaries_dir / "mc_eval_summary.json").write_text(json.dumps(mc_summary, indent=2) + "\n", encoding="utf-8")

    _plot_mc_rmse(episode_rows, figures_dir / "mc_rmse_boxplot.png")
    _plot_mc_scenario_rmse(episode_rows, figures_dir / "mc_scenario_rmse_boxplot.png")
    _plot_mc_violation_counts(episode_rows, figures_dir / "mc_violation_counts.png")
    _plot_mc_recovery(episode_rows, figures_dir / "mc_recovery_time_boxplot.png")
    _maybe_write_conclusion(output_dir, checkpoint, mc_summary)

    print(f"saved MC episode rows: {episode_output}")
    print(f"saved MC summary: {summary_output}")


if __name__ == "__main__":
    main()

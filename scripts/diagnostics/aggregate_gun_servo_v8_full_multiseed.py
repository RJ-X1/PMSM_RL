"""Aggregate full multi-seed v8 gun-servo randomized TD3 results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import math
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEEDS = (0, 1, 2)
RUN_TEMPLATE = "gun_servo_mc_sc_residual_td3_random_v8_full_seed{seed}"
FIXED_SUMMARY_NAME = "mc_sc_residual_td3_v8_full_fixed_summary.csv"
PID_COMPARISON_NAME = "mc_sc_residual_td3_v8_full_vs_pid_v2_summary.csv"
V7_COMPARISON_NAME = "mc_sc_residual_td3_v8_full_vs_v7_summary.csv"
MC_SUMMARY_NAME = "mc_eval_summary.csv"
MC_EPISODE_NAME = "mc_eval_episode_metrics.csv"

FIXED_METRICS = [
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "episode_return",
    "recovery_time_s",
    "recovery_time_0p02deg_s",
    "recovery_time_0p05deg_s",
    "recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
]

COMPARISON_METRICS = [
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "episode_return",
    "recovery_time_s",
    "recovery_time_0p02deg_s",
    "recovery_time_0p05deg_s",
    "recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
]

MC_METRICS = [
    "success_rate",
    "episode_limit_rate",
    "mean_rmse_theta_deg",
    "max_rmse_theta_deg",
    "mean_mae_theta_deg",
    "mean_flag_U_safe_count",
    "mean_recovery_time_s",
    "mean_recovery_time_0p02deg_s",
    "mean_recovery_time_0p05deg_s",
    "mean_recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
]

MC_COUNT_METRICS = [
    "omega_limit_violation_count",
    "hard_safety_violation_count",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "runs" / "gun_servo_mc_sc_residual_td3_random_v8_full_aggregate",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "outputs" / "runs",
    )
    return parser


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


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(values: list[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(value) for value in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean_std(values: list[Any]) -> tuple[float, float, int]:
    arr = _finite(values)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std, int(arr.size)


def _is_episode_limit(row: dict[str, Any]) -> bool:
    return str(row.get("done_reason", "")) == "episode_limit"


def _omega_violation(row: dict[str, Any]) -> bool:
    return "omega_limit_violation" in str(row.get("done_reason", ""))


def _hard_violation(row: dict[str, Any]) -> bool:
    if "violation" in str(row.get("done_reason", "")):
        return True
    return any(
        _safe_float(row.get(name)) > 0.5
        for name in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count")
    )


def _success(row: dict[str, Any]) -> bool:
    return _is_episode_limit(row) and not _omega_violation(row) and not _hard_violation(row)


def _run_dir(runs_root: Path, seed: int) -> Path:
    return runs_root / RUN_TEMPLATE.format(seed=seed)


def _train_completed(run_dir: Path) -> bool:
    rows = _read_rows(run_dir / "train_log.csv")
    if not rows:
        return False
    return _safe_float(rows[-1].get("global_steps", rows[-1].get("step"))) >= 600000


def _checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint_step_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _best_checkpoint_info(run_dir: Path) -> dict[str, str]:
    text_path = run_dir / "summaries" / "best_checkpoint.txt"
    out = {
        "checkpoint_path": str(run_dir / "checkpoints" / "checkpoint_best.pt"),
        "source_checkpoint_path": "",
        "global_step": "",
        "reason": "",
    }
    if not text_path.exists():
        return out
    for line in text_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in out:
            out[key] = value
        elif key == "selection_method":
            out["reason"] = value
        elif key == "best_checkpoint_reason":
            out["reason"] = value
    if not out["global_step"] and out["source_checkpoint_path"]:
        step = _checkpoint_step(out["source_checkpoint_path"])
        out["global_step"] = "" if step < 0 else str(step)
    return out


def aggregate_fixed(runs_root: Path, summaries_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in SEEDS:
        for row in _read_rows(_run_dir(runs_root, seed) / "summaries" / FIXED_SUMMARY_NAME):
            enriched = dict(row)
            enriched["seed"] = seed
            grouped[str(row.get("scenario", ""))].append(enriched)

    rows: list[dict[str, Any]] = []
    for scenario in sorted(grouped):
        items = grouped[scenario]
        out: dict[str, Any] = {
            "scenario": scenario,
            "n": len(items),
            "success_rate": sum(1 for row in items if _success(row)) / max(1, len(items)),
            "omega_limit_violation_count": sum(1 for row in items if _omega_violation(row)),
            "hard_safety_violation_count": sum(1 for row in items if _hard_violation(row)),
            "done_reason_summary": ";".join(
                f"{reason}:{sum(1 for row in items if str(row.get('done_reason', '')) == reason)}"
                for reason in sorted({str(row.get("done_reason", "")) for row in items})
            ),
        }
        for metric in FIXED_METRICS:
            mean, std, count = _mean_std([row.get(metric) for row in items])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_n"] = count
        rows.append(out)

    fieldnames = [
        "scenario",
        "n",
        "success_rate",
        "omega_limit_violation_count",
        "hard_safety_violation_count",
        "done_reason_summary",
    ]
    for metric in FIXED_METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_std", f"{metric}_n"]
    _write_rows(summaries_dir / "fixed_eval_mean_std.csv", rows, fieldnames)
    return rows


def aggregate_mc(runs_root: Path, summaries_dir: Path) -> list[dict[str, Any]]:
    seed_rows = []
    for seed in SEEDS:
        rows = _read_rows(_run_dir(runs_root, seed) / "summaries" / MC_SUMMARY_NAME)
        if rows:
            seed_rows.append({"seed": seed, **rows[0]})

    out_rows: list[dict[str, Any]] = []
    for metric in MC_METRICS:
        mean, std, count = _mean_std([row.get(metric) for row in seed_rows])
        out_rows.append({"metric": metric, "n": count, "mean": mean, "std": std, "total": ""})
    for metric in MC_COUNT_METRICS:
        values = [_safe_float(row.get(metric)) for row in seed_rows]
        finite = [value for value in values if math.isfinite(value)]
        mean, std, count = _mean_std(finite)
        out_rows.append({"metric": metric, "n": count, "mean": mean, "std": std, "total": sum(finite)})

    _write_rows(summaries_dir / "mc_eval_mean_std.csv", out_rows, ["metric", "n", "mean", "std", "total"])
    return out_rows


def aggregate_comparison(runs_root: Path, summaries_dir: Path, *, filename: str, output_name: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for seed in SEEDS:
        for row in _read_rows(_run_dir(runs_root, seed) / "summaries" / filename):
            enriched = dict(row)
            enriched["seed"] = seed
            grouped[(str(row.get("scenario", "")), str(row.get("controller", "")))].append(enriched)

    rows: list[dict[str, Any]] = []
    for (scenario, controller), items in sorted(grouped.items()):
        out: dict[str, Any] = {"scenario": scenario, "controller": controller, "n": len(items)}
        for metric in COMPARISON_METRICS:
            mean, std, count = _mean_std([row.get(metric) for row in items])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_n"] = count
        rows.append(out)

    fieldnames = ["scenario", "controller", "n"]
    for metric in COMPARISON_METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_std", f"{metric}_n"]
    _write_rows(summaries_dir / output_name, rows, fieldnames)
    return rows


def _plot_training_returns(runs_root: Path, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for seed in SEEDS:
        rows = _read_rows(_run_dir(runs_root, seed) / "train_log.csv")
        steps = _finite([row.get("global_steps") for row in rows])
        returns = _finite([row.get("episode_return") for row in rows])
        if steps.size == 0 or returns.size == 0:
            continue
        n = min(steps.size, returns.size)
        steps = steps[:n]
        returns = returns[:n]
        window = min(7, max(1, n))
        kernel = np.ones(window) / float(window)
        smooth = np.convolve(returns, kernel, mode="same")
        ax.plot(steps, smooth, linewidth=1.2, label=f"seed {seed}")
    ax.set_xlabel("global step")
    ax.set_ylabel("episode return (smoothed)")
    ax.set_title("v8 full randomized training returns")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "training_return_all_seeds.png", dpi=160)
    plt.close(fig)


def _plot_fixed_bar(rows: list[dict[str, Any]], figures_dir: Path, *, metric: str, output_name: str, ylabel: str) -> None:
    if not rows:
        return
    labels = [str(row["scenario"]) for row in rows]
    means = np.asarray([_safe_float(row.get(f"{metric}_mean")) for row in rows], dtype=np.float64)
    stds = np.asarray([_safe_float(row.get(f"{metric}_std")) for row in rows], dtype=np.float64)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(output_name.replace("_", " ").replace(".png", ""))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / output_name, dpi=160)
    plt.close(fig)


def _plot_mc_boxplot(runs_root: Path, figures_dir: Path) -> None:
    series = []
    labels = []
    for seed in SEEDS:
        rows = _read_rows(_run_dir(runs_root, seed) / "summaries" / MC_EPISODE_NAME)
        values = _finite([row.get("rmse_theta_deg") for row in rows])
        if values.size:
            series.append(values)
            labels.append(f"seed {seed}")
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.boxplot(series, labels=labels, showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("Monte Carlo RMSE across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "mc_rmse_boxplot_all_seeds.png", dpi=160)
    plt.close(fig)


def _plot_mc_seed_bars(runs_root: Path, figures_dir: Path) -> None:
    labels = []
    success = []
    omega = []
    hard = []
    for seed in SEEDS:
        rows = _read_rows(_run_dir(runs_root, seed) / "summaries" / MC_SUMMARY_NAME)
        if not rows:
            continue
        row = rows[0]
        labels.append(f"seed {seed}")
        success.append(_safe_float(row.get("success_rate")))
        omega.append(_safe_float(row.get("omega_limit_violation_count")))
        hard.append(_safe_float(row.get("hard_safety_violation_count")))
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x, success, color="tab:green", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("MC success rate by seed")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "mc_success_rate_bar.png", dpi=160)
    plt.close(fig)

    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - width / 2, omega, width, label="omega_limit")
    ax.bar(x + width / 2, hard, width, label="hard_safety")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("MC violation counts by seed")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "mc_violation_counts_bar.png", dpi=160)
    plt.close(fig)


def _plot_disturbance_recovery(runs_root: Path, figures_dir: Path) -> None:
    metrics = ["mean_recovery_time_s", "mean_recovery_time_0p05deg_s", "mean_recovery_time_0p10deg_s"]
    labels = ["legacy", "0.05 deg", "0.10 deg"]
    values_by_metric = []
    for metric in metrics:
        values = []
        for seed in SEEDS:
            rows = _read_rows(_run_dir(runs_root, seed) / "summaries" / MC_SUMMARY_NAME)
            if rows:
                values.append(rows[0].get(metric))
        values_by_metric.append(_mean_std(values)[:2])
    means = np.asarray([item[0] for item in values_by_metric], dtype=np.float64)
    stds = np.asarray([item[1] for item in values_by_metric], dtype=np.float64)
    if not np.any(np.isfinite(means)):
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:purple", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("recovery time [s]")
    ax.set_title("Disturbance recovery mean/std")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "disturbance_recovery_mean_std.png", dpi=160)
    plt.close(fig)


def make_figures(runs_root: Path, figures_dir: Path, fixed_rows: list[dict[str, Any]]) -> None:
    _plot_training_returns(runs_root, figures_dir)
    _plot_fixed_bar(
        fixed_rows,
        figures_dir,
        metric="rmse_theta_deg",
        output_name="fixed_eval_rmse_mean_std.png",
        ylabel="RMSE theta [deg]",
    )
    _plot_fixed_bar(
        fixed_rows,
        figures_dir,
        metric="flag_U_safe_count",
        output_name="fixed_eval_flag_U_mean_std.png",
        ylabel="flag_U_safe_count",
    )
    _plot_mc_boxplot(runs_root, figures_dir)
    _plot_mc_seed_bars(runs_root, figures_dir)
    _plot_disturbance_recovery(runs_root, figures_dir)


def _lookup_metric(rows: list[dict[str, Any]], scenario: str, controller: str, metric: str) -> float:
    for row in rows:
        if str(row.get("scenario", "")) == scenario and str(row.get("controller", "")) == controller:
            return _safe_float(row.get(metric))
    return float("nan")


def _mean_fixed_rmse(rows: list[dict[str, Any]]) -> float:
    values = [_safe_float(row.get("rmse_theta_deg_mean")) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def write_conclusion(
    *,
    runs_root: Path,
    summaries_dir: Path,
    fixed_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
    pid_rows: list[dict[str, Any]],
    v7_rows: list[dict[str, Any]],
) -> None:
    lines = ["# v8 Full Multi-Seed Result Conclusion", ""]
    lines.append("## Seed Completion")
    best_steps: list[int] = []
    for seed in SEEDS:
        run_dir = _run_dir(runs_root, seed)
        info = _best_checkpoint_info(run_dir)
        try:
            best_steps.append(int(info.get("global_step", "")))
        except (TypeError, ValueError):
            pass
        lines.append(
            f"- seed {seed}: training_600000={_train_completed(run_dir)}, "
            f"fixed_eval={(run_dir / 'summaries' / FIXED_SUMMARY_NAME).exists()}, "
            f"mc_eval={(run_dir / 'summaries' / MC_SUMMARY_NAME).exists()}, "
            f"best={info['checkpoint_path']}, source={info['source_checkpoint_path'] or 'periodic_eval_best'}"
        )

    early_safe_selection = bool(best_steps) and any(step < 600000 for step in best_steps)
    if early_safe_selection:
        lines += [
            "",
            "## Checkpoint Selection Note",
            "- At least one selected safe checkpoint is earlier than the 600000-step final policy.",
            "- Treat this as policy drift during long randomized training: the completed runs are useful, but the safe deployment candidate is the selector-chosen checkpoint_best.pt.",
        ]

    fixed_omega = sum(int(row.get("omega_limit_violation_count", 0)) for row in fixed_rows)
    fixed_hard = sum(int(row.get("hard_safety_violation_count", 0)) for row in fixed_rows)
    mc_lookup = {row["metric"]: row for row in mc_rows}
    mc_omega_total = _safe_float(mc_lookup.get("omega_limit_violation_count", {}).get("total"))
    mc_hard_total = _safe_float(mc_lookup.get("hard_safety_violation_count", {}).get("total"))
    success_mean = _safe_float(mc_lookup.get("success_rate", {}).get("mean"))
    success_std = _safe_float(mc_lookup.get("success_rate", {}).get("std"))

    lines += [
        "",
        "## Safety",
        f"- Fixed omega_limit_violation count: {fixed_omega}",
        f"- Fixed hard safety violation count: {fixed_hard}",
        f"- MC omega_limit_violation total: {mc_omega_total:g}",
        f"- MC hard safety violation total: {mc_hard_total:g}",
        f"- MC success_rate mean +/- std: {success_mean:.6g} +/- {success_std:.6g}",
        "",
        "## Fixed Evaluation Mean +/- Std",
    ]
    for row in fixed_rows:
        lines.append(
            f"- {row['scenario']}: RMSE {float(row['rmse_theta_deg_mean']):.6g} +/- "
            f"{float(row['rmse_theta_deg_std']):.6g} deg, "
            f"flag_U {float(row['flag_U_safe_count_mean']):.6g} +/- {float(row['flag_U_safe_count_std']):.6g}, "
            f"success {float(row['success_rate']):.3f}"
        )

    lines += ["", "## Monte Carlo Mean +/- Std"]
    for metric in ("mean_rmse_theta_deg", "max_rmse_theta_deg", "mean_mae_theta_deg", "mean_flag_U_safe_count"):
        row = mc_lookup.get(metric, {})
        lines.append(f"- {metric}: {_safe_float(row.get('mean')):.6g} +/- {_safe_float(row.get('std')):.6g}")

    partial_rows = _read_rows(runs_root / "gun_servo_mc_sc_residual_td3_random_v8" / "summaries" / "mc_sc_residual_td3_v8_fixed_summary.csv")
    partial_mean = float("nan")
    if partial_rows:
        partial_mean = float(np.mean(_finite([row.get("rmse_theta_deg") for row in partial_rows])))
    full_mean = _mean_fixed_rmse(fixed_rows)
    lines += [
        "",
        "## v8 Full vs Partial v8",
        f"- partial fixed mean RMSE: {partial_mean:.6g}",
        f"- full multi-seed fixed mean RMSE: {full_mean:.6g}",
        f"- full training improved fixed mean RMSE: {bool(math.isfinite(partial_mean) and math.isfinite(full_mean) and full_mean < partial_mean)}",
    ]

    v7_scenarios = sorted({row["scenario"] for row in v7_rows})
    v7_deltas = []
    for scenario in v7_scenarios:
        full_rmse = _lookup_metric(v7_rows, scenario, "mc_sc_residual_td3_v8_full", "rmse_theta_deg_mean")
        v7_rmse = _lookup_metric(v7_rows, scenario, "mc_sc_residual_td3_v7", "rmse_theta_deg_mean")
        if math.isfinite(full_rmse) and math.isfinite(v7_rmse):
            v7_deltas.append(full_rmse - v7_rmse)
    lines += [
        "",
        "## v8 Full vs v7",
        f"- mean fixed RMSE delta (v8_full - v7): {float(np.mean(v7_deltas)) if v7_deltas else float('nan'):.6g}",
        "- Randomized v8 adds Monte Carlo robustness evidence that fixed v7 did not measure.",
    ]

    competitive = []
    pid_better = []
    for scenario in sorted({row["scenario"] for row in pid_rows}):
        full_rmse = _lookup_metric(pid_rows, scenario, "mc_sc_residual_td3_v8_full", "rmse_theta_deg_mean")
        pid_rmse = _lookup_metric(pid_rows, scenario, "pid_v2", "rmse_theta_deg_mean")
        if not (math.isfinite(full_rmse) and math.isfinite(pid_rmse)):
            continue
        (competitive if full_rmse <= pid_rmse else pid_better).append(scenario)
    lines += [
        "",
        "## v8 Full vs PID v2",
        f"- v8 competitive/lower RMSE scenarios: {', '.join(competitive) if competitive else 'none'}",
        f"- PID lower RMSE scenarios: {', '.join(pid_better) if pid_better else 'none'}",
        "",
        "## Recommendation",
    ]
    if fixed_omega == 0 and fixed_hard == 0 and mc_omega_total == 0 and mc_hard_total == 0 and success_mean >= 0.95:
        if early_safe_selection:
            lines.append(
                "- Proceed to add the SMC baseline and paper tables, but tune v8 randomization/reward or checkpoint-selection cadence before claiming the latest 600000-step policy as final."
            )
        else:
            lines.append("- Proceed to add the SMC baseline and paper tables.")
    else:
        lines.append("- Tune randomization ranges or reward weights before paper-scale comparison.")
    lines.append("")
    (summaries_dir / "v8_full_result_conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fixed_rows = aggregate_fixed(args.runs_root, summaries_dir)
    mc_rows = aggregate_mc(args.runs_root, summaries_dir)
    pid_rows = aggregate_comparison(
        args.runs_root,
        summaries_dir,
        filename=PID_COMPARISON_NAME,
        output_name="v8_full_vs_pid_v2_mean_std.csv",
    )
    v7_rows = aggregate_comparison(
        args.runs_root,
        summaries_dir,
        filename=V7_COMPARISON_NAME,
        output_name="v8_full_vs_v7_mean_std.csv",
    )
    make_figures(args.runs_root, figures_dir, fixed_rows)
    write_conclusion(
        runs_root=args.runs_root,
        summaries_dir=summaries_dir,
        fixed_rows=fixed_rows,
        mc_rows=mc_rows,
        pid_rows=pid_rows,
        v7_rows=v7_rows,
    )
    print(f"saved aggregate summaries under {summaries_dir}")
    print(f"saved aggregate figures under {figures_dir}")


if __name__ == "__main__":
    main()

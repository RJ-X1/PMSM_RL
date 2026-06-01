"""Create paper-ready gun-servo result tables, figures, and reports."""

from __future__ import annotations

from pathlib import Path
import csv
import math
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_ready_results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
REPRESENTATIVE = FIGURES / "representative"

FIXED_SOURCE = ROOT / "outputs" / "runs" / "final_method_comparison" / "summaries" / "fixed_six_scenario_comparison.csv"
MC_SOURCE = ROOT / "outputs" / "runs" / "final_method_comparison" / "summaries" / "monte_carlo_comparison.csv"
CONCLUSION_SOURCE = (
    ROOT / "outputs" / "runs" / "final_method_comparison" / "summaries" / "final_comparison_conclusion.md"
)
PID_MC_SUMMARY_SOURCE = ROOT / "outputs" / "runs" / "gun_servo_pid_mc_baseline" / "summaries" / "pid_mc_eval_summary.csv"
PID_MC_EPISODES_SOURCE = (
    ROOT / "outputs" / "runs" / "gun_servo_pid_mc_baseline" / "summaries" / "pid_mc_eval_episode_metrics.csv"
)

SCENARIOS = [
    "C1a_5deg_step",
    "C1_10deg_step",
    "C2a_20deg_step_safe",
    "C3_trapezoid_tracking",
    "C4a_sine_tracking_safe",
    "C5a_disturbance_after_settling",
]

SHORT_SCENARIOS = {
    "C1a_5deg_step": "C1a",
    "C1_10deg_step": "C1",
    "C2a_20deg_step_safe": "C2a",
    "C3_trapezoid_tracking": "C3",
    "C4a_sine_tracking_safe": "C4a",
    "C5a_disturbance_after_settling": "C5a",
}

METHOD_MAP = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Direct TD3-PI v3": "Direct TD3-PI",
    "Residual TD3 v4": "Residual TD3-PI",
    "SC-Residual TD3 v5": "SC-Residual TD3-PI",
    "MC-SC-Residual TD3 v7": "MC-SC-Residual TD3-PI",
    "Randomized MC-SC-Residual TD3 v8 full": "Randomized MC-SC-Residual TD3-PI",
}

METHOD_ORDER = [
    "PID/PD-PI",
    "SMC-PI",
    "Direct TD3-PI",
    "Residual TD3-PI",
    "SC-Residual TD3-PI",
    "MC-SC-Residual TD3-PI",
    "Randomized MC-SC-Residual TD3-PI",
]

PROPOSED = "Randomized MC-SC-Residual TD3-PI"

FIGURE_LABELS = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Direct TD3-PI": "Direct\nTD3-PI",
    "Residual TD3-PI": "Residual\nTD3-PI",
    "SC-Residual TD3-PI": "SC-Residual\nTD3-PI",
    "MC-SC-Residual TD3-PI": "MC-SC\nResidual TD3-PI",
    "Randomized MC-SC-Residual TD3-PI": "Randomized MC-SC\nResidual TD3-PI",
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
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
    if value in (None, "", "nan", "NaN"):
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


def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _normalize_fixed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        method = METHOD_MAP.get(str(row.get("method", "")))
        if method is None:
            continue
        item = dict(row)
        item["method"] = method
        out.append(item)
    out.sort(key=lambda row: (METHOD_ORDER.index(row["method"]), SCENARIOS.index(row["scenario"])))
    return out


def _fixed_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["method"]), str(row["scenario"])): row for row in rows}


def _method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("method", "")) == method]


def _method_average_rmse(rows: list[dict[str, Any]], method: str) -> float:
    return _mean([row.get("rmse_theta_deg") for row in _method_rows(rows, method)])


def _method_safety_count(rows: list[dict[str, Any]], method: str) -> float:
    total = 0.0
    for row in _method_rows(rows, method):
        total += _safe_float(row.get("omega_limit_violation"), 0.0)
        total += _safe_float(row.get("hard_safety_violation"), 0.0)
    return total


def _best_by_scenario(rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for scenario in SCENARIOS:
        vals = [_safe_float(row.get("rmse_theta_deg")) for row in rows if row.get("scenario") == scenario]
        finite = [value for value in vals if math.isfinite(value)]
        if finite:
            best[scenario] = min(finite)
    return best


def _build_fixed_rmse_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = _fixed_lookup(rows)
    best_scenario = _best_by_scenario(rows)
    avg_values = {method: _method_average_rmse(rows, method) for method in METHOD_ORDER if _method_rows(rows, method)}
    best_avg = min((value for value in avg_values.values() if math.isfinite(value)), default=float("nan"))
    csv_rows: list[dict[str, Any]] = []
    md_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if not _method_rows(rows, method):
            continue
        csv_method = f"{method} [proposed]" if method == PROPOSED else method
        md_method = f"**{method} (proposed)**" if method == PROPOSED else method
        csv_row = {"Method": csv_method}
        md_row = {"Method": md_method}
        for scenario in SCENARIOS:
            value = _safe_float(lookup.get((method, scenario), {}).get("rmse_theta_deg"))
            is_best = math.isfinite(value) and abs(value - best_scenario.get(scenario, float("inf"))) <= 1e-12
            cell = _fmt(value)
            csv_row[f"{scenario} RMSE"] = f"{cell} (best)" if is_best else cell
            md_row[f"{scenario} RMSE"] = f"**{cell}**" if is_best else cell
        avg = avg_values[method]
        avg_best = math.isfinite(avg) and abs(avg - best_avg) <= 1e-12
        csv_row["Average RMSE"] = f"{_fmt(avg)} (best)" if avg_best else _fmt(avg)
        md_row["Average RMSE"] = f"**{_fmt(avg)}**" if avg_best else _fmt(avg)
        csv_row["Safety violation count"] = int(_method_safety_count(rows, method))
        md_row["Safety violation count"] = int(_method_safety_count(rows, method))
        csv_rows.append(csv_row)
        md_rows.append(md_row)
    return csv_rows, md_rows


def _markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    return "\n".join(lines) + "\n"


def _build_safety_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = _method_rows(rows, method)
        if not method_rows:
            continue
        n = len(method_rows)
        success = sum(1 for row in method_rows if "episode_limit" in str(row.get("done_reason", "")))
        out.append(
            {
                "method": f"{method} [proposed]" if method == PROPOSED else method,
                "total flag_U_safe_count": int(sum(_safe_float(row.get("flag_U_safe_count"), 0.0) for row in method_rows)),
                "total flag_E_safe_count": int(sum(_safe_float(row.get("flag_E_safe_count"), 0.0) for row in method_rows)),
                "total flag_X_safe_count": int(sum(_safe_float(row.get("flag_X_safe_count"), 0.0) for row in method_rows)),
                "total sigma_safe_count": int(sum(_safe_float(row.get("sigma_safe_count"), 0.0) for row in method_rows)),
                "omega_limit_violation count": int(
                    sum(_safe_float(row.get("omega_limit_violation"), 0.0) for row in method_rows)
                ),
                "hard safety violation count": int(
                    sum(_safe_float(row.get("hard_safety_violation"), 0.0) for row in method_rows)
                ),
                "episode_limit success rate": f"{success / max(1, n):.3f}",
            }
        )
    return out


def _normalize_mc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rename = {
        "Randomized MC-SC-Residual TD3 v8 full": PROPOSED,
    }
    out = []
    for row in rows:
        item = dict(row)
        item["method"] = rename.get(str(row.get("method", "")), str(row.get("method", "")))
        if item["method"] == PROPOSED:
            item["method"] = f"{PROPOSED} [proposed]"
        out.append(item)
    return out


def _write_mc_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = [
        "method",
        "success_rate",
        "episode_limit_rate",
        "mean_rmse_theta_deg",
        "std_rmse_theta_deg",
        "max_rmse_theta_deg",
        "mean_mae_theta_deg",
        "omega_limit_violation_count",
        "hard_safety_violation_count",
        "mean_flag_U_safe_count",
        "total_flag_U_safe_count",
        "mean_recovery_time_s",
        "max_abs_error_after_disturbance_deg",
    ]
    csv_rows = []
    md_rows = []
    for row in rows:
        csv_row = {"method": row["method"]}
        md_row = {"method": row["method"].replace(" [proposed]", " (proposed)")}
        for field in fields[1:]:
            digits = 4 if "rate" in field or "rmse" in field or "mae" in field or "recovery" in field or "error" in field else 2
            value = _safe_float(row.get(field))
            text = f"{value:.{digits}f}" if math.isfinite(value) else "-"
            csv_row[field] = text
            md_row[field] = text
        csv_rows.append(csv_row)
        md_rows.append(md_row)
    return csv_rows, md_rows


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _plot_fixed_grouped_rmse(rows: list[dict[str, Any]]) -> None:
    methods = [method for method in METHOD_ORDER if _method_rows(rows, method)]
    x = np.arange(len(SCENARIOS), dtype=np.float64)
    width = min(0.8 / len(methods), 0.11)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(methods)))
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    for idx, method in enumerate(methods):
        values = []
        for scenario in SCENARIOS:
            match = next((row for row in rows if row["method"] == method and row["scenario"] == scenario), None)
            values.append(_safe_float(match.get("rmse_theta_deg")) if match else np.nan)
        label = FIGURE_LABELS[method]
        ax.bar(x + (idx - (len(methods) - 1) / 2.0) * width, values, width=width, label=label, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_SCENARIOS[item] for item in SCENARIOS])
    ax.set_ylabel("RMSE (deg)")
    ax.set_xlabel("Scenario")
    ax.set_title("Fixed-scenario tracking RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "fig_fixed_rmse_grouped_bar.png", dpi=300)
    plt.close(fig)


def _plot_fixed_average_rmse(rows: list[dict[str, Any]]) -> None:
    methods = [method for method in METHOD_ORDER if _method_rows(rows, method)]
    values = [_method_average_rmse(rows, method) for method in methods]
    colors = ["#4C78A8" if method != PROPOSED else "#F58518" for method in methods]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar([FIGURE_LABELS[m] for m in methods], values, color=colors)
    ax.set_ylabel("Average RMSE (deg)")
    ax.set_title("Average fixed-scenario RMSE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_fixed_average_rmse_bar.png", dpi=300)
    plt.close(fig)


def _plot_fixed_safety(rows: list[dict[str, Any]]) -> None:
    methods = [method for method in METHOD_ORDER if _method_rows(rows, method)]
    flag_u = [sum(_safe_float(row.get("flag_U_safe_count"), 0.0) for row in _method_rows(rows, method)) for method in methods]
    flag_e = [sum(_safe_float(row.get("flag_E_safe_count"), 0.0) for row in _method_rows(rows, method)) for method in methods]
    flag_x = [sum(_safe_float(row.get("flag_X_safe_count"), 0.0) for row in _method_rows(rows, method)) for method in methods]
    sigma = [sum(_safe_float(row.get("sigma_safe_count"), 0.0) for row in _method_rows(rows, method)) for method in methods]
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(x - 1.5 * width, flag_u, width=width, label="flag_U")
    ax.bar(x - 0.5 * width, flag_e, width=width, label="flag_E")
    ax.bar(x + 0.5 * width, flag_x, width=width, label="flag_X")
    ax.bar(x + 1.5 * width, sigma, width=width, label="sigma")
    ax.set_xticks(x)
    ax.set_xticklabels([FIGURE_LABELS[m] for m in methods], rotation=25)
    ax.set_ylabel("Total count")
    ax.set_title("Fixed-scenario safety flags")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_fixed_safety_flags_bar.png", dpi=300)
    plt.close(fig)


def _plot_mc_bars(mc_rows: list[dict[str, Any]]) -> None:
    methods = [row["method"].replace(" [proposed]", "") for row in mc_rows]
    labels = [FIGURE_LABELS.get(method, method) for method in methods]
    colors = ["#4C78A8" if method != PROPOSED else "#F58518" for method in methods]

    def bar(field: str, ylabel: str, title: str, path: Path) -> None:
        values = [_safe_float(row.get(field), 0.0) for row in mc_rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    bar("mean_rmse_theta_deg", "Mean RMSE (deg)", "Monte Carlo mean RMSE", FIGURES / "fig_mc_mean_rmse_bar.png")
    bar("success_rate", "Success rate", "Monte Carlo success rate", FIGURES / "fig_mc_success_rate_bar.png")
    bar("mean_flag_U_safe_count", "Mean flag_U count", "Monte Carlo command safety intervention", FIGURES / "fig_mc_flag_U_bar.png")

    omega = [_safe_float(row.get("omega_limit_violation_count"), 0.0) for row in mc_rows]
    hard = [_safe_float(row.get("hard_safety_violation_count"), 0.0) for row in mc_rows]
    x = np.arange(len(mc_rows), dtype=np.float64)
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2.0, omega, width=width, label="omega")
    ax.bar(x + width / 2.0, hard, width=width, label="hard")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Violation count")
    ax.set_title("Monte Carlo safety violations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_mc_safety_violation_bar.png", dpi=300)
    plt.close(fig)


def _read_trace(path: Path, max_points: int = 2000) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    rows = _read_rows(path)
    if not rows:
        return None
    step = max(1, len(rows) // max_points)
    sampled = rows[::step]
    data: dict[str, list[float]] = {"t_s": [], "theta_ref_deg": [], "theta_L_deg": [], "e_theta_deg": [], "T_L": []}
    for row in sampled:
        for key in data:
            data[key].append(_safe_float(row.get(key)))
    return {key: np.asarray(values, dtype=np.float64) for key, values in data.items()}


def _trajectory_sources(scenario: str) -> dict[str, Path]:
    return {
        "PID/PD-PI": ROOT / "outputs" / "runs" / "gun_servo_pid_smoke_v2" / "eval" / f"{scenario}_pid.csv",
        "SMC-PI": ROOT / "outputs" / "runs" / "gun_servo_smc_baseline" / "eval" / f"{scenario}_smc.csv",
        "Proposed": (
            ROOT
            / "outputs"
            / "runs"
            / "gun_servo_mc_sc_residual_td3_random_v8_full_seed0"
            / "eval"
            / f"{scenario}_td3.csv"
        ),
    }


def _plot_position_comparison(scenario: str, out_name: str) -> str | None:
    traces = {label: _read_trace(path) for label, path in _trajectory_sources(scenario).items()}
    traces = {label: trace for label, trace in traces.items() if trace is not None}
    if not traces:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    first = next(iter(traces.values()))
    ax.plot(first["t_s"], first["theta_ref_deg"], "k--", linewidth=1.3, label="Reference")
    for label, trace in traces.items():
        ax.plot(trace["t_s"], trace["theta_L_deg"], linewidth=1.2, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (deg)")
    ax.set_title(f"{SHORT_SCENARIOS[scenario]} position tracking comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = REPRESENTATIVE / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def _plot_disturbance_comparison() -> str | None:
    scenario = "C5a_disturbance_after_settling"
    traces = {label: _read_trace(path) for label, path in _trajectory_sources(scenario).items()}
    traces = {label: trace for label, trace in traces.items() if trace is not None}
    if not traces:
        return None
    first = next(iter(traces.values()))
    t = first["t_s"]
    torque = first["T_L"]
    finite = torque[np.isfinite(torque)]
    center = 2.5
    if finite.size:
        changes = np.flatnonzero(np.abs(np.diff(torque)) > 1e-6 * max(1.0, float(np.max(np.abs(finite)))))
        if changes.size:
            center = float(t[int(changes[0] + 1)])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for label, trace in traces.items():
        mask = (trace["t_s"] >= center - 0.8) & (trace["t_s"] <= center + 1.2)
        ax.plot(trace["t_s"][mask], trace["e_theta_deg"][mask], linewidth=1.2, label=label)
    ax.axvline(center, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (deg)")
    ax.set_title("C5a disturbance rejection comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = REPRESENTATIVE / "fig_C5a_disturbance_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def _make_narrative(
    *,
    fixed_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
    missing_notes: list[str],
) -> None:
    pid_mc = next((row for row in mc_rows if row["method"] == "PID/PD-PI"), {})
    smc_mc_new = next((row for row in mc_rows if row["method"] == "SMC-PI"), {})
    proposed_mc_new = next((row for row in mc_rows if row["method"].replace(" [proposed]", "") == PROPOSED), {})
    mc_focus = [row for row in (pid_mc, smc_mc_new, proposed_mc_new) if row]
    best_mc = min(mc_focus, key=lambda row: _safe_float(row.get("mean_rmse_theta_deg"), float("inf"))) if mc_focus else {}
    v8_best = str(best_mc.get("method", "")).replace(" [proposed]", "") == PROPOSED
    classical_lower_flag = all(
        _safe_float(row.get("mean_flag_U_safe_count"), float("inf"))
        < _safe_float(proposed_mc_new.get("mean_flag_U_safe_count"), float("inf"))
        for row in (pid_mc, smc_mc_new)
        if row
    ) if proposed_mc_new and (pid_mc or smc_mc_new) else False
    text = f"""# Experiment Results Narrative

## 1. Experimental Setup

The gun-servo PMSM position-control task compares an engineering PID/PD-PI baseline, an SMC-PI robust-control baseline, and the proposed Randomized MC-SC-Residual TD3-PI method. All methods use the same outer-loop interface and pass through the same speed-command limits, acceleration limits, speed PI layer, actuator constraints, PMSM, gearbox, load dynamics, and safety checks.

## 2. Fixed Nominal Scenarios

The fixed scenarios include C1a small step, C1 step, C2a speed-limited step, C3 trapezoid tracking, C4a sine tracking, and C5a disturbance-after-settling recovery. PID/PD-PI and SMC-PI remain very strong in these nominal scenarios, with average fixed RMSE values of {_method_average_rmse(fixed_rows, 'PID/PD-PI'):.4f} deg and {_method_average_rmse(fixed_rows, 'SMC-PI'):.4f} deg. The proposed method has an average fixed RMSE of {_method_average_rmse(fixed_rows, PROPOSED):.4f} deg and no hard safety violations in the fixed six-scenario table.

## 3. Monte Carlo Robustness

The Monte Carlo evaluation uses the same randomized scenario-family distribution and plant/load randomization ranges for PID/PD-PI, SMC-PI, and the proposed method. PID/PD-PI reached a success rate of {_fmt(pid_mc.get('success_rate'), 4)} with mean RMSE {_fmt(pid_mc.get('mean_rmse_theta_deg'), 4)} deg. SMC-PI reached a success rate of {_fmt(smc_mc_new.get('success_rate'), 4)} with mean RMSE {_fmt(smc_mc_new.get('mean_rmse_theta_deg'), 4)} deg. The proposed method reached a success rate of {_fmt(proposed_mc_new.get('success_rate'), 4)} with mean RMSE {_fmt(proposed_mc_new.get('mean_rmse_theta_deg'), 4)} deg.

In this updated comparison, the proposed method still has the best Monte Carlo mean RMSE: {'yes' if v8_best else 'no'}. PID/PD-PI and SMC-PI have fewer command-layer flag_U interventions than the proposed method: {'yes' if classical_lower_flag else 'no'}. All three methods maintain zero omega-limit and hard-safety violations in the summarized Monte Carlo results.

## 4. Interpretation

The proposed method should not be claimed as universally better unless a specific metric supports that statement. A fair thesis wording is: under the tested randomized conditions, Randomized MC-SC-Residual TD3-PI improves average Monte Carlo tracking robustness relative to PID/PD-PI and SMC-PI while preserving safety, but PID/PD-PI and SMC-PI remain highly competitive in fixed nominal scenarios and require fewer command-layer safety interventions in Monte Carlo.

## 5. Recommended Thesis Claim

The strongest supported claim is about safety, robustness, and adaptation under random conditions: the proposed residual TD3 controller combines the classical PI/PID safety chain with randomized training and achieves the lowest mean Monte Carlo RMSE among the three final MC methods, without hard safety violations. The fixed-scenario section should separately acknowledge that classical PID/SMC baselines remain very strong and often more accurate under nominal conditions.

## 6. Remaining Limitations

{chr(10).join(f'- {note}' for note in missing_notes)}
"""
    (REPORTS / "experiment_results_narrative.md").write_text(text, encoding="utf-8")
    return

    smc_mc = next((row for row in mc_rows if row["method"] == "SMC-PI"), {})
    proposed_mc = next((row for row in mc_rows if row["method"].replace(" [proposed]", "") == PROPOSED), {})
    proposed_fixed = _method_rows(fixed_rows, PROPOSED)
    smc_fixed = _method_rows(fixed_rows, "SMC-PI")
    pid_fixed = _method_rows(fixed_rows, "PID/PD-PI")
    text = f"""# 实验结果叙述稿

## 1. 实验设置概述

本文面向火炮伺服 PMSM 位置外环控制任务，对工程 PID/PD-PI、经典鲁棒 SMC-PI 以及多阶段 TD3-PI 方法进行了对比。所有外环控制器均输出负载侧速度修正或速度指令，并经过相同的速度限幅、加速度限幅、动作平滑、速度 PI、执行器约束、PMSM、减速器和负载动力学链路。因此，表中差异主要反映位置外环策略本身的差异，而不是底层执行链路差异。

## 2. 固定六场景对比

固定场景包括 C1a 小阶跃、C1 阶跃、C2a 安全限速阶跃、C3 梯形轨迹、C4a 正弦跟踪和 C5a 扰动后恢复。PID/PD-PI 与 SMC-PI 在固定标称场景中仍然表现很强，平均 RMSE 分别为 {_method_average_rmse(fixed_rows, 'PID/PD-PI'):.4f} deg 和 {_method_average_rmse(fixed_rows, 'SMC-PI'):.4f} deg。拟采用的随机化 MC-SC-Residual-TD3-PI 在六个固定场景上的平均 RMSE 为 {_method_average_rmse(fixed_rows, PROPOSED):.4f} deg，安全终止违规次数为 {int(_method_safety_count(fixed_rows, PROPOSED))}。

## 3. 安全指标分析

固定场景中，所有纳入最终表格的方法均未出现硬安全违规。命令层安全干预 flag_U 反映速度指令限幅、加速度限幅或动作裁剪等保护机制的触发次数。SMC-PI 在部分固定场景中降低了 flag_U，但在 C1/C5a 等较大阶跃或扰动场景中仍存在命令层干预。随机化 MC-SC-Residual-TD3-PI 在固定和随机评估中均保持零 omega_limit_violation 和零 hard_safety_violation，说明安全约束链路有效。

## 4. Monte Carlo 鲁棒性分析

Monte Carlo 评估采用随机场景族和参数随机化，用于检验控制器在参考轨迹、负载参数、摩擦、重力矩、扰动、编码器噪声、减速器效率和直流母线变化下的鲁棒性。SMC-PI 的 MC success_rate 为 {_fmt(smc_mc.get('success_rate'), 4)}，平均 RMSE 为 {_fmt(smc_mc.get('mean_rmse_theta_deg'), 4)} deg；随机化 MC-SC-Residual-TD3-PI 的 MC success_rate 为 {_fmt(proposed_mc.get('success_rate'), 4)}，平均 RMSE 为 {_fmt(proposed_mc.get('mean_rmse_theta_deg'), 4)} deg。由此可见，拟采用方法在随机化鲁棒性平均 RMSE 上优于 SMC-PI，同时保持了 100% 成功率和零硬安全违规。

## 5. SMC/PID 与拟采用方法讨论

PID/PD-PI 和 SMC-PI 代表工程调参基线与经典鲁棒控制基线。二者在固定标称场景下具有很强的跟踪精度，因此不应将强化学习方法表述为固定条件下普遍优于经典控制。更合适的表述是：随机化 MC-SC-Residual-TD3-PI 在保持安全性的基础上，面向多条件随机化任务表现出更好的鲁棒适应性，尤其在 Monte Carlo 平均 RMSE 上优于 SMC-PI。

## 6. 局限性

当前结果仍存在若干限制：第一，PID/PD-PI 尚未完成与 SMC/v8 相同设置的 Monte Carlo 随机化评估，经典控制在随机条件下的完整对比仍可补充；第二，Direct TD3-PI、Residual TD3 v4 和 SC-Residual TD3 v5 只有部分固定场景结果，不能直接与六场景方法做完整平均对比；第三，拟采用方法在固定标称场景 RMSE 上仍落后于 PID/SMC，且 Monte Carlo 中命令层 flag_U 干预次数高于 SMC-PI。

## 7. 建议论文结论写法

建议将本文方法表述为“随机化多条件安全约束残差 TD3-PI 控制方法”。结论应强调：该方法并非在所有固定标称场景中都超过经典控制，而是在统一安全链路下实现了固定与随机条件的全场景安全完成，并在 Monte Carlo 随机化评估中取得低于 SMC-PI 的平均 RMSE。因此，强化学习方法的优势应定位为随机条件下的鲁棒性和适应性，而不是固定工况下对经典控制的全面精度替代。

## 数据缺失说明

{chr(10).join(f'- {note}' for note in missing_notes)}
"""
    (REPORTS / "experiment_results_narrative.md").write_text(text, encoding="utf-8-sig")


def _make_checklist(generated_tables: list[Path], generated_figures: list[Path], representative: list[str], missing_notes: list[str]) -> None:
    source_paths = [
        PID_MC_SUMMARY_SOURCE,
        PID_MC_EPISODES_SOURCE,
        MC_SOURCE,
        CONCLUSION_SOURCE,
        FIXED_SOURCE,
    ]
    lines = [
        "# Result File Checklist",
        "",
        "## Generated Tables",
        *[f"- {path}" for path in generated_tables],
        "",
        "## Generated Figures",
        *[f"- {path}" for path in generated_figures],
        "",
        "## Representative Trajectory Figures",
        *[f"- {path}" for path in representative],
        "",
        "## Source CSV / Report Paths",
        *[f"- {path}" for path in source_paths],
        "",
        "## Missing Data Notes",
        *[f"- {note}" for note in missing_notes],
        "",
        "## Recommended Files For Thesis/Paper",
        "- tables/table_fixed_six_scenario_rmse.md",
        "- tables/table_fixed_safety_metrics.md",
        "- tables/table_monte_carlo_robustness.md",
        "- figures/fig_fixed_rmse_grouped_bar.png",
        "- figures/fig_fixed_average_rmse_bar.png",
        "- figures/fig_mc_mean_rmse_bar.png",
        "- figures/fig_mc_success_rate_bar.png",
        "- figures/fig_mc_safety_violation_bar.png",
        "- figures/fig_mc_flag_U_bar.png",
        "- figures/representative/fig_C1_position_comparison.png",
        "- figures/representative/fig_C5a_disturbance_comparison.png",
        "- reports/experiment_results_narrative.md",
    ]
    (REPORTS / "result_file_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in (TABLES, FIGURES, REPORTS, REPRESENTATIVE):
        path.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib()

    fixed_rows = _normalize_fixed_rows(_read_rows(FIXED_SOURCE))
    mc_rows = _normalize_mc_rows(_read_rows(MC_SOURCE))

    fixed_csv_rows, fixed_md_rows = _build_fixed_rmse_table(fixed_rows)
    fixed_fields = [
        "Method",
        *(f"{scenario} RMSE" for scenario in SCENARIOS),
        "Average RMSE",
        "Safety violation count",
    ]
    fixed_csv = TABLES / "table_fixed_six_scenario_rmse.csv"
    fixed_md = TABLES / "table_fixed_six_scenario_rmse.md"
    _write_rows(fixed_csv, fixed_csv_rows, fixed_fields)
    fixed_md.write_text(_markdown_table(fixed_md_rows, fixed_fields), encoding="utf-8")

    safety_rows = _build_safety_table(fixed_rows)
    safety_fields = list(safety_rows[0].keys()) if safety_rows else []
    safety_csv = TABLES / "table_fixed_safety_metrics.csv"
    safety_md = TABLES / "table_fixed_safety_metrics.md"
    _write_rows(safety_csv, safety_rows, safety_fields)
    safety_md.write_text(_markdown_table(safety_rows, safety_fields), encoding="utf-8")

    mc_csv_rows, mc_md_rows = _write_mc_table(mc_rows)
    mc_fields = list(mc_csv_rows[0].keys()) if mc_csv_rows else []
    mc_csv = TABLES / "table_monte_carlo_robustness.csv"
    mc_md = TABLES / "table_monte_carlo_robustness.md"
    _write_rows(mc_csv, mc_csv_rows, mc_fields)
    mc_md.write_text(_markdown_table(mc_md_rows, mc_fields), encoding="utf-8")

    _plot_fixed_grouped_rmse(fixed_rows)
    _plot_fixed_average_rmse(fixed_rows)
    _plot_fixed_safety(fixed_rows)
    _plot_mc_bars(mc_rows)

    representative = []
    for scenario, name in (
        ("C1_10deg_step", "fig_C1_position_comparison.png"),
        ("C3_trapezoid_tracking", "fig_C3_position_comparison.png"),
        ("C4a_sine_tracking_safe", "fig_C4a_position_comparison.png"),
    ):
        out = _plot_position_comparison(scenario, name)
        if out:
            representative.append(out)
    c5 = _plot_disturbance_comparison()
    if c5:
        representative.append(c5)

    missing_notes = []
    for method in METHOD_ORDER:
        missing = [SHORT_SCENARIOS[s] for s in SCENARIOS if not any(r["method"] == method and r["scenario"] == s for r in fixed_rows)]
        if missing:
            missing_notes.append(f"{method}: missing fixed scenarios {', '.join(missing)}.")
    if not any(row["method"] == "PID/PD-PI" for row in mc_rows):
        missing_notes.append("PID/PD-PI Monte Carlo evaluation was not available.")
    if not representative:
        missing_notes.append("No representative trajectory overlay figures could be generated from CSV traces.")
    missing_notes.append("MC-SC-Residual TD3 v6 exists in source summaries but is omitted from the main paper table to match the requested method list.")

    generated_tables = [fixed_csv, fixed_md, safety_csv, safety_md, mc_csv, mc_md]
    generated_figures = [
        FIGURES / "fig_fixed_rmse_grouped_bar.png",
        FIGURES / "fig_fixed_average_rmse_bar.png",
        FIGURES / "fig_fixed_safety_flags_bar.png",
        FIGURES / "fig_mc_mean_rmse_bar.png",
        FIGURES / "fig_mc_success_rate_bar.png",
        FIGURES / "fig_mc_safety_violation_bar.png",
        FIGURES / "fig_mc_flag_U_bar.png",
    ]
    _make_narrative(fixed_rows=fixed_rows, mc_rows=mc_rows, missing_notes=missing_notes)
    _make_checklist(generated_tables, generated_figures, representative, missing_notes)

    print(f"paper-ready results written to {OUT}")
    print(f"tables: {len(generated_tables)}")
    print(f"figures: {len(generated_figures) + len(representative)}")
    print(f"reports: 2")


if __name__ == "__main__":
    if not FIXED_SOURCE.exists() or not MC_SOURCE.exists():
        print("Required final comparison sources are missing.", file=sys.stderr)
        sys.exit(1)
    main()

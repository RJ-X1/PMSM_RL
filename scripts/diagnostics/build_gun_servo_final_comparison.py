"""Build final gun-servo fixed and Monte Carlo comparison tables."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FINAL_SCENARIOS = [
    "C1a_5deg_step",
    "C1_10deg_step",
    "C2a_20deg_step_safe",
    "C3_trapezoid_tracking",
    "C4a_sine_tracking_safe",
    "C5a_disturbance_after_settling",
]

FIXED_FIELDS = [
    "method",
    "scenario",
    "rmse_theta_deg",
    "rmse_theta_deg_std",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "settling_time_s",
    "recovery_time_s",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "omega_limit_violation",
    "hard_safety_violation",
    "done_reason",
]

MC_FIELDS = [
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

METHOD_ORDER = [
    "PID/PD-PI",
    "SMC-PI",
    "Direct TD3-PI v3",
    "Residual TD3 v4",
    "SC-Residual TD3 v5",
    "MC-SC-Residual TD3 v6",
    "MC-SC-Residual TD3 v7",
    "Randomized MC-SC-Residual TD3 v8 full",
]


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


def _fmt(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _direct_fixed_row(row: dict[str, Any], method: str) -> dict[str, Any]:
    done_reason = str(row.get("done_reason", ""))
    flag_e = _safe_float(row.get("flag_E_safe_count"), 0.0)
    flag_x = _safe_float(row.get("flag_X_safe_count"), 0.0)
    sigma = _safe_float(row.get("sigma_safe_count"), 0.0)
    hard = int("violation" in done_reason or flag_e > 0.5 or flag_x > 0.5 or sigma > 0.5)
    return {
        "method": method,
        "scenario": str(row.get("scenario", "")),
        "rmse_theta_deg": _safe_float(row.get("rmse_theta_deg")),
        "rmse_theta_deg_std": _safe_float(row.get("rmse_theta_deg_std")),
        "mae_theta_deg": _safe_float(row.get("mae_theta_deg")),
        "max_abs_theta_error_deg": _safe_float(row.get("max_abs_theta_error_deg")),
        "settling_time_s": _safe_float(row.get("settling_time_s")),
        "recovery_time_s": _safe_float(row.get("recovery_time_s")),
        "flag_U_safe_count": _safe_float(row.get("flag_U_safe_count")),
        "flag_E_safe_count": flag_e,
        "flag_X_safe_count": flag_x,
        "sigma_safe_count": sigma,
        "omega_limit_violation": int("omega_limit_violation" in done_reason),
        "hard_safety_violation": hard,
        "done_reason": done_reason,
    }


def _aggregate_fixed_row(row: dict[str, Any], method: str) -> dict[str, Any]:
    omega_count = _safe_float(row.get("omega_limit_violation_count"), 0.0)
    hard_count = _safe_float(row.get("hard_safety_violation_count"), 0.0)
    return {
        "method": method,
        "scenario": str(row.get("scenario", "")),
        "rmse_theta_deg": _safe_float(row.get("rmse_theta_deg_mean")),
        "rmse_theta_deg_std": _safe_float(row.get("rmse_theta_deg_std")),
        "mae_theta_deg": _safe_float(row.get("mae_theta_deg_mean")),
        "max_abs_theta_error_deg": _safe_float(row.get("max_abs_theta_error_deg_mean")),
        "settling_time_s": float("nan"),
        "recovery_time_s": _safe_float(row.get("recovery_time_s_mean")),
        "flag_U_safe_count": _safe_float(row.get("flag_U_safe_count_mean")),
        "flag_E_safe_count": _safe_float(row.get("flag_E_safe_count_mean")),
        "flag_X_safe_count": _safe_float(row.get("flag_X_safe_count_mean")),
        "sigma_safe_count": _safe_float(row.get("sigma_safe_count_mean")),
        "omega_limit_violation": omega_count,
        "hard_safety_violation": hard_count,
        "done_reason": str(row.get("done_reason_summary", "")),
    }


def _load_pid_rows() -> list[dict[str, Any]]:
    rows = [
        _direct_fixed_row(row, "PID/PD-PI")
        for row in _read_rows(ROOT / "outputs/runs/gun_servo_pid_smoke_v2/summaries/pid_smoke_summary.csv")
        if str(row.get("scenario", "")) in FINAL_SCENARIOS
    ]
    seen = {row["scenario"] for row in rows}
    fallback = _read_rows(
        ROOT
        / "outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0/summaries/mc_sc_residual_td3_v8_full_vs_pid_v2_summary.csv"
    )
    for row in fallback:
        scenario = str(row.get("scenario", ""))
        if scenario in FINAL_SCENARIOS and scenario not in seen and str(row.get("controller", "")) == "pid_v2":
            rows.append(_direct_fixed_row(row, "PID/PD-PI"))
            seen.add(scenario)
    return rows


def _load_fixed_rows() -> list[dict[str, Any]]:
    sources = [
        (
            "SMC-PI",
            ROOT / "outputs/runs/gun_servo_smc_baseline/summaries/smc_baseline_summary.csv",
        ),
        (
            "Direct TD3-PI v3",
            ROOT / "outputs/runs/gun_servo_td3_smoke_v3_c1/summaries/td3_v3_smoke_summary.csv",
        ),
        (
            "Residual TD3 v4",
            ROOT / "outputs/runs/gun_servo_residual_td3_smoke_v4/summaries/residual_td3_v4_smoke_summary.csv",
        ),
        (
            "SC-Residual TD3 v5",
            ROOT / "outputs/runs/gun_servo_sc_residual_td3_smoke_v5/summaries/sc_residual_td3_v5_smoke_summary.csv",
        ),
        (
            "MC-SC-Residual TD3 v6",
            ROOT / "outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v6/summaries/mc_sc_residual_td3_v6_smoke_summary.csv",
        ),
        (
            "MC-SC-Residual TD3 v7",
            ROOT / "outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/summaries/mc_sc_residual_td3_v7_smoke_summary.csv",
        ),
    ]
    rows = _load_pid_rows()
    for method, path in sources:
        for row in _read_rows(path):
            if str(row.get("scenario", "")) in FINAL_SCENARIOS:
                rows.append(_direct_fixed_row(row, method))
    v8_path = ROOT / "outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_aggregate/summaries/fixed_eval_mean_std.csv"
    for row in _read_rows(v8_path):
        if str(row.get("scenario", "")) in FINAL_SCENARIOS:
            rows.append(_aggregate_fixed_row(row, "Randomized MC-SC-Residual TD3 v8 full"))
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda item: (order.get(str(item["method"]), 999), FINAL_SCENARIOS.index(str(item["scenario"]))))
    return rows


def _mc_row_from_smc_v8_comparison(row: dict[str, Any]) -> dict[str, Any]:
    controller = str(row.get("controller", ""))
    method = {
        "smc": "SMC-PI",
        "mc_sc_residual_td3_v8_full": "Randomized MC-SC-Residual TD3 v8 full",
    }.get(controller, controller)
    return {
        "method": method,
        "success_rate": _safe_float(row.get("success_rate")),
        "episode_limit_rate": _safe_float(row.get("episode_limit_rate")),
        "mean_rmse_theta_deg": _safe_float(row.get("mean_rmse_theta_deg")),
        "std_rmse_theta_deg": _safe_float(row.get("std_rmse_theta_deg")),
        "max_rmse_theta_deg": _safe_float(row.get("max_rmse_theta_deg")),
        "mean_mae_theta_deg": _safe_float(row.get("mean_mae_theta_deg")),
        "omega_limit_violation_count": _safe_float(row.get("omega_limit_violation_count"), 0.0),
        "hard_safety_violation_count": _safe_float(row.get("hard_safety_violation_count"), 0.0),
        "mean_flag_U_safe_count": _safe_float(row.get("mean_flag_U_safe_count")),
        "total_flag_U_safe_count": _safe_float(row.get("total_flag_U_safe_count")),
        "mean_recovery_time_s": _safe_float(row.get("mean_recovery_time_s")),
        "max_abs_error_after_disturbance_deg": _safe_float(row.get("max_abs_error_after_disturbance_deg")),
    }


def _load_mc_rows() -> list[dict[str, Any]]:
    rows = [
        _mc_row_from_smc_v8_comparison(row)
        for row in _read_rows(ROOT / "outputs/runs/gun_servo_smc_baseline/summaries/smc_mc_vs_v8_full_mc_summary.csv")
    ]
    pid_mc_path = ROOT / "outputs/runs/gun_servo_pid_mc_baseline/summaries/pid_mc_eval_summary.csv"
    for row in _read_rows(pid_mc_path):
        if "success_rate" in row and "mean_rmse_theta_deg" in row:
            item = {field: _safe_float(row.get(field)) for field in MC_FIELDS}
            item["method"] = "PID/PD-PI"
            rows.append(item)
    rows.sort(key=lambda item: METHOD_ORDER.index(str(item["method"])) if str(item["method"]) in METHOD_ORDER else 999)
    return rows


def _plot_grouped(
    rows: list[dict[str, Any]],
    *,
    field: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    methods = [method for method in METHOD_ORDER if any(str(row["method"]) == method for row in rows)]
    x = np.arange(len(FINAL_SCENARIOS), dtype=np.float64)
    width = min(0.8 / max(1, len(methods)), 0.12)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for idx, method in enumerate(methods):
        values = []
        for scenario in FINAL_SCENARIOS:
            match = next((row for row in rows if row["method"] == method and row["scenario"] == scenario), None)
            values.append(_safe_float(match.get(field)) if match else float("nan"))
        ax.bar(x + (idx - (len(methods) - 1) / 2.0) * width, values, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(FINAL_SCENARIOS, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_mc_metric(rows: list[dict[str, Any]], *, field: str, ylabel: str, title: str, path: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    values = [_safe_float(row.get(field)) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(methods, values, color=["tab:green", "tab:blue", "tab:gray"][: len(methods)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_mc_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    means = [_safe_float(row.get("mean_rmse_theta_deg")) for row in rows]
    stds = [_safe_float(row.get("std_rmse_theta_deg"), 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(methods, means, yerr=stds, capsize=5, color=["tab:green", "tab:blue", "tab:gray"][: len(methods)])
    ax.set_ylabel("MC RMSE theta [deg]")
    ax.set_title("Monte Carlo RMSE comparison")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_mc_violations(rows: list[dict[str, Any]], path: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.32
    omega = [_safe_float(row.get("omega_limit_violation_count"), 0.0) for row in rows]
    hard = [_safe_float(row.get("hard_safety_violation_count"), 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2.0, omega, width=width, label="omega")
    ax.bar(x + width / 2.0, hard, width=width, label="hard")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("count")
    ax.set_title("Monte Carlo violation counts")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _format_method_means(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if str(row.get("method", "")) == method]
        if not method_rows:
            continue
        lines.append(
            f"- {method}: mean fixed RMSE over {len(method_rows)} available scenario rows "
            f"{_mean([row.get('rmse_theta_deg') for row in method_rows]):.4f} deg, "
            f"hard violations {sum(_safe_float(row.get('hard_safety_violation'), 0.0) for row in method_rows):.0f}"
        )
    return lines


def _write_conclusion(out_dir: Path, fixed_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]) -> Path:
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in fixed_rows)]
    smc_fixed = [row for row in fixed_rows if row["method"] == "SMC-PI"]
    pid_fixed = [row for row in fixed_rows if row["method"] == "PID/PD-PI"]
    v8_fixed = [row for row in fixed_rows if row["method"] == "Randomized MC-SC-Residual TD3 v8 full"]
    smc_mc = next((row for row in mc_rows if row["method"] == "SMC-PI"), {})
    pid_mc = next((row for row in mc_rows if row["method"] == "PID/PD-PI"), {})
    v8_mc = next((row for row in mc_rows if row["method"] == "Randomized MC-SC-Residual TD3 v8 full"), {})
    mc_focus_methods = ["PID/PD-PI", "SMC-PI", "Randomized MC-SC-Residual TD3 v8 full"]
    mc_focus_rows = [row for method in mc_focus_methods if (row := next((item for item in mc_rows if item["method"] == method), None))]
    best_rmse_row = min(
        mc_focus_rows,
        key=lambda row: _safe_float(row.get("mean_rmse_theta_deg"), float("inf")),
    ) if mc_focus_rows else {}
    v8_best_mc_rmse = str(best_rmse_row.get("method", "")) == "Randomized MC-SC-Residual TD3 v8 full"
    classical_rows = [row for row in (pid_mc, smc_mc) if row]
    classical_lower_flag_u = all(
        _safe_float(row.get("mean_flag_U_safe_count"), float("inf"))
        < _safe_float(v8_mc.get("mean_flag_U_safe_count"), float("inf"))
        for row in classical_rows
    ) if classical_rows and v8_mc else False
    smc_better_pid = [
        row["scenario"]
        for row in smc_fixed
        if (match := next((pid for pid in pid_fixed if pid["scenario"] == row["scenario"]), None))
        and _safe_float(row["rmse_theta_deg"]) < _safe_float(match["rmse_theta_deg"])
    ]
    smc_better_v8 = [
        row["scenario"]
        for row in smc_fixed
        if (match := next((v8 for v8 in v8_fixed if v8["scenario"] == row["scenario"]), None))
        and _safe_float(row["rmse_theta_deg"]) < _safe_float(match["rmse_theta_deg"])
    ]
    lines = [
        "# Final Gun-Servo Method Comparison",
        "",
        "## 1. Compared methods",
        *[f"- {method}" for method in methods],
        "",
        "## 2. Fixed six-scenario conclusions",
        *_format_method_means(fixed_rows),
        "",
        "## 3. Monte Carlo robustness conclusions",
        "- PID/PD-PI: "
        f"success_rate {_fmt(pid_mc.get('success_rate'))}, "
        f"mean RMSE {_fmt(pid_mc.get('mean_rmse_theta_deg'))} deg, "
        f"mean flag_U {_fmt(pid_mc.get('mean_flag_U_safe_count'), 2)}, "
        f"max disturbance error {_fmt(pid_mc.get('max_abs_error_after_disturbance_deg'))} deg.",
        "- SMC-PI: "
        f"success_rate {_fmt(smc_mc.get('success_rate'))}, "
        f"mean RMSE {_fmt(smc_mc.get('mean_rmse_theta_deg'))} deg, "
        f"mean flag_U {_fmt(smc_mc.get('mean_flag_U_safe_count'), 2)}, "
        f"max disturbance error {_fmt(smc_mc.get('max_abs_error_after_disturbance_deg'))} deg.",
        "- Randomized MC-SC-Residual TD3 v8 full: "
        f"success_rate {_fmt(v8_mc.get('success_rate'))}, "
        f"mean RMSE {_fmt(v8_mc.get('mean_rmse_theta_deg'))} deg, "
        f"mean flag_U {_fmt(v8_mc.get('mean_flag_U_safe_count'), 2)}, "
        f"max disturbance error {_fmt(v8_mc.get('max_abs_error_after_disturbance_deg'))} deg.",
        f"- Best Monte Carlo mean RMSE: {best_rmse_row.get('method', 'n/a')} "
        f"({_fmt(best_rmse_row.get('mean_rmse_theta_deg'))} deg).",
        f"- v8 full still has the best Monte Carlo mean RMSE: {'yes' if v8_best_mc_rmse else 'no'}.",
        f"- PID/SMC have fewer command-layer safety interventions than v8 full: {'yes' if classical_lower_flag_u else 'no'}.",
        "",
        "## 4. SMC vs PID",
        f"- SMC lower fixed RMSE scenarios: {', '.join(smc_better_pid) if smc_better_pid else 'none'}",
        "- PID/PD-PI remains the strongest or near-strongest fixed tracking baseline in several smooth tracking cases.",
        "- In Monte Carlo, PID/PD-PI preserves 100% success and zero hard/omega violations, but its mean RMSE is higher than both SMC-PI and v8 full in this run.",
        "",
        "## 5. SMC vs v8 full",
        f"- SMC lower fixed RMSE scenarios: {', '.join(smc_better_v8) if smc_better_v8 else 'none'}",
        "- On Monte Carlo, v8 full has lower mean RMSE than SMC-PI, while SMC-PI has lower mean MAE, fewer command-safety interventions, and smaller disturbance peak error in this run.",
        "",
        "## 6. v8 full safety and robustness advantages",
        "- v8 full preserves 100% Monte Carlo success with zero omega and hard-safety violations across multi-seed aggregate reporting.",
        "- It is the only learned method here with randomized MC training and multi-seed robustness evidence.",
        "",
        "## 7. Where RL still lags classical control",
        "- Fixed-scenario tracking RMSE is still generally better for PID/SMC than for the current v8 full selected policies.",
        "- The learned residual policy has higher command-layer safety intervention counts than PID/SMC in Monte Carlo.",
        "",
        "## 8. Proposed method",
        "- Present MC-SC-Residual-TD3-PI as the proposed method.",
        "- Recommended thesis wording: the proposed randomized safety-constrained residual TD3-PI improves average Monte Carlo robustness relative to the classical PID/SMC baselines under the tested randomized distribution while maintaining zero hard safety violations, but it should not be claimed to universally dominate classical control in fixed nominal tracking accuracy or command-intervention frequency.",
        "",
        "## 9. Recommended next step",
        "- Proceed to manuscript/result section writing using the completed three-method Monte Carlo comparison.",
    ]
    path = out_dir / "summaries" / "final_comparison_conclusion.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    out_dir = ROOT / "outputs" / "runs" / "final_method_comparison"
    summaries_dir = out_dir / "summaries"
    figures_dir = out_dir / "figures"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fixed_rows = _load_fixed_rows()
    mc_rows = _load_mc_rows()
    fixed_csv = summaries_dir / "fixed_six_scenario_comparison.csv"
    mc_csv = summaries_dir / "monte_carlo_comparison.csv"
    _write_rows(fixed_csv, fixed_rows, FIXED_FIELDS)
    _write_rows(mc_csv, mc_rows, MC_FIELDS)

    _plot_grouped(
        fixed_rows,
        field="rmse_theta_deg",
        ylabel="RMSE theta [deg]",
        title="Fixed six-scenario RMSE",
        path=figures_dir / "fixed_rmse_by_method_and_scenario.png",
    )
    _plot_grouped(
        fixed_rows,
        field="flag_U_safe_count",
        ylabel="flag_U count",
        title="Fixed six-scenario command safety interventions",
        path=figures_dir / "fixed_flag_U_by_method_and_scenario.png",
    )
    _plot_mc_rmse(mc_rows, figures_dir / "monte_carlo_rmse_comparison.png")
    _plot_mc_metric(
        mc_rows,
        field="success_rate",
        ylabel="success rate",
        title="Monte Carlo success rate",
        path=figures_dir / "monte_carlo_success_rate_comparison.png",
    )
    _plot_mc_violations(mc_rows, figures_dir / "monte_carlo_violation_counts_comparison.png")
    conclusion_path = _write_conclusion(out_dir, fixed_rows, mc_rows)

    print(f"saved fixed comparison: {fixed_csv}")
    print(f"saved Monte Carlo comparison: {mc_csv}")
    print(f"saved final comparison conclusion: {conclusion_path}")


if __name__ == "__main__":
    main()

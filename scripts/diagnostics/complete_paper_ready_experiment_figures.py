"""Generate missing paper-ready figures for the gun-servo PMSM study.

This script only reads existing logs, summaries, trajectories, and configs. It
does not rerun training/evaluation or modify raw experiment result files.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import shutil
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - local environments normally have PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"
PAPER = ROOT / "outputs" / "paper_ready_results"
FIGURES = PAPER / "figures"
TRAINING = FIGURES / "training_diagnostics"
REPRESENTATIVE = FIGURES / "representative"
SINGLE_METHOD_COPIES = REPRESENTATIVE / "copied_single_method"
ACTION = FIGURES / "action_smoothness"
RANDOMIZATION = FIGURES / "randomization"
DISTURBANCE = FIGURES / "disturbance"
REPORTS = PAPER / "reports"

FIXED_COMPARISON = RUNS / "final_method_comparison" / "summaries" / "fixed_six_scenario_comparison.csv"
MC_COMPARISON = RUNS / "final_method_comparison" / "summaries" / "monte_carlo_comparison.csv"
ENV_RANDOMIZATION_CONFIG = ROOT / "configs" / "env" / "gun_servo_position_mc_sc_residual_td3_v8_rand.yaml"
EVAL_MC_CONFIG = ROOT / "configs" / "eval" / "gun_servo_mc_sc_residual_td3_random_v8_eval_mc.yaml"

SEED_RUNS = {
    "seed0": RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed0",
    "seed1": RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed1",
    "seed2": RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed2",
}

METHOD_ORDER = [
    "PID/PD-PI",
    "SMC-PI",
    "Residual TD3-PI",
    "SC-Residual TD3-PI",
    "MC-SC-Residual TD3-PI",
    "Randomized MC-SC-Residual TD3-PI",
]

FIXED_METHOD_MAP = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Residual TD3 v4": "Residual TD3-PI",
    "SC-Residual TD3 v5": "SC-Residual TD3-PI",
    "MC-SC-Residual TD3 v7": "MC-SC-Residual TD3-PI",
    "Randomized MC-SC-Residual TD3 v8 full": "Randomized MC-SC-Residual TD3-PI",
}

MC_METHOD_MAP = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Randomized MC-SC-Residual TD3 v8 full": "Randomized MC-SC-Residual TD3-PI",
}

PLOT_LABELS = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Residual TD3-PI": "Residual TD3-PI",
    "SC-Residual TD3-PI": "SC-Residual TD3-PI",
    "MC-SC-Residual TD3-PI": "MC-SC-Residual TD3-PI",
    "Randomized MC-SC-Residual TD3-PI": "Randomized MC-SC-Residual TD3-PI",
}

SHORT_PLOT_LABELS = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Residual TD3-PI": "Residual\nTD3-PI",
    "SC-Residual TD3-PI": "SC-Residual\nTD3-PI",
    "MC-SC-Residual TD3-PI": "MC-SC\nResidual TD3-PI",
    "Randomized MC-SC-Residual TD3-PI": "Randomized MC-SC\nResidual TD3-PI",
}

SCENARIOS = {
    "C1a_5deg_step": "C1a",
    "C1_10deg_step": "C1",
    "C2a_20deg_step_safe": "C2a",
    "C3_trapezoid_tracking": "C3",
    "C4a_sine_tracking_safe": "C4a",
    "C5a_disturbance_after_settling": "C5a",
}

REPRESENTATIVE_SCENARIOS = {
    "C1_10deg_step": "C1",
    "C3_trapezoid_tracking": "C3",
    "C4a_sine_tracking_safe": "C4a",
    "C5a_disturbance_after_settling": "C5a",
}

MC_SCENARIOS = [
    "random_step_safe",
    "random_trapezoid_safe",
    "random_sine_safe",
    "random_disturbance_after_settling",
]

METHOD_COLORS = {
    "PID/PD-PI": "#4C78A8",
    "SMC-PI": "#54A24B",
    "Residual TD3-PI": "#B279A2",
    "SC-Residual TD3-PI": "#72B7B2",
    "MC-SC-Residual TD3-PI": "#E45756",
    "Randomized MC-SC-Residual TD3-PI": "#F58518",
}


@dataclass
class GenerationState:
    generated: list[Path]
    skipped: list[str]
    warnings: list[str]
    sources: set[Path]

    def add_generated(self, path: Path) -> None:
        self.generated.append(path)

    def add_source(self, path: Path) -> None:
        self.sources.add(path)

    def skip(self, message: str) -> None:
        self.skipped.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def ensure_dirs() -> None:
    for path in (FIGURES, TRAINING, REPRESENTATIVE, ACTION, RANDOMIZATION, DISTURBANCE, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, float):
        return value if math.isfinite(value) else default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def get_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(column)) for row in rows], dtype=float)


def sample_rows(rows: list[dict[str, str]], max_points: int = 2500) -> list[dict[str, str]]:
    if len(rows) <= max_points:
        return rows
    step = max(1, len(rows) // max_points)
    return rows[::step]


def save_figure(fig: plt.Figure, path: Path, state: GenerationState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    state.add_generated(path)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ordered_methods(methods: Iterable[str]) -> list[str]:
    present = set(methods)
    return [method for method in METHOD_ORDER if method in present] + sorted(present - set(METHOD_ORDER))


def load_train_logs(state: GenerationState) -> dict[str, list[dict[str, str]]]:
    logs: dict[str, list[dict[str, str]]] = {}
    for seed, run_dir in SEED_RUNS.items():
        path = run_dir / "train_log.csv"
        rows = read_rows(path)
        if rows:
            logs[seed] = rows
            state.add_source(path)
        else:
            state.warn(f"missing training log: {relative(path)}")
    return logs


def plot_training_returns(logs: dict[str, list[dict[str, str]]], state: GenerationState) -> bool:
    if not logs:
        state.skip("training return curve was not generated because no seed train_log.csv files were available.")
        return False
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for seed, rows in logs.items():
        x = get_column(rows, "global_steps")
        y = get_column(rows, "episode_return")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.plot(x[mask], y[mask], linewidth=1.4, label=seed)
    if not ax.lines:
        plt.close(fig)
        state.skip("training return curve was not generated because global_steps/episode_return were missing.")
        return False
    ax.set_xlabel("Global steps")
    ax.set_ylabel("Episode return")
    ax.set_title("Training return curves of Randomized MC-SC-Residual-TD3-PI")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, TRAINING / "fig_training_return_all_seeds.png", state)
    return True


def plot_eval_rmse(logs: dict[str, list[dict[str, str]]], state: GenerationState) -> bool:
    candidates = ["eval_rmse_theta_deg", "eval_rmse_all", "eval_rmse_theta"]
    available = []
    for rows in logs.values():
        if rows:
            available.extend([col for col in candidates if col in rows[0]])
    column = next((col for col in candidates if col in available), "")
    if not column:
        state.skip(
            "eval RMSE curve cannot be generated because checkpoint-level fixed evaluation records are not available."
        )
        return False
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for seed, rows in logs.items():
        x = get_column(rows, "global_steps")
        y = get_column(rows, column)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.plot(x[mask], y[mask], linewidth=1.4, label=seed)
    if not ax.lines:
        plt.close(fig)
        state.skip(
            "eval RMSE curve cannot be generated because checkpoint-level fixed evaluation records are not available."
        )
        return False
    ax.set_xlabel("Global steps")
    ylabel = "Fixed evaluation average RMSE [deg]" if column.endswith("_deg") else "Fixed evaluation average RMSE"
    ax.set_ylabel(ylabel)
    ax.set_title("Fixed-evaluation RMSE during training")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, TRAINING / "fig_eval_rmse_all_seeds.png", state)
    return True


def plot_loss_curve(
    logs: dict[str, list[dict[str, str]]],
    column: str,
    ylabel: str,
    title: str,
    filename: str,
    state: GenerationState,
) -> bool:
    if not any(rows and column in rows[0] for rows in logs.values()):
        return False
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for seed, rows in logs.items():
        if not rows or column not in rows[0]:
            continue
        x = get_column(rows, "global_steps")
        y = get_column(rows, column)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.plot(x[mask], y[mask], linewidth=1.4, label=seed)
    if not ax.lines:
        plt.close(fig)
        return False
    ax.set_xlabel("Global steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, TRAINING / filename, state)
    return True


def trajectory_sources(scenario: str) -> dict[str, Path]:
    return {
        "PID/PD-PI": RUNS / "gun_servo_pid_smoke_v2" / "eval" / f"{scenario}_pid.csv",
        "SMC-PI": RUNS / "gun_servo_smc_baseline" / "eval" / f"{scenario}_smc.csv",
        "Randomized MC-SC-Residual TD3-PI": SEED_RUNS["seed0"] / "eval" / f"{scenario}_td3.csv",
    }


def read_trace(path: Path, state: GenerationState, max_points: int = 2500) -> dict[str, np.ndarray] | None:
    rows = read_rows(path)
    if not rows:
        return None
    state.add_source(path)
    rows = sample_rows(rows, max_points=max_points)
    keys = [
        "t_s",
        "time_s",
        "theta_ref_deg",
        "theta_L_deg",
        "e_theta_deg",
        "a_raw",
        "action_raw",
        "a_safe",
        "action_safe",
        "omega_L_cmd_safe",
        "omega_L_cmd_safe_deg_s",
        "omega_cmd_deg_s",
        "T_L",
        "disturbance_torque_Nm",
        "disturbance_step_time_s",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key in rows[0]:
            out[key] = get_column(rows, key)
    if "t_s" not in out and "time_s" in out:
        out["t_s"] = out["time_s"]
    return out


def pick_series(trace: dict[str, np.ndarray], candidates: list[str]) -> np.ndarray | None:
    for column in candidates:
        values = trace.get(column)
        if values is not None and np.isfinite(values).any():
            return values
    return None


def omega_command_deg_s(trace: dict[str, np.ndarray]) -> np.ndarray | None:
    values = pick_series(trace, ["omega_L_cmd_safe_deg_s", "omega_cmd_deg_s"])
    if values is not None:
        return values
    raw = trace.get("omega_L_cmd_safe")
    if raw is not None and np.isfinite(raw).any():
        return np.rad2deg(raw)
    return None


def plot_position_overlay(scenario: str, filename: str, state: GenerationState) -> bool:
    traces = {
        method: read_trace(path, state)
        for method, path in trajectory_sources(scenario).items()
    }
    traces = {method: trace for method, trace in traces.items() if trace is not None}
    if not traces:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ref_trace = next((trace for trace in traces.values() if "theta_ref_deg" in trace), None)
    if ref_trace is not None:
        ax.plot(ref_trace["t_s"], ref_trace["theta_ref_deg"], color="black", linestyle="--", linewidth=1.2, label="Reference")
    for method in ordered_methods(traces.keys()):
        trace = traces[method]
        if "theta_L_deg" not in trace:
            continue
        ax.plot(
            trace["t_s"],
            trace["theta_L_deg"],
            linewidth=1.2,
            color=METHOD_COLORS.get(method),
            label=PLOT_LABELS.get(method, method),
        )
    if len(ax.lines) <= 1:
        plt.close(fig)
        return False
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [deg]")
    ax.set_title(f"{SCENARIOS[scenario]} position tracking comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, REPRESENTATIVE / filename, state)
    return True


def plot_error_overlay(scenario: str, filename: str, state: GenerationState) -> bool:
    traces = {
        method: read_trace(path, state)
        for method, path in trajectory_sources(scenario).items()
    }
    traces = {method: trace for method, trace in traces.items() if trace is not None}
    if not traces:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in ordered_methods(traces.keys()):
        trace = traces[method]
        error = trace.get("e_theta_deg")
        if error is None:
            continue
        ax.plot(
            trace["t_s"],
            error,
            linewidth=1.2,
            color=METHOD_COLORS.get(method),
            label=PLOT_LABELS.get(method, method),
        )
    if not ax.lines:
        plt.close(fig)
        return False
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [deg]")
    ax.set_title(f"{SCENARIOS[scenario]} position error comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, REPRESENTATIVE / filename, state)
    return True


def detect_disturbance_time(traces: dict[str, dict[str, np.ndarray]]) -> float:
    for trace in traces.values():
        step_time = trace.get("disturbance_step_time_s")
        if step_time is not None:
            finite_step = finite(step_time)
            if finite_step.size:
                return float(finite_step[0])
        for column in ("disturbance_torque_Nm", "T_L"):
            y = trace.get(column)
            t = trace.get("t_s")
            if y is None or t is None or len(y) < 3:
                continue
            mask = np.isfinite(y) & np.isfinite(t)
            if mask.sum() < 3:
                continue
            yy = y[mask]
            tt = t[mask]
            diff = np.abs(np.diff(yy))
            threshold = max(1e-8, 0.1 * np.nanmax(diff)) if diff.size else 0.0
            idx = np.flatnonzero(diff > threshold)
            if idx.size:
                return float(tt[idx[0] + 1])
    return 1.0


def plot_c5a_disturbance_overlay(state: GenerationState) -> bool:
    scenario = "C5a_disturbance_after_settling"
    traces = {
        method: read_trace(path, state)
        for method, path in trajectory_sources(scenario).items()
    }
    traces = {method: trace for method, trace in traces.items() if trace is not None}
    if not traces:
        return False
    t0 = detect_disturbance_time(traces)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in ordered_methods(traces.keys()):
        trace = traces[method]
        error = trace.get("e_theta_deg")
        t = trace.get("t_s")
        if error is None or t is None:
            continue
        mask = np.isfinite(t) & np.isfinite(error) & (t >= t0 - 0.5) & (t <= t0 + 1.5)
        if mask.any():
            ax.plot(
                t[mask],
                error[mask],
                linewidth=1.2,
                color=METHOD_COLORS.get(method),
                label=PLOT_LABELS.get(method, method),
            )
    if not ax.lines:
        plt.close(fig)
        return False
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(t0, color="black", linestyle="--", linewidth=1.0, label="Disturbance step")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [deg]")
    ax.set_title("C5a disturbance response comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, REPRESENTATIVE / "fig_C5a_disturbance_comparison.png", state)
    return True


def copy_single_method_figures(state: GenerationState) -> None:
    SINGLE_METHOD_COPIES.mkdir(parents=True, exist_ok=True)
    candidates = [
        SEED_RUNS["seed0"] / "figures" / "C1_10deg_step_td3_position.png",
        SEED_RUNS["seed0"] / "figures" / "C3_trapezoid_tracking_td3_position.png",
        SEED_RUNS["seed0"] / "figures" / "C4a_sine_tracking_safe_td3_position.png",
        SEED_RUNS["seed0"] / "figures" / "C5a_disturbance_after_settling_td3_disturbance_zoom.png",
    ]
    copied = 0
    for src in candidates:
        if src.exists():
            dst = SINGLE_METHOD_COPIES / src.name
            shutil.copy2(src, dst)
            state.add_source(src)
            state.add_generated(dst)
            copied += 1
    if copied:
        state.warn("representative overlay comparison could not be generated for all requested figures, so single-method figures were copied.")


def generate_representative_figures(state: GenerationState) -> bool:
    results = [
        plot_position_overlay("C1_10deg_step", "fig_C1_position_comparison.png", state),
        plot_error_overlay("C1_10deg_step", "fig_C1_error_comparison.png", state),
        plot_position_overlay("C3_trapezoid_tracking", "fig_C3_position_comparison.png", state),
        plot_position_overlay("C4a_sine_tracking_safe", "fig_C4a_position_comparison.png", state),
        plot_c5a_disturbance_overlay(state),
    ]
    if not any(results):
        copy_single_method_figures(state)
        state.skip("representative comparison overlays were not generated because overlay trajectory CSVs were not available.")
        return False
    if not all(results):
        state.warn("some representative comparison overlays were skipped because one or more required trajectory columns were missing.")
    return all(results)


def action_traces(state: GenerationState) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scenario in ("C1_10deg_step", "C4a_sine_tracking_safe"):
        out[scenario] = {}
        for method, path in trajectory_sources(scenario).items():
            trace = read_trace(path, state)
            if trace is not None:
                out[scenario][method] = trace
    return out


def plot_action_raw_safe(traces: dict[str, dict[str, dict[str, np.ndarray]]], state: GenerationState) -> bool:
    if not traces:
        state.skip("action raw/safe comparison was not generated because trajectory CSVs were unavailable.")
        return False
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=False)
    plotted = False
    for ax, scenario in zip(axes, ("C1_10deg_step", "C4a_sine_tracking_safe")):
        scenario_traces = traces.get(scenario, {})
        for method in ordered_methods(scenario_traces.keys()):
            trace = scenario_traces[method]
            t = trace.get("t_s")
            raw = pick_series(trace, ["a_raw", "action_raw"])
            safe = pick_series(trace, ["a_safe", "action_safe"])
            color = METHOD_COLORS.get(method)
            if t is None:
                continue
            if safe is not None:
                ax.plot(t, safe, linewidth=1.0, color=color, label=f"{PLOT_LABELS[method]} safe")
                plotted = True
            if raw is not None:
                ax.plot(t, raw, linewidth=0.9, color=color, linestyle="--", alpha=0.65, label=f"{PLOT_LABELS[method]} raw")
                plotted = True
        ax.set_ylabel("Action")
        ax.set_title(f"{SCENARIOS[scenario]} raw and safe action")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 6:
        handles = handles[:6]
        labels = labels[:6]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    if not plotted:
        plt.close(fig)
        state.skip("action raw/safe comparison was not generated because a_raw/a_safe columns were unavailable.")
        return False
    save_figure(fig, ACTION / "fig_action_raw_safe_comparison.png", state)
    return True


def plot_speed_command(traces: dict[str, dict[str, dict[str, np.ndarray]]], state: GenerationState) -> bool:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=False)
    plotted = False
    for ax, scenario in zip(axes, ("C1_10deg_step", "C4a_sine_tracking_safe")):
        for method in ordered_methods(traces.get(scenario, {}).keys()):
            trace = traces[scenario][method]
            t = trace.get("t_s")
            cmd = omega_command_deg_s(trace)
            if t is None or cmd is None:
                continue
            ax.plot(t, cmd, linewidth=1.1, color=METHOD_COLORS.get(method), label=PLOT_LABELS[method])
            plotted = True
        ax.set_ylabel("Commanded speed [deg/s]")
        ax.set_title(f"{SCENARIOS[scenario]} safe speed command")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    if not plotted:
        plt.close(fig)
        state.skip("speed command comparison was not generated because omega_L_cmd_safe columns were unavailable.")
        return False
    save_figure(fig, ACTION / "fig_speed_command_comparison.png", state)
    return True


def variation(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(vals))))


def plot_action_variation_bar(traces: dict[str, dict[str, dict[str, np.ndarray]]], state: GenerationState) -> bool:
    methods = ordered_methods(
        method
        for scenario_traces in traces.values()
        for method in scenario_traces.keys()
    )
    scenarios = ["C1_10deg_step", "C4a_sine_tracking_safe"]
    if not methods:
        state.skip("action variation bar was not generated because action trajectory data was unavailable.")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    plotted = False
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    x = np.arange(len(methods), dtype=float)
    width = 0.35
    for ax, scenario in zip(axes, scenarios):
        action_vals = []
        speed_vals = []
        for method in methods:
            trace = traces.get(scenario, {}).get(method, {})
            action_vals.append(variation(pick_series(trace, ["a_safe", "action_safe"])))
            speed_vals.append(variation(omega_command_deg_s(trace)))
        action_bars = ax.bar(x - width / 2, action_vals, width=width, label="Action variation", color="#4C78A8")
        ax2 = ax.twinx()
        speed_bars = ax2.bar(
            x + width / 2,
            speed_vals,
            width=width,
            label="Speed-command variation",
            color="#F58518",
            alpha=0.85,
        )
        if not legend_handles:
            legend_handles = [action_bars[0], speed_bars[0]]
            legend_labels = ["Action variation", "Speed-command variation"]
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_PLOT_LABELS.get(method, method) for method in methods], rotation=22, ha="right")
        ax.set_ylabel("Sum |diff(a_safe)|")
        ax2.set_ylabel("Sum |diff(speed cmd)| [deg/s]")
        ax.set_title(f"{SCENARIOS[scenario]} variation")
        ax.grid(True, axis="y", alpha=0.25)
        plotted = plotted or np.isfinite(action_vals).any() or np.isfinite(speed_vals).any()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncols=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    if not plotted:
        plt.close(fig)
        state.skip("action variation bar was not generated because variation metrics were unavailable.")
        return False
    save_figure(fig, ACTION / "fig_action_variation_bar.png", state)
    return True


def normalize_fixed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        method = FIXED_METHOD_MAP.get(str(row.get("method", "")))
        if method is None:
            continue
        item = dict(row)
        item["method"] = method
        normalized.append(item)
    return normalized


def normalize_mc_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        method = MC_METHOD_MAP.get(str(row.get("method", "")))
        if method is None:
            continue
        item = dict(row)
        item["method"] = method
        normalized.append(item)
    return normalized


def plot_fixed_safety_flags(state: GenerationState) -> bool:
    rows = normalize_fixed_rows(read_rows(FIXED_COMPARISON))
    if not rows:
        state.skip("fixed safety flags bar was not generated because fixed_six_scenario_comparison.csv was missing.")
        return False
    state.add_source(FIXED_COMPARISON)
    methods = ordered_methods(row["method"] for row in rows)
    metrics = [
        ("flag_U_safe_count", "flag_U"),
        ("flag_E_safe_count", "flag_E"),
        ("flag_X_safe_count", "flag_X"),
        ("sigma_safe_count", "sigma"),
        ("omega_limit_violation", "omega limit"),
        ("hard_safety_violation", "hard safety"),
    ]
    x = np.arange(len(methods), dtype=float)
    width = 0.12
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(metrics)))
    for idx, (column, label) in enumerate(metrics):
        values = [
            sum(safe_float(row.get(column), 0.0) for row in rows if row["method"] == method)
            for method in methods
        ]
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, values, width=width, label=label, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_PLOT_LABELS.get(method, method) for method in methods], rotation=22, ha="right")
    ax.set_ylabel("Total count")
    ax.set_title("Fixed-scenario safety flags and violations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_fixed_safety_flags_bar.png", state)
    return True


def plot_mc_safety(state: GenerationState) -> tuple[bool, bool]:
    rows = normalize_mc_rows(read_rows(MC_COMPARISON))
    if not rows:
        state.skip("Monte Carlo safety bars were not generated because monte_carlo_comparison.csv was missing.")
        return False, False
    state.add_source(MC_COMPARISON)
    methods = ordered_methods(row["method"] for row in rows)
    lookup = {row["method"]: row for row in rows}
    x = np.arange(len(methods), dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    width = 0.30
    omega = [safe_float(lookup[method].get("omega_limit_violation_count"), 0.0) for method in methods]
    hard = [safe_float(lookup[method].get("hard_safety_violation_count"), 0.0) for method in methods]
    ax.bar(x - width / 2, omega, width=width, label="omega limit", color="#4C78A8")
    ax.bar(x + width / 2, hard, width=width, label="hard safety", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_PLOT_LABELS.get(method, method) for method in methods], rotation=15, ha="right")
    ax.set_ylabel("Violation count")
    ax.set_title("Monte Carlo safety violations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_mc_safety_violation_bar.png", state)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    flag = [
        safe_float(lookup[method].get("mean_flag_U_safe_count"), safe_float(lookup[method].get("total_flag_U_safe_count"), 0.0))
        for method in methods
    ]
    ax.bar(
        [SHORT_PLOT_LABELS.get(method, method) for method in methods],
        flag,
        color=[METHOD_COLORS.get(method, "#4C78A8") for method in methods],
    )
    ax.set_ylabel("Mean flag_U count")
    ax.set_title("Monte Carlo command safety intervention")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_mc_flag_U_bar.png", state)

    for missing in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count"):
        if all(missing not in row for row in rows):
            state.warn(f"Monte Carlo final comparison does not contain {missing}; MC safety plots use flag_U and violation counts only.")
    return True, True


def load_yaml_config(path: Path, state: GenerationState) -> dict[str, Any]:
    if not path.exists():
        state.warn(f"missing config: {relative(path)}")
        return {}
    state.add_source(path)
    if yaml is None:
        state.warn("PyYAML is unavailable; randomization config could not be parsed.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def randomization_ranges(config: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    dr = config.get("domain_randomization", {})
    if not isinstance(dr, dict):
        return []
    mapping = [
        ("J_L", "kg m^2", "inertia_range"),
        ("B_L", "N m s/rad", "damping_range"),
        ("T_c", "N m", "coulomb_friction_range"),
        ("K_g", "N m", "gravity_torque_range"),
        ("eta", "", "gear_efficiency_range"),
        ("Disturbance torque", "N m", "disturbance_torque_range"),
        ("Encoder noise", "rad", "encoder_noise_std_range"),
        ("V_dc scale", "", "vdc_scale_range"),
    ]
    ranges = []
    for label, unit, key in mapping:
        value = dr.get(key)
        if value is None and key == "encoder_noise_std_range":
            value = dr.get("encoder_noise_rad_range")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lo = safe_float(value[0])
            hi = safe_float(value[1])
            if math.isfinite(lo) and math.isfinite(hi):
                ranges.append((label, unit, lo, hi))
    return ranges


def plot_randomization_ranges(state: GenerationState) -> bool:
    config = load_yaml_config(ENV_RANDOMIZATION_CONFIG, state)
    eval_config = load_yaml_config(EVAL_MC_CONFIG, state)
    if eval_config:
        state.add_source(EVAL_MC_CONFIG)
    ranges = randomization_ranges(config)
    if not ranges:
        state.skip("parameter randomization ranges were not generated because configured ranges were unavailable.")
        return False
    mechanical = [item for item in ranges if item[0] in {"J_L", "B_L", "T_c", "K_g", "Disturbance torque"}]
    small = [item for item in ranges if item[0] not in {"J_L", "B_L", "T_c", "K_g", "Disturbance torque"}]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, group, title in (
        (axes[0], mechanical, "Mechanical/load ranges"),
        (axes[1], small, "Dimensionless and sensor ranges"),
    ):
        y = np.arange(len(group), dtype=float)
        lo_all = [lo for _, _, lo, _ in group]
        hi_all = [hi for _, _, _, hi in group]
        if lo_all and hi_all:
            span = max(hi_all) - min(lo_all)
            pad = 0.12 * span if span > 0 else 1.0
            ax.set_xlim(min(lo_all) - pad, max(hi_all) + 1.6 * pad)
        for idx, (label, unit, lo, hi) in enumerate(group):
            ax.hlines(idx, lo, hi, color="#4C78A8", linewidth=4)
            ax.plot([lo, hi], [idx, idx], "o", color="#4C78A8", markersize=4)
            unit_text = f" {unit}" if unit else ""
            ax.text(hi, idx + 0.10, f"[{lo:g}, {hi:g}]{unit_text}", fontsize=8, va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, _, _, _ in group])
        ax.set_ylim(-0.35, max(0.65, len(group) - 0.15))
        ax.set_title(title)
        ax.set_xlabel("Configured range")
        ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Parameter randomization ranges")
    fig.tight_layout()
    save_figure(fig, RANDOMIZATION / "fig_parameter_randomization_ranges.png", state)
    return True


def load_v8_mc_episode_rows(state: GenerationState) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed, run_dir in SEED_RUNS.items():
        path = run_dir / "summaries" / "mc_eval_episode_metrics.csv"
        seed_rows = read_rows(path)
        if seed_rows:
            for row in seed_rows:
                item = dict(row)
                item["source_seed"] = seed
                rows.append(item)
            state.add_source(path)
    return rows


def plot_randomization_distribution(v8_rows: list[dict[str, str]], state: GenerationState) -> bool:
    columns = [
        ("active_J_load", "J_L"),
        ("active_B_load", "B_L"),
        ("active_coulomb_friction", "T_c"),
        ("active_gravity_torque_coeff", "K_g"),
        ("gear_efficiency", "eta"),
        ("disturbance_step_Nm", "Disturbance step"),
        ("encoder_noise_std_rad", "Encoder noise"),
        ("vdc_scale", "V_dc scale"),
    ]
    available = [(column, label) for column, label in columns if any(row.get(column) not in ("", None) for row in v8_rows)]
    if not available:
        state.skip("sampled parameter distribution was not generated because MC episode metrics do not record sampled parameters.")
        return False
    ncols = 4
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.0 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (column, label) in zip(axes_arr, available):
        values = finite(row.get(column) for row in v8_rows)
        if values.size:
            ax.hist(values, bins=min(18, max(5, int(np.sqrt(values.size)))), color="#4C78A8", alpha=0.82)
        ax.set_title(label)
        ax.set_ylabel("Episodes")
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes_arr[len(available):]:
        ax.axis("off")
    fig.suptitle("Sampled parameter distributions in Monte Carlo episode metrics")
    fig.tight_layout()
    save_figure(fig, RANDOMIZATION / "fig_parameter_randomization_distribution.png", state)
    return True


def load_mc_episode_comparison(state: GenerationState) -> dict[str, list[dict[str, str]]]:
    paths = {
        "PID/PD-PI": RUNS / "gun_servo_pid_mc_baseline" / "summaries" / "pid_mc_eval_episode_metrics.csv",
        "SMC-PI": RUNS / "gun_servo_smc_baseline" / "summaries" / "smc_mc_eval_episode_metrics.csv",
    }
    out: dict[str, list[dict[str, str]]] = {}
    for method, path in paths.items():
        rows = read_rows(path)
        if rows:
            out[method] = rows
            state.add_source(path)
        else:
            state.warn(f"missing MC episode metrics: {relative(path)}")
    v8_rows = load_v8_mc_episode_rows(state)
    if v8_rows:
        out["Randomized MC-SC-Residual TD3-PI"] = v8_rows
    else:
        state.warn("v8 aggregate episode-level data is unavailable and per-seed MC episode metrics were not found.")
    return out


def plot_mc_scenario_rmse_boxplot(mc_rows: dict[str, list[dict[str, str]]], state: GenerationState) -> bool:
    if not mc_rows:
        state.skip("Monte Carlo scenario-family RMSE boxplot was not generated because episode-level MC data is missing.")
        return False
    methods = ordered_methods(mc_rows.keys())
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    positions = []
    data = []
    colors = []
    labels = []
    width = 0.18
    for scenario_idx, scenario in enumerate(MC_SCENARIOS):
        center = scenario_idx + 1
        for method_idx, method in enumerate(methods):
            values = finite(
                row.get("rmse_theta_deg")
                for row in mc_rows.get(method, [])
                if row.get("scenario") == scenario
            )
            if values.size == 0:
                continue
            pos = center + (method_idx - (len(methods) - 1) / 2) * width
            positions.append(pos)
            data.append(values)
            colors.append(METHOD_COLORS.get(method, "#4C78A8"))
            labels.append(method)
    if not data:
        plt.close(fig)
        state.skip("Monte Carlo scenario-family RMSE boxplot was not generated because no rmse_theta_deg values were available.")
        return False
    boxes = ax.boxplot(data, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(MC_SCENARIOS) + 1))
    ax.set_xticklabels(["step", "trapezoid", "sine", "disturbance"])
    ax.set_xlabel("Scenario family")
    ax.set_ylabel("RMSE [deg]")
    ax.set_title("Monte Carlo scenario-family RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS.get(method, "#4C78A8"), linewidth=7, alpha=0.75)
        for method in methods
    ]
    ax.legend(legend_handles, [PLOT_LABELS.get(method, method) for method in methods], frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_mc_scenario_rmse_boxplot.png", state)
    return True


def disturbance_rows(mc_rows: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    return {
        method: [row for row in rows if row.get("scenario") == "random_disturbance_after_settling"]
        for method, rows in mc_rows.items()
    }


def plot_disturbance_recovery_thresholds(mc_rows: dict[str, list[dict[str, str]]], state: GenerationState) -> bool:
    rows_by_method = disturbance_rows(mc_rows)
    thresholds = [
        ("recovery_time_0p02deg_s", "0.02 deg"),
        ("recovery_time_0p05deg_s", "0.05 deg"),
        ("recovery_time_0p10deg_s", "0.10 deg"),
    ]
    methods = ordered_methods(rows_by_method.keys())
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    x = np.arange(len(thresholds), dtype=float)
    width = min(0.8 / max(len(methods), 1), 0.22)
    plotted = False
    for idx, method in enumerate(methods):
        means = []
        stds = []
        for column, _ in thresholds:
            values = finite(row.get(column) for row in rows_by_method.get(method, []))
            means.append(float(np.mean(values)) if values.size else float("nan"))
            stds.append(float(np.std(values)) if values.size else 0.0)
        if np.isfinite(means).any():
            ax.bar(
                x + (idx - (len(methods) - 1) / 2) * width,
                means,
                yerr=stds,
                capsize=2,
                width=width,
                color=METHOD_COLORS.get(method),
                label=PLOT_LABELS.get(method, method),
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        state.skip("disturbance recovery thresholds were not generated because threshold recovery metrics are unavailable.")
        return False
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in thresholds])
    ax.set_xlabel("Recovery threshold")
    ax.set_ylabel("Recovery time [s]")
    ax.set_title("Disturbance recovery time thresholds")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, DISTURBANCE / "fig_disturbance_recovery_time_thresholds.png", state)
    return True


def plot_disturbance_peak_error(mc_rows: dict[str, list[dict[str, str]]], state: GenerationState) -> bool:
    rows_by_method = disturbance_rows(mc_rows)
    methods = ordered_methods(rows_by_method.keys())
    metrics = [
        ("max_abs_error_after_disturbance_deg", "Max abs error after disturbance [deg]"),
        ("peak_error_in_1s_after_disturbance_deg", "Peak error in 1 s after disturbance [deg]"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    plotted_any = False
    for ax, (column, title) in zip(axes, metrics):
        data = []
        labels = []
        colors = []
        for method in methods:
            values = finite(row.get(column) for row in rows_by_method.get(method, []))
            if values.size:
                data.append(values)
                labels.append(SHORT_PLOT_LABELS.get(method, method))
                colors.append(METHOD_COLORS.get(method, "#4C78A8"))
        if data:
            boxes = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.78)
            ax.tick_params(axis="x", rotation=20)
            plotted_any = True
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Error [deg]")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    if not plotted_any:
        plt.close(fig)
        state.skip("disturbance peak error figure was not generated because disturbance error metrics are unavailable.")
        return False
    if not any(
        row.get("peak_error_in_1s_after_disturbance_deg") not in ("", None)
        for rows in rows_by_method.values()
        for row in rows
    ):
        state.warn("peak_error_in_1s_after_disturbance_deg is not available in the disturbance episode metrics.")
    save_figure(fig, DISTURBANCE / "fig_disturbance_peak_error_after_disturbance.png", state)
    return True


def read_existing_table_paths() -> list[Path]:
    tables = sorted((PAPER / "tables").glob("*")) if (PAPER / "tables").exists() else []
    return [path for path in tables if path.suffix.lower() in {".csv", ".md"}]


def recommendation_lists() -> tuple[list[str], list[str]]:
    main = [
        "figures/training_diagnostics/fig_training_return_all_seeds.png",
        "figures/training_diagnostics/fig_eval_rmse_all_seeds.png",
        "figures/representative/fig_C1_position_comparison.png",
        "figures/representative/fig_C1_error_comparison.png",
        "figures/representative/fig_C3_position_comparison.png",
        "figures/representative/fig_C4a_position_comparison.png",
        "figures/representative/fig_C5a_disturbance_comparison.png",
        "figures/fig_mc_scenario_rmse_boxplot.png",
        "figures/randomization/fig_parameter_randomization_ranges.png",
        "figures/disturbance/fig_disturbance_recovery_time_thresholds.png",
    ]
    appendix = [
        "figures/training_diagnostics/fig_actor_loss_all_seeds.png",
        "figures/training_diagnostics/fig_critic_loss_all_seeds.png",
        "figures/action_smoothness/fig_action_raw_safe_comparison.png",
        "figures/action_smoothness/fig_speed_command_comparison.png",
        "figures/action_smoothness/fig_action_variation_bar.png",
        "figures/fig_fixed_safety_flags_bar.png",
        "figures/fig_mc_safety_violation_bar.png",
        "figures/fig_mc_flag_U_bar.png",
        "figures/randomization/fig_parameter_randomization_distribution.png",
        "figures/disturbance/fig_disturbance_peak_error_after_disturbance.png",
    ]
    return main, appendix


def write_generation_report(
    state: GenerationState,
    status: dict[str, bool],
) -> Path:
    main, appendix = recommendation_lists()
    lines = [
        "# Completed Figure Generation Report",
        "",
        "## Generated Figure Paths",
    ]
    for path in sorted(state.generated, key=lambda item: str(item)):
        if path.suffix.lower() == ".png":
            lines.append(f"- {path}")
    lines.extend(["", "## Source Files Used"])
    for path in sorted(state.sources, key=lambda item: str(item)):
        lines.append(f"- {path}")
    lines.extend(["", "## Figures Successfully Generated"])
    for key, ok in status.items():
        if ok:
            lines.append(f"- {key}")
    lines.extend(["", "## Figures Skipped And Why"])
    skipped = state.skipped if state.skipped else ["No requested figure groups were skipped."]
    for note in skipped:
        lines.append(f"- {note}")
    lines.extend(["", "## Missing Data Warnings"])
    warnings = state.warnings if state.warnings else ["No missing-data warnings beyond unavailable optional columns."]
    for warning in warnings:
        lines.append(f"- {warning}")
    lines.extend(["", "## Recommended Main-Text Figures"])
    for item in main:
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Appendix Figures"])
    for item in appendix:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Status Summary",
            f"- Training return curve generated: {'yes' if status.get('training return') else 'no'}",
            f"- Eval RMSE curve generated: {'yes' if status.get('eval RMSE') else 'no'}",
            f"- Actor loss curve generated: {'yes' if status.get('actor loss') else 'no'}",
            f"- Critic loss curve generated: {'yes' if status.get('critic loss') else 'no'}",
            f"- Representative overlays generated: {'yes' if status.get('representative overlays') else 'no'}",
            f"- Action smoothness figures generated: {'yes' if status.get('action smoothness') else 'no'}",
            f"- Parameter randomization range figure generated: {'yes' if status.get('randomization ranges') else 'no'}",
            f"- Parameter randomization distribution generated: {'yes' if status.get('randomization distribution') else 'no'}",
            f"- Disturbance recovery figures generated: {'yes' if status.get('disturbance recovery') else 'no'}",
        ]
    )
    path = REPORTS / "completed_figure_generation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_checklist(state: GenerationState, report_path: Path) -> Path:
    main, appendix = recommendation_lists()
    generated_figures = sorted([path for path in set(state.generated) if path.suffix.lower() == ".png"], key=lambda item: str(item))
    existing_figures = sorted(FIGURES.glob("*.png"))
    representative = sorted(REPRESENTATIVE.glob("*.png"))
    training = sorted(TRAINING.glob("*.png"))
    action = sorted(ACTION.glob("*.png"))
    randomization = sorted(RANDOMIZATION.glob("*.png"))
    disturbance = sorted(DISTURBANCE.glob("*.png"))
    all_figures = []
    seen = set()
    for path in existing_figures + representative + training + action + randomization + disturbance + generated_figures:
        if path not in seen:
            all_figures.append(path)
            seen.add(path)
    lines = [
        "# Result File Checklist",
        "",
        "## Generated Tables",
    ]
    tables = read_existing_table_paths()
    lines.extend([f"- {path}" for path in tables] or ["- No generated tables found."])
    lines.extend(["", "## Generated Figures"])
    lines.extend([f"- {path}" for path in all_figures] or ["- No generated figures found."])
    lines.extend(["", "## Source CSV / Config / Report Paths"])
    for path in sorted(state.sources | {report_path}, key=lambda item: str(item)):
        lines.append(f"- {path}")
    lines.extend(["", "## Missing Data Notes"])
    notes = state.warnings + state.skipped
    lines.extend([f"- {note}" for note in notes] or ["- No missing data warnings."])
    lines.extend(["", "## Recommended Files For Thesis Main Text"])
    for item in main:
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Files For Appendix"])
    for item in appendix:
        lines.append(f"- {item}")
    path = REPORTS / "result_file_checklist.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    configure_matplotlib()
    ensure_dirs()
    state = GenerationState(generated=[], skipped=[], warnings=[], sources=set())

    logs = load_train_logs(state)
    status: dict[str, bool] = {}
    status["training return"] = plot_training_returns(logs, state)
    status["eval RMSE"] = plot_eval_rmse(logs, state)
    actor = plot_loss_curve(
        logs,
        "actor_loss",
        "Actor loss",
        "Actor loss during training",
        "fig_actor_loss_all_seeds.png",
        state,
    )
    critic = plot_loss_curve(
        logs,
        "critic_loss",
        "Critic loss",
        "Critic loss during training",
        "fig_critic_loss_all_seeds.png",
        state,
    )
    status["actor loss"] = actor
    status["critic loss"] = critic
    if not actor or not critic:
        state.skip("actor/critic loss curves were not generated because the training logs do not record actor_loss/critic_loss.")

    status["representative overlays"] = generate_representative_figures(state)

    traces = action_traces(state)
    action_statuses = [
        plot_action_raw_safe(traces, state),
        plot_speed_command(traces, state),
        plot_action_variation_bar(traces, state),
    ]
    status["action smoothness"] = all(action_statuses)

    status["fixed safety"] = plot_fixed_safety_flags(state)
    mc_safety_status = plot_mc_safety(state)
    status["MC safety violations"] = mc_safety_status[0]
    status["MC flag_U"] = mc_safety_status[1]

    status["randomization ranges"] = plot_randomization_ranges(state)
    v8_mc_rows = load_v8_mc_episode_rows(state)
    status["randomization distribution"] = plot_randomization_distribution(v8_mc_rows, state)

    mc_rows = load_mc_episode_comparison(state)
    status["MC scenario RMSE"] = plot_mc_scenario_rmse_boxplot(mc_rows, state)

    disturbance_statuses = [
        plot_disturbance_recovery_thresholds(mc_rows, state),
        plot_disturbance_peak_error(mc_rows, state),
    ]
    status["disturbance recovery"] = all(disturbance_statuses)

    report_path = write_generation_report(state, status)
    checklist_path = write_checklist(state, report_path)
    state.add_generated(report_path)
    state.add_generated(checklist_path)

    print(f"generated_png_count={sum(1 for path in state.generated if path.suffix.lower() == '.png')}")
    print(f"report={report_path}")
    print(f"checklist={checklist_path}")
    if state.skipped:
        print("skipped:")
        for item in state.skipped:
            print(f"- {item}")
    if state.warnings:
        print("warnings:")
        for item in state.warnings:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Audit and package final paper-ready gun-servo figures.

The script reads existing logs, summaries, and trajectory CSVs, regenerates
derived PNG figures where source CSV data is available, and writes audit
manifests/reports under outputs/figure_audit. It does not rerun training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import re
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"
PAPER = ROOT / "outputs" / "paper_ready_results"
PAPER_FIGURES = PAPER / "figures"
BY_METHOD = PAPER_FIGURES / "by_method"
REPRESENTATIVE = PAPER_FIGURES / "representative"
TRAINING = PAPER_FIGURES / "training_diagnostics"
ACTION_DIR = PAPER_FIGURES / "action_smoothness"
RANDOMIZATION_DIR = PAPER_FIGURES / "randomization"
DISTURBANCE_DIR = PAPER_FIGURES / "disturbance"
PAPER_REPORTS = PAPER / "reports"

AUDIT = ROOT / "outputs" / "figure_audit"
AUDIT_SUMMARIES = AUDIT / "summaries"
AUDIT_REPORTS = AUDIT / "reports"
AUDIT_MANIFESTS = AUDIT / "manifests"

FIXED_COMPARISON = RUNS / "final_method_comparison" / "summaries" / "fixed_six_scenario_comparison.csv"
MC_COMPARISON = RUNS / "final_method_comparison" / "summaries" / "monte_carlo_comparison.csv"
PID_MC_EPISODES = RUNS / "gun_servo_pid_mc_baseline" / "summaries" / "pid_mc_eval_episode_metrics.csv"
SMC_MC_EPISODES = RUNS / "gun_servo_smc_baseline" / "summaries" / "smc_mc_eval_episode_metrics.csv"
ENV_RANDOMIZATION_CONFIG = ROOT / "configs" / "env" / "gun_servo_position_mc_sc_residual_td3_v8_rand.yaml"

RUN_DIRS = [
    RUNS / "gun_servo_pid_smoke_v2",
    RUNS / "gun_servo_pid_mc_baseline",
    RUNS / "gun_servo_smc_baseline",
    RUNS / "gun_servo_td3_smoke_v3_c1",
    RUNS / "gun_servo_residual_td3_smoke_v4",
    RUNS / "gun_servo_sc_residual_td3_smoke_v5",
    RUNS / "gun_servo_mc_sc_residual_td3_smoke_v6",
    RUNS / "gun_servo_mc_sc_residual_td3_smoke_v7",
    RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed0",
    RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed1",
    RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed2",
    RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_aggregate",
    RUNS / "final_method_comparison",
    PAPER,
]

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

MC_SCENARIOS = [
    "random_step_safe",
    "random_trapezoid_safe",
    "random_sine_safe",
    "random_disturbance_after_settling",
]

MAIN_METHODS = [
    "PID/PD-PI",
    "SMC-PI",
    "Direct TD3-PI",
    "Randomized MC-SC-Residual-TD3-PI",
]

METHOD_COLORS = {
    "PID/PD-PI": "#4C78A8",
    "SMC-PI": "#54A24B",
    "Direct TD3-PI": "#B279A2",
    "Randomized MC-SC-Residual-TD3-PI": "#F58518",
}

METHOD_SHORT_LABELS = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Direct TD3-PI": "Direct TD3-PI",
    "Randomized MC-SC-Residual-TD3-PI": "Randomized MC-SC\nResidual TD3-PI",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    short: str
    run_dir: Path
    controller_suffix: str
    rl_method: bool = False
    mc_method: bool = False
    notes: str = ""

    def trace_path(self, scenario: str) -> Path:
        return self.run_dir / "eval" / f"{scenario}_{self.controller_suffix}.csv"

    def figure_path(self, scenario: str, figure_type: str) -> Path:
        return self.run_dir / "figures" / f"{scenario}_{self.controller_suffix}_{figure_type}.png"

    def by_method_path(self, scenario: str, figure_type: str) -> Path:
        return BY_METHOD / f"{self.short}_{scenario}_{figure_type}.png"


METHODS = {
    "PID/PD-PI": MethodSpec(
        "PID/PD-PI",
        "PID",
        RUNS / "gun_servo_pid_smoke_v2",
        "pid",
        mc_method=True,
    ),
    "SMC-PI": MethodSpec(
        "SMC-PI",
        "SMC",
        RUNS / "gun_servo_smc_baseline",
        "smc",
        mc_method=True,
    ),
    "Direct TD3-PI": MethodSpec(
        "Direct TD3-PI",
        "TD3",
        RUNS / "gun_servo_td3_smoke_v3_c1",
        "td3",
        rl_method=True,
    ),
    "Randomized MC-SC-Residual-TD3-PI": MethodSpec(
        "Randomized MC-SC-Residual-TD3-PI",
        "PROPOSED",
        RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed0",
        "td3",
        rl_method=True,
        mc_method=True,
    ),
}

PID_C1A_FALLBACK_TRACE = (
    RUNS / "gun_servo_mc_sc_residual_td3_random_v8_full_seed0" / "eval" / "C1a_5deg_step_pid.csv"
)

FIXED_METHOD_MAP = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Direct TD3-PI v3": "Direct TD3-PI",
    "Randomized MC-SC-Residual TD3 v8 full": "Randomized MC-SC-Residual-TD3-PI",
}

MC_METHOD_MAP = {
    "PID/PD-PI": "PID/PD-PI",
    "SMC-PI": "SMC-PI",
    "Randomized MC-SC-Residual TD3 v8 full": "Randomized MC-SC-Residual-TD3-PI",
}


@dataclass
class AuditState:
    regenerated: list[Path] = field(default_factory=list)
    packaged: list[Path] = field(default_factory=list)
    source_files: set[Path] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def source(self, path: Path) -> None:
        if path.exists():
            self.source_files.add(path)

    def warn(self, note: str) -> None:
        if note not in self.warnings:
            self.warnings.append(note)


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
        }
    )


def ensure_dirs() -> None:
    for path in (
        AUDIT_SUMMARIES,
        AUDIT_REPORTS,
        AUDIT_MANIFESTS,
        PAPER_FIGURES,
        BY_METHOD,
        REPRESENTATIVE,
        TRAINING,
        ACTION_DIR,
        RANDOMIZATION_DIR,
        DISTURBANCE_DIR,
        PAPER_REPORTS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
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


def column(rows: list[dict[str, str]], *names: str) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    for name in names:
        if name in rows[0]:
            return np.asarray([safe_float(row.get(name)) for row in rows], dtype=float)
    return np.full(len(rows), np.nan, dtype=float)


def sample_rows(rows: list[dict[str, str]], max_points: int = 3000) -> list[dict[str, str]]:
    if len(rows) <= max_points:
        return rows
    return rows[:: max(1, len(rows) // max_points)]


def save_fig(fig: plt.Figure, path: Path, state: AuditState, *, regenerated: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if regenerated:
        state.regenerated.append(path)
    if PAPER_FIGURES in path.parents:
        state.packaged.append(path)


def detect_method_from_run(path: Path) -> str:
    text = str(path).lower()
    if "pid" in text and "smc" not in text and "td3" not in text:
        return "PID/PD-PI"
    if "smc_baseline" in text:
        return "SMC-PI"
    if "td3_smoke_v3" in text:
        return "Direct TD3-PI"
    if "random_v8_full" in text or "paper_ready_results" in text:
        return "Randomized MC-SC-Residual-TD3-PI"
    if "residual_td3_smoke_v4" in text:
        return "Residual TD3-PI"
    if "sc_residual_td3_smoke_v5" in text:
        return "SC-Residual TD3-PI"
    if "mc_sc_residual_td3_smoke_v6" in text:
        return "MC-SC-Residual TD3-PI v6"
    if "mc_sc_residual_td3_smoke_v7" in text:
        return "MC-SC-Residual TD3-PI"
    return ""


def scenario_from_name(name: str) -> str:
    for scenario in sorted(SCENARIOS + MC_SCENARIOS, key=len, reverse=True):
        if scenario in name:
            return scenario
    for short, scenario in (("C1a", "C1a_5deg_step"), ("C4a", "C4a_sine_tracking_safe"), ("C5a", "C5a_disturbance_after_settling")):
        if short in name:
            return scenario
    if re.search(r"(^|_)C1(_|$)", name):
        return "C1_10deg_step"
    if re.search(r"(^|_)C3(_|$)", name):
        return "C3_trapezoid_tracking"
    return ""


def figure_type_from_name(name: str) -> str:
    lower = name.lower()
    if "disturbance_recovery" in lower or "recovery_time" in lower:
        return "disturbance recovery comparison"
    if "disturbance_zoom" in lower or "disturbance_comparison" in lower:
        return "disturbance_zoom"
    if "position" in lower:
        return "position"
    if "error" in lower and "peak_error" not in lower:
        return "error"
    if "action_raw_safe" in lower or "action_variation" in lower or "action_smoothness" in lower:
        return "action smoothness"
    if "action_trace" in lower:
        return "action_trace"
    if "speed_command" in lower:
        return "speed_command"
    if "safety_flags" in lower:
        return "safety_flags"
    if "torque" in lower:
        return "torque"
    if "training_return" in lower or "return_curve" in lower:
        return "training_return"
    if "eval_rmse" in lower:
        return "eval_rmse"
    if "actor_loss" in lower:
        return "actor_loss"
    if "critic_loss" in lower:
        return "critic_loss"
    if "scenario_rmse" in lower:
        return "Monte Carlo scenario RMSE"
    if "rmse_boxplot" in lower:
        return "Monte Carlo RMSE"
    if "success_rate" in lower:
        return "Monte Carlo success rate"
    if "violation" in lower:
        return "Monte Carlo violation counts"
    if "flag_u" in lower:
        return "Monte Carlo flag_U"
    if "randomization" in lower or "parameter" in lower:
        return "randomization distribution"
    return "other"


def build_inventory(state: AuditState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in RUN_DIRS:
        if not run_dir.exists():
            rows.append(
                {
                    "run_name": run_dir.name,
                    "method": detect_method_from_run(run_dir),
                    "run_dir": str(run_dir),
                    "figure_dir": "",
                    "figure_file": "",
                    "scenario": "",
                    "figure_type": "",
                    "exists": "no",
                    "file_size": "",
                    "notes": "run directory missing",
                }
            )
            continue
        pngs = sorted(run_dir.rglob("*.png"))
        if not pngs:
            rows.append(
                {
                    "run_name": run_dir.name,
                    "method": detect_method_from_run(run_dir),
                    "run_dir": str(run_dir),
                    "figure_dir": "",
                    "figure_file": "",
                    "scenario": "",
                    "figure_type": "",
                    "exists": "no",
                    "file_size": "",
                    "notes": "no PNG figures found",
                }
            )
            continue
        for path in pngs:
            rows.append(
                {
                    "run_name": run_dir.name,
                    "method": detect_method_from_run(path),
                    "run_dir": str(run_dir),
                    "figure_dir": str(path.parent),
                    "figure_file": path.name,
                    "scenario": scenario_from_name(path.name),
                    "figure_type": figure_type_from_name(path.name),
                    "exists": "yes",
                    "file_size": path.stat().st_size,
                    "notes": "",
                }
            )
    path = AUDIT_SUMMARIES / "current_figure_inventory.csv"
    write_rows(
        path,
        rows,
        ["run_name", "method", "run_dir", "figure_dir", "figure_file", "scenario", "figure_type", "exists", "file_size", "notes"],
    )
    state.source(path)
    return rows


def required_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_types = ["position", "error", "speed_command", "safety_flags"]
    for method in MAIN_METHODS:
        spec = METHODS[method]
        for scenario in SCENARIOS:
            for figure_type in fixed_types:
                rows.append(
                    {
                        "figure_id": f"{spec.short}_{scenario}_{figure_type}",
                        "method": method,
                        "method_short": spec.short,
                        "scenario": scenario,
                        "figure_type": figure_type,
                        "required_file": str(spec.by_method_path(scenario, figure_type)),
                        "required_scope": "fixed_scenario_by_method",
                        "source_run": str(spec.run_dir),
                        "applicable": "yes",
                        "notes": "",
                    }
                )
            if scenario == "C5a_disturbance_after_settling":
                rows.append(
                    {
                        "figure_id": f"{spec.short}_{scenario}_disturbance_zoom",
                        "method": method,
                        "method_short": spec.short,
                        "scenario": scenario,
                        "figure_type": "disturbance_zoom",
                        "required_file": str(spec.by_method_path(scenario, "disturbance_zoom")),
                        "required_scope": "fixed_scenario_by_method",
                        "source_run": str(spec.run_dir),
                        "applicable": "yes",
                        "notes": "",
                    }
                )
            if spec.rl_method:
                rows.append(
                    {
                        "figure_id": f"{spec.short}_{scenario}_action_trace",
                        "method": method,
                        "method_short": spec.short,
                        "scenario": scenario,
                        "figure_type": "action_trace",
                        "required_file": str(spec.by_method_path(scenario, "action_trace")),
                        "required_scope": "fixed_scenario_by_method_rl",
                        "source_run": str(spec.run_dir),
                        "applicable": "yes",
                        "notes": "RL-specific action trace",
                    }
                )
    proposed = METHODS["Randomized MC-SC-Residual-TD3-PI"]
    proposed_extras = [
        ("training_return_all_seeds", TRAINING / "fig_training_return_all_seeds.png", "training_diagnostics"),
        ("eval_rmse_all_seeds", TRAINING / "fig_eval_rmse_all_seeds.png", "training_diagnostics"),
        ("actor_loss_all_seeds", TRAINING / "fig_actor_loss_all_seeds.png", "training_diagnostics_optional"),
        ("critic_loss_all_seeds", TRAINING / "fig_critic_loss_all_seeds.png", "training_diagnostics_optional"),
        ("action_raw_safe_comparison", ACTION_DIR / "fig_action_raw_safe_comparison.png", "proposed_action_smoothness"),
        ("parameter_randomization_ranges", RANDOMIZATION_DIR / "fig_parameter_randomization_ranges.png", "randomization"),
        ("parameter_randomization_distribution", RANDOMIZATION_DIR / "fig_parameter_randomization_distribution.png", "randomization_optional"),
        ("disturbance_recovery_statistics", DISTURBANCE_DIR / "fig_disturbance_recovery_time_thresholds.png", "disturbance_statistics"),
    ]
    for figure_type, path, scope in proposed_extras:
        rows.append(
            {
                "figure_id": figure_type,
                "method": proposed.method,
                "method_short": proposed.short,
                "scenario": "",
                "figure_type": figure_type,
                "required_file": str(path),
                "required_scope": scope,
                "source_run": str(proposed.run_dir),
                "applicable": "yes",
                "notes": "",
            }
        )
    mc_methods = "PID/PD-PI; SMC-PI; Randomized MC-SC-Residual-TD3-PI"
    mc_required = [
        ("mc_mean_rmse_comparison", PAPER_FIGURES / "fig_mc_mean_rmse_bar.png"),
        ("mc_rmse_boxplot", PAPER_FIGURES / "fig_mc_scenario_rmse_boxplot.png"),
        ("mc_scenario_family_rmse_boxplot", PAPER_FIGURES / "fig_mc_scenario_rmse_boxplot.png"),
        ("mc_success_rate_comparison", PAPER_FIGURES / "fig_mc_success_rate_bar.png"),
        ("mc_violation_counts_comparison", PAPER_FIGURES / "fig_mc_safety_violation_bar.png"),
        ("mc_flag_U_comparison", PAPER_FIGURES / "fig_mc_flag_U_bar.png"),
        ("mc_disturbance_recovery_comparison", DISTURBANCE_DIR / "fig_disturbance_recovery_time_thresholds.png"),
    ]
    for figure_type, path in mc_required:
        rows.append(
            {
                "figure_id": figure_type,
                "method": mc_methods,
                "method_short": "MC",
                "scenario": "Monte Carlo",
                "figure_type": figure_type,
                "required_file": str(path),
                "required_scope": "monte_carlo_comparison",
                "source_run": str(RUNS / "final_method_comparison"),
                "applicable": "yes",
                "notes": "",
            }
        )
    overlay_required = [
        ("overlay_C1_position", REPRESENTATIVE / "fig_C1_position_comparison.png", "C1_10deg_step", "position"),
        ("overlay_C1_error", REPRESENTATIVE / "fig_C1_error_comparison.png", "C1_10deg_step", "error"),
        ("overlay_C3_position", REPRESENTATIVE / "fig_C3_position_comparison.png", "C3_trapezoid_tracking", "position"),
        ("overlay_C4a_position", REPRESENTATIVE / "fig_C4a_position_comparison.png", "C4a_sine_tracking_safe", "position"),
        ("overlay_C5a_disturbance", REPRESENTATIVE / "fig_C5a_disturbance_comparison.png", "C5a_disturbance_after_settling", "disturbance_zoom"),
    ]
    for figure_id, path, scenario, figure_type in overlay_required:
        rows.append(
            {
                "figure_id": figure_id,
                "method": "; ".join(MAIN_METHODS),
                "method_short": "OVERLAY",
                "scenario": scenario,
                "figure_type": figure_type,
                "required_file": str(path),
                "required_scope": "representative_overlay",
                "source_run": "multiple",
                "applicable": "yes",
                "notes": "",
            }
        )
    return rows


def write_required_matrix(rows: list[dict[str, Any]]) -> None:
    write_rows(
        AUDIT_SUMMARIES / "required_figure_matrix.csv",
        rows,
        [
            "figure_id",
            "method",
            "method_short",
            "scenario",
            "figure_type",
            "required_file",
            "required_scope",
            "source_run",
            "applicable",
            "notes",
        ],
    )


def trace_path_for(method: str, scenario: str) -> Path:
    primary = METHODS[method].trace_path(scenario)
    if primary.exists():
        return primary
    if method == "PID/PD-PI" and scenario == "C1a_5deg_step" and PID_C1A_FALLBACK_TRACE.exists():
        return PID_C1A_FALLBACK_TRACE
    return primary


def trace_exists(method: str, scenario: str) -> bool:
    return trace_path_for(method, scenario).exists()


def source_available(row: dict[str, Any]) -> tuple[bool, bool, str]:
    scope = row["required_scope"]
    method = row["method"]
    scenario = row["scenario"]
    figure_type = row["figure_type"]
    if scope.startswith("fixed_scenario_by_method"):
        if method not in METHODS:
            return False, False, "method source is not configured"
        trace = trace_path_for(method, scenario)
        if trace.exists():
            if figure_type == "action_trace" and method in {"PID/PD-PI", "SMC-PI"}:
                return False, False, "classical controllers do not require RL action traces"
            return True, True, f"trajectory CSV available: {trace}"
        checkpoint = METHODS[method].run_dir / "checkpoints" / "checkpoint_best.pt"
        if method == "Direct TD3-PI" and checkpoint.exists():
            return False, True, "trajectory CSV missing; Direct TD3 checkpoint exists for possible evaluation"
        return False, False, f"trajectory CSV missing: {trace}"
    if scope.startswith("training_diagnostics"):
        train_logs = [RUNS / f"gun_servo_mc_sc_residual_td3_random_v8_full_seed{i}" / "train_log.csv" for i in range(3)]
        if all(path.exists() for path in train_logs):
            first = read_rows(train_logs[0])
            if figure_type.startswith("actor_loss") and first and "actor_loss" not in first[0]:
                return False, False, "actor_loss column is not recorded"
            if figure_type.startswith("critic_loss") and first and "critic_loss" not in first[0]:
                return False, False, "critic_loss column is not recorded"
            return True, True, "three seed train_log.csv files available"
        return False, False, "one or more seed train_log.csv files are missing"
    if scope in {"randomization", "randomization_optional"}:
        if figure_type == "parameter_randomization_distribution":
            v8 = [RUNS / f"gun_servo_mc_sc_residual_td3_random_v8_full_seed{i}" / "summaries" / "mc_eval_episode_metrics.csv" for i in range(3)]
            return all(path.exists() for path in v8), all(path.exists() for path in v8), "sampled MC episode parameter logs"
        return ENV_RANDOMIZATION_CONFIG.exists(), ENV_RANDOMIZATION_CONFIG.exists(), "randomization config"
    if scope in {"proposed_action_smoothness", "disturbance_statistics", "monte_carlo_comparison"}:
        return True, True, "summary/trajectory data available"
    if scope == "representative_overlay":
        available = [method for method in MAIN_METHODS if trace_exists(method, scenario)]
        return bool(available), bool(available), f"available methods: {', '.join(available)}"
    return Path(row["required_file"]).exists(), False, "existing packaged file"


def write_missing_outputs(required: list[dict[str, Any]], state: AuditState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in required:
        required_path = Path(row["required_file"])
        data_ok, can_regenerate, source_note = source_available(row)
        if row.get("applicable") == "no":
            status = "not_applicable"
            action = "not applicable"
        elif required_path.exists():
            status = "found"
            action = "use packaged figure"
        else:
            status = "missing"
            action = "regenerate from existing data" if can_regenerate else "report missing; do not fabricate"
        rows.append(
            {
                "required_file": str(required_path),
                "method": row["method"],
                "scenario": row["scenario"],
                "figure_type": row["figure_type"],
                "current_status": status,
                "source_data_available": "yes" if data_ok else "no",
                "can_regenerate": "yes" if can_regenerate else "no",
                "recommended_action": action,
                "notes": source_note,
            }
        )
    fields = [
        "required_file",
        "method",
        "scenario",
        "figure_type",
        "current_status",
        "source_data_available",
        "can_regenerate",
        "recommended_action",
        "notes",
    ]
    write_rows(AUDIT_SUMMARIES / "missing_figure_list.csv", rows, fields)
    missing = [row for row in rows if row["current_status"] == "missing"]
    lines = [
        "# Missing Figure Report",
        "",
        f"Required figure records: {len(rows)}",
        f"Missing records after regeneration/package step: {len(missing)}",
        "",
        "## Missing Figures",
    ]
    if missing:
        for item in missing:
            lines.append(
                f"- {item['method']} | {item['scenario']} | {item['figure_type']} | "
                f"source_data={item['source_data_available']} | action={item['recommended_action']} | {item['notes']}"
            )
    else:
        lines.append("- No required figure records remain missing.")
    (AUDIT_REPORTS / "missing_figure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.source(AUDIT_SUMMARIES / "missing_figure_list.csv")
    state.source(AUDIT_REPORTS / "missing_figure_report.md")
    return rows


def read_trace(path: Path, state: AuditState, max_points: int = 3000) -> list[dict[str, str]]:
    rows = read_rows(path)
    if rows:
        state.source(path)
    return sample_rows(rows, max_points)


def omega_cmd_deg_s(rows: list[dict[str, str]]) -> np.ndarray:
    values = column(rows, "omega_L_cmd_safe_deg_s", "omega_cmd_deg_s")
    if np.isfinite(values).any():
        return values
    raw = column(rows, "omega_L_cmd_safe")
    if np.isfinite(raw).any():
        return np.rad2deg(raw)
    return raw


def plot_trace(rows: list[dict[str, str]], path: Path, state: AuditState, *, title: str, figure_type: str) -> bool:
    if not rows:
        return False
    t = column(rows, "t_s", "time_s")
    if not np.isfinite(t).any():
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if figure_type == "position":
        ax.plot(t, column(rows, "theta_ref_deg"), color="black", linestyle="--", linewidth=1.2, label="Reference")
        ax.plot(t, column(rows, "theta_L_deg"), linewidth=1.2, label="Actual")
        ax.set_ylabel("Position [deg]")
    elif figure_type == "error":
        error = column(rows, "e_theta_deg")
        if not np.isfinite(error).any():
            error = column(rows, "theta_ref_deg") - column(rows, "theta_L_deg")
        ax.plot(t, error, linewidth=1.2, label="Error")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Error [deg]")
    elif figure_type == "speed_command":
        ax.plot(t, omega_cmd_deg_s(rows), linewidth=1.1, label="Commanded speed")
        omega = column(rows, "omega_L_deg_s")
        if np.isfinite(omega).any():
            ax.plot(t, omega, linewidth=1.0, label="Load speed")
        ax.set_ylabel("Speed [deg/s]")
    elif figure_type == "safety_flags":
        for idx, name in enumerate(("flag_U_safe", "flag_E_safe", "flag_X_safe", "sigma_safe")):
            y = column(rows, name)
            if np.isfinite(y).any():
                ax.step(t, y + 0.04 * idx, where="post", linewidth=1.0, label=name)
        ax.set_ylim(-0.1, 1.35)
        ax.set_ylabel("Flag")
    elif figure_type == "action_trace":
        raw = column(rows, "a_raw", "action_raw")
        safe = column(rows, "a_safe", "action_safe")
        if np.isfinite(raw).any():
            ax.plot(t, raw, linewidth=1.1, linestyle="--", label="Raw action")
        if np.isfinite(safe).any():
            ax.plot(t, safe, linewidth=1.1, label="Safe action")
        ax.set_ylabel("Action")
    elif figure_type == "disturbance_zoom":
        error = column(rows, "e_theta_deg")
        if not np.isfinite(error).any():
            error = column(rows, "theta_ref_deg") - column(rows, "theta_L_deg")
        disturbance = column(rows, "disturbance_torque_Nm", "disturbance_step_Nm", "T_L")
        mask = np.isfinite(t) & np.isfinite(error)
        if np.isfinite(disturbance).any():
            diff = np.abs(np.diff(disturbance))
            idx = int(np.flatnonzero(diff > max(1e-9, 0.2 * np.nanmax(diff)))[0] + 1) if diff.size and np.nanmax(diff) > 0 else 0
            center = float(t[idx])
            mask &= (t >= center - 0.75) & (t <= center + 1.5)
        ax.plot(t[mask], error[mask], linewidth=1.2, label="Error")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Error [deg]")
        if np.isfinite(disturbance).any():
            ax2 = ax.twinx()
            ax2.plot(t[mask], disturbance[mask], color="#E45756", linewidth=1.0, alpha=0.75, label="Disturbance")
            ax2.set_ylabel("Disturbance [N m]")
    else:
        plt.close(fig)
        return False
    ax.set_xlabel("Time [s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if figure_type == "disturbance_zoom" and len(fig.axes) > 1:
        h2, l2 = fig.axes[1].get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels, frameon=False)
    fig.tight_layout()
    save_fig(fig, path, state)
    return True


def generate_fixed_by_method(state: AuditState) -> None:
    for method in MAIN_METHODS:
        spec = METHODS[method]
        for scenario in SCENARIOS:
            trace_path = trace_path_for(method, scenario)
            trace = read_trace(trace_path, state)
            if not trace:
                state.warn(f"{method} missing trajectory CSV for {scenario}: {spec.trace_path(scenario)}")
                continue
            figure_types = ["position", "error", "speed_command", "safety_flags"]
            if spec.rl_method:
                figure_types.append("action_trace")
            if scenario == "C5a_disturbance_after_settling":
                figure_types.append("disturbance_zoom")
            for figure_type in figure_types:
                title = f"{method} {SHORT_SCENARIOS[scenario]} {figure_type.replace('_', ' ')}"
                original_path = spec.figure_path(scenario, figure_type)
                if not original_path.exists():
                    plot_trace(trace, original_path, state, title=title, figure_type=figure_type)
                paper_path = spec.by_method_path(scenario, figure_type)
                plot_trace(trace, paper_path, state, title=title, figure_type=figure_type)


def overlay_traces(scenario: str, *, mode: str, out: Path, title: str, state: AuditState) -> bool:
    traces = []
    for method in MAIN_METHODS:
        spec = METHODS[method]
        rows = read_trace(trace_path_for(method, scenario), state)
        if rows:
            traces.append((method, rows))
    if not traces:
        return False
    starts = []
    ends = []
    for _, rows in traces:
        t_values = column(rows, "t_s", "time_s")
        finite_t = t_values[np.isfinite(t_values)]
        if finite_t.size:
            starts.append(float(np.min(finite_t)))
            ends.append(float(np.max(finite_t)))
    common_start = max(starts) if starts else float("-inf")
    common_end = min(ends) if ends else float("inf")
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    if mode == "position":
        first = traces[0][1]
        ref_t = column(first, "t_s", "time_s")
        ref_y = column(first, "theta_ref_deg")
        ref_mask = np.isfinite(ref_t) & np.isfinite(ref_y) & (ref_t >= common_start) & (ref_t <= common_end)
        ax.plot(ref_t[ref_mask], ref_y[ref_mask], color="black", linestyle="--", linewidth=1.2, label="Reference")
    for method, rows in traces:
        t = column(rows, "t_s", "time_s")
        if mode == "position":
            y = column(rows, "theta_L_deg")
            ylabel = "Position [deg]"
        else:
            y = column(rows, "e_theta_deg")
            if not np.isfinite(y).any():
                y = column(rows, "theta_ref_deg") - column(rows, "theta_L_deg")
            ylabel = "Error [deg]"
        if scenario == "C5a_disturbance_after_settling":
            disturbance = column(rows, "disturbance_torque_Nm", "disturbance_step_Nm", "T_L")
            if np.isfinite(disturbance).any():
                diff = np.abs(np.diff(disturbance))
                if diff.size and np.nanmax(diff) > 0:
                    idx = int(np.flatnonzero(diff > max(1e-9, 0.2 * np.nanmax(diff)))[0] + 1)
                    center = float(t[idx])
                    mask = (t >= center - 0.75) & (t <= center + 1.5)
                else:
                    mask = np.isfinite(t) & np.isfinite(y)
            else:
                mask = np.isfinite(t) & np.isfinite(y)
        else:
            mask = np.isfinite(t) & np.isfinite(y)
        mask &= (t >= common_start) & (t <= common_end)
        ax.plot(t[mask], y[mask], linewidth=1.15, color=METHOD_COLORS.get(method), label=METHOD_SHORT_LABELS[method])
    if mode == "error":
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out, state)
    return True


def generate_overlays(state: AuditState) -> list[Path]:
    outputs = [
        ("C1_10deg_step", "position", REPRESENTATIVE / "fig_C1_position_comparison.png", "C1 position tracking comparison"),
        ("C1_10deg_step", "error", REPRESENTATIVE / "fig_C1_error_comparison.png", "C1 position error comparison"),
        ("C3_trapezoid_tracking", "position", REPRESENTATIVE / "fig_C3_position_comparison.png", "C3 position tracking comparison"),
        ("C4a_sine_tracking_safe", "position", REPRESENTATIVE / "fig_C4a_position_comparison.png", "C4a position tracking comparison"),
        ("C5a_disturbance_after_settling", "error", REPRESENTATIVE / "fig_C5a_disturbance_comparison.png", "C5a disturbance response comparison"),
    ]
    made = []
    for scenario, mode, path, title in outputs:
        if overlay_traces(scenario, mode=mode, out=path, title=title, state=state):
            made.append(path)
    return made


def load_fixed_rows(state: AuditState) -> list[dict[str, Any]]:
    rows = []
    for row in read_rows(FIXED_COMPARISON):
        method = FIXED_METHOD_MAP.get(str(row.get("method", "")))
        if method in MAIN_METHODS:
            item = dict(row)
            item["method"] = method
            rows.append(item)
    state.source(FIXED_COMPARISON)
    direct_extra_paths = [
        METHODS["Direct TD3-PI"].run_dir / "summaries" / "td3_v3_smoke_summary.csv",
        METHODS["Direct TD3-PI"].run_dir / "summaries" / "td3_v3_missing_fixed_summary.csv",
    ]
    seen = {(row["method"], row["scenario"]) for row in rows}
    for path in direct_extra_paths:
        for row in read_rows(path):
            scenario = row.get("scenario", "")
            key = ("Direct TD3-PI", scenario)
            if scenario in SCENARIOS and key not in seen:
                item = dict(row)
                item["method"] = "Direct TD3-PI"
                item["omega_limit_violation"] = item.get("omega_limit_violation", 0)
                item["hard_safety_violation"] = item.get("hard_safety_violation", 0)
                rows.append(item)
                seen.add(key)
        state.source(path)
    return rows


def method_scenario_value(rows: list[dict[str, Any]], method: str, scenario: str, column_name: str) -> float:
    for row in rows:
        if row.get("method") == method and row.get("scenario") == scenario:
            return safe_float(row.get(column_name))
    return float("nan")


def generate_fixed_stats(rows: list[dict[str, Any]], state: AuditState) -> None:
    methods = [method for method in MAIN_METHODS if any(row.get("method") == method for row in rows)]
    x = np.arange(len(SCENARIOS), dtype=float)
    width = min(0.78 / max(1, len(methods)), 0.18)
    fig, ax = plt.subplots(figsize=(11.0, 5.3))
    for idx, method in enumerate(methods):
        values = [method_scenario_value(rows, method, scenario, "rmse_theta_deg") for scenario in SCENARIOS]
        ax.bar(
            x + (idx - (len(methods) - 1) / 2) * width,
            values,
            width=width,
            color=METHOD_COLORS.get(method),
            label=METHOD_SHORT_LABELS[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_SCENARIOS[s] for s in SCENARIOS])
    ax.set_ylabel("RMSE [deg]")
    ax.set_xlabel("Scenario")
    ax.set_title("Fixed-scenario tracking RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    save_fig(fig, PAPER_FIGURES / "fig_fixed_rmse_grouped_bar.png", state)

    averages = []
    for method in methods:
        vals = finite(row.get("rmse_theta_deg") for row in rows if row.get("method") == method)
        averages.append(float(np.mean(vals)) if vals.size else float("nan"))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar([METHOD_SHORT_LABELS[m] for m in methods], averages, color=[METHOD_COLORS.get(m) for m in methods])
    ax.set_ylabel("Average RMSE [deg]")
    ax.set_title("Average fixed-scenario RMSE")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, PAPER_FIGURES / "fig_fixed_average_rmse_bar.png", state)

    metrics = [
        ("flag_U_safe_count", "flag_U"),
        ("flag_E_safe_count", "flag_E"),
        ("flag_X_safe_count", "flag_X"),
        ("sigma_safe_count", "sigma"),
        ("omega_limit_violation", "omega limit"),
        ("hard_safety_violation", "hard safety"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(methods), dtype=float)
    width = 0.12
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(metrics)))
    for idx, (metric, label) in enumerate(metrics):
        values = [
            sum(safe_float(row.get(metric), 0.0) for row in rows if row.get("method") == method)
            for method in methods
        ]
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, values, width=width, color=colors[idx], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_SHORT_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Total count")
    ax.set_title("Fixed-scenario safety flags and violations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    save_fig(fig, PAPER_FIGURES / "fig_fixed_safety_flags_bar.png", state)


def load_mc_rows(state: AuditState) -> list[dict[str, Any]]:
    rows = []
    for row in read_rows(MC_COMPARISON):
        method = MC_METHOD_MAP.get(str(row.get("method", "")))
        if method:
            item = dict(row)
            item["method"] = method
            rows.append(item)
    state.source(MC_COMPARISON)
    return rows


def generate_mc_stats(rows: list[dict[str, Any]], state: AuditState) -> None:
    methods = [method for method in MAIN_METHODS if any(row.get("method") == method for row in rows)]
    labels = [METHOD_SHORT_LABELS.get(method, method) for method in methods]
    colors = [METHOD_COLORS.get(method, "#4C78A8") for method in methods]
    lookup = {row["method"]: row for row in rows}

    def simple_bar(column_name: str, ylabel: str, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.bar(labels, [safe_float(lookup[m].get(column_name), 0.0) for m in methods], color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        save_fig(fig, path, state)

    simple_bar("mean_rmse_theta_deg", "Mean RMSE [deg]", "Monte Carlo mean RMSE", PAPER_FIGURES / "fig_mc_mean_rmse_bar.png")
    simple_bar("success_rate", "Success rate", "Monte Carlo success rate", PAPER_FIGURES / "fig_mc_success_rate_bar.png")
    simple_bar("mean_flag_U_safe_count", "Mean flag_U count", "Monte Carlo command safety intervention", PAPER_FIGURES / "fig_mc_flag_U_bar.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(methods), dtype=float)
    width = 0.30
    ax.bar(
        x - width / 2,
        [safe_float(lookup[m].get("omega_limit_violation_count"), 0.0) for m in methods],
        width=width,
        label="omega limit",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        [safe_float(lookup[m].get("hard_safety_violation_count"), 0.0) for m in methods],
        width=width,
        label="hard safety",
        color="#E45756",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Violation count")
    ax.set_title("Monte Carlo safety violations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, PAPER_FIGURES / "fig_mc_safety_violation_bar.png", state)
    for optional in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count"):
        if all(optional not in row for row in rows):
            state.warn(f"Monte Carlo final comparison lacks {optional}; MC safety figures use flag_U and violation counts.")


def mc_episode_rows(state: AuditState) -> dict[str, list[dict[str, str]]]:
    data = {
        "PID/PD-PI": read_rows(PID_MC_EPISODES),
        "SMC-PI": read_rows(SMC_MC_EPISODES),
        "Randomized MC-SC-Residual-TD3-PI": [],
    }
    state.source(PID_MC_EPISODES)
    state.source(SMC_MC_EPISODES)
    for seed in range(3):
        path = RUNS / f"gun_servo_mc_sc_residual_td3_random_v8_full_seed{seed}" / "summaries" / "mc_eval_episode_metrics.csv"
        rows = read_rows(path)
        for row in rows:
            item = dict(row)
            item["source_seed"] = str(seed)
            data["Randomized MC-SC-Residual-TD3-PI"].append(item)
        state.source(path)
    return data


def generate_mc_scenario_boxplot(data: dict[str, list[dict[str, str]]], state: AuditState) -> None:
    methods = [method for method in ("PID/PD-PI", "SMC-PI", "Randomized MC-SC-Residual-TD3-PI") if data.get(method)]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    positions: list[float] = []
    box_data: list[np.ndarray] = []
    box_colors: list[str] = []
    width = 0.18
    for sidx, scenario in enumerate(MC_SCENARIOS):
        for midx, method in enumerate(methods):
            vals = finite(row.get("rmse_theta_deg") for row in data.get(method, []) if row.get("scenario") == scenario)
            if vals.size:
                positions.append(sidx + 1 + (midx - (len(methods) - 1) / 2) * width)
                box_data.append(vals)
                box_colors.append(METHOD_COLORS.get(method, "#4C78A8"))
    if not box_data:
        plt.close(fig)
        state.warn("Monte Carlo scenario-family RMSE boxplot skipped because episode-level RMSE values were unavailable.")
        return
    boxes = ax.boxplot(box_data, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(MC_SCENARIOS) + 1))
    ax.set_xticklabels(["step", "trapezoid", "sine", "disturbance"])
    ax.set_xlabel("Scenario family")
    ax.set_ylabel("RMSE [deg]")
    ax.set_title("Monte Carlo scenario-family RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=7, alpha=0.75) for m in methods]
    ax.legend(handles, [METHOD_SHORT_LABELS[m] for m in methods], frameon=False, loc="upper right")
    fig.tight_layout()
    save_fig(fig, PAPER_FIGURES / "fig_mc_scenario_rmse_boxplot.png", state)


def generate_training_figures(state: AuditState) -> None:
    logs: dict[str, list[dict[str, str]]] = {}
    for seed in range(3):
        path = RUNS / f"gun_servo_mc_sc_residual_td3_random_v8_full_seed{seed}" / "train_log.csv"
        rows = read_rows(path)
        if rows:
            logs[f"seed{seed}"] = rows
            state.source(path)
    if not logs:
        state.warn("proposed training diagnostics skipped because seed train logs are missing.")
        return

    def line_plot(column_name: str, ylabel: str, title: str, path: Path, *, fallback: list[str] | None = None) -> None:
        candidates = [column_name] + (fallback or [])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for seed, rows in logs.items():
            x = column(rows, "global_steps", "step")
            y = np.full(len(rows), np.nan)
            for candidate in candidates:
                y = column(rows, candidate)
                if np.isfinite(y).any():
                    break
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                ax.plot(x[mask], y[mask], linewidth=1.25, label=seed)
        if not ax.lines:
            plt.close(fig)
            state.warn(f"{path.name} skipped because {column_name} was not available.")
            return
        ax.set_xlabel("Global steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, path, state)

    line_plot("episode_return", "Episode return", "Training return curves of Randomized MC-SC-Residual-TD3-PI", TRAINING / "fig_training_return_all_seeds.png")
    line_plot("eval_rmse_theta_deg", "Fixed evaluation average RMSE [deg]", "Fixed-evaluation RMSE during training", TRAINING / "fig_eval_rmse_all_seeds.png", fallback=["eval_rmse_all", "eval_rmse_theta"])
    line_plot("actor_loss", "Actor loss", "Actor loss during training", TRAINING / "fig_actor_loss_all_seeds.png")
    line_plot("critic_loss", "Critic loss", "Critic loss during training", TRAINING / "fig_critic_loss_all_seeds.png")


def direct_td3_complete() -> bool:
    spec = METHODS["Direct TD3-PI"]
    return all(spec.trace_path(scenario).exists() for scenario in SCENARIOS)


def final_index_rows(state: AuditState, overlay_paths: list[Path]) -> list[dict[str, Any]]:
    main_paths = {
        TRAINING / "fig_training_return_all_seeds.png",
        TRAINING / "fig_eval_rmse_all_seeds.png",
        REPRESENTATIVE / "fig_C1_position_comparison.png",
        REPRESENTATIVE / "fig_C1_error_comparison.png",
        REPRESENTATIVE / "fig_C3_position_comparison.png",
        REPRESENTATIVE / "fig_C4a_position_comparison.png",
        REPRESENTATIVE / "fig_C5a_disturbance_comparison.png",
        PAPER_FIGURES / "fig_fixed_rmse_grouped_bar.png",
        PAPER_FIGURES / "fig_fixed_average_rmse_bar.png",
        PAPER_FIGURES / "fig_mc_mean_rmse_bar.png",
        PAPER_FIGURES / "fig_mc_success_rate_bar.png",
        PAPER_FIGURES / "fig_mc_scenario_rmse_boxplot.png",
        RANDOMIZATION_DIR / "fig_parameter_randomization_ranges.png",
        DISTURBANCE_DIR / "fig_disturbance_recovery_time_thresholds.png",
    }
    candidates = sorted(
        set(PAPER_FIGURES.rglob("*.png")) | set(state.packaged) | set(overlay_paths),
        key=lambda path: str(path),
    )
    rows = []
    for idx, path in enumerate(candidates, start=1):
        name = path.stem
        scenario = scenario_from_name(path.name)
        if path in main_paths:
            section = "main text"
            use_main = "yes"
        elif "by_method" in path.parts:
            section = "appendix by-method figure package"
            use_main = "no"
        else:
            section = "appendix"
            use_main = "no"
        rows.append(
            {
                "figure_id": f"F{idx:03d}",
                "figure_title": name.replace("_", " "),
                "file_path": str(path),
                "source_run": source_run_for_path(path),
                "source_data": source_data_for_path(path),
                "methods_included": methods_for_path(path),
                "scenario": scenario,
                "recommended_section": section,
                "use_in_main_text": use_main,
                "notes": "Direct TD3 complete fixed trajectories available" if "comparison" in name and direct_td3_complete() else "",
            }
        )
    return rows


def source_run_for_path(path: Path) -> str:
    if "by_method" in path.parts:
        stem = path.name
        for method, spec in METHODS.items():
            if stem.startswith(spec.short + "_"):
                return str(spec.run_dir)
    if "representative" in path.parts:
        return "multiple"
    if "training_diagnostics" in path.parts:
        return "v8_full_seed0/seed1/seed2 train logs"
    if path.name.startswith("fig_mc") or path.name.startswith("fig_fixed"):
        return str(RUNS / "final_method_comparison")
    return "derived"


def source_data_for_path(path: Path) -> str:
    if "by_method" in path.parts:
        for method, spec in METHODS.items():
            if path.name.startswith(spec.short + "_"):
                scenario = scenario_from_name(path.name)
                return str(trace_path_for(method, scenario))
    if "training_diagnostics" in path.parts:
        return "train_log.csv for seed0, seed1, seed2"
    if "randomization" in path.parts:
        return str(ENV_RANDOMIZATION_CONFIG)
    if path.name.startswith("fig_mc_scenario"):
        return "PID/SMC/v8 MC episode metrics"
    if path.name.startswith("fig_mc") or path.name.startswith("fig_fixed"):
        return "final comparison summary CSVs"
    return "trajectory and summary CSVs"


def methods_for_path(path: Path) -> str:
    if "by_method" in path.parts:
        for method, spec in METHODS.items():
            if path.name.startswith(spec.short + "_"):
                return method
    if "representative" in path.parts or path.name.startswith("fig_fixed"):
        return "; ".join(MAIN_METHODS)
    if path.name.startswith("fig_mc"):
        return "PID/PD-PI; SMC-PI; Randomized MC-SC-Residual-TD3-PI"
    if "training" in path.parts or "randomization" in path.parts:
        return "Randomized MC-SC-Residual-TD3-PI"
    return ""


def write_final_report(
    *,
    inventory_count: int,
    required_count: int,
    missing_rows: list[dict[str, Any]],
    state: AuditState,
    overlay_paths: list[Path],
) -> None:
    missing = [row for row in missing_rows if row["current_status"] == "missing"]
    critical = [
        row
        for row in missing
        if row["figure_type"] in {"position", "error", "speed_command", "safety_flags", "mean RMSE comparison"}
    ]
    optional = [row for row in missing if row not in critical]
    direct_complete = direct_td3_complete()
    lines = [
        "# Final Figure Package Report",
        "",
        f"Figures found in inventory: {inventory_count}",
        f"Required figure records: {required_count}",
        f"Regenerated figures: {len(state.regenerated)}",
        f"Copied/packaged paper-ready figures: {len(set(state.packaged))}",
        f"Missing critical figures: {len(critical)}",
        f"Missing optional figures: {len(optional)}",
        f"Direct TD3 complete fixed six-scenario figures: {'yes' if direct_complete else 'no'}",
        f"Direct TD3 can be included in main comparison figures: {'yes' if direct_complete else 'partial only; use in ablation or partial overlays'}",
        "",
        "## Overlay Comparison Figures",
    ]
    lines.extend([f"- {path}" for path in overlay_paths])
    lines.extend(["", "## Missing Critical Figures"])
    lines.extend(
        [
            f"- {row['method']} | {row['scenario']} | {row['figure_type']} | {row['notes']}"
            for row in critical
        ]
        or ["- None."]
    )
    lines.extend(["", "## Missing Optional Figures"])
    lines.extend(
        [
            f"- {row['method']} | {row['scenario']} | {row['figure_type']} | {row['notes']}"
            for row in optional
        ]
        or ["- None."]
    )
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {note}" for note in state.warnings] or ["- No audit warnings."])
    lines.extend(
        [
            "",
            "## Recommended Thesis Main-Text Figures",
            "- figures/training_diagnostics/fig_training_return_all_seeds.png",
            "- figures/training_diagnostics/fig_eval_rmse_all_seeds.png",
            "- figures/representative/fig_C1_position_comparison.png",
            "- figures/representative/fig_C1_error_comparison.png",
            "- figures/representative/fig_C3_position_comparison.png",
            "- figures/representative/fig_C4a_position_comparison.png",
            "- figures/representative/fig_C5a_disturbance_comparison.png",
            "- figures/fig_fixed_rmse_grouped_bar.png",
            "- figures/fig_fixed_average_rmse_bar.png",
            "- figures/fig_mc_mean_rmse_bar.png",
            "- figures/fig_mc_success_rate_bar.png",
            "- figures/fig_mc_scenario_rmse_boxplot.png",
            "- figures/randomization/fig_parameter_randomization_ranges.png",
            "- figures/disturbance/fig_disturbance_recovery_time_thresholds.png",
            "",
            "## Recommended Appendix Figures",
            "- figures/by_method/",
            "- figures/training_diagnostics/fig_actor_loss_all_seeds.png",
            "- figures/training_diagnostics/fig_critic_loss_all_seeds.png",
            "- figures/action_smoothness/",
            "- figures/fig_fixed_safety_flags_bar.png",
            "- figures/fig_mc_safety_violation_bar.png",
            "- figures/fig_mc_flag_U_bar.png",
            "- figures/randomization/fig_parameter_randomization_distribution.png",
            "- figures/disturbance/fig_disturbance_peak_error_after_disturbance.png",
        ]
    )
    (AUDIT_REPORTS / "final_figure_package_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_checklist(missing_rows: list[dict[str, Any]], index_path: Path, state: AuditState) -> None:
    figures = sorted(PAPER_FIGURES.rglob("*.png"), key=lambda path: str(path))
    tables = sorted((PAPER / "tables").glob("*")) if (PAPER / "tables").exists() else []
    missing = [row for row in missing_rows if row["current_status"] == "missing"]
    lines = [
        "# Result File Checklist",
        "",
        "## Generated Tables",
    ]
    lines.extend([f"- {path}" for path in tables] or ["- No generated tables found."])
    lines.extend(["", "## Generated Figures"])
    lines.extend([f"- {path}" for path in figures] or ["- No generated figures found."])
    lines.extend(
        [
            "",
            "## Figure Audit Outputs",
            f"- {AUDIT_SUMMARIES / 'current_figure_inventory.csv'}",
            f"- {AUDIT_SUMMARIES / 'required_figure_matrix.csv'}",
            f"- {AUDIT_SUMMARIES / 'missing_figure_list.csv'}",
            f"- {AUDIT_REPORTS / 'missing_figure_report.md'}",
            f"- {AUDIT_REPORTS / 'final_figure_package_report.md'}",
            f"- {index_path}",
            "",
            "## Direct TD3 Inclusion Recommendation",
            "- Direct TD3-PI now has fixed six-scenario trajectory CSVs and by-method figures.",
            "- Include Direct TD3-PI in fixed-scenario overlays and fixed RMSE/safety figures.",
            "- Do not include Direct TD3-PI in Monte Carlo figures unless Direct TD3 Monte Carlo episode data is added.",
            "",
            "## Missing But Non-Critical Figures",
        ]
    )
    lines.extend(
        [
            f"- {row['method']} | {row['scenario']} | {row['figure_type']} | {row['notes']}"
            for row in missing
        ]
        or ["- None."]
    )
    lines.extend(["", "## Missing Data Notes"])
    lines.extend([f"- {note}" for note in state.warnings] or ["- No missing data warnings."])
    lines.extend(
        [
            "",
            "## Recommended Files For Thesis Main Text",
            "- figures/training_diagnostics/fig_training_return_all_seeds.png",
            "- figures/training_diagnostics/fig_eval_rmse_all_seeds.png",
            "- figures/representative/fig_C1_position_comparison.png",
            "- figures/representative/fig_C1_error_comparison.png",
            "- figures/representative/fig_C3_position_comparison.png",
            "- figures/representative/fig_C4a_position_comparison.png",
            "- figures/representative/fig_C5a_disturbance_comparison.png",
            "- figures/fig_fixed_rmse_grouped_bar.png",
            "- figures/fig_fixed_average_rmse_bar.png",
            "- figures/fig_mc_mean_rmse_bar.png",
            "- figures/fig_mc_success_rate_bar.png",
            "- figures/fig_mc_scenario_rmse_boxplot.png",
            "- figures/randomization/fig_parameter_randomization_ranges.png",
            "- figures/disturbance/fig_disturbance_recovery_time_thresholds.png",
            "",
            "## Recommended Files For Appendix",
            "- figures/by_method/",
            "- figures/training_diagnostics/fig_actor_loss_all_seeds.png",
            "- figures/training_diagnostics/fig_critic_loss_all_seeds.png",
            "- figures/action_smoothness/",
            "- figures/fig_fixed_safety_flags_bar.png",
            "- figures/fig_mc_safety_violation_bar.png",
            "- figures/fig_mc_flag_U_bar.png",
            "- figures/randomization/fig_parameter_randomization_distribution.png",
            "- figures/disturbance/fig_disturbance_peak_error_after_disturbance.png",
        ]
    )
    (PAPER_REPORTS / "result_file_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_matplotlib()
    ensure_dirs()
    state = AuditState()

    inventory = build_inventory(state)
    required = required_rows()
    write_required_matrix(required)

    generate_fixed_by_method(state)
    overlay_paths = generate_overlays(state)
    fixed_rows = load_fixed_rows(state)
    generate_fixed_stats(fixed_rows, state)
    mc_rows = load_mc_rows(state)
    generate_mc_stats(mc_rows, state)
    mc_data = mc_episode_rows(state)
    generate_mc_scenario_boxplot(mc_data, state)
    generate_training_figures(state)

    # Re-run status after packaging/regeneration so the missing list reflects the final package.
    missing_rows = write_missing_outputs(required, state)
    inventory = build_inventory(state)

    index_rows = final_index_rows(state, overlay_paths)
    index_path = AUDIT_MANIFESTS / "final_figure_index.csv"
    write_rows(
        index_path,
        index_rows,
        [
            "figure_id",
            "figure_title",
            "file_path",
            "source_run",
            "source_data",
            "methods_included",
            "scenario",
            "recommended_section",
            "use_in_main_text",
            "notes",
        ],
    )
    write_final_report(
        inventory_count=sum(1 for row in inventory if row["exists"] == "yes"),
        required_count=len(required),
        missing_rows=missing_rows,
        state=state,
        overlay_paths=overlay_paths,
    )
    update_checklist(missing_rows, index_path, state)

    missing_count = sum(1 for row in missing_rows if row["current_status"] == "missing")
    print(f"inventory_figures_found={sum(1 for row in inventory if row['exists'] == 'yes')}")
    print(f"required_figure_records={len(required)}")
    print(f"regenerated_figures={len(state.regenerated)}")
    print(f"packaged_paper_ready_figures={len(set(state.packaged))}")
    print(f"missing_required_records={missing_count}")
    print(f"direct_td3_complete={'yes' if direct_td3_complete() else 'no'}")
    print(f"final_figure_index={index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

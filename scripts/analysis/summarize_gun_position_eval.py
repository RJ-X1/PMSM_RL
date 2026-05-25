"""Summarize gun-servo position evaluation CSV files."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import glob
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import summarize_eval_csv

SUMMARY_FIELDS = [
    "source",
    "algorithm",
    "controller",
    "scenario",
    "seed",
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "overshoot_percent",
    "settling_time_s",
    "recovery_time_s",
    "max_abs_iq_a",
    "max_abs_Te_nm",
    "max_abs_Tout_nm",
    "torque_usage_peak",
    "action_variation_sum",
    "constraint_violation_count",
    "episode_return",
    "done_reason",
    "episode_length",
    "rmse_theta",
    "mae_theta",
    "max_abs_theta_error",
    "steady_state_error_deg",
    "rmse_omega",
    "max_iq",
    "control_energy",
    "overshoot",
    "settling_time",
    "disturbance_recovery_time",
    "speed_saturation_count",
    "current_saturation_count",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", "--input-glob", dest="glob", nargs="+", required=True, help="Input CSV glob(s)")
    parser.add_argument("--output", "--output-csv", dest="output", type=Path, required=True)
    return parser


def expand_globs(patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    csvs = sorted(
        {
            path
            for path in matches
            if path.lower().endswith(".csv") and "summary" not in Path(path).stem.lower()
        }
    )
    if not csvs:
        raise FileNotFoundError(f"No CSV files found for patterns: {patterns}")
    return csvs


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: object, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _first_row_value(rows: list[dict[str, str]], column: str, default: str) -> str:
    if not rows:
        return default
    value = rows[0].get(column, "")
    return str(value) if value not in (None, "") else default


def _last_window_mean_abs(rows: list[dict[str, str]], column: str, *, fraction: float = 0.1) -> float:
    if not rows or column not in rows[0]:
        return float("nan")
    count = max(1, int(math.ceil(len(rows) * fraction)))
    values = [_to_float(row.get(column)) for row in rows[-count:]]
    finite_values = [abs(value) for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))


def _sum_flag(rows: list[dict[str, str]], column: str) -> int:
    if not rows or column not in rows[0]:
        return 0
    return int(sum(1 for row in rows if _to_float(row.get(column), default=0.0) > 0.5))


def _series(rows: list[dict[str, str]], *columns: str) -> list[float]:
    for column in columns:
        if rows and column in rows[0]:
            return [_to_float(row.get(column)) for row in rows]
    return []


def _max_abs(values: list[float]) -> float:
    finite = [abs(value) for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def _action_variation_sum(rows: list[dict[str, str]]) -> float:
    delta = _series(rows, "delta_a_safe")
    finite_delta = [abs(value) for value in delta if math.isfinite(value)]
    if finite_delta:
        return float(sum(finite_delta))
    action = _series(rows, "a_safe", "action_safe")
    finite_action = [value for value in action if math.isfinite(value)]
    if len(finite_action) < 2:
        return float("nan")
    return float(sum(abs(curr - prev) for prev, curr in zip(finite_action, finite_action[1:])))


def summarize_one(path: str) -> dict[str, float | int | str]:
    rows = _read_csv_rows(path)
    summary = summarize_eval_csv(path)
    algorithm = str(_first_row_value(rows, "algorithm", _first_row_value(rows, "controller", "unknown")))
    seed_raw = _first_row_value(rows, "seed", "")
    max_tout = _max_abs(_series(rows, "T_out_Nm", "T_out"))
    tout_limit = _max_abs(_series(rows, "T_out_max_Nm", "gear_output_torque_limit_nm"))
    torque_usage_peak = max_tout / tout_limit if math.isfinite(max_tout) and tout_limit > 0.0 else float("nan")
    constraint_count = max(
        _sum_flag(rows, "constraint_violation"),
        _sum_flag(rows, "sigma_safe"),
        _sum_flag(rows, "flag_X_safe"),
    )
    row: dict[str, float | int | str] = {
        "source": path,
        "algorithm": algorithm,
        "controller": str(summary.get("controller", _first_row_value(rows, "controller", "unknown"))),
        "scenario": str(summary.get("scenario", _first_row_value(rows, "scenario", "default"))),
        "seed": seed_raw,
        "rmse_theta_deg": float(summary.get("rmse_theta", float("nan"))),
        "mae_theta_deg": float(summary.get("mae_theta", float("nan"))),
        "max_abs_theta_error_deg": float(summary.get("max_abs_theta_error", float("nan"))),
        "overshoot_percent": float(summary.get("overshoot", float("nan"))),
        "settling_time_s": float(summary.get("settling_time", float("nan"))),
        "recovery_time_s": float(summary.get("disturbance_recovery_time", float("nan"))),
        "max_abs_iq_a": float(summary.get("max_iq", _max_abs(_series(rows, "iq_A", "i_q", "iq")))),
        "max_abs_Te_nm": _max_abs(_series(rows, "Te_Nm", "T_e", "Te")),
        "max_abs_Tout_nm": max_tout,
        "torque_usage_peak": torque_usage_peak,
        "action_variation_sum": _action_variation_sum(rows),
        "constraint_violation_count": constraint_count,
        "episode_return": float(summary.get("episode_return", float("nan"))),
        "done_reason": str(summary.get("done_reason", "")),
        "episode_length": int(summary.get("episode_length", len(rows))),
        "rmse_theta": float(summary.get("rmse_theta", float("nan"))),
        "mae_theta": float(summary.get("mae_theta", float("nan"))),
        "max_abs_theta_error": float(summary.get("max_abs_theta_error", float("nan"))),
        "steady_state_error_deg": _last_window_mean_abs(rows, "e_theta_deg"),
        "rmse_omega": float(summary.get("rmse_omega", float("nan"))),
        "max_iq": float(summary.get("max_iq", float("nan"))),
        "control_energy": float(summary.get("control_energy", float("nan"))),
        "overshoot": float(summary.get("overshoot", float("nan"))),
        "settling_time": float(summary.get("settling_time", float("nan"))),
        "disturbance_recovery_time": float(summary.get("disturbance_recovery_time", float("nan"))),
        "speed_saturation_count": int(summary.get("speed_saturation_count", 0)),
        "current_saturation_count": int(summary.get("current_saturation_count", 0)),
    }
    return row


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = sorted(
        [summarize_one(path) for path in expand_globs(args.glob)],
        key=lambda row: (str(row["scenario"]), str(row["controller"]), str(row["source"])),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved gun-servo summary: {args.output}")


if __name__ == "__main__":
    main()

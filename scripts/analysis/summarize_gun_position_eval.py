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
    "controller",
    "scenario",
    "episode_return",
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
    "constraint_violation_count",
    "done_reason",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", nargs="+", required=True, help="Input CSV glob(s)")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def expand_globs(patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    csvs = sorted({path for path in matches if path.lower().endswith(".csv")})
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


def summarize_one(path: str) -> dict[str, float | int | str]:
    rows = _read_csv_rows(path)
    summary = summarize_eval_csv(path)
    row: dict[str, float | int | str] = {
        "source": path,
        "controller": str(summary.get("controller", _first_row_value(rows, "controller", "unknown"))),
        "scenario": str(summary.get("scenario", _first_row_value(rows, "scenario", "default"))),
        "episode_return": float(summary.get("episode_return", float("nan"))),
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
        "constraint_violation_count": _sum_flag(rows, "constraint_violation"),
        "done_reason": str(summary.get("done_reason", "")),
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

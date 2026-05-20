"""Summarize gun-servo position evaluation CSV files."""

from __future__ import annotations

from pathlib import Path
import argparse
import glob
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import summarize_eval_csv


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


def _last_window_mean_abs(df: pd.DataFrame, column: str, *, fraction: float = 0.1) -> float:
    if column not in df.columns or df.empty:
        return float("nan")
    count = max(1, int(np.ceil(len(df) * fraction)))
    return float(np.mean(np.abs(df[column].tail(count).astype(float))))


def summarize_one(path: str) -> dict[str, float | int | str]:
    df = pd.read_csv(path)
    summary = summarize_eval_csv(path)
    row: dict[str, float | int | str] = {
        "source": path,
        "controller": str(summary.get("controller", df.get("controller", ["unknown"])[0])),
        "scenario": str(summary.get("scenario", df.get("scenario", ["default"])[0])),
        "episode_return": float(summary.get("episode_return", float("nan"))),
        "episode_length": int(summary.get("episode_length", len(df))),
        "rmse_theta": float(summary.get("rmse_theta", float("nan"))),
        "mae_theta": float(summary.get("mae_theta", float("nan"))),
        "max_abs_theta_error": float(summary.get("max_abs_theta_error", float("nan"))),
        "steady_state_error_deg": _last_window_mean_abs(df, "e_theta_deg"),
        "rmse_omega": float(summary.get("rmse_omega", float("nan"))),
        "max_iq": float(summary.get("max_iq", float("nan"))),
        "control_energy": float(summary.get("control_energy", float("nan"))),
        "overshoot": float(summary.get("overshoot", float("nan"))),
        "settling_time": float(summary.get("settling_time", float("nan"))),
        "disturbance_recovery_time": float(summary.get("disturbance_recovery_time", float("nan"))),
        "speed_saturation_count": int(summary.get("speed_saturation_count", 0)),
        "current_saturation_count": int(summary.get("current_saturation_count", 0)),
        "constraint_violation_count": int(df.get("constraint_violation", pd.Series(dtype=float)).fillna(0).sum()),
        "done_reason": str(summary.get("done_reason", "")),
    }
    return row


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = [summarize_one(path) for path in expand_globs(args.glob)]
    out = pd.DataFrame(rows).sort_values(["scenario", "controller", "source"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved gun-servo summary: {args.output}")


if __name__ == "__main__":
    main()

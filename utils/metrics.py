"""Evaluation metrics for exported trajectory CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence


CURRENT_REFERENCE_CANDIDATES = [
    ("i_d", "ref_i_d"),
    ("i_q", "ref_i_q"),
    ("i_a", "ref_i_a"),
    ("i_b", "ref_i_b"),
    ("i_c", "ref_i_c"),
]



def load_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def infer_tracking_pairs(fieldnames: Iterable[str]) -> list[tuple[str, str]]:
    names = set(fieldnames)
    return [(state, ref) for state, ref in CURRENT_REFERENCE_CANDIDATES if state in names and ref in names]



def summarize_eval_csv(path: str | Path) -> dict[str, float | int | str]:
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    fieldnames = rows[0].keys()
    pairs = infer_tracking_pairs(fieldnames)
    summary: dict[str, float | int | str] = {
        "path": str(path),
        "controller": rows[0].get("controller", "unknown"),
        "env_id": rows[0].get("env_id", "unknown"),
        "layout": rows[0].get("layout", "unknown"),
        "steps": len(rows),
        "episode_return": float(rows[-1].get("cum_reward", 0.0) or 0.0),
        "done": int(float(rows[-1].get("done", 0.0) or 0.0)),
        "done_reason": rows[-1].get("done_reason") or rows[-1].get("termination_reason") or "",
        "num_tracking_pairs": len(pairs),
    }
    if not pairs:
        return summary

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    max_abs = 0.0

    for state_name, ref_name in pairs:
        channel_abs = 0.0
        channel_sq = 0.0
        channel_max = 0.0
        for row in rows:
            err = float(row[ref_name]) - float(row[state_name])
            abs_err = abs(err)
            channel_abs += abs_err
            channel_sq += err * err
            channel_max = max(channel_max, abs_err)
            total_abs += abs_err
            total_sq += err * err
            total_count += 1
            max_abs = max(max_abs, abs_err)
        summary[f"mae_{state_name}"] = channel_abs / len(rows)
        summary[f"rmse_{state_name}"] = math.sqrt(channel_sq / len(rows))
        summary[f"max_abs_err_{state_name}"] = channel_max

    summary["mae_all"] = total_abs / total_count
    summary["rmse_all"] = math.sqrt(total_sq / total_count)
    summary["max_abs_err_all"] = max_abs
    return summary


def _to_float(value: float | int | str | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _mean_ignore_nan(values: Iterable[float | int | str | None]) -> float:
    numeric = [x for x in (_to_float(value) for value in values) if not math.isnan(x)]
    if not numeric:
        return float("nan")
    return sum(numeric) / len(numeric)


def aggregate_repeated_eval_summaries(
    summaries: Sequence[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    """Aggregate repeated evaluation summaries for one checkpoint."""
    if not summaries:
        raise ValueError("Expected at least one evaluation summary to aggregate")

    done_count = sum(int(float(summary.get("done", 0.0) or 0.0)) for summary in summaries)
    num_runs = len(summaries)
    success_count = num_runs - done_count

    aggregate: dict[str, float | int | str] = {
        "checkpoint_path": str(summaries[0].get("checkpoint_path", "")),
        "num_runs": num_runs,
        "done_count": done_count,
        "success_count": success_count,
        "success_rate": success_count / num_runs,
        "mean_steps": _mean_ignore_nan(summary.get("steps") for summary in summaries),
        "mean_mae_i_d": _mean_ignore_nan(summary.get("mae_i_d") for summary in summaries),
        "mean_mae_i_q": _mean_ignore_nan(summary.get("mae_i_q") for summary in summaries),
        "mean_mae_all": _mean_ignore_nan(summary.get("mae_all") for summary in summaries),
        "mean_rmse_all": _mean_ignore_nan(summary.get("rmse_all") for summary in summaries),
    }
    return aggregate


def select_best_checkpoint_aggregate(
    aggregates: Sequence[dict[str, float | int | str]],
) -> dict[str, float | int | str] | None:
    """Select best checkpoint by highest success_rate, then lowest mean_mae_all."""
    if not aggregates:
        return None

    def _sort_key(row: dict[str, float | int | str]) -> tuple[float, float, float, str]:
        success_rate = _to_float(row.get("success_rate"))
        mean_mae_all = _to_float(row.get("mean_mae_all"))
        mean_rmse_all = _to_float(row.get("mean_rmse_all"))
        if math.isnan(success_rate):
            success_rate = float("-inf")
        if math.isnan(mean_mae_all):
            mean_mae_all = float("inf")
        if math.isnan(mean_rmse_all):
            mean_rmse_all = float("inf")
        return (-success_rate, mean_mae_all, mean_rmse_all, str(row.get("checkpoint_path", "")))

    return min(aggregates, key=_sort_key)

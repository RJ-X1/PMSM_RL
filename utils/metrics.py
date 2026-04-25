"""Evaluation metrics for exported trajectory CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Canonical summary fields are the project-wide primary contract.
# Compatibility aliases are still emitted where older scripts expect them.
TRACKING_CHANNEL_ALIASES = {
    "i_d": ["i_d_phys", "i_d", "id", "id_actual", "i_d_meas"],
    "ref_i_d": ["ref_i_d_phys", "ref_i_d", "i_d_ref", "id_ref", "i_d_star", "id_star"],
    "i_q": ["i_q_phys", "i_q", "iq", "iq_actual", "i_q_meas"],
    "ref_i_q": ["ref_i_q_phys", "ref_i_q", "i_q_ref", "iq_ref", "i_q_star", "iq_star"],
    "i_a": ["i_a", "ia", "i_a_meas"],
    "ref_i_a": ["ref_i_a", "i_a_ref", "ia_ref"],
    "i_b": ["i_b", "ib", "i_b_meas"],
    "ref_i_b": ["ref_i_b", "i_b_ref", "ib_ref"],
    "i_c": ["i_c", "ic", "i_c_meas"],
    "ref_i_c": ["ref_i_c", "i_c_ref", "ic_ref"],
}
TRACKING_PAIRS = (
    ("i_d", "ref_i_d"),
    ("i_q", "ref_i_q"),
    ("i_a", "ref_i_a"),
    ("i_b", "ref_i_b"),
    ("i_c", "ref_i_c"),
)
TIME_COLUMN_ALIASES = ("time_s", "time", "time_sec", "t")
STEP_COLUMN_ALIASES = ("step", "global_step", "global_steps", "steps")
CONTROL_COLUMN_ALIASES = {
    "u_d": ("u_d", "u_d_cmd", "prev_u_d", "act_u_d"),
    "u_q": ("u_q", "u_q_cmd", "prev_u_q", "act_u_q"),
    "act_u_d": ("act_u_d",),
    "act_u_q": ("act_u_q",),
    "active_Umax": ("active_Umax", "Umax", "u_max"),
    "action_saturated": ("action_saturated",),
}



def load_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def _first_existing(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {str(name).lower(): str(name) for name in fieldnames}
    for candidate in candidates:
        match = lowered.get(str(candidate).lower())
        if match is not None:
            return match
    return None


def infer_tracking_pairs(fieldnames: Iterable[str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    names = list(fieldnames)
    for state_key, ref_key in TRACKING_PAIRS:
        state_col = _first_existing(names, TRACKING_CHANNEL_ALIASES.get(state_key, (state_key,)))
        ref_col = _first_existing(names, TRACKING_CHANNEL_ALIASES.get(ref_key, (ref_key,)))
        if state_col is not None and ref_col is not None:
            pairs.append((state_key, state_col, ref_col))
    return pairs


def _read_series(
    rows: Sequence[dict[str, str]],
    fieldnames: Iterable[str],
    candidates: Iterable[str],
) -> tuple[np.ndarray | None, str | None]:
    column_name = _first_existing(fieldnames, candidates)
    if column_name is None:
        return None, None

    values: list[float] = []
    for row in rows:
        raw = row.get(column_name, "")
        if raw in ("", None):
            return None, column_name
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, column_name
        if not math.isfinite(value):
            return None, column_name
        values.append(value)
    return np.asarray(values, dtype=np.float64), column_name


def _resolve_axis(rows: Sequence[dict[str, str]], fieldnames: Iterable[str]) -> tuple[np.ndarray, float]:
    axis, _ = _read_series(rows, fieldnames, TIME_COLUMN_ALIASES)
    if axis is None:
        axis, _ = _read_series(rows, fieldnames, STEP_COLUMN_ALIASES)
    if axis is None:
        axis = np.arange(len(rows), dtype=np.float64)
        return axis, 1.0
    if axis.size <= 1:
        return axis, 1.0
    diffs = np.diff(axis)
    finite_diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-12)]
    if finite_diffs.size == 0:
        return axis, 1.0
    return axis, float(np.mean(finite_diffs))


def _compute_step_response_metrics(
    *,
    axis: np.ndarray,
    response: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    if response.size == 0 or reference.size == 0 or response.size != reference.size:
        return float("nan"), float("nan")

    tolerance = max(1e-9, 1e-6 * max(1.0, float(np.max(np.abs(reference)))))
    change_indices = np.flatnonzero(np.abs(np.diff(reference)) > tolerance)
    if change_indices.size == 0:
        return float("nan"), float("nan")

    step_index = int(change_indices[0] + 1)
    initial_ref = float(reference[step_index - 1])
    final_ref = float(reference[-1])
    step_amplitude = final_ref - initial_ref

    if np.max(np.abs(reference[:step_index] - initial_ref)) > tolerance:
        return float("nan"), float("nan")
    if np.max(np.abs(reference[step_index:] - final_ref)) > tolerance:
        return float("nan"), float("nan")

    if abs(step_amplitude) <= tolerance:
        overshoot = float("nan")
    else:
        excursion_direction = 1.0 if step_amplitude > 0.0 else -1.0
        overshoot = max(
            0.0,
            float(np.max(excursion_direction * (response[step_index:] - final_ref))),
        )

    settling_band = 0.02 * abs(final_ref)
    if settling_band <= tolerance:
        settling_time = float("nan")
    else:
        within_band = np.abs(response[step_index:] - final_ref) <= settling_band
        settled_from_here = np.logical_and.accumulate(within_band[::-1])[::-1]
        settling_candidates = np.flatnonzero(settled_from_here)
        if settling_candidates.size == 0:
            settling_time = float("nan")
        else:
            settling_time = float(axis[step_index:][settling_candidates[0]] - axis[step_index])

    return float(overshoot), float(settling_time)


def _compute_saturation_count(
    rows: Sequence[dict[str, str]],
    fieldnames: Iterable[str],
) -> int | float:
    action_saturated, _ = _read_series(
        rows,
        fieldnames,
        CONTROL_COLUMN_ALIASES["action_saturated"],
    )
    if action_saturated is not None:
        return int(np.sum(action_saturated > 0.5))

    u_d, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["u_d"])
    u_q, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["u_q"])
    if u_d is None or u_q is None:
        return float("nan")

    act_u_d, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["act_u_d"])
    act_u_q, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["act_u_q"])
    active_umax, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["active_Umax"])
    if act_u_d is not None and act_u_q is not None and active_umax is not None:
        raw_u_d = np.clip(act_u_d, -1.0, 1.0) * active_umax
        raw_u_q = np.clip(act_u_q, -1.0, 1.0) * active_umax
        delta = np.sqrt((raw_u_d - u_d) ** 2 + (raw_u_q - u_q) ** 2)
        tolerance = 1e-6 * np.maximum(1.0, np.abs(active_umax))
        return int(np.sum(delta > tolerance))

    active_limit = (
        active_umax
        if active_umax is not None
        else np.full_like(u_d, fill_value=max(float(np.max(np.abs(u_d))), float(np.max(np.abs(u_q))), 1e-8))
    )
    voltage_norm = np.sqrt(u_d ** 2 + u_q ** 2)
    return int(np.sum(voltage_norm >= (0.98 * active_limit)))



def summarize_eval_csv(path: str | Path) -> dict[str, float | int | str]:
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    fieldnames = list(rows[0].keys())
    pairs = infer_tracking_pairs(fieldnames)
    terminated = int(float(rows[-1].get("terminated", 0.0) or 0.0))
    truncated = int(float(rows[-1].get("truncated", 0.0) or 0.0))
    done = int(float(rows[-1].get("done", float(bool(terminated or truncated))) or 0.0))
    episode_return_raw = rows[-1].get("cum_reward", rows[-1].get("episode_return", 0.0))
    episode_return = float(episode_return_raw or 0.0)
    axis, dt = _resolve_axis(rows, fieldnames)
    summary: dict[str, float | int | str] = {
        "path": str(path),
        "controller": rows[0].get("controller", "unknown"),
        "env_id": rows[0].get("env_id", "unknown"),
        "layout": rows[0].get("layout", "unknown"),
        "scenario": rows[-1].get("scenario") or rows[0].get("scenario") or "default",
        "steps": len(rows),
        "episode_length": len(rows),
        "episode_return": episode_return,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "done_reason": rows[-1].get("done_reason") or rows[-1].get("termination_reason") or "",
        "num_tracking_pairs": len(pairs),
    }
    if not pairs:
        summary["control_energy"] = float("nan")
        summary["action_delta_energy"] = float("nan")
        summary["saturation_count"] = _compute_saturation_count(rows, fieldnames)
        return summary

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    max_abs = 0.0
    total_iae = 0.0

    for state_name, state_col, ref_col in pairs:
        response, _ = _read_series(rows, fieldnames, [state_col])
        reference, _ = _read_series(rows, fieldnames, [ref_col])
        if response is None or reference is None:
            continue

        errors = reference - response
        abs_errors = np.abs(errors)
        channel_abs = float(np.sum(abs_errors))
        channel_sq = float(np.sum(errors ** 2))
        channel_max = float(np.max(abs_errors))
        channel_iae = float(channel_abs * dt)
        overshoot, settling_time = _compute_step_response_metrics(
            axis=axis,
            response=response,
            reference=reference,
        )

        total_abs += channel_abs
        total_sq += channel_sq
        total_count += int(errors.size)
        max_abs = max(max_abs, channel_max)
        total_iae += channel_iae

        summary[f"mae_{state_name}"] = channel_abs / len(rows)
        summary[f"rmse_{state_name}"] = math.sqrt(channel_sq / len(rows))
        summary[f"iae_{state_name}"] = channel_iae
        summary[f"max_abs_error_{state_name}"] = channel_max
        # Legacy alias retained for older summaries; prefer max_abs_error_* above.
        summary[f"max_abs_err_{state_name}"] = channel_max
        summary[f"overshoot_{state_name}"] = overshoot
        summary[f"settling_time_{state_name}"] = settling_time

    summary["mae_all"] = total_abs / total_count
    summary["rmse_all"] = math.sqrt(total_sq / total_count)
    summary["iae_all"] = total_iae
    summary["max_abs_error_all"] = max_abs
    # Legacy alias retained for older summaries; prefer max_abs_error_all above.
    summary["max_abs_err_all"] = max_abs

    u_d, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["u_d"])
    u_q, _ = _read_series(rows, fieldnames, CONTROL_COLUMN_ALIASES["u_q"])
    if u_d is not None and u_q is not None:
        delta_u_d = np.diff(u_d, prepend=u_d[0])
        delta_u_q = np.diff(u_q, prepend=u_q[0])
        summary["control_energy"] = float(np.sum(u_d ** 2 + u_q ** 2) * dt)
        summary["action_delta_energy"] = float(np.sum(delta_u_d ** 2 + delta_u_q ** 2) * dt)
    else:
        summary["control_energy"] = float("nan")
        summary["action_delta_energy"] = float("nan")
    summary["saturation_count"] = _compute_saturation_count(rows, fieldnames)
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

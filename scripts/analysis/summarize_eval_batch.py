from __future__ import annotations

import argparse
import glob
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import summarize_eval_csv


COLUMN_ALIASES = {
    "step": ["global_steps", "global_step", "step", "steps", "episode_steps"],
    "time": ["time", "time_s", "t"],
    "method": ["method", "controller", "agent", "algo", "algorithm", "variant"],
    "seed": ["seed", "random_seed", "run_seed"],
    "scenario": ["scenario", "test_case", "case", "condition", "done_reason"],
    "speed": ["speed_rpm", "omega_m_rpm", "omega_rpm", "omega_me_rpm", "omega_m_phys", "omega_m", "omega"],
    "load": ["load_torque_nm", "load_nm", "torque_load_nm", "load_torque_nm", "load_torque", "T_L", "torque"],
    "i_d": ["i_d_phys", "i_d", "id", "id_actual", "i_d_meas"],
    "i_q": ["i_q_phys", "i_q", "iq", "iq_actual", "i_q_meas"],
    "i_d_ref": ["ref_i_d_phys", "ref_i_d", "i_d_ref", "id_ref", "i_d_star", "id_star"],
    "i_q_ref": ["ref_i_q_phys", "ref_i_q", "i_q_ref", "iq_ref", "i_q_star", "iq_star"],
    "u_d": ["u_d", "ud", "v_d", "u_d_cmd", "prev_u_d", "act_u_d"],
    "u_q": ["u_q", "uq", "v_q", "u_q_cmd", "prev_u_q", "act_u_q"],
}


def infer_method_from_path(path: str | Path) -> str:
    joined = "/".join(part.lower() for part in Path(path).parts)
    if "eval_pi" in joined or "/pi/" in joined:
        return "PI"
    if "smooth" in joined or "cdr" in joined or "ours" in joined or "full" in joined:
        return "Ours"
    if "td3" in joined:
        return "TD3"
    if "ddpg" in joined:
        return "DDPG"
    if "eval_rl" in joined or "/rl/" in joined:
        return "RL"
    return Path(path).parent.name or "Unknown"


def infer_seed_from_path(path: str | Path) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", str(path).lower())
    return int(match.group(1)) if match else None


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def resolve_column(df: pd.DataFrame, canonical_name: str) -> str:
    col = first_existing(df, COLUMN_ALIASES.get(canonical_name, [canonical_name]))
    if col is None:
        raise KeyError(f"Could not resolve '{canonical_name}'. Available columns: {list(df.columns)}")
    return col


def maybe_resolve_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    return first_existing(df, COLUMN_ALIASES.get(canonical_name, [canonical_name]))


def add_metadata(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    out = df.copy()
    method_col = maybe_resolve_column(out, "method")
    if method_col is None:
        out["method"] = infer_method_from_path(path)
    else:
        controller_value = str(out[method_col].iloc[0]).strip().lower()
        if controller_value == "pi":
            out["method"] = "PI"
        elif controller_value == "rl":
            out["method"] = infer_method_from_path(path)
        else:
            out["method"] = out[method_col]
    if maybe_resolve_column(out, "seed") is None:
        out["seed"] = -1 if infer_seed_from_path(path) is None else infer_seed_from_path(path)
    return out


def summarize_one(path: str, scenario_override: str | None) -> dict[str, float | str | int]:
    df = add_metadata(pd.read_csv(path), path)
    speed_col = maybe_resolve_column(df, "speed")
    load_col = maybe_resolve_column(df, "load")
    scenario_col = maybe_resolve_column(df, "scenario")
    metric_summary = summarize_eval_csv(path)

    scenario = scenario_override
    if scenario is None and scenario_col is not None:
        scenario = str(df[scenario_col].iloc[-1])
    if scenario is None:
        scenario = "default"

    canonical_row = {
        "source": path,
        "method": str(df["method"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
        "scenario": scenario,
        "speed_marker": float(df[speed_col].mean()) if speed_col is not None else float("nan"),
        "load_marker": float(df[load_col].mean()) if load_col is not None else float("nan"),
        "rmse_i_d": float(metric_summary.get("rmse_i_d", float("nan"))),
        "rmse_i_q": float(metric_summary.get("rmse_i_q", float("nan"))),
        "rmse_all": float(metric_summary.get("rmse_all", float("nan"))),
        "mae_i_d": float(metric_summary.get("mae_i_d", float("nan"))),
        "mae_i_q": float(metric_summary.get("mae_i_q", float("nan"))),
        "mae_all": float(metric_summary.get("mae_all", float("nan"))),
        "iae_i_d": float(metric_summary.get("iae_i_d", float("nan"))),
        "iae_i_q": float(metric_summary.get("iae_i_q", float("nan"))),
        "iae_all": float(metric_summary.get("iae_all", float("nan"))),
        "max_abs_error_i_d": float(metric_summary.get("max_abs_error_i_d", float("nan"))),
        "max_abs_error_i_q": float(metric_summary.get("max_abs_error_i_q", float("nan"))),
        "max_abs_error_all": float(metric_summary.get("max_abs_error_all", float("nan"))),
        "overshoot_i_d": float(metric_summary.get("overshoot_i_d", float("nan"))),
        "overshoot_i_q": float(metric_summary.get("overshoot_i_q", float("nan"))),
        "settling_time_i_d": float(metric_summary.get("settling_time_i_d", float("nan"))),
        "settling_time_i_q": float(metric_summary.get("settling_time_i_q", float("nan"))),
        "control_energy": float(metric_summary.get("control_energy", float("nan"))),
        "action_delta_energy": float(metric_summary.get("action_delta_energy", float("nan"))),
        "saturation_count": metric_summary.get("saturation_count", float("nan")),
        "episode_return": float(metric_summary.get("episode_return", float("nan"))),
        "episode_length": int(metric_summary.get("episode_length", len(df))),
        "terminated": int(metric_summary.get("terminated", 0)),
        "truncated": int(metric_summary.get("truncated", 0)),
        "done_reason": str(metric_summary.get("done_reason", "")),
    }
    compatibility_aliases = {
        # Compatibility aliases retained for older tables/scripts; prefer the canonical names above.
        "rmse_id": float(metric_summary.get("rmse_i_d", float("nan"))),
        "rmse_iq": float(metric_summary.get("rmse_i_q", float("nan"))),
        "rmse": float(metric_summary.get("rmse_all", float("nan"))),
        "mae_id": float(metric_summary.get("mae_i_d", float("nan"))),
        "mae_iq": float(metric_summary.get("mae_i_q", float("nan"))),
        "iae": float(metric_summary.get("iae_all", float("nan"))),
        "max_abs_error": float(metric_summary.get("max_abs_error_all", float("nan"))),
        "action_smoothness": float(metric_summary.get("action_delta_energy", float("nan"))),
        "steps": int(metric_summary.get("episode_length", len(df))),
    }
    canonical_row.update(compatibility_aliases)
    return canonical_row


def expand_globs(patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    csvs = sorted({path for path in matches if path.lower().endswith('.csv')})
    if not csvs:
        raise FileNotFoundError(f"No CSV files found for patterns: {patterns}")
    return csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PMSM evaluation traces into paper-ready metrics.")
    parser.add_argument("--glob", nargs="+", required=True, help="Input trace glob(s), e.g. outputs/runs/**/eval/*.csv")
    parser.add_argument("--output", required=True, help="Output summary CSV path.")
    parser.add_argument("--scenario", default=None, help="Optional forced scenario label.")
    args = parser.parse_args()

    rows = [summarize_one(path, args.scenario) for path in expand_globs(args.glob)]
    out_df = pd.DataFrame(rows).sort_values(["method", "scenario", "seed", "source"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"saved summary: {out_path}")


if __name__ == "__main__":
    main()

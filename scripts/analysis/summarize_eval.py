from __future__ import annotations

import argparse
import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "step": ["global_steps", "global_step", "step", "steps", "episode_steps"],
    "time": ["time", "time_s", "t"],
    "method": ["method", "controller", "agent", "algo", "algorithm", "variant"],
    "seed": ["seed", "random_seed", "run_seed"],
    "scenario": ["scenario", "test_case", "case", "condition", "done_reason"],
    "speed": ["speed_rpm", "omega_rpm", "omega_me_rpm", "omega_m", "omega"],
    "load": ["load_nm", "torque_load_nm", "load_torque_nm", "load_torque", "T_L", "torque"],
    "i_d": ["i_d", "id", "id_actual", "i_d_meas"],
    "i_q": ["i_q", "iq", "iq_actual", "i_q_meas"],
    "i_d_ref": ["ref_i_d", "i_d_ref", "id_ref", "i_d_star", "id_star"],
    "i_q_ref": ["ref_i_q", "i_q_ref", "iq_ref", "i_q_star", "iq_star"],
    "u_d": ["act_u_d", "u_d", "ud", "v_d", "u_d_cmd", "prev_u_d"],
    "u_q": ["act_u_q", "u_q", "uq", "v_q", "u_q_cmd", "prev_u_q"],
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
    step_col = maybe_resolve_column(df, "step")
    time_col = maybe_resolve_column(df, "time")
    i_d_col = resolve_column(df, "i_d")
    i_q_col = resolve_column(df, "i_q")
    i_d_ref_col = resolve_column(df, "i_d_ref")
    i_q_ref_col = resolve_column(df, "i_q_ref")
    u_d_col = resolve_column(df, "u_d")
    u_q_col = resolve_column(df, "u_q")
    speed_col = maybe_resolve_column(df, "speed")
    load_col = maybe_resolve_column(df, "load")
    scenario_col = maybe_resolve_column(df, "scenario")

    if time_col is not None and len(df) > 1:
        axis = df[time_col].to_numpy(dtype=float)
        dt = float(np.mean(np.diff(axis)))
    elif step_col is not None and len(df) > 1:
        axis = df[step_col].to_numpy(dtype=float)
        dt = float(np.mean(np.diff(axis)))
    else:
        dt = 1.0

    e_d = df[i_d_ref_col].to_numpy(dtype=float) - df[i_d_col].to_numpy(dtype=float)
    e_q = df[i_q_ref_col].to_numpy(dtype=float) - df[i_q_col].to_numpy(dtype=float)
    u_d = df[u_d_col].to_numpy(dtype=float)
    u_q = df[u_q_col].to_numpy(dtype=float)
    err_norm = np.sqrt(e_d ** 2 + e_q ** 2)
    delta_u = np.sqrt(np.diff(u_d, prepend=u_d[0]) ** 2 + np.diff(u_q, prepend=u_q[0]) ** 2)
    sat_threshold = 0.98 * max(float(np.max(np.abs(u_d))), float(np.max(np.abs(u_q))), 1e-8)
    saturation_count = int(np.sum((np.abs(u_d) >= sat_threshold) | (np.abs(u_q) >= sat_threshold)))

    scenario = scenario_override
    if scenario is None and scenario_col is not None:
        scenario = str(df[scenario_col].iloc[-1])
    if scenario is None:
        scenario = "default"

    return {
        "source": path,
        "method": str(df["method"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
        "scenario": scenario,
        "speed_marker": float(df[speed_col].mean()) if speed_col is not None else float("nan"),
        "load_marker": float(df[load_col].mean()) if load_col is not None else float("nan"),
        "rmse_id": float(np.sqrt(np.mean(e_d ** 2))),
        "rmse_iq": float(np.sqrt(np.mean(e_q ** 2))),
        "rmse": float(np.sqrt(np.mean(err_norm ** 2))),
        "mae_id": float(np.mean(np.abs(e_d))),
        "mae_iq": float(np.mean(np.abs(e_q))),
        "iae": float(np.sum(np.abs(err_norm)) * dt),
        "max_abs_error": float(np.max(np.abs(err_norm))),
        "control_energy": float(np.sum(u_d ** 2 + u_q ** 2) * dt),
        "action_smoothness": float(np.sum(delta_u ** 2) * dt),
        "saturation_count": saturation_count,
        "steps": int(len(df)),
    }


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

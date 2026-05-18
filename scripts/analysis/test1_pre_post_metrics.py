# Time : 2026/4/28 14:14
# Author : 熊
# File : test1_pre_post_metrics
# Software : PyCharm

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STEP_TIME_S = 0.02


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def compute_segment_metrics(df: pd.DataFrame, label: str, segment_name: str, mask: pd.Series) -> dict:
    seg = df.loc[mask].copy()
    if seg.empty:
        return {
            "run": label,
            "segment": segment_name,
            "segment_length": 0,
        }

    i_d = _safe_col(seg, "i_d_phys")
    i_q = _safe_col(seg, "i_q_phys")
    ref_i_d = _safe_col(seg, "ref_i_d_phys")
    ref_i_q = _safe_col(seg, "ref_i_q_phys")

    e_d = ref_i_d - i_d
    e_q = ref_i_q - i_q

    u_d = _safe_col(seg, "u_d")
    u_q = _safe_col(seg, "u_q")
    action_saturated = _safe_col(seg, "action_saturated")

    rmse_i_d = float(np.sqrt(np.mean(np.square(e_d))))
    rmse_i_q = float(np.sqrt(np.mean(np.square(e_q))))
    rmse_all = float(np.sqrt(np.mean(np.concatenate([np.square(e_d), np.square(e_q)]))))

    mae_i_d = float(np.mean(np.abs(e_d)))
    mae_i_q = float(np.mean(np.abs(e_q)))
    mae_all = float(np.mean(np.concatenate([np.abs(e_d), np.abs(e_q)])))

    return {
        "run": label,
        "segment": segment_name,
        "segment_length": int(len(seg)),
        "time_start": float(seg["time_s"].iloc[0]) if "time_s" in seg.columns else np.nan,
        "time_end": float(seg["time_s"].iloc[-1]) if "time_s" in seg.columns else np.nan,
        "rmse_i_d": rmse_i_d,
        "rmse_i_q": rmse_i_q,
        "rmse_all": rmse_all,
        "mae_i_d": mae_i_d,
        "mae_i_q": mae_i_q,
        "mae_all": mae_all,
        "final_error_i_d": float(e_d.iloc[-1]),
        "final_error_i_q": float(e_q.iloc[-1]),
        "mean_abs_u_d": float(np.mean(np.abs(u_d))),
        "mean_abs_u_q": float(np.mean(np.abs(u_q))),
        "max_abs_u_d": float(np.max(np.abs(u_d))),
        "max_abs_u_q": float(np.max(np.abs(u_q))),
        "saturation_count": int(np.sum(action_saturated > 0.5)),
    }


def analyze_file(path: Path, label: str) -> list[dict]:
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise KeyError(f"{path} does not contain time_s column")

    time_s = pd.to_numeric(df["time_s"], errors="coerce")

    return [
        compute_segment_metrics(df, label, "pre_step", time_s < STEP_TIME_S),
        compute_segment_metrics(df, label, "post_step", time_s >= STEP_TIME_S),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Test-1 eval metrics into pre/post step segments.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Pairs of label=csv_path, e.g. td3_500k_best=outputs/.../eval.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/diagnostics/test1_pre_post_metrics.csv"),
    )
    args = parser.parse_args()

    rows: list[dict] = []

    for item in args.input:
        if "=" not in item:
            raise ValueError(f"Input must be label=path, got: {item}")
        label, raw_path = item.split("=", 1)
        rows.extend(analyze_file(Path(raw_path), label))

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved: {args.output}")
    print(out_df.to_string(index=False))

    print("\nKey diagnosis:")
    for label in out_df["run"].unique():
        sub = out_df[out_df["run"] == label]
        pre = sub[sub["segment"] == "pre_step"]
        post = sub[sub["segment"] == "post_step"]
        if not pre.empty and not post.empty:
            print(
                f"- {label}: pre_rmse_all={pre['rmse_all'].iloc[0]:.4f}, "
                f"post_rmse_all={post['rmse_all'].iloc[0]:.4f}"
            )


if __name__ == "__main__":
    main()
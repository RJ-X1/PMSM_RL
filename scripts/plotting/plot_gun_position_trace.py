"""Plot gun-servo position, speed, action, and torque traces."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser


def _series(df: pd.DataFrame, name: str, fallback: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series([fallback] * len(df), index=df.index, dtype=float)


def main() -> None:
    args = build_arg_parser().parse_args()
    df = pd.read_csv(args.input)
    x = _series(df, "time_s") if "time_s" in df.columns else _series(df, "step")
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(x, _series(df, "theta_ref_deg"), "k--", linewidth=1.2, label="theta_ref")
    axes[0].plot(x, _series(df, "theta_L_deg"), label="theta_L")
    if "theta_meas_deg" in df.columns:
        axes[0].plot(x, _series(df, "theta_meas_deg"), alpha=0.6, label="theta_meas")
    axes[1].plot(x, _series(df, "e_theta_deg"), label="e_theta")
    axes[2].plot(x, _series(df, "omega_L_deg_s"), label="omega_L")
    axes[2].plot(x, _series(df, "omega_cmd_deg_s"), "--", label="omega_cmd")
    axes[3].plot(x, _series(df, "act_delta_omega"), label="action")
    if "action_safe" in df.columns:
        axes[3].plot(x, _series(df, "action_safe"), "--", alpha=0.7, label="action_safe")
    axes[4].plot(x, _series(df, "Te_Nm"), label="T_motor")
    if "T_out_Nm" in df.columns:
        axes[4].plot(x, _series(df, "T_out_Nm"), label="T_out")
    axes[4].plot(x, _series(df, "disturbance_torque_Nm"), "--", label="disturbance")

    labels = ["theta [deg]", "error [deg]", "omega [deg/s]", "action [-]", "torque [Nm]"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]" if "time_s" in df.columns else "step")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()

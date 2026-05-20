"""Plot PID/TD3/proposed gun-servo position-control comparison CSVs."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="label=csv_path pairs")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_pair(item: str) -> tuple[str, pd.DataFrame]:
    if "=" not in item:
        raise ValueError(f"Expected label=csv_path, got {item!r}")
    label, raw_path = item.split("=", 1)
    return label, pd.read_csv(raw_path)


def main() -> None:
    args = build_arg_parser().parse_args()
    traces = [_load_pair(item) for item in args.input]
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    for label, df in traces:
        x = df["time_s"] if "time_s" in df.columns else df["step"]
        axes[0].plot(x, df["theta_L_deg"], label=label)
        axes[1].plot(x, df["e_theta_deg"], label=label)
        axes[2].plot(x, df["omega_L_deg_s"], label=f"{label} omega_L")
        if "omega_cmd_deg_s" in df.columns:
            axes[2].plot(x, df["omega_cmd_deg_s"], linestyle="--", alpha=0.6, label=f"{label} omega_cmd")
        axes[3].plot(x, df["iq_A"], label=label)
        axes[4].plot(x, df["disturbance_torque_Nm"], label=label)
    if traces:
        ref_df = traces[0][1]
        x_ref = ref_df["time_s"] if "time_s" in ref_df.columns else ref_df["step"]
        axes[0].plot(x_ref, ref_df["theta_ref_deg"], "k--", linewidth=1.2, label="theta_ref")
    labels = ["theta [deg]", "e_theta [deg]", "omega [deg/s]", "iq [A]", "disturbance [Nm]"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]" if "time_s" in traces[0][1].columns else "step")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()

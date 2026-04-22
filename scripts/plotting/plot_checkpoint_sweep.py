"""Plot aggregate checkpoint sweep metrics for one run."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.run_layout import infer_run_dir_from_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot aggregate checkpoint sweep metrics")
    parser.add_argument("aggregate_summary_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _default_output_path(aggregate_summary_csv: Path) -> Path:
    run_dir = infer_run_dir_from_path(aggregate_summary_csv)
    if run_dir is not None:
        return run_dir / "figures" / "checkpoint_sweep.png"
    return aggregate_summary_csv.with_name("checkpoint_sweep.png")


def main() -> None:
    args = build_arg_parser().parse_args()
    with args.aggregate_summary_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {args.aggregate_summary_csv}")

    labels = [Path(row["checkpoint_path"]).stem for row in rows]
    success_rate = np.asarray([float(row["success_rate"]) for row in rows], dtype=np.float32)
    mean_mae_all = np.asarray([float(row["mean_mae_all"]) for row in rows], dtype=np.float32)
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - 0.2, success_rate, width=0.4, label="success_rate", color="tab:green", alpha=0.75)
    ax1.set_ylabel("success_rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x + 0.2, mean_mae_all, label="mean_mae_all", color="tab:red", marker="o", linewidth=1.8)
    ax2.set_ylabel("mean_mae_all")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()
    output_path = args.output or _default_output_path(args.aggregate_summary_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved plot: {output_path}")


if __name__ == "__main__":
    main()

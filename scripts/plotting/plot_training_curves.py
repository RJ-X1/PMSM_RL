"""Plot training curves from a run train_log.csv file."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
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
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.csv")
    parser.add_argument("train_log_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _default_output_path(train_log_csv: Path) -> Path:
    run_dir = infer_run_dir_from_path(train_log_csv)
    if run_dir is not None:
        return run_dir / "figures" / "training_curves.png"
    return train_log_csv.with_name("training_curves.png")


def main() -> None:
    args = build_arg_parser().parse_args()
    with args.train_log_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {args.train_log_csv}")

    episodes = np.asarray([float(row["episode"]) for row in rows], dtype=np.float32)
    returns = np.asarray([float(row["episode_return"]) for row in rows], dtype=np.float32)
    global_steps = np.asarray([float(row["global_steps"]) for row in rows], dtype=np.float32)
    actor_losses = np.asarray([float(row["avg_actor_loss"]) for row in rows], dtype=np.float32)
    critic_losses = np.asarray([float(row["avg_critic_loss"]) for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(episodes, returns, label="episode_return", color="tab:blue", linewidth=1.8)
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(global_steps, actor_losses, label="avg_actor_loss", linewidth=1.6)
    axes[1].plot(global_steps, critic_losses, label="avg_critic_loss", linewidth=1.6)
    axes[1].set_xlabel("global_steps")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    output_path = args.output or _default_output_path(args.train_log_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved plot: {output_path}")


if __name__ == "__main__":
    main()

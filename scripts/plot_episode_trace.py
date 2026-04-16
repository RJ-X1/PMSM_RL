"""Plot one evaluation episode trace with a task-aware CLI."""

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

from envs.obs_parser import required_eval_trace_columns
from utils.run_layout import infer_run_dir_from_path


SUPPORTED_TASKS = ("pmsm_cc",)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one evaluation episode trace from CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--task", default="pmsm_cc", choices=SUPPORTED_TASKS)
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path")
    return parser


def _default_output_path(csv_path: Path) -> Path:
    run_dir = infer_run_dir_from_path(csv_path)
    if run_dir is not None:
        return run_dir / "figures" / f"{csv_path.stem}_trace.png"
    return csv_path.with_name(f"{csv_path.stem}_trace.png")


def _load_eval_trace(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], int | None, str]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in required_eval_trace_columns() if name not in fieldnames]
        if missing:
            raise KeyError(f"CSV is missing required trace columns: {missing}")

        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV contains no data rows: {csv_path}")

    x = np.asarray(
        [float(row["step"]) if row.get("step", "") != "" else float(idx) for idx, row in enumerate(rows)],
        dtype=np.float32,
    )
    data = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        for name in required_eval_trace_columns()
    }

    done_step: int | None = None
    done_reason = ""
    for row in rows:
        if row.get("done", "") == "1":
            done_step = int(float(row["step"])) if row.get("step", "") != "" else None
            done_reason = row.get("done_reason") or row.get("termination_reason") or ""
            break
    return x, data, done_step, done_reason


def create_episode_trace_plot(*, csv_path: Path, output_path: Path, task: str = "pmsm_cc", title_suffix: str = "") -> Path:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    x, data, done_step, done_reason = _load_eval_trace(csv_path)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(x, data["i_d"], label="i_d", color="tab:blue", linewidth=1.8)
    axes[0].plot(x, data["ref_i_d"], label="ref_i_d", color="tab:orange", linestyle="--", linewidth=1.6)
    axes[0].set_ylabel("d current")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, data["i_q"], label="i_q", color="tab:green", linewidth=1.8)
    axes[1].plot(x, data["ref_i_q"], label="ref_i_q", color="tab:red", linestyle="--", linewidth=1.6)
    axes[1].set_ylabel("q current")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, data["act_u_a"], label="act_u_a", linewidth=1.5)
    axes[2].plot(x, data["act_u_b"], label="act_u_b", linewidth=1.5)
    axes[2].plot(x, data["act_u_c"], label="act_u_c", linewidth=1.5)
    axes[2].set_ylabel("abc action")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, data["epsilon"], label="epsilon", color="tab:purple", linewidth=1.8)
    axes[3].set_ylabel("epsilon")
    axes[3].set_xlabel("step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    title = csv_path.name
    if done_step is not None:
        for ax in axes:
            ax.axvline(done_step, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.8)
        if done_reason:
            title = f"{title} | done_reason={done_reason}"
        else:
            title = f"{title} | done_step={done_step}"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    axes[0].set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output or _default_output_path(args.csv_path)
    saved_path = create_episode_trace_plot(
        csv_path=args.csv_path,
        output_path=output_path,
        task=str(args.task),
    )
    print(f"saved plot: {saved_path}")


if __name__ == "__main__":
    main()

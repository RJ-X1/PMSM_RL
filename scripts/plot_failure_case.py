"""Plot a failure-case episode trace with an explicit failure-focused output name."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

from scripts.plot_episode_trace import SUPPORTED_TASKS, create_episode_trace_plot
from utils.run_layout import infer_run_dir_from_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a failure-case evaluation trace")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--task", default="pmsm_cc", choices=SUPPORTED_TASKS)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _default_output_path(csv_path: Path) -> Path:
    run_dir = infer_run_dir_from_path(csv_path)
    if run_dir is not None:
        return run_dir / "figures" / f"{csv_path.stem}_failure.png"
    return csv_path.with_name(f"{csv_path.stem}_failure.png")


def main() -> None:
    args = build_arg_parser().parse_args()
    with args.csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {args.csv_path}")
    if rows[-1].get("done", "") != "1":
        raise ValueError(f"Expected a terminating/failure case CSV, got done={rows[-1].get('done', '')}")

    output_path = args.output or _default_output_path(args.csv_path)
    saved_path = create_episode_trace_plot(
        csv_path=args.csv_path,
        output_path=output_path,
        task=str(args.task),
        title_suffix="failure_case",
    )
    print(f"saved plot: {saved_path}")


if __name__ == "__main__":
    main()

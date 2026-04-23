"""Batch-evaluate checkpoints with repeated seeded rollouts and aggregate ranking."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

from scripts.core.evaluate import run_evaluation
from utils.config import parse_env_config
from utils.metrics import (
    aggregate_repeated_eval_summaries,
    select_best_checkpoint_aggregate,
    summarize_eval_csv,
)
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_PI_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
)
from utils.run_layout import ensure_run_layout, make_run_layout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-evaluate all checkpoints in a directory")
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--per-run-summary-csv", type=Path, default=None)
    parser.add_argument("--aggregate-summary-csv", type=Path, default=None)
    parser.add_argument("--eval-csv-dir", type=Path, default=None)
    return parser


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    if stem == "checkpoint_latest":
        return (2, 0, stem)
    if stem.startswith("checkpoint_ep_"):
        try:
            episode_idx = int(stem.split("checkpoint_ep_", 1)[1])
        except ValueError:
            episode_idx = 10**9
        return (1, episode_idx, stem)
    return (0, 0, stem)


def _resolve_dirs(path: Path) -> tuple[Path, Path]:
    if (path / "checkpoints").is_dir():
        return path, path / "checkpoints"
    if path.is_dir():
        return path.parent, path
    raise FileNotFoundError(f"Checkpoint directory not found: {path}")


def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    return sorted((path for path in checkpoint_dir.glob("*.pt") if path.is_file()), key=_checkpoint_sort_key)


def _make_eval_csv_path(eval_csv_dir: Path, checkpoint_path: Path, repeat_idx: int, seed: int) -> Path:
    return eval_csv_dir / checkpoint_path.stem / f"run_{repeat_idx:02d}_seed_{seed}.csv"


def _safe_str(value: object) -> str:
    return "" if value is None else str(value)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be >= 1")

    run_dir, checkpoint_dir = _resolve_dirs(args.checkpoint_dir)
    env_cfg = parse_env_config(args.env_config)
    seed_start = int(args.seed_start if args.seed_start is not None else env_cfg.seed)
    layout = ensure_run_layout(make_run_layout(run_dir.name, run_dir.parent.parent))
    per_run_summary_csv = args.per_run_summary_csv or (layout.summaries_dir / "per_run_summary.csv")
    aggregate_summary_csv = args.aggregate_summary_csv or (layout.summaries_dir / "aggregate_summary.csv")
    eval_csv_dir = args.eval_csv_dir or (layout.eval_dir / "checkpoint_sweep")

    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")

    per_run_rows: list[dict[str, object]] = []
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    eval_csv_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_path in checkpoints:
        for repeat_idx in range(args.num_repeats):
            eval_seed = seed_start + repeat_idx
            eval_csv_path = _make_eval_csv_path(eval_csv_dir, checkpoint_path, repeat_idx, eval_seed)
            eval_result = run_evaluation(
                env_config_path=args.env_config,
                train_config_path=args.train_config,
                pi_config_path=DEFAULT_PI_CONFIG_PATH,
                controller_name="rl",
                checkpoint_path=checkpoint_path,
                checkpoint_tag="best",
                max_steps_override=args.max_steps,
                output_csv_path=eval_csv_path,
                seed_override=eval_seed,
            )
            summary = summarize_eval_csv(eval_csv_path)
            row = {
                "checkpoint_path": str(checkpoint_path),
                "repeat_idx": repeat_idx,
                "seed": eval_seed,
                "steps": int(summary["steps"]),
                "done": int(summary["done"]),
                "done_reason": _safe_str(summary.get("done_reason", eval_result["done_reason"])),
                "mae_i_d": float(summary.get("mae_i_d", float("nan"))),
                "mae_i_q": float(summary.get("mae_i_q", float("nan"))),
                "mae_all": float(summary.get("mae_all", float("nan"))),
                "rmse_all": float(summary.get("rmse_all", float("nan"))),
                "eval_csv_path": str(eval_csv_path),
            }
            per_run_rows.append(row)
            grouped_rows[str(checkpoint_path)].append(row)
            print(
                f"evaluated checkpoint={checkpoint_path} repeat={repeat_idx} seed={eval_seed} "
                f"steps={row['steps']} done={row['done']} done_reason={row['done_reason']} "
                f"mae_all={row['mae_all']:.6f}"
            )

    aggregate_rows = [
        aggregate_repeated_eval_summaries(grouped_rows[str(checkpoint_path)])
        for checkpoint_path in checkpoints
    ]

    per_run_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_run_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_path",
                "repeat_idx",
                "seed",
                "steps",
                "done",
                "done_reason",
                "mae_i_d",
                "mae_i_q",
                "mae_all",
                "rmse_all",
                "eval_csv_path",
            ],
        )
        writer.writeheader()
        writer.writerows(per_run_rows)

    aggregate_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_path",
                "num_runs",
                "done_count",
                "success_count",
                "success_rate",
                "mean_steps",
                "mean_mae_i_d",
                "mean_mae_i_q",
                "mean_mae_all",
                "mean_rmse_all",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    best = select_best_checkpoint_aggregate(aggregate_rows)
    print(f"saved per-run summary csv: {per_run_summary_csv}")
    print(f"saved aggregate summary csv: {aggregate_summary_csv}")
    if best is None:
        print("best checkpoint: none available")
        return
    print(
        f"best checkpoint: {best['checkpoint_path']} "
        f"(success_rate={float(best['success_rate']):.3f} "
        f"done_count={int(best['done_count'])} "
        f"mean_mae_all={float(best['mean_mae_all']):.6f})"
    )


if __name__ == "__main__":
    main()

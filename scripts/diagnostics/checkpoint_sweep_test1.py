"""Evaluate saved TD3 checkpoints on canonical Test-1 and summarize the sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import summarize_eval_csv


DEFAULT_RUN_DIR = Path("outputs/runs/td3_main_2m_external_step_randomtime_seed0")
DEFAULT_ENV_CONFIG = Path("configs/env/pmsm_cc_train_external_step_randomtime.yaml")
DEFAULT_TRAIN_CONFIG = Path("configs/train/td3_main_2m_external_step_randomtime_seed0.yaml")
DEFAULT_EVAL_CONFIG = Path("configs/eval/default.yaml")
SCENARIO = "test_1_nominal_step_tracking"
PREVIOUS_500K_RMSE = 0.285


@dataclass(frozen=True)
class CheckpointItem:
    checkpoint_name: str
    checkpoint_path: Path
    global_step: int | None
    is_checkpoint_best: bool
    is_checkpoint_latest: bool
    eval_csv: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot-title", default="TD3 checkpoint sweep on Test-1")
    parser.add_argument("--reference-rmse", type=float, default=None)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def infer_seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", str(path).lower())
    return int(match.group(1)) if match else None


def _read_best_step(run_dir: Path) -> int | None:
    path = run_dir / "best_checkpoint.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("global_step")
    return int(value) if value is not None else None


def _read_latest_step(run_dir: Path) -> int | None:
    path = run_dir / "train_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda name: name in {"global_steps", "global_step"})
    col = "global_steps" if "global_steps" in df.columns else "global_step" if "global_step" in df.columns else None
    if col is None or df.empty:
        return None
    return int(pd.to_numeric(df[col], errors="coerce").max())


def discover_checkpoints(run_dir: Path, eval_dir: Path) -> list[CheckpointItem]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {ckpt_dir}")

    items: list[CheckpointItem] = []
    for path in sorted(ckpt_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))):
        match = re.search(r"checkpoint_step_(\d+)", path.name)
        if match is None:
            continue
        step = int(match.group(1))
        items.append(
            CheckpointItem(
                checkpoint_name=f"step_{step}",
                checkpoint_path=path,
                global_step=step,
                is_checkpoint_best=False,
                is_checkpoint_latest=False,
                eval_csv=eval_dir / f"eval_step_{step}.csv",
            )
        )

    best_path = ckpt_dir / "checkpoint_best.pt"
    if best_path.exists():
        items.append(
            CheckpointItem(
                checkpoint_name="checkpoint_best",
                checkpoint_path=best_path,
                global_step=_read_best_step(run_dir),
                is_checkpoint_best=True,
                is_checkpoint_latest=False,
                eval_csv=eval_dir / "eval_checkpoint_best.csv",
            )
        )

    latest_path = ckpt_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        items.append(
            CheckpointItem(
                checkpoint_name="checkpoint_latest",
                checkpoint_path=latest_path,
                global_step=_read_latest_step(run_dir),
                is_checkpoint_best=False,
                is_checkpoint_latest=True,
                eval_csv=eval_dir / "eval_checkpoint_latest.csv",
            )
        )
    return items


def run_evaluation(args: argparse.Namespace, item: CheckpointItem) -> None:
    if args.skip_existing and item.eval_csv.exists():
        print(f"skip existing {item.eval_csv}")
        return
    item.eval_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/core/evaluate.py"),
        "--env-config",
        str(args.env_config),
        "--train-config",
        str(args.train_config),
        "--eval-config",
        str(args.eval_config),
        "--controller",
        "rl",
        "--checkpoint",
        str(item.checkpoint_path),
        "--scenario",
        str(args.scenario),
        "--max-steps",
        str(args.max_steps),
        "--output-csv",
        str(item.eval_csv),
    ]
    print("running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _segment_metrics(df: pd.DataFrame, mask: pd.Series, prefix: str) -> dict[str, float | int]:
    seg = df.loc[mask].copy()
    if seg.empty:
        keys = [
            "rmse_all",
            "rmse_i_d",
            "rmse_i_q",
            "mae_all",
            "final_error_i_d",
            "final_error_i_q",
        ]
        return {f"{prefix}_{key}": float("nan") for key in keys} | {f"{prefix}_segment_length": 0}

    e_d = seg["ref_i_d_phys"].to_numpy(dtype=np.float64) - seg["i_d_phys"].to_numpy(dtype=np.float64)
    e_q = seg["ref_i_q_phys"].to_numpy(dtype=np.float64) - seg["i_q_phys"].to_numpy(dtype=np.float64)
    abs_d = np.abs(e_d)
    abs_q = np.abs(e_q)
    return {
        f"{prefix}_rmse_i_d": float(math.sqrt(float(np.mean(e_d**2)))),
        f"{prefix}_rmse_i_q": float(math.sqrt(float(np.mean(e_q**2)))),
        f"{prefix}_rmse_all": float(math.sqrt(float(np.mean(np.concatenate([e_d, e_q]) ** 2)))),
        f"{prefix}_mae_all": float(np.mean(np.concatenate([abs_d, abs_q]))),
        f"{prefix}_final_error_i_d": float(e_d[-1]),
        f"{prefix}_final_error_i_q": float(e_q[-1]),
        f"{prefix}_segment_length": int(len(seg)),
    }


def summarize_item(item: CheckpointItem, *, seed: int | None = None) -> dict[str, float | int | str | bool | None]:
    df = pd.read_csv(item.eval_csv)
    summary = summarize_eval_csv(item.eval_csv)
    row: dict[str, float | int | str | bool | None] = {
        "seed": seed,
        "checkpoint_name": item.checkpoint_name,
        "checkpoint_path": str(item.checkpoint_path),
        "eval_csv": str(item.eval_csv),
        "global_step": item.global_step,
        "is_checkpoint_best": item.is_checkpoint_best,
        "is_checkpoint_latest": item.is_checkpoint_latest,
        "episode_return": float(summary.get("episode_return", float("nan"))),
        "rmse_i_d": float(summary.get("rmse_i_d", float("nan"))),
        "rmse_i_q": float(summary.get("rmse_i_q", float("nan"))),
        "rmse_all": float(summary.get("rmse_all", float("nan"))),
        "mae_all": float(summary.get("mae_all", float("nan"))),
        "overshoot_i_d": float(summary.get("overshoot_i_d", float("nan"))),
        "overshoot_i_q": float(summary.get("overshoot_i_q", float("nan"))),
        "settling_time_i_d": float(summary.get("settling_time_i_d", float("nan"))),
        "settling_time_i_q": float(summary.get("settling_time_i_q", float("nan"))),
        "control_energy": float(summary.get("control_energy", float("nan"))),
        "action_delta_energy": float(summary.get("action_delta_energy", float("nan"))),
        "saturation_count": int(summary.get("saturation_count", 0)),
        "speed_marker": float(df["speed_rpm"].mean()) if "speed_rpm" in df.columns else float("nan"),
        "load_marker": float(df["load_torque_nm"].mean()) if "load_torque_nm" in df.columns else float("nan"),
        "episode_length": int(summary.get("episode_length", len(df))),
        "done_reason": str(summary.get("done_reason", "")),
    }
    row.update(_segment_metrics(df, df["time_s"] < 0.02, "pre"))
    row.update(_segment_metrics(df, df["time_s"] >= 0.02, "post"))
    return row


def write_plot(
    summary_df: pd.DataFrame,
    figure_path: Path,
    *,
    title: str = "TD3 checkpoint sweep on Test-1",
    reference_rmse: float | None = None,
    reference_label: str = "reference",
) -> None:
    step_df = summary_df[summary_df["checkpoint_name"].str.startswith("step_")].copy()
    step_df = step_df.sort_values("global_step")
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(step_df["global_step"], step_df["rmse_all"], marker="o", label="full")
    axes[0].plot(step_df["global_step"], step_df["pre_rmse_all"], marker=".", label="pre-step")
    axes[0].plot(step_df["global_step"], step_df["post_rmse_all"], marker=".", label="post-step")
    if reference_rmse is not None:
        axes[0].axhline(
            float(reference_rmse),
            color="tab:gray",
            linestyle="--",
            linewidth=1,
            label=reference_label,
        )
    axes[0].set_ylabel("RMSE all [A]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(step_df["global_step"], step_df["episode_return"], marker="o", color="tab:green")
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("episode return")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def _fmt(value: object, digits: int = 6) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def write_diagnosis(summary_df: pd.DataFrame, diagnosis_path: Path) -> None:
    best_rmse = summary_df.loc[summary_df["rmse_all"].idxmin()]
    best_post = summary_df.loc[summary_df["post_rmse_all"].idxmin()]
    best_return = summary_df.loc[summary_df["episode_return"].idxmax()]
    checkpoint_best = summary_df.loc[summary_df["is_checkpoint_best"] == True].iloc[0]
    checkpoint_latest = summary_df.loc[summary_df["is_checkpoint_latest"] == True].iloc[0]
    beats_500k = float(best_rmse["rmse_all"]) < PREVIOUS_500K_RMSE
    aligned = str(best_rmse["checkpoint_name"]) == str(checkpoint_best["checkpoint_name"])

    lines = [
        "# Checkpoint Sweep Test-1 Diagnosis",
        "",
        f"- Total checkpoints evaluated: {len(summary_df)}",
        f"- Best by rmse_all: {best_rmse['checkpoint_name']} at step {best_rmse['global_step']}",
        f"- Best rmse_i_d/rmse_i_q/rmse_all: {_fmt(best_rmse['rmse_i_d'])} / {_fmt(best_rmse['rmse_i_q'])} / {_fmt(best_rmse['rmse_all'])}",
        f"- Best episode_return: {_fmt(best_rmse['episode_return'])}",
        f"- Best saturation_count: {best_rmse['saturation_count']}",
        f"- Best by post_step rmse_all: {best_post['checkpoint_name']} at step {best_post['global_step']} with post_rmse_all={_fmt(best_post['post_rmse_all'])}",
        f"- Best by episode_return: {best_return['checkpoint_name']} at step {best_return['global_step']} with episode_return={_fmt(best_return['episode_return'])}",
        f"- checkpoint_best.pt rmse_all: {_fmt(checkpoint_best['rmse_all'])} at step {checkpoint_best['global_step']}",
        f"- checkpoint_latest.pt rmse_all: {_fmt(checkpoint_latest['rmse_all'])} at step {checkpoint_latest['global_step']}",
        f"- Beats previous 500k random-step-time best rmse_all ~= {PREVIOUS_500K_RMSE}: {beats_500k}",
        f"- checkpoint_best aligned with lowest rmse_all: {aligned}",
        "",
        "## Recommendation",
    ]
    if beats_500k:
        lines.append("- Use the best 2M intermediate checkpoint for Test-1 reporting, then verify broader scenarios before multi-seed runs.")
    else:
        lines.append("- Keep the current 500k random-step-time best as the reference result and investigate long-run stability before multi-seed 2M.")
    if float(checkpoint_latest["rmse_all"]) > float(best_rmse["rmse_all"]) * 1.2:
        lines.append("- The latest checkpoint is degraded relative to the best intermediate checkpoint, so late training is not monotonic.")
    if not aligned:
        lines.append("- Mean-return checkpoint selection is misaligned with minimum rmse_all for this sweep.")

    diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosis_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir
    eval_dir = run_dir / "eval" / "checkpoint_sweep_test1"
    summary_path = run_dir / "summaries" / "checkpoint_sweep_test1_summary.csv"
    prepost_path = run_dir / "summaries" / "checkpoint_sweep_test1_prepost.csv"
    figure_path = run_dir / "figures" / "checkpoint_sweep_test1.png"
    diagnosis_path = run_dir / "summaries" / "checkpoint_sweep_test1_diagnosis.md"

    checkpoints = discover_checkpoints(run_dir, eval_dir)
    print(f"discovered {len(checkpoints)} checkpoints")
    for item in checkpoints:
        run_evaluation(args, item)

    seed = args.seed if args.seed is not None else infer_seed_from_path(run_dir)
    rows = [summarize_item(item, seed=seed) for item in checkpoints]
    summary_df = pd.DataFrame(rows).sort_values(
        ["is_checkpoint_best", "is_checkpoint_latest", "global_step", "checkpoint_name"],
        ascending=[True, True, True, True],
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    prepost_cols = [
        "checkpoint_name",
        "checkpoint_path",
        "global_step",
        "pre_rmse_all",
        "post_rmse_all",
        "pre_rmse_i_d",
        "pre_rmse_i_q",
        "post_rmse_i_d",
        "post_rmse_i_q",
        "pre_mae_all",
        "post_mae_all",
        "pre_final_error_i_d",
        "pre_final_error_i_q",
        "post_final_error_i_d",
        "post_final_error_i_q",
    ]
    summary_df[prepost_cols].to_csv(prepost_path, index=False)
    write_plot(
        summary_df,
        figure_path,
        title=args.plot_title,
        reference_rmse=args.reference_rmse,
        reference_label=args.reference_label,
    )
    write_diagnosis(summary_df, diagnosis_path)

    best_rmse = summary_df.loc[summary_df["rmse_all"].idxmin()]
    best_post = summary_df.loc[summary_df["post_rmse_all"].idxmin()]
    best_return = summary_df.loc[summary_df["episode_return"].idxmax()]
    print(f"saved summary: {summary_path}")
    print(f"saved prepost: {prepost_path}")
    print(f"saved figure: {figure_path}")
    print(f"saved diagnosis: {diagnosis_path}")
    print(
        "best_rmse_all "
        f"{best_rmse['checkpoint_name']} step={best_rmse['global_step']} "
        f"rmse_all={_fmt(best_rmse['rmse_all'])} episode_return={_fmt(best_rmse['episode_return'])}"
    )
    print(
        "best_post_rmse_all "
        f"{best_post['checkpoint_name']} step={best_post['global_step']} "
        f"post_rmse_all={_fmt(best_post['post_rmse_all'])}"
    )
    print(
        "best_episode_return "
        f"{best_return['checkpoint_name']} step={best_return['global_step']} "
        f"episode_return={_fmt(best_return['episode_return'])} rmse_all={_fmt(best_return['rmse_all'])}"
    )


if __name__ == "__main__":
    main()

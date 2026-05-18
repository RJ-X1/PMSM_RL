"""Run Test-1 checkpoint sweeps for 500k TD3 seeds and select by RMSE."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_PREFIX = "td3_pretrain_500k_external_step_randomtime_test1_eval_seed"
ENV_CONFIG = Path("configs/env/pmsm_cc_train_external_step_randomtime.yaml")
EVAL_CONFIG = Path("configs/eval/default.yaml")
SCENARIO = "test_1_nominal_step_tracking"


@dataclass(frozen=True)
class SeedRun:
    seed: int
    run_name: str
    train_config: Path


def default_seed_runs() -> list[SeedRun]:
    return [
        SeedRun(
            seed=0,
            run_name=f"{RUN_PREFIX}0",
            train_config=Path("configs/train/td3_pretrain_500k_external_step_randomtime_test1_eval.yaml"),
        ),
        SeedRun(
            seed=1,
            run_name=f"{RUN_PREFIX}1",
            train_config=Path("configs/train/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1.yaml"),
        ),
        SeedRun(
            seed=2,
            run_name=f"{RUN_PREFIX}2",
            train_config=Path("configs/train/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2.yaml"),
        ),
    ]


def build_seed_runs(args: argparse.Namespace) -> list[SeedRun]:
    seeds = list(args.seeds)
    if args.run_names is not None and len(args.run_names) != len(seeds):
        raise ValueError("--run-names length must match --seeds length")
    if args.train_configs is not None and len(args.train_configs) != len(seeds):
        raise ValueError("--train-configs length must match --seeds length")

    if args.run_names is None and args.train_configs is None and args.run_prefix == RUN_PREFIX and seeds == [0, 1, 2]:
        return default_seed_runs()

    if args.train_configs is None:
        raise ValueError("--train-configs is required when using custom runs")

    runs: list[SeedRun] = []
    for idx, seed in enumerate(seeds):
        run_name = args.run_names[idx] if args.run_names is not None else f"{args.run_prefix}{seed}"
        runs.append(SeedRun(seed=seed, run_name=run_name, train_config=args.train_configs[idx]))
    return runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=ENV_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--run-names", nargs="+", default=None)
    parser.add_argument("--train-configs", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--output-prefix",
        default="td3_pretrain_500k_external_step_randomtime_checkpoint_sweep",
        help="Prefix for aggregate files under outputs/runs",
    )
    parser.add_argument("--plot-title-prefix", default="TD3 500k")
    return parser


def run_sweep(args: argparse.Namespace, seed_run: SeedRun) -> None:
    run_dir = Path("outputs/runs") / seed_run.run_name
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/diagnostics/checkpoint_sweep_test1.py"),
        "--run-dir",
        str(run_dir),
        "--env-config",
        str(args.env_config),
        "--train-config",
        str(seed_run.train_config),
        "--eval-config",
        str(args.eval_config),
        "--scenario",
        str(args.scenario),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(seed_run.seed),
        "--plot-title",
        f"{args.plot_title_prefix} seed {seed_run.seed} checkpoint sweep on Test-1",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    print("running sweep: " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _metric_row(df: pd.DataFrame, idx: int) -> pd.Series:
    return df.loc[idx].copy()


def collect_seed_rows(seed_runs: list[SeedRun]) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    selected_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    per_seed: dict[int, pd.DataFrame] = {}

    for seed_run in seed_runs:
        summary_path = Path("outputs/runs") / seed_run.run_name / "summaries/checkpoint_sweep_test1_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing sweep summary: {summary_path}")
        df = pd.read_csv(summary_path)
        per_seed[seed_run.seed] = df

        rmse_row = _metric_row(df, int(df["rmse_all"].idxmin()))
        post_row = _metric_row(df, int(df["post_rmse_all"].idxmin()))
        return_row = _metric_row(df, int(df["episode_return"].idxmax()))
        best_row = df.loc[df["is_checkpoint_best"] == True].iloc[0]
        latest_row = df.loc[df["is_checkpoint_latest"] == True].iloc[0]

        selected_rows.append(
            {
                "seed": seed_run.seed,
                "selected_checkpoint_name": rmse_row["checkpoint_name"],
                "selected_checkpoint_path": rmse_row["checkpoint_path"],
                "selected_eval_csv": rmse_row["eval_csv"],
                "selected_global_step": rmse_row["global_step"],
                "episode_return": rmse_row["episode_return"],
                "rmse_i_d": rmse_row["rmse_i_d"],
                "rmse_i_q": rmse_row["rmse_i_q"],
                "rmse_all": rmse_row["rmse_all"],
                "mae_all": rmse_row["mae_all"],
                "pre_rmse_all": rmse_row["pre_rmse_all"],
                "post_rmse_all": rmse_row["post_rmse_all"],
                "saturation_count": rmse_row["saturation_count"],
                "control_energy": rmse_row["control_energy"],
                "action_delta_energy": rmse_row["action_delta_energy"],
                "done_reason": rmse_row["done_reason"],
                "best_post_checkpoint_name": post_row["checkpoint_name"],
                "best_post_global_step": post_row["global_step"],
                "best_post_rmse_all": post_row["post_rmse_all"],
                "best_return_checkpoint_name": return_row["checkpoint_name"],
                "best_return_global_step": return_row["global_step"],
                "best_return_episode_return": return_row["episode_return"],
                "checkpoint_best_rmse_all": best_row["rmse_all"],
                "checkpoint_latest_rmse_all": latest_row["rmse_all"],
            }
        )

        best_rmse = float(best_row["rmse_all"])
        selected_rmse = float(rmse_row["rmse_all"])
        improvement_abs = best_rmse - selected_rmse
        comparison_rows.append(
            {
                "seed": seed_run.seed,
                "checkpoint_best_rmse_all": best_rmse,
                "rmse_selected_rmse_all": selected_rmse,
                "improvement_abs": improvement_abs,
                "improvement_percent": 100.0 * improvement_abs / best_rmse if best_rmse > 0.0 else float("nan"),
                "checkpoint_best_step": best_row["global_step"],
                "rmse_selected_step": rmse_row["global_step"],
                "checkpoint_best_episode_return": best_row["episode_return"],
                "rmse_selected_episode_return": rmse_row["episode_return"],
                "checkpoint_best_name": best_row["checkpoint_name"],
                "rmse_selected_name": rmse_row["checkpoint_name"],
            }
        )

    return pd.DataFrame(selected_rows), pd.DataFrame(comparison_rows), per_seed


def write_stats(selected_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    metrics = [
        "rmse_i_d",
        "rmse_i_q",
        "rmse_all",
        "mae_all",
        "pre_rmse_all",
        "post_rmse_all",
        "episode_return",
        "saturation_count",
        "control_energy",
        "action_delta_energy",
    ]
    rows = []
    for metric in metrics:
        values = pd.to_numeric(selected_df[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(path, index=False)
    return stats_df


def _timeline_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    step_df = df[df["checkpoint_name"].astype(str).str.startswith("step_")].copy()
    best_df = df[df["is_checkpoint_best"] == True].copy()
    if not best_df.empty:
        missing = ~best_df["global_step"].isin(step_df["global_step"])
        step_df = pd.concat([step_df, best_df.loc[missing]], ignore_index=True)
    return step_df.sort_values("global_step")


def write_aggregate_figure(per_seed: dict[int, pd.DataFrame], path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for seed, df in sorted(per_seed.items()):
        plot_df = _timeline_for_plot(df)
        axes[0].plot(plot_df["global_step"], plot_df["rmse_all"], marker="o", label=f"seed{seed}")
        axes[1].plot(plot_df["global_step"], plot_df["episode_return"], marker="o", label=f"seed{seed}")
    axes[0].set_ylabel("RMSE all [A]")
    axes[1].set_ylabel("episode return")
    axes[1].set_xlabel("global step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt(value: object, digits: int = 6) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def write_diagnosis(
    *,
    selected_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    per_seed: dict[int, pd.DataFrame],
    path: Path,
    title: str,
) -> None:
    rmse_stats = stats_df.loc[stats_df["metric"] == "rmse_all"].iloc[0]
    lines = [
        f"# {title}",
        "",
        "## Coverage",
    ]
    for seed, df in sorted(per_seed.items()):
        lines.append(f"- seed{seed}: evaluated {len(df)} checkpoints")

    lines.extend(
        [
            "",
            "## RMSE-Selected Checkpoints",
            "",
            "| seed | checkpoint | step | rmse_all | pre_rmse_all | post_rmse_all | episode_return | saturation |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in selected_df.sort_values("seed").iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['selected_checkpoint_name']} | {row['selected_global_step']} | "
            f"{_fmt(row['rmse_all'])} | {_fmt(row['pre_rmse_all'])} | {_fmt(row['post_rmse_all'])} | "
            f"{_fmt(row['episode_return'])} | {int(row['saturation_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoint Best vs RMSE Selection",
            "",
            "| seed | checkpoint_best rmse_all | RMSE-selected rmse_all | improvement | checkpoint_best step | RMSE-selected step |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparison_df.sort_values("seed").iterrows():
        lines.append(
            f"| {int(row['seed'])} | {_fmt(row['checkpoint_best_rmse_all'])} | "
            f"{_fmt(row['rmse_selected_rmse_all'])} | {_fmt(row['improvement_abs'])} "
            f"({_fmt(row['improvement_percent'], 2)}%) | {row['checkpoint_best_step']} | {row['rmse_selected_step']} |"
        )

    aligned_count = int((comparison_df["improvement_abs"].abs() <= 1e-12).sum())
    latest_degraded = []
    for _, row in selected_df.iterrows():
        best_rmse = float(row["rmse_all"])
        latest_rmse = float(row["checkpoint_latest_rmse_all"])
        ratio = latest_rmse / best_rmse if best_rmse > 0.0 else float("nan")
        if math.isfinite(ratio) and ratio > 1.2:
            latest_degraded.append(f"seed{int(row['seed'])} latest/best={ratio:.3f}")

    lines.extend(
        [
            "",
            "## Answers",
            f"- Lowest-rmse checkpoint differs from checkpoint_best for {len(comparison_df) - aligned_count}/{len(comparison_df)} seeds.",
            "- RMSE-selected checkpoints do not improve over checkpoint_best when improvement_abs is 0.",
            f"- RMSE-selected rmse_all mean +/- std: {_fmt(rmse_stats['mean'])} +/- {_fmt(rmse_stats['std'])}.",
            "- checkpoint_best selection is aligned with post-hoc rmse_all when the difference count above is 0.",
            f"- Latest checkpoint degradation: {', '.join(latest_degraded) if latest_degraded else 'none above 20%'}",
            "",
            "## Recommendation",
            "- Use checkpoint_best.pt / RMSE-selected checkpoints for reporting.",
            "- Do not use checkpoint_latest.pt for reporting when latest degradation is present.",
            "- If this sweep is the final baseline gate, freeze the configuration as a baseline candidate only after accepting the observed latest-checkpoint instability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_runs = build_seed_runs(args)
    for seed_run in seed_runs:
        run_sweep(args, seed_run)

    selected_df, comparison_df, per_seed = collect_seed_rows(seed_runs)
    out_root = Path("outputs/runs")
    selected_path = out_root / f"{args.output_prefix}_multiseed_summary.csv"
    stats_path = out_root / f"{args.output_prefix}_multiseed_stats.csv"
    comparison_path = out_root / f"{args.output_prefix}_checkpoint_best_vs_rmse_selected.csv"
    figure_path = out_root / f"{args.output_prefix}_multiseed.png"
    diagnosis_path = out_root / f"{args.output_prefix}_diagnosis.md"

    selected_df.to_csv(selected_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    stats_df = write_stats(selected_df, stats_path)
    write_aggregate_figure(
        per_seed,
        figure_path,
        title=f"{args.plot_title_prefix} checkpoint sweep on Test-1",
    )
    write_diagnosis(
        selected_df=selected_df,
        comparison_df=comparison_df,
        stats_df=stats_df,
        per_seed=per_seed,
        path=diagnosis_path,
        title=f"{args.plot_title_prefix} Checkpoint Sweep Diagnosis",
    )

    print(f"saved selected summary: {selected_path}")
    print(f"saved stats: {stats_path}")
    print(f"saved comparison: {comparison_path}")
    print(f"saved aggregate figure: {figure_path}")
    print(f"saved diagnosis: {diagnosis_path}")
    print(selected_df[["seed", "selected_checkpoint_name", "selected_global_step", "rmse_all", "pre_rmse_all", "post_rmse_all", "episode_return"]].to_string(index=False))
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()

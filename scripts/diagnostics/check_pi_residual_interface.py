"""Smoke-check the PI-RL residual control interface on canonical Test-1."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.core.evaluate import run_evaluation
from utils.metrics import summarize_eval_csv


DEFAULT_ENV_CONFIG = Path("configs/env/pmsm_cc_train_external_step_randomtime.yaml")
DEFAULT_TRAIN_CONFIG = Path("configs/train/td3_main_500k_external_step_randomtime_seed0_low_actor_lr.yaml")
DEFAULT_EVAL_CONFIG = Path("configs/eval/default.yaml")
DEFAULT_PI_CONFIG = Path("configs/pi/pi_default.yaml")
DEFAULT_CHECKPOINT = Path(
    "outputs/runs/td3_main_500k_external_step_randomtime_seed0_low_actor_lr/checkpoints/checkpoint_best.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics/pi_residual_interface_check")
SCENARIO = "test_1_nominal_step_tracking"
STEP_TIME_S = 0.02
EQUIVALENCE_TOL = 1e-6


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--pi-config", type=Path, default=DEFAULT_PI_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-evaluations", action="store_true")
    return parser


def _run_eval_pair(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_evaluation(
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        pi_config_path=args.pi_config,
        eval_config_path=args.eval_config,
        controller_name="pi",
        checkpoint_path=None,
        checkpoint_tag="best",
        max_steps_override=args.max_steps,
        output_csv_path=args.output_dir / "pi_eval.csv",
        seed_override=None,
        scenario_name=args.scenario,
    )
    run_evaluation(
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        pi_config_path=args.pi_config,
        eval_config_path=args.eval_config,
        controller_name="residual",
        checkpoint_path=args.checkpoint,
        checkpoint_tag="best",
        max_steps_override=args.max_steps,
        output_csv_path=args.output_dir / "residual_zero_eval.csv",
        seed_override=None,
        scenario_name=args.scenario,
        residual_zero_test_override=True,
    )


def _series(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise KeyError(f"Missing required column: {column}")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _optional_series(df: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.nanmean(values**2))))


def _rmse_all_for_mask(df: pd.DataFrame, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return float("nan")
    e_d = _series(df, "ref_i_d_phys") - _series(df, "i_d_phys")
    e_q = _series(df, "ref_i_q_phys") - _series(df, "i_q_phys")
    return _rmse(np.concatenate([e_d[mask], e_q[mask]]))


def _compare_pi_residual_zero(output_dir: Path) -> dict[str, Any]:
    pi = pd.read_csv(output_dir / "pi_eval.csv")
    rz = pd.read_csv(output_dir / "residual_zero_eval.csv")
    n = min(len(pi), len(rz))
    pi = pi.iloc[:n].copy()
    rz = rz.iloc[:n].copy()

    current_cols = ("i_d_phys", "i_q_phys", "ref_i_d_phys", "ref_i_q_phys")
    current_diffs = []
    for col in current_cols:
        current_diffs.append(_series(pi, col) - _series(rz, col))
    current_diff = np.concatenate(current_diffs)

    pi_action = np.column_stack([_series(pi, "act_u_d"), _series(pi, "act_u_q")])
    rz_action = np.column_stack([
        _series(rz, "action_total_clipped_d"),
        _series(rz, "action_total_clipped_q"),
    ])
    action_diff = (pi_action - rz_action).reshape(-1)

    pi_summary = summarize_eval_csv(output_dir / "pi_eval.csv")
    rz_summary = summarize_eval_csv(output_dir / "residual_zero_eval.csv")
    row = {
        "pi_rows": int(len(pi)),
        "residual_zero_rows": int(len(rz)),
        "max_abs_current_trace_diff": float(np.nanmax(np.abs(current_diff))),
        "rmse_current_trace_diff": _rmse(current_diff),
        "max_abs_total_action_diff": float(np.nanmax(np.abs(action_diff))),
        "rmse_total_action_diff": _rmse(action_diff),
        "pi_rmse_all": float(pi_summary.get("rmse_all", float("nan"))),
        "residual_zero_rmse_all": float(rz_summary.get("rmse_all", float("nan"))),
        "rmse_all_diff": float(rz_summary.get("rmse_all", float("nan"))) - float(pi_summary.get("rmse_all", float("nan"))),
        "pi_episode_return": float(pi_summary.get("episode_return", float("nan"))),
        "residual_zero_episode_return": float(rz_summary.get("episode_return", float("nan"))),
        "episode_return_diff": float(rz_summary.get("episode_return", float("nan")))
        - float(pi_summary.get("episode_return", float("nan"))),
    }
    row["equivalent"] = bool(
        row["max_abs_current_trace_diff"] <= EQUIVALENCE_TOL
        and row["max_abs_total_action_diff"] <= EQUIVALENCE_TOL
    )
    pd.DataFrame([row]).to_csv(output_dir / "pi_residual_equivalence_summary.csv", index=False)
    return row


def _summarize_one(label: str, path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    summary = summarize_eval_csv(path)
    time_s = _optional_series(df, "time_s")
    if time_s is None:
        time_s = np.arange(len(df), dtype=np.float64)
    row: dict[str, Any] = {
        "run": label,
        "csv_path": str(path),
        "rmse_i_d": float(summary.get("rmse_i_d", float("nan"))),
        "rmse_i_q": float(summary.get("rmse_i_q", float("nan"))),
        "rmse_all": float(summary.get("rmse_all", float("nan"))),
        "mae_all": float(summary.get("mae_all", float("nan"))),
        "pre_rmse_all": _rmse_all_for_mask(df, time_s < STEP_TIME_S),
        "post_rmse_all": _rmse_all_for_mask(df, time_s >= STEP_TIME_S),
        "episode_return": float(summary.get("episode_return", float("nan"))),
        "saturation_count": int(summary.get("saturation_count", 0)),
        "action_delta_energy": float(summary.get("action_delta_energy", float("nan"))),
        "control_energy": float(summary.get("control_energy", float("nan"))),
        "episode_length": int(summary.get("episode_length", len(df))),
        "done_reason": str(summary.get("done_reason", "")),
    }
    return row


def _write_smoke_summary(output_dir: Path) -> pd.DataFrame:
    items = [
        ("pi", output_dir / "pi_eval.csv"),
        ("residual_zero", output_dir / "residual_zero_eval.csv"),
        ("residual_td3_smoke", output_dir / "residual_td3_smoke_eval.csv"),
    ]
    rows = [_summarize_one(label, path) for label, path in items if path.exists()]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "pi_residual_smoke_summary.csv", index=False)
    return df


def _plot_pi_vs_residual_zero(output_dir: Path) -> None:
    pi = pd.read_csv(output_dir / "pi_eval.csv")
    rz = pd.read_csv(output_dir / "residual_zero_eval.csv")
    time = _series(pi, "time_s") if "time_s" in pi.columns else np.arange(len(pi))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, _series(pi, "ref_i_d_phys"), "k--", linewidth=1, label="ref i_d")
    axes[0].plot(time, _series(pi, "i_d_phys"), label="PI i_d")
    axes[0].plot(time[: len(rz)], _series(rz, "i_d_phys"), ":", label="residual-zero i_d")
    axes[0].set_ylabel("i_d [A]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time, _series(pi, "ref_i_q_phys"), "k--", linewidth=1, label="ref i_q")
    axes[1].plot(time, _series(pi, "i_q_phys"), label="PI i_q")
    axes[1].plot(time[: len(rz)], _series(rz, "i_q_phys"), ":", label="residual-zero i_q")
    axes[1].set_ylabel("i_q [A]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time, _series(pi, "act_u_d"), label="PI action d")
    axes[2].plot(time, _series(pi, "act_u_q"), label="PI action q")
    axes[2].plot(time[: len(rz)], _series(rz, "action_total_clipped_d"), ":", label="residual-zero total d")
    axes[2].plot(time[: len(rz)], _series(rz, "action_total_clipped_q"), ":", label="residual-zero total q")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("normalized action")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle("PI vs Residual-Zero Interface Check")
    fig.tight_layout()
    fig.savefig(output_dir / "pi_vs_residual_zero_waveform.png", dpi=160)
    plt.close(fig)


def _plot_residual_td3(output_dir: Path) -> None:
    path = output_dir / "residual_td3_smoke_eval.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    time = _series(df, "time_s") if "time_s" in df.columns else np.arange(len(df))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, _series(df, "ref_i_d_phys"), "k--", linewidth=1, label="ref i_d")
    axes[0].plot(time, _series(df, "i_d_phys"), label="i_d")
    axes[0].set_ylabel("i_d [A]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time, _series(df, "ref_i_q_phys"), "k--", linewidth=1, label="ref i_q")
    axes[1].plot(time, _series(df, "i_q_phys"), label="i_q")
    axes[1].set_ylabel("i_q [A]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time, _series(df, "action_pi_d"), label="PI d")
    axes[2].plot(time, _series(df, "action_pi_q"), label="PI q")
    axes[2].plot(time, _series(df, "action_residual_d"), label="RL residual d")
    axes[2].plot(time, _series(df, "action_residual_q"), label="RL residual q")
    axes[2].set_ylabel("normalized action")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(time, _series(df, "action_total_clipped_d"), label="total d")
    axes[3].plot(time, _series(df, "action_total_clipped_q"), label="total q")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("normalized action")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle("Residual TD3 Smoke Waveform")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_td3_smoke_waveform.png", dpi=160)
    plt.close(fig)


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def _csv_has_columns(path: Path, columns: set[str]) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return columns.issubset(set(reader.fieldnames or []))


def _write_report(output_dir: Path, equivalence: dict[str, Any], smoke_df: pd.DataFrame) -> None:
    residual_csv = output_dir / "residual_td3_smoke_eval.csv"
    rl_csv = output_dir / "rl_smoke_eval.csv"
    residual_fields = {
        "action_pi_d",
        "action_pi_q",
        "action_residual_d",
        "action_residual_q",
        "action_total_d",
        "action_total_q",
        "action_total_clipped_d",
        "action_total_clipped_q",
        "residual_scale",
        "residual_clip_applied",
    }
    residual_fields_ok = _csv_has_columns(residual_csv, residual_fields)
    rl_ok = rl_csv.exists() and len(pd.read_csv(rl_csv)) > 0
    pi_ok = (output_dir / "pi_eval.csv").exists() and len(pd.read_csv(output_dir / "pi_eval.csv")) > 0
    ready = bool(equivalence.get("equivalent", False) and residual_fields_ok and pi_ok and rl_ok)
    residual_summary = smoke_df.loc[smoke_df["run"] == "residual_td3_smoke"]

    lines = [
        "# PI-RL Residual Interface Diagnosis",
        "",
        "## 1. Files Changed",
        "- `utils/config.py`: added backward-compatible residual config fields.",
        "- `utils/residual_control.py`: added normalized residual action composition helpers.",
        "- `scripts/core/evaluate.py`: added `--controller residual` evaluation mode and residual CSV fields.",
        "- `scripts/core/train.py`: added gated residual training action interface.",
        "- `agents/td3_agent.py` and `agents/ddpg_agent.py`: clamp actor-update actions to configured agent bounds; unchanged for the default `[-1, 1]` baseline bounds.",
        "- `configs/eval/residual_smoke.yaml` and `configs/eval/residual_zero_smoke.yaml`: residual smoke eval configs.",
        "- `configs/train/td3_residual_smoke.yaml` and `configs/train/td3_residual_200k_external_step_randomtime_seed0.yaml`: residual train configs.",
        "- `scripts/diagnostics/check_pi_residual_interface.py`: this smoke/equivalence diagnostic harness.",
        "",
        "## 2. Total Action Formula",
        "`action_total = action_pi + residual_scale * clip(action_residual_raw, -residual_action_clip, residual_action_clip)`; `action_total_clipped = clip(action_total, env.action_space.low, env.action_space.high)` is passed to `env.step()`.",
        "",
        "## 3. Residual Units",
        "The residual action is normalized dq action, not physical voltage. The custom environment maps the final normalized action to physical dq voltage internally; `u_d` and `u_q` in CSV are the applied physical voltages reported by the environment.",
        "",
        "## 4. Residual Settings",
        "- `residual_scale`: 0.2",
        "- `residual_action_clip`: 0.3 normalized action units",
        "- `residual_zero_test`: false for TD3 smoke, true for equivalence validation",
        "",
        "## 5. PI Equivalence",
        f"- Equivalent within tolerance {EQUIVALENCE_TOL:g}: {bool(equivalence.get('equivalent', False))}",
        f"- max_abs_current_trace_diff: {_fmt(equivalence.get('max_abs_current_trace_diff'))}",
        f"- max_abs_total_action_diff: {_fmt(equivalence.get('max_abs_total_action_diff'))}",
        f"- PI rmse_all: {_fmt(equivalence.get('pi_rmse_all'))}",
        f"- residual-zero rmse_all: {_fmt(equivalence.get('residual_zero_rmse_all'))}",
        f"- episode_return diff: {_fmt(equivalence.get('episode_return_diff'))}",
        "",
        "## 6. CSV Recording",
        f"- Residual TD3 CSV contains requested PI/residual/total fields: {residual_fields_ok}",
        "- Existing `act_u_d` and `act_u_q` fields are preserved as the final clipped normalized total action.",
        "- Existing `u_d` and `u_q` fields remain the final applied physical voltage action from env info.",
        "",
        "## 7. Ordinary PI Evaluation",
        f"- Pure PI evaluation produced rows: {pi_ok}",
        "",
        "## 8. Ordinary RL Evaluation",
        f"- Pure RL smoke evaluation produced rows: {rl_ok}",
        "",
        "## 9. Training Readiness",
        f"- Ready for a 200k residual TD3 seed0 training test: {ready}",
    ]
    if not residual_summary.empty:
        row = residual_summary.iloc[0]
        lines.extend(
            [
                f"- Residual TD3 smoke rmse_all: {_fmt(row.get('rmse_all'))}",
                f"- Residual TD3 smoke episode_return: {_fmt(row.get('episode_return'))}",
                f"- Residual TD3 smoke saturation_count: {row.get('saturation_count')}",
            ]
        )
    lines.extend(
        [
            "",
            "## 10. Next Command",
            "Do not run this as part of smoke validation. If the interface is accepted, use:",
            "",
            "```powershell",
            "python -B scripts/core/train.py `",
            "  --env-config configs/env/pmsm_cc_train_external_step_randomtime.yaml `",
            "  --train-config configs/train/td3_residual_200k_external_step_randomtime_seed0.yaml `",
            "  --eval-config configs/eval/default.yaml `",
            "  --pi-config configs/pi/pi_default.yaml `",
            "  --eval-scenario test_1_nominal_step_tracking",
            "```",
        ]
    )
    (output_dir / "pi_residual_interface_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_evaluations:
        _run_eval_pair(args)
    equivalence = _compare_pi_residual_zero(args.output_dir)
    smoke_df = _write_smoke_summary(args.output_dir)
    _plot_pi_vs_residual_zero(args.output_dir)
    _plot_residual_td3(args.output_dir)
    _write_report(args.output_dir, equivalence, smoke_df)
    print(f"saved: {args.output_dir / 'pi_residual_equivalence_summary.csv'}")
    print(f"saved: {args.output_dir / 'pi_residual_smoke_summary.csv'}")
    print(f"saved: {args.output_dir / 'pi_vs_residual_zero_waveform.png'}")
    if (args.output_dir / "residual_td3_smoke_waveform.png").exists():
        print(f"saved: {args.output_dir / 'residual_td3_smoke_waveform.png'}")
    print(f"saved: {args.output_dir / 'pi_residual_interface_diagnosis.md'}")
    print(
        "equivalence "
        f"ok={bool(equivalence.get('equivalent', False))} "
        f"max_abs_current_diff={_fmt(equivalence.get('max_abs_current_trace_diff'))} "
        f"max_abs_action_diff={_fmt(equivalence.get('max_abs_total_action_diff'))}"
    )
    if not bool(equivalence.get("equivalent", False)):
        raise SystemExit("Residual-zero interface does not match pure PI within tolerance.")


if __name__ == "__main__":
    main()

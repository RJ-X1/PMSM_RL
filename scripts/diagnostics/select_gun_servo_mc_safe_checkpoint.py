"""Select the safest multi-condition gun-servo TD3 checkpoint."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import re
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.evaluate import run_evaluation
from scripts.diagnostics.run_gun_servo_td3_smoke_eval import SUMMARY_FIELDS, summarize_trace
from utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Legacy unsafe-penalty weight. Prefer --unsafe-gamma for new runs.",
    )
    parser.add_argument("--recovery-beta", type=float, default=0.05)
    parser.add_argument("--disturbance-beta", type=float, default=0.0)
    parser.add_argument("--unsafe-gamma", type=float, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    return p


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 10**12


def _discover_checkpoints(run_dir: Path, max_checkpoints: int | None) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"), key=_checkpoint_step)
    if not checkpoints:
        latest = checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists():
            checkpoints = [latest]
    if max_checkpoints is not None and int(max_checkpoints) > 0:
        checkpoints = checkpoints[-int(max_checkpoints) :]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
    return checkpoints


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_episode_limit(summary: dict[str, Any]) -> bool:
    return str(summary.get("done_reason", "")) == "episode_limit"


def _has_hard_violation(summary: dict[str, Any]) -> bool:
    done_reason = str(summary.get("done_reason", ""))
    if "violation" in done_reason:
        return True
    if _safe_float(summary.get("max_abs_omega_L_deg_s")) > 30.000001:
        return True
    return any(
        _safe_float(summary.get(name), default=0.0) > 0.5
        for name in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count")
    )


def _selection_row(
    *,
    checkpoint: Path,
    scenario_rows: list[dict[str, Any]],
    alpha: float,
    recovery_beta: float,
    disturbance_beta: float,
    unsafe_gamma: float,
    max_steps: int,
) -> dict[str, Any]:
    scenario_count = max(1, len(scenario_rows))
    all_episode_limit = all(_is_episode_limit(row) for row in scenario_rows)
    omega_limit_violation = any("omega_limit_violation" in str(row.get("done_reason", "")) for row in scenario_rows)
    hard_violation = any(_has_hard_violation(row) for row in scenario_rows)
    recovery_values = [
        _safe_float(row.get("recovery_time_s"))
        for row in scenario_rows
        if "disturbance" in str(row.get("scenario", "")).lower()
    ]
    recovery_required = bool(recovery_values)
    recovery_finite = (not recovery_required) or all(math.isfinite(value) for value in recovery_values)
    safe = bool(all_episode_limit and not omega_limit_violation and not hard_violation and recovery_finite)
    rmse_values = [_safe_float(row.get("rmse_theta_deg")) for row in scenario_rows]
    finite_rmse = [value for value in rmse_values if math.isfinite(value)]
    mean_rmse = sum(finite_rmse) / len(finite_rmse) if finite_rmse else float("nan")
    total_flag_u = sum(int(_safe_float(row.get("flag_U_safe_count"), default=0.0)) for row in scenario_rows)
    normalized_flag_u = float(total_flag_u) / float(max(1, scenario_count * int(max_steps)))
    episode_time_s = float(max_steps) * 1.0e-3
    finite_recovery = [value for value in recovery_values if math.isfinite(value)]
    c5a_recovery_time = finite_recovery[0] if finite_recovery else float("nan")
    disturbance_peak_values = [
        _safe_float(row.get("max_abs_error_after_disturbance_deg"))
        for row in scenario_rows
        if "disturbance" in str(row.get("scenario", "")).lower()
    ]
    finite_disturbance_peak = [value for value in disturbance_peak_values if math.isfinite(value)]
    mean_disturbance_peak_error = (
        sum(finite_disturbance_peak) / len(finite_disturbance_peak)
        if finite_disturbance_peak
        else 0.0
    )
    normalized_recovery = (
        min(max(float(c5a_recovery_time) / max(episode_time_s, 1e-9), 0.0), 1.0)
        if math.isfinite(c5a_recovery_time)
        else (1.0 if recovery_required else 0.0)
    )
    unsafe_penalty = 0.0 if safe else 1.0
    combined_score = (
        float(mean_rmse)
        + float(alpha) * normalized_flag_u
        + float(recovery_beta) * normalized_recovery
        + float(disturbance_beta) * float(mean_disturbance_peak_error)
        + float(unsafe_gamma) * unsafe_penalty
    )
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "global_step": "" if _checkpoint_step(checkpoint) >= 10**12 else _checkpoint_step(checkpoint),
        "safe": int(safe),
        "all_episode_limit": int(all_episode_limit),
        "omega_limit_violation": int(omega_limit_violation),
        "hard_safety_violation": int(hard_violation),
        "c5a_recovery_finite": int(recovery_finite),
        "combined_score": combined_score,
        "mean_rmse_theta_deg": mean_rmse,
        "normalized_total_flag_U": normalized_flag_u,
        "total_flag_U_safe_count": total_flag_u,
        "c5a_recovery_time_s": c5a_recovery_time,
        "normalized_recovery_time_C5a": normalized_recovery,
        "mean_disturbance_peak_error": mean_disturbance_peak_error,
        "unsafe_penalty": unsafe_penalty,
    }
    for summary in scenario_rows:
        scenario = str(summary["scenario"])
        row[f"{scenario}_rmse_theta_deg"] = summary.get("rmse_theta_deg", "")
        row[f"{scenario}_done_reason"] = summary.get("done_reason", "")
        row[f"{scenario}_flag_U_safe_count"] = summary.get("flag_U_safe_count", "")
        row[f"{scenario}_flag_E_safe_count"] = summary.get("flag_E_safe_count", "")
        row[f"{scenario}_flag_X_safe_count"] = summary.get("flag_X_safe_count", "")
        row[f"{scenario}_sigma_safe_count"] = summary.get("sigma_safe_count", "")
        row[f"{scenario}_max_abs_omega_L_deg_s"] = summary.get("max_abs_omega_L_deg_s", "")
        row[f"{scenario}_recovery_time_s"] = summary.get("recovery_time_s", "")
        row[f"{scenario}_max_abs_error_after_disturbance_deg"] = summary.get(
            "max_abs_error_after_disturbance_deg",
            "",
        )
    return row


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = _resolve_path(args.run_dir)
    env_config = _resolve_path(args.env_config)
    eval_config = _resolve_path(args.eval_config)
    eval_data = load_yaml(eval_config)
    train_config = _resolve_path(args.train_config or eval_data.get("train_config", "configs/train/gun_servo_td3_smoke.yaml"))
    scenarios = [str(item) for item in eval_data.get("scenarios", [])]
    if not scenarios:
        raise ValueError("Expected eval config to define scenarios")
    max_steps = int(eval_data.get("max_steps", 5000))
    unsafe_gamma = float(args.unsafe_gamma if args.unsafe_gamma is not None else (args.beta if args.beta is not None else 1000.0))
    controller = str(eval_data.get("controller", "residual"))
    controller_label = str(eval_data.get("td3_controller_label", "td3"))
    seed = int(eval_data.get("seed", 0))
    checkpoints = _discover_checkpoints(run_dir, args.max_checkpoints)
    eval_dir = run_dir / "eval" / "checkpoint_selection"
    summaries_dir = run_dir / "summaries"
    selection_rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        scenario_summaries: list[dict[str, Any]] = []
        for scenario in scenarios:
            output_csv = eval_dir / checkpoint.stem / f"{scenario}_td3.csv"
            result = run_evaluation(
                env_config_path=env_config,
                eval_config_path=eval_config,
                train_config_path=train_config,
                pi_config_path=Path("configs/pi/pi_default.yaml"),
                controller_name=controller,
                checkpoint_path=checkpoint,
                checkpoint_tag="best",
                max_steps_override=max_steps,
                output_csv_path=output_csv,
                seed_override=seed,
                scenario_name=scenario,
            )
            summary = summarize_trace(output_csv, scenario=scenario, controller=controller_label, checkpoint_path=checkpoint)
            summary["episode_return"] = float(result.get("episode_return", summary["episode_return"]))
            summary["done_reason"] = str(result.get("done_reason", summary["done_reason"]))
            scenario_summaries.append(summary)
        row = _selection_row(
            checkpoint=checkpoint,
            scenario_rows=scenario_summaries,
            alpha=float(args.alpha),
            recovery_beta=float(args.recovery_beta),
            disturbance_beta=float(args.disturbance_beta),
            unsafe_gamma=unsafe_gamma,
            max_steps=max_steps,
        )
        selection_rows.append(row)
        print(
            f"checkpoint={checkpoint.name} safe={row['safe']} "
            f"mean_rmse={float(row['mean_rmse_theta_deg']):.6f} "
            f"flag_U={row['total_flag_U_safe_count']} score={float(row['combined_score']):.6f}"
        )

    safe_rows = [row for row in selection_rows if int(row["safe"]) == 1]
    candidates = safe_rows if safe_rows else selection_rows
    best = min(candidates, key=lambda row: float(row["combined_score"]))
    best_checkpoint = Path(str(best["checkpoint"]))
    checkpoint_best = run_dir / "checkpoints" / "checkpoint_best.pt"
    best_alias = run_dir / "checkpoints" / "best.pt"
    checkpoint_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, checkpoint_best)
    shutil.copy2(best_checkpoint, best_alias)

    fieldnames = list(selection_rows[0].keys())
    selection_csv = summaries_dir / "checkpoint_selection_mc_safety_score.csv"
    _write_csv(selection_csv, selection_rows, fieldnames)

    reason = [
        "selection_method: multi_condition_safe_combined_score",
        f"checkpoint_path: {checkpoint_best}",
        f"source_checkpoint_path: {best_checkpoint}",
        f"global_step: {best.get('global_step', '')}",
        f"safe: {bool(int(best['safe']))}",
        f"all_episode_limit: {bool(int(best['all_episode_limit']))}",
        f"omega_limit_violation: {bool(int(best['omega_limit_violation']))}",
        f"hard_safety_violation: {bool(int(best['hard_safety_violation']))}",
        f"c5a_recovery_finite: {bool(int(best['c5a_recovery_finite']))}",
        "combined_score = mean_RMSE(all_scenarios) + alpha * normalized_total_flag_U + recovery_beta * normalized_recovery_time_C5a + disturbance_beta * mean_disturbance_peak_error + gamma * unsafe_penalty",
        f"alpha: {float(args.alpha)}",
        f"recovery_beta: {float(args.recovery_beta)}",
        f"disturbance_beta: {float(args.disturbance_beta)}",
        f"gamma: {unsafe_gamma}",
        f"combined_score: {float(best['combined_score']):.9g}",
        f"mean_rmse_theta_deg: {float(best['mean_rmse_theta_deg']):.9g}",
        f"normalized_total_flag_U: {float(best['normalized_total_flag_U']):.9g}",
        f"total_flag_U_safe_count: {best['total_flag_U_safe_count']}",
        f"c5a_recovery_time_s: {best['c5a_recovery_time_s']}",
        f"normalized_recovery_time_C5a: {best['normalized_recovery_time_C5a']}",
        f"mean_disturbance_peak_error: {best['mean_disturbance_peak_error']}",
    ]
    for scenario in scenarios:
        reason.append(
            f"{scenario}: rmse={best.get(f'{scenario}_rmse_theta_deg', '')}, "
            f"done_reason={best.get(f'{scenario}_done_reason', '')}, "
            f"flag_U={best.get(f'{scenario}_flag_U_safe_count', '')}, "
            f"recovery_time_s={best.get(f'{scenario}_recovery_time_s', '')}"
        )
    (summaries_dir / "best_checkpoint.txt").write_text("\n".join(reason) + "\n", encoding="utf-8")
    (run_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_best),
                "source_checkpoint_path": str(best_checkpoint),
                "global_step": best.get("global_step", ""),
                "metric_name": "multi_condition_safe_combined_score",
                "metric_value": float(best["combined_score"]),
                "selection_reason": "multi_condition_safe_combined_score",
                "safe": bool(int(best["safe"])),
                "scenarios": scenarios,
                "alpha": float(args.alpha),
                "recovery_beta": float(args.recovery_beta),
                "disturbance_beta": float(args.disturbance_beta),
                "gamma": unsafe_gamma,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved selection csv: {selection_csv}")
    print(f"selected checkpoint_best.pt from: {best_checkpoint}")


if __name__ == "__main__":
    main()

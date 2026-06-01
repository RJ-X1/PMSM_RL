"""Monte Carlo evaluation for the SMC-PI gun-servo baseline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.make_env import make_eval_env  # noqa: E402
from envs.obs_parser import build_observation_spec  # noqa: E402
from scripts.core.evaluate import _apply_gun_servo_env_overrides, _extract_done_reason  # noqa: E402
from scripts.diagnostics.run_gun_servo_pid_smoke import (  # noqa: E402
    CSV_FIELDS,
    _as_scenarios,
    _episode_steps_from_config,
    _find_condition,
    _load_test_conditions,
    _make_csv_row as _make_pid_csv_row,
    _max_abs,
    _recovery_time,
    _resolve_path,
    _safe_float,
    _series,
    _with_episode_steps,
    _write_rows,
)
from scripts.diagnostics.run_gun_servo_smc_baseline import (  # noqa: E402
    SMC_SCENARIO_MAP,
    _make_smc_controller,
    _speed_rad_s,
)
from utils.config import load_yaml, parse_env_config  # noqa: E402
from utils.experiment_factory import make_env_build_config  # noqa: E402
from utils.seed import set_seed  # noqa: E402


EPISODE_FIELDS = [
    "episode",
    "seed",
    "scenario",
    "controller",
    "success",
    "episode_limit",
    "omega_limit_violation",
    "hard_safety_violation",
    "rmse_theta_deg",
    "mae_theta_deg",
    "max_abs_theta_error_deg",
    "settling_time_s",
    "recovery_time_s",
    "recovery_time_0p02deg_s",
    "recovery_time_0p05deg_s",
    "recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
    "max_abs_omega_L_deg_s",
    "max_abs_omega_L_cmd_safe_deg_s",
    "max_abs_Tout_nm",
    "max_abs_iq_a",
    "flag_U_safe_count",
    "flag_E_safe_count",
    "flag_X_safe_count",
    "sigma_safe_count",
    "constraint_violation_count",
    "episode_return",
    "done_reason",
]

SUMMARY_FIELDS = [
    "num_episodes",
    "disturbance_episode_count",
    "success_rate",
    "episode_limit_rate",
    "mean_rmse_theta_deg",
    "std_rmse_theta_deg",
    "max_rmse_theta_deg",
    "mean_mae_theta_deg",
    "max_abs_theta_error_deg",
    "omega_limit_violation_count",
    "hard_safety_violation_count",
    "total_flag_U_safe_count",
    "mean_flag_U_safe_count",
    "mean_recovery_time_s",
    "recovery_time_0p02deg_s",
    "recovery_time_0p05deg_s",
    "recovery_time_0p10deg_s",
    "max_abs_error_after_disturbance_deg",
]

MC_COMPARISON_FIELDS = [
    "controller",
    "success_rate",
    "episode_limit_rate",
    "mean_rmse_theta_deg",
    "std_rmse_theta_deg",
    "max_rmse_theta_deg",
    "mean_mae_theta_deg",
    "omega_limit_violation_count",
    "hard_safety_violation_count",
    "mean_flag_U_safe_count",
    "total_flag_U_safe_count",
    "mean_recovery_time_s",
    "max_abs_error_after_disturbance_deg",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--smc-config", type=Path, default=Path("configs/eval/gun_servo_smc_baseline_eval.yaml"))
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite(values: list[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(value) for value in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean(values: list[Any]) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: list[Any]) -> float:
    arr = _finite(values)
    return float(np.std(arr)) if arr.size else float("nan")


def _max(values: list[Any]) -> float:
    arr = _finite(values)
    return float(np.max(arr)) if arr.size else float("nan")


def _scenario_probabilities(scenarios: list[str], weights_cfg: Any) -> np.ndarray | None:
    if not scenarios or not isinstance(weights_cfg, dict):
        return None
    weights = np.asarray([float(weights_cfg.get(name, 0.0)) for name in scenarios], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else None


def _load_smc_data(eval_data: dict[str, Any], smc_config_path: Path) -> dict[str, Any]:
    if isinstance(eval_data.get("smc_controller"), dict):
        return dict(eval_data["smc_controller"])
    path = _resolve_path(smc_config_path)
    if path.exists():
        raw = load_yaml(path)
        if isinstance(raw.get("smc_controller"), dict):
            return dict(raw["smc_controller"])
    return {
        "lambda_smc": 6.0,
        "k_e": 4.0,
        "k_s_deg_s": 4.0,
        "phi_rad_s": 0.06,
        "k_d": 0.25,
    }


def _apply_v8_randomization_template(base_env_cfg: Any, eval_data: dict[str, Any]) -> Any:
    template_path = eval_data.get("env_config")
    if template_path in (None, ""):
        domain_randomization = dict(getattr(base_env_cfg, "gun_domain_randomization", {}) or {})
        domain_randomization["enabled"] = True
        return replace(base_env_cfg, gun_domain_randomization=domain_randomization)
    template = parse_env_config(_resolve_path(template_path))
    return replace(
        base_env_cfg,
        gun_domain_randomization=dict(getattr(template, "gun_domain_randomization", {}) or {}),
    )


def _make_csv_row(**kwargs: Any) -> dict[str, Any]:
    row = _make_pid_csv_row(**kwargs)
    row["algorithm"] = "smc"
    row["controller"] = "smc"
    return row


def _disturbance_change_index(rows: list[dict[str, Any]]) -> int | None:
    disturbance = _series(rows, "T_L")
    if disturbance.size < 2:
        return None
    finite = disturbance[np.isfinite(disturbance)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    changes = np.flatnonzero(np.abs(np.diff(disturbance)) > 1e-6 * scale)
    return int(changes[0] + 1) if changes.size else None


def _max_abs_error_after_disturbance(rows: list[dict[str, Any]], error: np.ndarray) -> float:
    idx = _disturbance_change_index(rows)
    if idx is None or idx >= error.size:
        return float("nan")
    return _max_abs(error[idx:])


def _summarize_episode(rows: list[dict[str, Any]], *, scenario: str) -> dict[str, Any]:
    theta_ref = _series(rows, "theta_ref_deg")
    theta_l = _series(rows, "theta_L_deg")
    error = theta_ref - theta_l
    time_s = _series(rows, "t_s")
    disturbance = _series(rows, "T_L")
    is_disturbance = "disturbance" in scenario.lower()
    recovery_0p02 = (
        _recovery_time(time_s=time_s, error=error, disturbance=disturbance, threshold_deg=0.02)
        if is_disturbance
        else float("nan")
    )
    recovery_0p05 = (
        _recovery_time(time_s=time_s, error=error, disturbance=disturbance, threshold_deg=0.05)
        if is_disturbance
        else float("nan")
    )
    recovery_0p10 = (
        _recovery_time(time_s=time_s, error=error, disturbance=disturbance, threshold_deg=0.10)
        if is_disturbance
        else float("nan")
    )
    return {
        "rmse_theta_deg": float(math.sqrt(float(np.nanmean(error**2)))) if error.size else float("nan"),
        "mae_theta_deg": float(np.nanmean(np.abs(error))) if error.size else float("nan"),
        "max_abs_theta_error_deg": _max_abs(error),
        "settling_time_s": float("nan"),
        "recovery_time_s": recovery_0p02,
        "recovery_time_0p02deg_s": recovery_0p02,
        "recovery_time_0p05deg_s": recovery_0p05,
        "recovery_time_0p10deg_s": recovery_0p10,
        "max_abs_error_after_disturbance_deg": _max_abs_error_after_disturbance(rows, error),
        "max_abs_omega_L_deg_s": _max_abs(_series(rows, "omega_L_deg_s")),
        "max_abs_omega_L_cmd_safe_deg_s": _max_abs(_series(rows, "omega_L_cmd_safe_deg_s")),
        "max_abs_Tout_nm": _max_abs(_series(rows, "T_out")),
        "max_abs_iq_a": _max_abs(_series(rows, "i_q")),
        "flag_U_safe_count": int(np.sum(_series(rows, "flag_U_safe") > 0.5)),
        "flag_E_safe_count": int(np.sum(_series(rows, "flag_E_safe") > 0.5)),
        "flag_X_safe_count": int(np.sum(_series(rows, "flag_X_safe") > 0.5)),
        "sigma_safe_count": int(np.sum(_series(rows, "sigma_safe") > 0.5)),
        "constraint_violation_count": int(np.sum(_series(rows, "constraint_violation") > 0.5)),
        "episode_return": float(np.nansum(_series(rows, "reward"))),
        "done_reason": str(rows[-1].get("done_reason", "")) if rows else "",
    }


def _has_omega_violation(row: dict[str, Any]) -> bool:
    return "omega_limit_violation" in str(row.get("done_reason", ""))


def _has_hard_violation(row: dict[str, Any]) -> bool:
    if "violation" in str(row.get("done_reason", "")):
        return True
    return any(
        _safe_float(row.get(name), default=0.0) > 0.5
        for name in ("flag_E_safe_count", "flag_X_safe_count", "sigma_safe_count")
    )


def _run_episode(
    *,
    base_env_cfg: Any,
    condition: dict[str, Any],
    scenario: str,
    seed: int,
    episode: int,
    run_name: str,
    episode_steps: int | None,
    smc_data: dict[str, Any],
    save_trace: bool,
    eval_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env_cfg = _apply_gun_servo_env_overrides(base_env_cfg, condition)
    env_cfg = _with_episode_steps(env_cfg, episode_steps)
    set_seed(seed)
    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed, apply_domain_randomization=True))
    obs, _info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    controller = _make_smc_controller(spec=spec, env=env, smc_data=smc_data)
    max_steps = int(episode_steps or dict(getattr(env_cfg, "gun_servo_env", {}) or {}).get("episode_steps", 5000))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    cum_reward = 0.0
    for step in range(max_steps):
        action = np.asarray(controller.compute_action(obs), dtype=np.float32).reshape(env.action_space.shape)
        action = np.clip(action, action_low, action_high)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"SMC produced non-finite action for {scenario} episode {episode}")
        obs, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(obs).all() or not math.isfinite(float(reward)):
            raise FloatingPointError(f"Env produced non-finite data for {scenario} episode {episode}")
        cum_reward += float(reward)
        done_reason = _extract_done_reason(info, terminated=bool(terminated), truncated=bool(truncated))
        rows.append(
            _make_csv_row(
                run_id=run_name,
                scenario=scenario,
                seed=seed,
                episode=episode,
                step=step,
                action=action,
                reward=float(reward),
                cum_reward=cum_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
                done_reason=done_reason,
                info=dict(info),
                env_id=str(env_cfg.env_id),
                layout=spec.layout,
            )
        )
        if bool(terminated or truncated):
            break
    if save_trace:
        _write_rows(eval_dir / f"episode_{episode:04d}_{scenario}_smc.csv", rows, CSV_FIELDS)
    summary = _summarize_episode(rows, scenario=scenario)
    env.close()
    return rows, summary


def _aggregate_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    disturbance_rows = [row for row in rows if "disturbance" in str(row.get("scenario", "")).lower()]
    success_count = sum(int(row.get("success", 0)) for row in rows)
    episode_limit_count = sum(int(row.get("episode_limit", 0)) for row in rows)
    return {
        "num_episodes": len(rows),
        "disturbance_episode_count": len(disturbance_rows),
        "success_rate": float(success_count) / float(n),
        "episode_limit_rate": float(episode_limit_count) / float(n),
        "mean_rmse_theta_deg": _mean([row.get("rmse_theta_deg") for row in rows]),
        "std_rmse_theta_deg": _std([row.get("rmse_theta_deg") for row in rows]),
        "max_rmse_theta_deg": _max([row.get("rmse_theta_deg") for row in rows]),
        "mean_mae_theta_deg": _mean([row.get("mae_theta_deg") for row in rows]),
        "max_abs_theta_error_deg": _max([row.get("max_abs_theta_error_deg") for row in rows]),
        "omega_limit_violation_count": sum(int(row.get("omega_limit_violation", 0)) for row in rows),
        "hard_safety_violation_count": sum(int(row.get("hard_safety_violation", 0)) for row in rows),
        "total_flag_U_safe_count": sum(int(_safe_float(row.get("flag_U_safe_count"), default=0.0)) for row in rows),
        "mean_flag_U_safe_count": _mean([row.get("flag_U_safe_count") for row in rows]),
        "mean_recovery_time_s": _mean([row.get("recovery_time_s") for row in disturbance_rows]),
        "recovery_time_0p02deg_s": _mean([row.get("recovery_time_0p02deg_s") for row in disturbance_rows]),
        "recovery_time_0p05deg_s": _mean([row.get("recovery_time_0p05deg_s") for row in disturbance_rows]),
        "recovery_time_0p10deg_s": _mean([row.get("recovery_time_0p10deg_s") for row in disturbance_rows]),
        "max_abs_error_after_disturbance_deg": _max(
            [row.get("max_abs_error_after_disturbance_deg") for row in disturbance_rows]
        ),
    }


def _plot_mc_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    rmse = _finite([row.get("rmse_theta_deg") for row in rows])
    if not rmse.size:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot([rmse], tick_labels=["SMC-PI"], showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("SMC Monte Carlo RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_scenario_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    labels = sorted({str(row.get("scenario", "")) for row in rows})
    series = [_finite([row.get("rmse_theta_deg") for row in rows if str(row.get("scenario", "")) == label]) for label in labels]
    pairs = [(label, values) for label, values in zip(labels, series) if values.size]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.boxplot([values for _, values in pairs], tick_labels=[label for label, _ in pairs], showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("SMC Monte Carlo RMSE by scenario family")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_violation_counts(rows: list[dict[str, Any]], path: Path) -> None:
    labels = ["omega", "hard", "not_episode_limit", "flag_U_total"]
    values = [
        sum(int(row.get("omega_limit_violation", 0)) for row in rows),
        sum(int(row.get("hard_safety_violation", 0)) for row in rows),
        sum(1 for row in rows if int(row.get("episode_limit", 0)) == 0),
        sum(int(_safe_float(row.get("flag_U_safe_count"), default=0.0)) for row in rows),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color=["tab:red", "tab:orange", "tab:gray", "tab:blue"])
    ax.set_ylabel("count")
    ax.set_title("SMC Monte Carlo violation counts")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_recovery(rows: list[dict[str, Any]], path: Path) -> None:
    disturbance_rows = [row for row in rows if "disturbance" in str(row.get("scenario", "")).lower()]
    columns = ["recovery_time_s", "recovery_time_0p02deg_s", "recovery_time_0p05deg_s", "recovery_time_0p10deg_s"]
    labels = ["legacy", "0.02deg", "0.05deg", "0.10deg"]
    pairs = [(label, _finite([row.get(column) for row in disturbance_rows])) for label, column in zip(labels, columns)]
    pairs = [(label, values) for label, values in pairs if values.size]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot([values for _, values in pairs], tick_labels=[label for label, _ in pairs], showmeans=True)
    ax.set_ylabel("recovery time [s]")
    ax.set_title("SMC Monte Carlo disturbance recovery")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _long_metric_value(rows: list[dict[str, Any]], metric: str, field: str = "mean") -> float:
    for row in rows:
        if str(row.get("metric", "")) == metric:
            return _safe_float(row.get(field))
    return float("nan")


def _v8_mc_comparison_row(aggregate_path: Path) -> dict[str, Any]:
    rows = _read_rows(aggregate_path)
    seed_rows: list[dict[str, Any]] = []
    for seed_path in sorted((ROOT / "outputs" / "runs").glob("gun_servo_mc_sc_residual_td3_random_v8_full_seed*/summaries/mc_eval_summary.csv")):
        seed_rows.extend(_read_rows(seed_path))
    return {
        "controller": "mc_sc_residual_td3_v8_full",
        "success_rate": _long_metric_value(rows, "success_rate"),
        "episode_limit_rate": _long_metric_value(rows, "episode_limit_rate"),
        "mean_rmse_theta_deg": _long_metric_value(rows, "mean_rmse_theta_deg"),
        "std_rmse_theta_deg": _mean([row.get("std_rmse_theta_deg") for row in seed_rows]),
        "max_rmse_theta_deg": _long_metric_value(rows, "max_rmse_theta_deg"),
        "mean_mae_theta_deg": _long_metric_value(rows, "mean_mae_theta_deg"),
        "omega_limit_violation_count": _long_metric_value(rows, "omega_limit_violation_count", field="total"),
        "hard_safety_violation_count": _long_metric_value(rows, "hard_safety_violation_count", field="total"),
        "mean_flag_U_safe_count": _long_metric_value(rows, "mean_flag_U_safe_count"),
        "total_flag_U_safe_count": _mean([row.get("total_flag_U_safe_count") for row in seed_rows]),
        "mean_recovery_time_s": _long_metric_value(rows, "mean_recovery_time_s"),
        "max_abs_error_after_disturbance_deg": _long_metric_value(rows, "max_abs_error_after_disturbance_deg"),
    }


def _write_mc_comparison(summary: dict[str, Any], summaries_dir: Path) -> Path:
    smc_row = {field: summary.get(field, float("nan")) for field in MC_COMPARISON_FIELDS}
    smc_row["controller"] = "smc"
    v8_path = ROOT / "outputs" / "runs" / "gun_servo_mc_sc_residual_td3_random_v8_full_aggregate" / "summaries" / "mc_eval_mean_std.csv"
    rows = [smc_row]
    if v8_path.exists():
        rows.append(_v8_mc_comparison_row(v8_path))
    out = summaries_dir / "smc_mc_vs_v8_full_mc_summary.csv"
    _write_rows(out, rows, MC_COMPARISON_FIELDS)
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    env_config_path = _resolve_path(args.env_config)
    eval_config_path = _resolve_path(args.eval_config)
    eval_data = load_yaml(eval_config_path)
    run_name = "gun_servo_smc_baseline"
    output_dir = _resolve_path(args.output_dir or (ROOT / "outputs" / "runs" / run_name))
    eval_dir = output_dir / "eval" / "mc"
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    for path in (eval_dir, summaries_dir, figures_dir, output_dir / "configs"):
        path.mkdir(parents=True, exist_ok=True)

    base_env_cfg = _apply_v8_randomization_template(parse_env_config(env_config_path), eval_data)
    test_conditions_path = _resolve_path(eval_data.get("test_conditions", eval_data.get("test_conditions_path")))
    conditions = _load_test_conditions(test_conditions_path)
    scenario_map = {
        **SMC_SCENARIO_MAP,
        **dict(eval_data.get("scenario_condition_map", {}) or {}),
    }
    scenarios = _as_scenarios(eval_data.get("scenario_families", eval_data.get("scenarios")))
    probabilities = _scenario_probabilities(
        scenarios,
        eval_data.get("scenario_family_weights", eval_data.get("scenario_weights")),
    )
    seed = int(eval_data.get("seed", 8008))
    num_episodes = int(args.num_episodes)
    rng = np.random.default_rng(seed)
    smc_data = _load_smc_data(eval_data, args.smc_config)
    save_trace = bool(eval_data.get("save_mc_traces", False))

    episode_rows: list[dict[str, Any]] = []
    for episode in range(num_episodes):
        scenario = str(rng.choice(scenarios, p=probabilities))
        condition = _find_condition(scenario, conditions=conditions, scenario_map=scenario_map)
        episode_steps = _episode_steps_from_config(base_env_cfg, eval_data, scenario=scenario, condition=condition)
        episode_seed = seed + episode
        _trace_rows, summary = _run_episode(
            base_env_cfg=base_env_cfg,
            condition=condition,
            scenario=scenario,
            seed=episode_seed,
            episode=episode,
            run_name=run_name,
            episode_steps=episode_steps,
            smc_data=smc_data,
            save_trace=save_trace,
            eval_dir=eval_dir,
        )
        episode_limit = int(str(summary.get("done_reason", "")) == "episode_limit")
        omega = int(_has_omega_violation(summary))
        hard = int(_has_hard_violation(summary))
        row = {
            "episode": episode,
            "seed": episode_seed,
            "scenario": scenario,
            "controller": "smc",
            "success": int(episode_limit and not omega and not hard),
            "episode_limit": episode_limit,
            "omega_limit_violation": omega,
            "hard_safety_violation": hard,
            **summary,
        }
        episode_rows.append(row)
        print(
            f"smc_mc_episode={episode + 1}/{num_episodes} scenario={scenario} "
            f"rmse={float(row['rmse_theta_deg']):.4f} done_reason={row['done_reason']}"
        )

    episode_csv = summaries_dir / "smc_mc_eval_episode_metrics.csv"
    summary_csv = summaries_dir / "smc_mc_eval_summary.csv"
    _write_rows(episode_csv, episode_rows, EPISODE_FIELDS)
    summary = _aggregate_episode_rows(episode_rows)
    _write_rows(summary_csv, [summary], SUMMARY_FIELDS)
    (summaries_dir / "smc_mc_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    comparison_path = _write_mc_comparison(summary, summaries_dir)

    _plot_mc_rmse(episode_rows, figures_dir / "smc_mc_rmse_boxplot.png")
    _plot_mc_scenario_rmse(episode_rows, figures_dir / "smc_mc_scenario_rmse_boxplot.png")
    _plot_mc_violation_counts(episode_rows, figures_dir / "smc_mc_violation_counts.png")
    _plot_mc_recovery(episode_rows, figures_dir / "smc_mc_recovery_time_boxplot.png")

    shutil.copy2(eval_config_path, output_dir / "configs" / eval_config_path.name)
    print(f"saved SMC MC episode metrics: {episode_csv}")
    print(f"saved SMC MC summary: {summary_csv}")
    print(f"saved SMC MC vs v8 full MC comparison: {comparison_path}")


if __name__ == "__main__":
    main()

"""Monte Carlo evaluation for the PID/PD-PI gun-servo baseline."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
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
from scripts.core.evaluate import _apply_gun_servo_env_overrides, _extract_done_reason, _make_cascade_servo_controller  # noqa: E402
from scripts.diagnostics.run_gun_servo_pid_smoke import (  # noqa: E402
    CSV_FIELDS,
    _as_scenarios,
    _episode_steps_from_config,
    _find_condition,
    _load_test_conditions,
    _make_csv_row,
    _resolve_path,
    _safe_float,
    _with_episode_steps,
    _write_rows,
)
from scripts.diagnostics.run_gun_servo_smc_baseline import SMC_SCENARIO_MAP  # noqa: E402
from scripts.diagnostics.run_gun_servo_smc_mc_eval import (  # noqa: E402
    EPISODE_FIELDS,
    SUMMARY_FIELDS,
    _aggregate_episode_rows,
    _has_hard_violation,
    _has_omega_violation,
    _scenario_probabilities,
    _summarize_episode,
)
from utils.config import load_yaml, parse_env_config  # noqa: E402
from utils.experiment_factory import make_env_build_config  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-episodes", type=int, default=100)
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite(values: list[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(value) for value in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _apply_v8_eval_template(base_env_cfg: Any, eval_data: dict[str, Any]) -> Any:
    """Apply the env template named by the v8 MC eval config.

    The command still accepts the nominal env config, but the MC eval config is
    the source of the randomized v8 plant ranges and PID/PD gains.
    """
    template_path = eval_data.get("env_config")
    if template_path in (None, ""):
        domain_randomization = dict(getattr(base_env_cfg, "gun_domain_randomization", {}) or {})
        domain_randomization["enabled"] = True
        return replace(base_env_cfg, gun_domain_randomization=domain_randomization)

    template = parse_env_config(_resolve_path(template_path))
    return replace(
        base_env_cfg,
        gun_motor=dict(getattr(template, "gun_motor", {}) or {}),
        gun_servo_env=dict(getattr(template, "gun_servo_env", {}) or {}),
        gun_servo_drive=dict(getattr(template, "gun_servo_drive", {}) or {}),
        gun_gearbox=dict(getattr(template, "gun_gearbox", {}) or {}),
        gun_load=dict(getattr(template, "gun_load", {}) or {}),
        gun_load_encoder=dict(getattr(template, "gun_load_encoder", {}) or {}),
        gun_reference=dict(getattr(template, "gun_reference", {}) or {}),
        gun_rl_action=dict(getattr(template, "gun_rl_action", {}) or {}),
        gun_rl_controller=dict(getattr(template, "gun_rl_controller", {}) or {}),
        gun_reward=dict(getattr(template, "gun_reward", {}) or {}),
        gun_safety=dict(getattr(template, "gun_safety", {}) or {}),
        gun_normalization=dict(getattr(template, "gun_normalization", {}) or {}),
        gun_domain_randomization=dict(getattr(template, "gun_domain_randomization", {}) or {}),
        gun_environment=dict(getattr(template, "gun_environment", {}) or {}),
        gun_speed_controller=dict(getattr(template, "gun_speed_controller", {}) or {}),
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
    save_trace: bool,
    eval_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env_cfg = _apply_gun_servo_env_overrides(base_env_cfg, condition)
    env_cfg = _with_episode_steps(env_cfg, episode_steps)
    set_seed(seed)
    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed, apply_domain_randomization=True))
    obs, _info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    controller = _make_cascade_servo_controller(spec=spec, env=env)
    controller.reset()

    max_steps = int(episode_steps or dict(getattr(env_cfg, "gun_servo_env", {}) or {}).get("episode_steps", 5000))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    cum_reward = 0.0
    for step in range(max_steps):
        action = np.asarray(controller.compute_action(obs), dtype=np.float32).reshape(env.action_space.shape)
        action = np.clip(action, action_low, action_high)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"PID produced non-finite action for {scenario} episode {episode}")
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
        _write_rows(eval_dir / f"episode_{episode:04d}_{scenario}_pid.csv", rows, CSV_FIELDS)
    summary = _summarize_episode(rows, scenario=scenario)
    env.close()
    return rows, summary


def _plot_mc_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    rmse = _finite([row.get("rmse_theta_deg") for row in rows])
    if not rmse.size:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot([rmse], tick_labels=["PID/PD-PI"], showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("PID Monte Carlo RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_scenario_rmse(rows: list[dict[str, Any]], path: Path) -> None:
    labels = sorted({str(row.get("scenario", "")) for row in rows})
    pairs = []
    for label in labels:
        values = _finite([row.get("rmse_theta_deg") for row in rows if str(row.get("scenario", "")) == label])
        if values.size:
            pairs.append((label, values))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.boxplot([values for _, values in pairs], tick_labels=[label for label, _ in pairs], showmeans=True)
    ax.set_ylabel("RMSE theta [deg]")
    ax.set_title("PID Monte Carlo RMSE by scenario family")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
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
    ax.set_title("PID Monte Carlo violation counts")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mc_recovery(rows: list[dict[str, Any]], path: Path) -> None:
    disturbance_rows = [row for row in rows if "disturbance" in str(row.get("scenario", "")).lower()]
    columns = ["recovery_time_s", "recovery_time_0p02deg_s", "recovery_time_0p05deg_s", "recovery_time_0p10deg_s"]
    labels = ["legacy", "0.02deg", "0.05deg", "0.10deg"]
    pairs = []
    for label, column in zip(labels, columns):
        values = _finite([row.get(column) for row in disturbance_rows])
        if values.size:
            pairs.append((label, values))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot([values for _, values in pairs], tick_labels=[label for label, _ in pairs], showmeans=True)
    ax.set_ylabel("recovery time [s]")
    ax.set_title("PID Monte Carlo disturbance recovery")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _copy_configs(
    *,
    env_config: Path,
    eval_config: Path,
    output_dir: Path,
    eval_data: dict[str, Any],
    test_conditions_path: Path,
) -> dict[str, str]:
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "env": env_config,
        "eval": eval_config,
        "test_conditions": test_conditions_path,
        "script": Path(__file__),
    }
    if eval_data.get("env_config") not in (None, ""):
        sources["eval_env_template"] = _resolve_path(eval_data["env_config"])
    snapshots: dict[str, str] = {}
    for label, src in sources.items():
        if Path(src).exists():
            dst = configs_dir / Path(src).name
            shutil.copy2(src, dst)
            snapshots[label] = str(dst)
    return snapshots


def _write_metadata(
    *,
    output_dir: Path,
    run_name: str,
    env_config: Path,
    eval_config: Path,
    snapshots: dict[str, str],
    summary: dict[str, Any],
    episode_csv: Path,
    summary_csv: Path,
) -> None:
    payload = {
        "run_name": run_name,
        "algorithm": "pid",
        "controller": "pid",
        "task_name": "gun_servo_position",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config_paths": {"env": str(env_config), "eval": str(eval_config)},
        "snapshot_config_paths": snapshots,
        "episode_metrics_csv": str(episode_csv),
        "summary_csv": str(summary_csv),
        "summary": summary,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = build_arg_parser().parse_args()
    env_config_path = _resolve_path(args.env_config)
    eval_config_path = _resolve_path(args.eval_config)
    eval_data = load_yaml(eval_config_path)
    run_name = "gun_servo_pid_mc_baseline"
    output_dir = _resolve_path(args.output_dir or (ROOT / "outputs" / "runs" / run_name))
    eval_dir = output_dir / "eval" / "mc"
    summaries_dir = output_dir / "summaries"
    figures_dir = output_dir / "figures"
    for path in (eval_dir, summaries_dir, figures_dir, output_dir / "configs"):
        path.mkdir(parents=True, exist_ok=True)

    base_env_cfg = _apply_v8_eval_template(parse_env_config(env_config_path), eval_data)
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
            "controller": "pid",
            "success": int(episode_limit and not omega and not hard),
            "episode_limit": episode_limit,
            "omega_limit_violation": omega,
            "hard_safety_violation": hard,
            **summary,
        }
        episode_rows.append(row)
        print(
            f"pid_mc_episode={episode + 1}/{num_episodes} scenario={scenario} "
            f"rmse={float(row['rmse_theta_deg']):.4f} done_reason={row['done_reason']}"
        )

    episode_csv = summaries_dir / "pid_mc_eval_episode_metrics.csv"
    summary_csv = summaries_dir / "pid_mc_eval_summary.csv"
    _write_rows(episode_csv, episode_rows, EPISODE_FIELDS)
    summary = _aggregate_episode_rows(episode_rows)
    _write_rows(summary_csv, [summary], SUMMARY_FIELDS)
    (summaries_dir / "pid_mc_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _plot_mc_rmse(episode_rows, figures_dir / "pid_mc_rmse_boxplot.png")
    _plot_mc_scenario_rmse(episode_rows, figures_dir / "pid_mc_scenario_rmse_boxplot.png")
    _plot_mc_violation_counts(episode_rows, figures_dir / "pid_mc_violation_counts.png")
    _plot_mc_recovery(episode_rows, figures_dir / "pid_mc_recovery_time_boxplot.png")

    snapshots = _copy_configs(
        env_config=env_config_path,
        eval_config=eval_config_path,
        output_dir=output_dir,
        eval_data=eval_data,
        test_conditions_path=test_conditions_path,
    )
    _write_metadata(
        output_dir=output_dir,
        run_name=run_name,
        env_config=env_config_path,
        eval_config=eval_config_path,
        snapshots=snapshots,
        summary=summary,
        episode_csv=episode_csv,
        summary_csv=summary_csv,
    )

    print(f"saved PID MC episode metrics: {episode_csv}")
    print(f"saved PID MC summary: {summary_csv}")
    print(f"saved PID MC outputs: {output_dir}")


if __name__ == "__main__":
    main()

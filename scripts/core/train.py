"""Canonical training entrypoint for PMSM current-control RL baselines."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from typing import Any

import numpy as np

from baselines.pi_current_controller import PICurrentController
from agents.replay_buffer import ReplayBuffer
from envs.make_env import make_eval_env, make_train_env
from envs.obs_parser import build_observation_spec
from scripts.core.evaluate import _maybe_apply_eval_scenario
from utils.config import (
    apply_train_action_smoothness_override,
    apply_train_domain_randomization_override,
    apply_train_reward_override,
    parse_env_config,
    parse_pi_config,
    parse_train_config,
)
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_PI_CONFIG_PATH,
    DEFAULT_TASK_NAME,
    DEFAULT_TRAIN_CONFIG_PATH,
    build_rl_agent,
    make_env_build_config,
    resolve_agent_name,
)
from utils.residual_control import (
    compose_residual_action,
    is_residual_controller,
    normalize_controller_name,
    residual_agent_action_bounds,
    resolve_residual_settings,
)
from utils.run_layout import ensure_run_layout, make_run_layout, snapshot_run_configs, write_run_metadata
from utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train TD3/DDPG for PMSM current control")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--eval-config", type=Path, default=None, help="Optional eval-config override")
    parser.add_argument("--pi-config", type=Path, default=DEFAULT_PI_CONFIG_PATH)
    parser.add_argument("--eval-scenario", default=None, help="Optional periodic-eval scenario override")
    parser.add_argument("--controller-mode", choices=("rl", "residual", "pi_rl_residual"), default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--residual-action-clip", type=float, default=None)
    parser.add_argument("--residual-zero-test", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None, help="Optional total-step override")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional legacy episode override")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Optional warmup override")
    parser.add_argument("--update-after", type=int, default=None, help="Optional update-after override")
    parser.add_argument("--save-every-steps", type=int, default=None, help="Optional save interval override")
    parser.add_argument("--eval-every-steps", type=int, default=None, help="Optional eval interval override")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Optional eval-episode override")
    return parser


def _resolve_episode_horizon(env_cfg: Any, train_cfg: Any, *, override: int | None) -> int:
    if override is not None:
        return int(override)
    max_steps_per_episode = getattr(train_cfg, "max_steps_per_episode", None)
    if max_steps_per_episode is not None:
        return int(max_steps_per_episode)
    environment_cfg = getattr(env_cfg, "environment", None)
    if environment_cfg is not None and getattr(environment_cfg, "episode_steps", None) is not None:
        return int(environment_cfg.episode_steps)
    return 500


def _resolve_total_steps(train_cfg: Any, *, episode_horizon: int, override: int | None, max_episodes_override: int | None) -> int:
    if override is not None:
        return int(override)
    if getattr(train_cfg, "total_steps", None) is not None:
        return int(train_cfg.total_steps)
    max_episodes = int(
        max_episodes_override
        if max_episodes_override is not None
        else (train_cfg.max_episodes if getattr(train_cfg, "max_episodes", None) is not None else 1)
    )
    return int(max_episodes) * int(episode_horizon)


def _resolve_periodic_steps(
    *,
    direct_value: int | None,
    legacy_episodes: int | None,
    episode_horizon: int,
    override: int | None,
) -> int | None:
    if override is not None:
        return int(override)
    if direct_value is not None:
        return int(direct_value)
    if legacy_episodes is not None:
        return int(legacy_episodes) * int(episode_horizon)
    return None


def _resolve_eval_config_path(train_cfg: Any, *, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    configured = getattr(train_cfg, "eval_config", None)
    if configured not in (None, ""):
        return Path(str(configured))
    return DEFAULT_EVAL_CONFIG_PATH


def _resolve_eval_scenario(train_cfg: Any, *, override: str | None) -> str | None:
    if override not in (None, ""):
        return str(override)
    configured = getattr(train_cfg, "eval_scenario", None)
    if configured not in (None, ""):
        return str(configured)
    return None


def _resolve_best_checkpoint_metric(train_cfg: Any) -> str:
    requested = str(getattr(train_cfg, "best_checkpoint_metric", "mean_return")).strip().lower()
    if requested in {"mean_return", "mean_eval_return"}:
        return "mean_return"
    if requested == "rmse_all":
        return "rmse_all"
    print(
        f"best_checkpoint_metric={requested!r} is not supported by the lightweight "
        "training evaluator; using mean_return."
    )
    return "mean_return"


def _current_exploration_noise(train_cfg: Any, global_step: int) -> float:
    initial = float(getattr(train_cfg, "exploration_noise", 0.0))
    final = getattr(train_cfg, "exploration_noise_final", None)
    decay_steps = getattr(train_cfg, "exploration_noise_decay_steps", None)
    if final is None or decay_steps is None or int(decay_steps) <= 0:
        return initial
    progress = min(max(float(global_step) / float(decay_steps), 0.0), 1.0)
    return float(initial + progress * (float(final) - initial))


def _set_agent_exploration_noise(agent: Any, value: float) -> None:
    hparams = getattr(agent, "hparams", None)
    if hparams is None:
        return
    if hasattr(hparams, "exploration_noise"):
        hparams.exploration_noise = float(value)
    if hasattr(hparams, "policy_noise_std"):
        hparams.policy_noise_std = float(value)


def _resolve_controller_mode(train_cfg: Any, *, override: str | None) -> str:
    configured = override if override not in (None, "") else getattr(train_cfg, "controller_mode", None)
    mode = normalize_controller_name(configured if configured not in (None, "") else "rl")
    if mode not in {"rl", "residual"}:
        raise ValueError(f"Unsupported training controller_mode: {configured}")
    return mode


def _make_pi_controller(
    *,
    env: Any,
    env_cfg: Any,
    pi_cfg: Any,
) -> PICurrentController:
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    motor_params = env_cfg.motor.to_motor_params() if bool(getattr(env_cfg, "use_custom_env", False)) else None
    controller = PICurrentController(
        action_dim=int(env.action_space.shape[0]),
        signal_names=spec.signal_names,
        config=pi_cfg.to_controller_config(),
        motor_params=motor_params,
    )
    controller.reset()
    return controller


def _compose_env_action_for_training(
    *,
    obs: np.ndarray,
    policy_action: np.ndarray,
    pi_controller: PICurrentController | None,
    action_low: float,
    action_high: float,
    residual_settings: Any,
) -> np.ndarray:
    if pi_controller is None:
        return np.asarray(policy_action, dtype=np.float32)
    action_pi = pi_controller.compute_action(obs)
    residual_trace = compose_residual_action(
        action_pi=action_pi,
        action_residual_raw=policy_action,
        action_low=action_low,
        action_high=action_high,
        settings=residual_settings,
    )
    return np.asarray(residual_trace["action_total_clipped"], dtype=np.float32)


def _safe_metric_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _collect_eval_metric_row(
    *,
    obs: np.ndarray,
    env: Any,
    action: np.ndarray,
    info: dict[str, Any] | None,
    step: int,
    action_low: float,
    action_high: float,
) -> dict[str, float]:
    base_env = getattr(env, "unwrapped", env)
    scales = getattr(base_env, "observation_normalization_scales", {})
    motor_params = getattr(base_env, "motor_params", None)
    current_scale = _safe_metric_float(scales.get("current", float("nan")))
    ts = _safe_metric_float(getattr(motor_params, "Ts", float("nan")))
    row: dict[str, float] = {"time_s": float(step) * ts if np.isfinite(ts) else float("nan")}
    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if obs_arr.size >= 5 and np.isfinite(current_scale):
        row.update(
            {
                "i_d_phys": float(obs_arr[0] * current_scale),
                "i_q_phys": float(obs_arr[1] * current_scale),
                "ref_i_d_phys": float(obs_arr[3] * current_scale),
                "ref_i_q_phys": float(obs_arr[4] * current_scale),
            }
        )
    if isinstance(info, dict):
        u_d = _safe_metric_float(info.get("prev_u_d"))
        u_q = _safe_metric_float(info.get("prev_u_q"))
        active_umax = _safe_metric_float(info.get("active_Umax"))
        row["u_d"] = u_d
        row["u_q"] = u_q
        row["active_Umax"] = active_umax
        row["action_saturated"] = float("nan")
        if np.isfinite(active_umax) and np.isfinite(u_d) and np.isfinite(u_q) and len(action) >= 2:
            raw_command = np.clip(np.asarray(action[:2], dtype=np.float64), action_low, action_high) * active_umax
            applied_command = np.asarray([u_d, u_q], dtype=np.float64)
            saturation_error = float(np.linalg.norm(raw_command - applied_command))
            row["action_saturated"] = float(saturation_error > (1e-6 * max(1.0, abs(active_umax))))
    return row


def _summarize_eval_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "rmse_i_d": float("nan"),
            "rmse_i_q": float("nan"),
            "rmse_all": float("nan"),
            "mae_all": float("nan"),
            "pre_rmse_all": float("nan"),
            "post_rmse_all": float("nan"),
            "saturation_count": float("nan"),
        }
    required = {"i_d_phys", "i_q_phys", "ref_i_d_phys", "ref_i_q_phys"}
    if any(not required.issubset(row.keys()) for row in rows):
        return {
            "rmse_i_d": float("nan"),
            "rmse_i_q": float("nan"),
            "rmse_all": float("nan"),
            "mae_all": float("nan"),
            "pre_rmse_all": float("nan"),
            "post_rmse_all": float("nan"),
            "saturation_count": float("nan"),
        }

    i_d = np.asarray([row["i_d_phys"] for row in rows], dtype=np.float64)
    i_q = np.asarray([row["i_q_phys"] for row in rows], dtype=np.float64)
    ref_i_d = np.asarray([row["ref_i_d_phys"] for row in rows], dtype=np.float64)
    ref_i_q = np.asarray([row["ref_i_q_phys"] for row in rows], dtype=np.float64)
    time_s = np.asarray([row.get("time_s", np.nan) for row in rows], dtype=np.float64)
    e_d = ref_i_d - i_d
    e_q = ref_i_q - i_q

    def rmse_all_for_mask(mask: np.ndarray) -> float:
        if not bool(np.any(mask)):
            return float("nan")
        return float(np.sqrt(np.mean(np.concatenate([e_d[mask], e_q[mask]]) ** 2)))

    saturation_values = np.asarray([row.get("action_saturated", np.nan) for row in rows], dtype=np.float64)
    return {
        "rmse_i_d": float(np.sqrt(np.mean(e_d**2))),
        "rmse_i_q": float(np.sqrt(np.mean(e_q**2))),
        "rmse_all": float(np.sqrt(np.mean(np.concatenate([e_d, e_q]) ** 2))),
        "mae_all": float(np.mean(np.concatenate([np.abs(e_d), np.abs(e_q)]))),
        "pre_rmse_all": rmse_all_for_mask(time_s < 0.02),
        "post_rmse_all": rmse_all_for_mask(time_s >= 0.02),
        "saturation_count": float(np.nansum(saturation_values > 0.5)),
    }


def _evaluate_policy_with_metrics(
    *,
    agent: Any,
    env_cfg: Any,
    train_cfg: Any,
    controller_mode: str,
    pi_cfg: Any | None,
    residual_settings: Any,
    eval_episodes: int,
    episode_horizon: int,
    seed: int,
) -> float:
    eval_env = make_eval_env(
        make_env_build_config(
            env_cfg,
            seed_override=seed,
            apply_domain_randomization=False,
        )
    )
    returns: list[float] = []
    metric_rows: list[dict[str, float]] = []
    action_low = float(eval_env.action_space.low.min())
    action_high = float(eval_env.action_space.high.max())
    pi_controller = (
        _make_pi_controller(env=eval_env, env_cfg=env_cfg, pi_cfg=pi_cfg)
        if is_residual_controller(controller_mode)
        else None
    )
    for episode_idx in range(int(eval_episodes)):
        obs, _info = eval_env.reset(seed=seed + episode_idx)
        if pi_controller is not None:
            pi_controller.reset()
        episode_return = 0.0
        for step in range(int(episode_horizon)):
            policy_action = agent.select_action(obs, add_noise=False)
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(eval_env.action_space.shape)
            if not np.isfinite(policy_action).all():
                raise FloatingPointError(f"Evaluation policy produced non-finite action: {policy_action}")
            action = _compose_env_action_for_training(
                obs=obs,
                policy_action=policy_action,
                pi_controller=pi_controller,
                action_low=action_low,
                action_high=action_high,
                residual_settings=residual_settings,
            )
            action = np.asarray(action, dtype=np.float32).reshape(eval_env.action_space.shape)
            metric_obs = obs
            obs, reward, terminated, truncated, _info = eval_env.step(action)
            metric_rows.append(
                _collect_eval_metric_row(
                    obs=metric_obs,
                    env=eval_env,
                    action=action,
                    info=_info if isinstance(_info, dict) else None,
                    step=step,
                    action_low=action_low,
                    action_high=action_high,
                )
            )
            episode_return += float(reward)
            if bool(terminated or truncated):
                break
        returns.append(float(episode_return))
    out = _summarize_eval_metric_rows(metric_rows)
    out["mean_return"] = float(sum(returns) / max(len(returns), 1))
    return out


def _best_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "rmse_all":
        return float(metrics.get("rmse_all", float("nan")))
    return float(metrics.get("mean_return", float("nan")))


def _is_better_metric(candidate: float, current_best: float | None, metric_name: str) -> bool:
    if not np.isfinite(candidate):
        return False
    if current_best is None or not np.isfinite(float(current_best)):
        return True
    if metric_name == "rmse_all":
        return float(candidate) < float(current_best)
    return float(candidate) > float(current_best)


def _write_best_checkpoint_metadata(
    path: Path,
    *,
    checkpoint_path: Path,
    global_step: int,
    metric_name: str,
    metric_value: float | None,
    agent_name: str,
    env_id: str,
    reward_mode: str,
    selection_reason: str,
    eval_scenario: str | None = None,
    eval_config_path: Path | None = None,
) -> None:
    """Write a compact JSON summary for the best checkpoint selection."""
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "global_step": int(global_step),
        "metric_name": str(metric_name),
        "metric_value": None if metric_value is None else float(metric_value),
        "agent_name": str(agent_name),
        "env_id": str(env_id),
        "reward_mode": str(reward_mode),
        "selection_reason": str(selection_reason),
        "eval_scenario": None if eval_scenario in (None, "") else str(eval_scenario),
        "eval_config_path": None if eval_config_path is None else str(eval_config_path),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run TD3/DDPG training loop with replay, periodic checkpointing, and lightweight eval."""
    args = build_arg_parser().parse_args()

    env_cfg = parse_env_config(args.env_config)
    train_cfg = parse_train_config(args.train_config)
    pi_cfg = parse_pi_config(args.pi_config)
    controller_mode = _resolve_controller_mode(train_cfg, override=args.controller_mode)
    residual_settings = resolve_residual_settings(
        train_cfg,
        residual_scale_override=args.residual_scale,
        residual_action_clip_override=args.residual_action_clip,
        residual_zero_test_override=True if args.residual_zero_test else None,
    )
    if train_cfg.seed is not None:
        env_cfg.seed = int(train_cfg.seed)
    env_cfg = apply_train_reward_override(env_cfg, train_cfg)
    env_cfg = apply_train_action_smoothness_override(env_cfg, train_cfg)
    env_cfg = apply_train_domain_randomization_override(env_cfg, train_cfg)
    effective_eval_config_path = _resolve_eval_config_path(train_cfg, override=args.eval_config)
    eval_scenario = _resolve_eval_scenario(train_cfg, override=args.eval_scenario)
    eval_env_cfg = _maybe_apply_eval_scenario(env_cfg, effective_eval_config_path, eval_scenario)
    best_checkpoint_metric = _resolve_best_checkpoint_metric(train_cfg)
    set_seed(int(env_cfg.seed))

    env = make_train_env(
        make_env_build_config(env_cfg, apply_domain_randomization=None)
    )
    base_env = getattr(env, "unwrapped", env)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low.min())
    action_high = float(env.action_space.high.max())
    residual_mode = is_residual_controller(controller_mode)
    agent_action_low, agent_action_high = (
        residual_agent_action_bounds(
            action_low=action_low,
            action_high=action_high,
            settings=residual_settings,
        )
        if residual_mode
        else (action_low, action_high)
    )
    pi_controller = _make_pi_controller(env=env, env_cfg=env_cfg, pi_cfg=pi_cfg) if residual_mode else None

    episode_horizon = _resolve_episode_horizon(env_cfg, train_cfg, override=args.max_steps)
    eval_episode_horizon = _resolve_episode_horizon(eval_env_cfg, train_cfg, override=args.max_steps)
    total_steps = _resolve_total_steps(
        train_cfg,
        episode_horizon=episode_horizon,
        override=args.total_steps,
        max_episodes_override=args.max_episodes,
    )
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.batch_size)
    warmup_steps = int(args.warmup_steps if args.warmup_steps is not None else train_cfg.warmup_steps)
    update_after = int(args.update_after if args.update_after is not None else train_cfg.update_after)
    eval_every_steps = _resolve_periodic_steps(
        direct_value=getattr(train_cfg, "eval_every_steps", None),
        legacy_episodes=getattr(train_cfg, "eval_every_episodes", None),
        episode_horizon=episode_horizon,
        override=args.eval_every_steps,
    )
    save_every_steps = _resolve_periodic_steps(
        direct_value=getattr(train_cfg, "save_every_steps", None),
        legacy_episodes=getattr(train_cfg, "save_every_episodes", None),
        episode_horizon=episode_horizon,
        override=args.save_every_steps,
    )
    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else train_cfg.eval_episodes)

    agent = build_rl_agent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        train_cfg=train_cfg,
        action_low=agent_action_low,
        action_high=agent_action_high,
    )
    replay = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=int(train_cfg.replay_capacity))

    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    snapshot_paths = snapshot_run_configs(
        layout,
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        eval_config_path=effective_eval_config_path,
    )
    agent_name = resolve_agent_name(train_cfg)
    write_run_metadata(
        layout,
        agent_name=agent_name,
        task_name=DEFAULT_TASK_NAME,
        env_id=str(env_cfg.env_id),
        canonical_config_paths={
            "env": str(args.env_config),
            "train": str(args.train_config),
            "eval": str(effective_eval_config_path),
        },
        snapshot_config_paths=snapshot_paths,
    )

    latest_ckpt = layout.checkpoints_dir / "checkpoint_latest.pt"
    best_ckpt = layout.checkpoints_dir / "checkpoint_best.pt"
    best_ckpt_meta = layout.run_dir / "best_checkpoint.json"
    reward_mode = str(getattr(getattr(env_cfg, "reward", None), "mode", ""))
    print(
        "components "
        f"controller_mode={controller_mode} "
        f"residual_scale={float(residual_settings.residual_scale):.4f} "
        f"residual_action_clip={residual_settings.residual_action_clip} "
        f"residual_zero_test={bool(residual_settings.residual_zero_test)} "
        f"reward_mode={reward_mode} "
        f"action_smoothness={bool(getattr(base_env, 'action_smoothness_enabled', False))} "
        f"action_smoothness_weight={float(getattr(base_env, 'action_smoothness_weight', 0.0)):.4f} "
        f"domain_randomization={bool(getattr(base_env, 'apply_domain_randomization', False))} "
        f"eval_scenario={eval_scenario or ''} "
        f"best_checkpoint_metric={best_checkpoint_metric}"
    )
    obs, _info = env.reset(seed=int(env_cfg.seed))
    if pi_controller is not None:
        pi_controller.reset()
    episode_idx = 0
    episode_steps = 0
    episode_return = 0.0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    actor_update_counts: list[float] = []
    latest_eval_return: float | None = None
    latest_eval_metrics: dict[str, float] = {}
    best_eval_metric_value: float | None = None
    current_exploration_noise = float(train_cfg.exploration_noise)
    next_eval_step = int(eval_every_steps) if eval_every_steps is not None and eval_every_steps > 0 else None
    next_save_step = int(save_every_steps) if save_every_steps is not None and save_every_steps > 0 else None

    with layout.train_log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_steps",
                "global_steps",
                "episode_return",
                "buffer_size",
                "avg_actor_loss",
                "avg_critic_loss",
                "avg_actor_updates",
                "current_exploration_noise",
                "latest_eval_return",
                "eval_rmse_i_d",
                "eval_rmse_i_q",
                "eval_rmse_all",
                "eval_mae_all",
                "eval_pre_rmse_all",
                "eval_post_rmse_all",
                "eval_saturation_count",
                "saved_checkpoint",
                "checkpoint_path",
            ],
        )
        writer.writeheader()

        for global_step in range(1, total_steps + 1):
            current_exploration_noise = _current_exploration_noise(train_cfg, global_step)
            _set_agent_exploration_noise(agent, current_exploration_noise)
            if global_step <= warmup_steps:
                if residual_mode:
                    action = np.random.uniform(agent_action_low, agent_action_high, size=act_dim).astype(np.float32)
                else:
                    action = env.action_space.sample()
            else:
                action = agent.select_action(obs, add_noise=True)

            action = np.asarray(action, dtype=np.float32).reshape(act_dim)
            if not np.isfinite(action).all():
                raise FloatingPointError(f"Training policy produced non-finite action at step {global_step}: {action}")

            env_action = _compose_env_action_for_training(
                obs=obs,
                policy_action=action,
                pi_controller=pi_controller,
                action_low=action_low,
                action_high=action_high,
                residual_settings=residual_settings,
            )
            next_obs, reward, terminated, truncated, _info = env.step(env_action)
            episode_steps += 1
            episode_return += float(reward)
            done = bool(terminated or truncated)
            reached_step_limit = episode_steps >= episode_horizon
            replay.push(obs=obs, act=action, rew=float(reward), next_obs=next_obs, done=done or reached_step_limit)
            obs = next_obs

            if global_step >= update_after and len(replay) >= batch_size:
                losses = agent.update_step(replay.sample(batch_size))
                actor_losses.append(float(losses.get("actor_loss", 0.0)))
                critic_losses.append(float(losses.get("critic_loss", 0.0)))
                actor_update_counts.append(float(losses.get("actor_updated", 1.0)))

            if int(train_cfg.log_every_steps) > 0 and global_step % int(train_cfg.log_every_steps) == 0:
                print(
                    f"algo={agent_name} step={global_step}/{total_steps} "
                    f"episode={episode_idx} ep_step={episode_steps} "
                    f"return={episode_return:.3f} buffer={len(replay)}"
                )

            if next_eval_step is not None and global_step >= next_eval_step:
                latest_eval_metrics = _evaluate_policy_with_metrics(
                    agent=agent,
                    env_cfg=eval_env_cfg,
                    train_cfg=train_cfg,
                    controller_mode=controller_mode,
                    pi_cfg=pi_cfg,
                    residual_settings=residual_settings,
                    eval_episodes=eval_episodes,
                    episode_horizon=eval_episode_horizon,
                    seed=int(env_cfg.seed) + 10_000 + global_step,
                )
                latest_eval_return = float(latest_eval_metrics.get("mean_return", float("nan")))
                print(
                    f"eval algo={agent_name} step={global_step} "
                    f"episodes={eval_episodes} scenario={eval_scenario or 'env-config'} "
                    f"mean_return={latest_eval_return:.3f} "
                    f"rmse_all={float(latest_eval_metrics.get('rmse_all', float('nan'))):.6f} "
                    f"pre_rmse_all={float(latest_eval_metrics.get('pre_rmse_all', float('nan'))):.6f} "
                    f"post_rmse_all={float(latest_eval_metrics.get('post_rmse_all', float('nan'))):.6f}"
                )
                candidate_metric = _best_metric_value(latest_eval_metrics, best_checkpoint_metric)
                if _is_better_metric(candidate_metric, best_eval_metric_value, best_checkpoint_metric):
                    best_eval_metric_value = float(candidate_metric)
                    agent.save_checkpoint(best_ckpt)
                    metadata_metric_name = (
                        "eval_rmse_all" if best_checkpoint_metric == "rmse_all" else "mean_eval_return"
                    )
                    selection_reason = (
                        "periodic_eval_rmse_all_improved"
                        if best_checkpoint_metric == "rmse_all"
                        else "periodic_eval_return_improved"
                    )
                    _write_best_checkpoint_metadata(
                        best_ckpt_meta,
                        checkpoint_path=best_ckpt,
                        global_step=global_step,
                        metric_name=metadata_metric_name,
                        metric_value=best_eval_metric_value,
                        agent_name=agent_name,
                        env_id=str(env_cfg.env_id),
                        reward_mode=reward_mode,
                        selection_reason=selection_reason,
                        eval_scenario=eval_scenario,
                        eval_config_path=effective_eval_config_path,
                    )
                    print(
                        f"updated best checkpoint: {best_ckpt} "
                        f"({metadata_metric_name}={best_eval_metric_value:.6f}, step={global_step})"
                    )
                next_eval_step += int(eval_every_steps)

            saved_checkpoint = 0
            checkpoint_path = ""
            if next_save_step is not None and global_step >= next_save_step:
                checkpoint = layout.checkpoints_dir / f"checkpoint_step_{global_step}.pt"
                agent.save_checkpoint(checkpoint)
                agent.save_checkpoint(latest_ckpt)
                saved_checkpoint = 1
                checkpoint_path = str(checkpoint)
                print(f"saved checkpoints: {checkpoint} and {latest_ckpt}")
                next_save_step += int(save_every_steps)

            if done or reached_step_limit:
                avg_actor = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
                avg_critic = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
                avg_actor_updates = (
                    sum(actor_update_counts) / len(actor_update_counts) if actor_update_counts else 0.0
                )
                writer.writerow(
                    {
                        "episode": episode_idx,
                        "episode_steps": episode_steps,
                        "global_steps": global_step,
                        "episode_return": episode_return,
                        "buffer_size": len(replay),
                        "avg_actor_loss": avg_actor,
                        "avg_critic_loss": avg_critic,
                        "avg_actor_updates": avg_actor_updates,
                        "current_exploration_noise": current_exploration_noise,
                        "latest_eval_return": "" if latest_eval_return is None else latest_eval_return,
                        "eval_rmse_i_d": latest_eval_metrics.get("rmse_i_d", ""),
                        "eval_rmse_i_q": latest_eval_metrics.get("rmse_i_q", ""),
                        "eval_rmse_all": latest_eval_metrics.get("rmse_all", ""),
                        "eval_mae_all": latest_eval_metrics.get("mae_all", ""),
                        "eval_pre_rmse_all": latest_eval_metrics.get("pre_rmse_all", ""),
                        "eval_post_rmse_all": latest_eval_metrics.get("post_rmse_all", ""),
                        "eval_saturation_count": latest_eval_metrics.get("saturation_count", ""),
                        "saved_checkpoint": saved_checkpoint,
                        "checkpoint_path": checkpoint_path,
                    }
                )
                f.flush()

                print(
                    f"episode={episode_idx} episode_steps={episode_steps} global_steps={global_step} "
                    f"return={episode_return:.3f} buffer={len(replay)} "
                    f"actor_loss={avg_actor:.6f} critic_loss={avg_critic:.6f}"
                )

                episode_idx += 1
                obs, _info = env.reset()
                if pi_controller is not None:
                    pi_controller.reset()
                episode_steps = 0
                episode_return = 0.0
                actor_losses = []
                critic_losses = []
                actor_update_counts = []

        if episode_steps > 0:
            avg_actor = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
            avg_critic = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
            avg_actor_updates = sum(actor_update_counts) / len(actor_update_counts) if actor_update_counts else 0.0
            writer.writerow(
                {
                    "episode": episode_idx,
                    "episode_steps": episode_steps,
                    "global_steps": total_steps,
                    "episode_return": episode_return,
                    "buffer_size": len(replay),
                    "avg_actor_loss": avg_actor,
                    "avg_critic_loss": avg_critic,
                    "avg_actor_updates": avg_actor_updates,
                    "current_exploration_noise": current_exploration_noise,
                    "latest_eval_return": "" if latest_eval_return is None else latest_eval_return,
                    "eval_rmse_i_d": latest_eval_metrics.get("rmse_i_d", ""),
                    "eval_rmse_i_q": latest_eval_metrics.get("rmse_i_q", ""),
                    "eval_rmse_all": latest_eval_metrics.get("rmse_all", ""),
                    "eval_mae_all": latest_eval_metrics.get("mae_all", ""),
                    "eval_pre_rmse_all": latest_eval_metrics.get("pre_rmse_all", ""),
                    "eval_post_rmse_all": latest_eval_metrics.get("post_rmse_all", ""),
                    "eval_saturation_count": latest_eval_metrics.get("saturation_count", ""),
                    "saved_checkpoint": 0,
                    "checkpoint_path": "",
                }
            )
            f.flush()

    agent.save_checkpoint(latest_ckpt)
    if best_eval_metric_value is None:
        agent.save_checkpoint(best_ckpt)
        _write_best_checkpoint_metadata(
            best_ckpt_meta,
            checkpoint_path=best_ckpt,
            global_step=total_steps,
            metric_name="eval_rmse_all" if best_checkpoint_metric == "rmse_all" else "mean_eval_return",
            metric_value=None,
            agent_name=agent_name,
            env_id=str(env_cfg.env_id),
            reward_mode=reward_mode,
            selection_reason="fallback_latest_no_periodic_eval",
            eval_scenario=eval_scenario,
            eval_config_path=effective_eval_config_path,
        )
        print(f"saved fallback best checkpoint: {best_ckpt}")
    print(f"saved final checkpoint: {latest_ckpt}")


if __name__ == "__main__":
    main()

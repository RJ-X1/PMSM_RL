"""Canonical training entrypoint for PMSM current-control RL baselines."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from typing import Any

import numpy as np

from agents.replay_buffer import ReplayBuffer
from envs.make_env import make_eval_env, make_train_env
from utils.config import (
    apply_train_domain_randomization_override,
    apply_train_reward_override,
    parse_env_config,
    parse_train_config,
)
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_TASK_NAME,
    DEFAULT_TRAIN_CONFIG_PATH,
    build_rl_agent,
    make_env_build_config,
    resolve_agent_name,
)
from utils.run_layout import ensure_run_layout, make_run_layout, snapshot_run_configs, write_run_metadata
from utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train TD3/DDPG for PMSM current control")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
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


def _evaluate_policy(
    *,
    agent: Any,
    env_cfg: Any,
    train_cfg: Any,
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
    for episode_idx in range(int(eval_episodes)):
        obs, _info = eval_env.reset(seed=seed + episode_idx)
        episode_return = 0.0
        for _step in range(int(episode_horizon)):
            action = agent.select_action(obs, add_noise=False)
            action = np.asarray(action, dtype=np.float32).reshape(eval_env.action_space.shape)
            if not np.isfinite(action).all():
                raise FloatingPointError(f"Evaluation policy produced non-finite action: {action}")
            obs, reward, terminated, truncated, _info = eval_env.step(action)
            episode_return += float(reward)
            if bool(terminated or truncated):
                break
        returns.append(float(episode_return))
    return float(sum(returns) / max(len(returns), 1))


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
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run TD3/DDPG training loop with replay, periodic checkpointing, and lightweight eval."""
    args = build_arg_parser().parse_args()

    env_cfg = parse_env_config(args.env_config)
    train_cfg = parse_train_config(args.train_config)
    env_cfg = apply_train_reward_override(env_cfg, train_cfg)
    env_cfg = apply_train_domain_randomization_override(env_cfg, train_cfg)
    set_seed(int(env_cfg.seed))

    env = make_train_env(
        make_env_build_config(env_cfg, apply_domain_randomization=None)
    )
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low.min())
    action_high = float(env.action_space.high.max())

    episode_horizon = _resolve_episode_horizon(env_cfg, train_cfg, override=args.max_steps)
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
        action_low=action_low,
        action_high=action_high,
    )
    replay = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=int(train_cfg.replay_capacity))

    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    snapshot_paths = snapshot_run_configs(
        layout,
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        eval_config_path=args.eval_config,
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
            "eval": str(args.eval_config),
        },
        snapshot_config_paths=snapshot_paths,
    )

    latest_ckpt = layout.checkpoints_dir / "checkpoint_latest.pt"
    best_ckpt = layout.checkpoints_dir / "checkpoint_best.pt"
    best_ckpt_meta = layout.run_dir / "best_checkpoint.json"
    reward_mode = str(getattr(getattr(env_cfg, "reward", None), "mode", ""))
    obs, _info = env.reset(seed=int(env_cfg.seed))
    episode_idx = 0
    episode_steps = 0
    episode_return = 0.0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    actor_update_counts: list[float] = []
    latest_eval_return: float | None = None
    best_eval_return: float | None = None
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
                "latest_eval_return",
                "saved_checkpoint",
                "checkpoint_path",
            ],
        )
        writer.writeheader()

        for global_step in range(1, total_steps + 1):
            if global_step <= warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, add_noise=True)

            action = np.asarray(action, dtype=np.float32).reshape(act_dim)
            if not np.isfinite(action).all():
                raise FloatingPointError(f"Training policy produced non-finite action at step {global_step}: {action}")

            next_obs, reward, terminated, truncated, _info = env.step(action)
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
                latest_eval_return = _evaluate_policy(
                    agent=agent,
                    env_cfg=env_cfg,
                    train_cfg=train_cfg,
                    eval_episodes=eval_episodes,
                    episode_horizon=episode_horizon,
                    seed=int(env_cfg.seed) + 10_000 + global_step,
                )
                print(
                    f"eval algo={agent_name} step={global_step} "
                    f"episodes={eval_episodes} mean_return={latest_eval_return:.3f}"
                )
                if best_eval_return is None or float(latest_eval_return) > float(best_eval_return):
                    best_eval_return = float(latest_eval_return)
                    agent.save_checkpoint(best_ckpt)
                    _write_best_checkpoint_metadata(
                        best_ckpt_meta,
                        checkpoint_path=best_ckpt,
                        global_step=global_step,
                        metric_name="mean_eval_return",
                        metric_value=best_eval_return,
                        agent_name=agent_name,
                        env_id=str(env_cfg.env_id),
                        reward_mode=reward_mode,
                        selection_reason="periodic_eval_return_improved",
                    )
                    print(
                        f"updated best checkpoint: {best_ckpt} "
                        f"(mean_eval_return={best_eval_return:.3f}, step={global_step})"
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
                        "latest_eval_return": "" if latest_eval_return is None else latest_eval_return,
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
                    "latest_eval_return": "" if latest_eval_return is None else latest_eval_return,
                    "saved_checkpoint": 0,
                    "checkpoint_path": "",
                }
            )
            f.flush()

    agent.save_checkpoint(latest_ckpt)
    if best_eval_return is None:
        agent.save_checkpoint(best_ckpt)
        _write_best_checkpoint_metadata(
            best_ckpt_meta,
            checkpoint_path=best_ckpt,
            global_step=total_steps,
            metric_name="mean_eval_return",
            metric_value=None,
            agent_name=agent_name,
            env_id=str(env_cfg.env_id),
            reward_mode=reward_mode,
            selection_reason="fallback_latest_no_periodic_eval",
        )
        print(f"saved fallback best checkpoint: {best_ckpt}")
    print(f"saved final checkpoint: {latest_ckpt}")


if __name__ == "__main__":
    main()

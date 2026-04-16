"""Canonical training entrypoint for PMSM current-control DDPG."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

from agents.replay_buffer import ReplayBuffer
from envs.make_env import make_train_env
from utils.config import parse_env_config, parse_train_config
from utils.experiment_factory import (
    DEFAULT_AGENT_NAME,
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_TASK_NAME,
    DEFAULT_TRAIN_CONFIG_PATH,
    build_agent_hparams,
    build_ddpg_agent,
    make_env_build_config,
)
from utils.run_layout import ensure_run_layout, make_run_layout, snapshot_run_configs, write_run_metadata
from utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train DDPG for PMSM current control")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional override for smoke tests")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    return parser


def main() -> None:
    """Run training loop: collect rollouts, update agent, and save artifacts."""
    args = build_arg_parser().parse_args()

    env_cfg = parse_env_config(args.env_config)
    train_cfg = parse_train_config(args.train_config)

    max_episodes = int(args.max_episodes if args.max_episodes is not None else train_cfg.max_episodes)
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.max_steps_per_episode)
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.batch_size)

    set_seed(int(env_cfg.seed))

    env = make_train_env(make_env_build_config(env_cfg))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent = build_ddpg_agent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        train_cfg=train_cfg,
        action_low=float(env.action_space.low.min()),
        action_high=float(env.action_space.high.max()),
    )
    replay = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=int(train_cfg.replay_capacity))

    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    snapshot_paths = snapshot_run_configs(
        layout,
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        eval_config_path=args.eval_config,
    )
    write_run_metadata(
        layout,
        agent_name=DEFAULT_AGENT_NAME,
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

    # CSV schema:
    # episode,episode_steps,global_steps,episode_return,buffer_size,avg_actor_loss,avg_critic_loss,saved_checkpoint,checkpoint_path
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
                "saved_checkpoint",
                "checkpoint_path",
            ],
        )
        writer.writeheader()

        global_step = 0
        for episode in range(max_episodes):
            obs, _info = env.reset()
            episode_return = 0.0
            episode_steps = 0
            actor_losses: list[float] = []
            critic_losses: list[float] = []

            for _t in range(max_steps):
                use_noise = global_step >= int(train_cfg.warmup_steps)
                if use_noise:
                    action = agent.select_action(obs, add_noise=True)
                else:
                    action = env.action_space.sample()

                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

                replay.push(obs=obs, act=action, rew=float(reward), next_obs=next_obs, done=done)
                obs = next_obs
                episode_return += float(reward)
                episode_steps += 1
                global_step += 1

                if len(replay) >= batch_size:
                    batch = replay.sample(batch_size)
                    losses = agent.update_step(batch)
                    actor_losses.append(losses["actor_loss"])
                    critic_losses.append(losses["critic_loss"])

                if done:
                    break

            avg_actor = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
            avg_critic = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
            should_save_checkpoint = (episode + 1) % int(train_cfg.save_every_episodes) == 0
            checkpoint_path = layout.checkpoints_dir / f"checkpoint_ep_{episode + 1}.pt" if should_save_checkpoint else None

            writer.writerow(
                {
                    "episode": episode,
                    "episode_steps": episode_steps,
                    "global_steps": global_step,
                    "episode_return": episode_return,
                    "buffer_size": len(replay),
                    "avg_actor_loss": avg_actor,
                    "avg_critic_loss": avg_critic,
                    "saved_checkpoint": int(should_save_checkpoint),
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
                }
            )
            f.flush()

            print(
                f"episode={episode} episode_steps={episode_steps} global_steps={global_step} "
                f"return={episode_return:.3f} buffer={len(replay)} "
                f"actor_loss={avg_actor:.6f} critic_loss={avg_critic:.6f}"
            )

            if should_save_checkpoint:
                agent.save_checkpoint(checkpoint_path)
                agent.save_checkpoint(latest_ckpt)
                print(f"saved checkpoints: {checkpoint_path} and {latest_ckpt}")

        agent.save_checkpoint(latest_ckpt)
        print(f"saved final checkpoint: {latest_ckpt}")


if __name__ == "__main__":
    main()

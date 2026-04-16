"""Canonical evaluation entrypoint for RL and PI baselines."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
from typing import Any

import numpy as np

from baselines.pi_current_controller import PICurrentController
from envs.make_env import make_eval_env
from envs.obs_parser import build_observation_spec, parse_flat_observation, required_eval_trace_columns
from utils.config import parse_env_config, parse_train_config
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    build_ddpg_agent,
    make_env_build_config,
)
from utils.run_layout import ensure_run_layout, make_run_layout
from utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DDPG or PI baseline and export trajectory")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/runs/ddpg_baseline/checkpoints/checkpoint_latest.pt"),
    )
    parser.add_argument("--controller", choices=("rl", "pi"), default="rl")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed override")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def _extract_done_reason(info: Any, *, terminated: bool, truncated: bool) -> str:
    """Best-effort termination reason from env info with clear fallbacks."""
    if isinstance(info, dict):
        for key in (
            "done_reason",
            "termination_reason",
            "terminated_reason",
            "truncation_reason",
            "reason",
        ):
            value = info.get(key)
            if value not in (None, ""):
                return str(value)
        if bool(info.get("TimeLimit.truncated", False)):
            return "time_limit"
        for key, value in info.items():
            if "reason" in str(key).lower() and value not in (None, ""):
                return f"{key}={value}"
    if terminated and truncated:
        return "terminated+truncated"
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return ""


def _default_output_csv_path(train_cfg: Any, controller_name: str) -> Path:
    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    return layout.eval_dir / f"eval_{controller_name}.csv"


def run_evaluation(
    *,
    env_config_path: Path,
    train_config_path: Path,
    controller_name: str,
    checkpoint_path: Path,
    max_steps_override: int | None,
    output_csv_path: Path | None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Run one evaluation episode and export the trajectory CSV."""
    env_cfg = parse_env_config(env_config_path)
    train_cfg = parse_train_config(train_config_path)
    eval_seed = int(seed_override if seed_override is not None else env_cfg.seed)
    set_seed(eval_seed)

    env = make_eval_env(make_env_build_config(env_cfg, seed_override=eval_seed))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low.min())
    action_high = float(env.action_space.high.max())
    max_steps = int(max_steps_override if max_steps_override is not None else train_cfg.max_steps_per_episode)

    obs, _info = env.reset(seed=eval_seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    signal_fieldnames = list(parse_flat_observation(obs, spec=spec).keys())
    missing_trace_columns = [
        name
        for name in required_eval_trace_columns(layout=spec.layout)
        if name not in signal_fieldnames and name not in spec.action_names
    ]
    if missing_trace_columns:
        raise KeyError(f"Evaluation export is missing required trace columns: {missing_trace_columns}")

    if controller_name == "rl":
        agent = build_ddpg_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            train_cfg=train_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        agent.load_checkpoint(checkpoint_path)

        def policy_fn(policy_obs: np.ndarray) -> np.ndarray:
            return agent.select_action(policy_obs, add_noise=False)

    else:
        pi = PICurrentController(action_dim=act_dim, signal_names=spec.signal_names)
        pi.reset()

        def policy_fn(policy_obs: np.ndarray) -> np.ndarray:
            return pi.compute_action(policy_obs)

    final_output_csv = output_csv_path or _default_output_csv_path(train_cfg, controller_name)
    final_output_csv.parent.mkdir(parents=True, exist_ok=True)
    cum_reward = 0.0
    terminated = False
    truncated = False
    final_done_reason = ""

    fieldnames = [
        "controller",
        "env_id",
        "layout",
        "step",
        "reward",
        "cum_reward",
        "terminated",
        "truncated",
        "done",
        "done_reason",
        "termination_reason",
    ]
    fieldnames += signal_fieldnames + spec.action_names

    with final_output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(max_steps):
            action = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(act_dim)
            action = np.clip(action, action_low, action_high)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            cum_reward += float(reward)
            done_reason = _extract_done_reason(_info, terminated=bool(terminated), truncated=bool(truncated))
            final_done_reason = done_reason

            row = {
                "controller": controller_name,
                "env_id": env_cfg.env_id,
                "layout": spec.layout,
                "step": step,
                "reward": float(reward),
                "cum_reward": float(cum_reward),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "done": int(done),
                "done_reason": done_reason,
                "termination_reason": done_reason,
            }
            row.update(parse_flat_observation(obs, spec=spec))
            row.update({name: float(value) for name, value in zip(spec.action_names, action)})
            writer.writerow(row)

            obs = next_obs
            if done:
                break

    return {
        "controller": controller_name,
        "env_id": env_cfg.env_id,
        "layout": spec.layout,
        "episode_return": float(cum_reward),
        "steps": int(step + 1),
        "done_reason": final_done_reason or "not_done",
        "output_csv": final_output_csv,
        "checkpoint_path": checkpoint_path,
        "seed": eval_seed,
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "done": int(bool(terminated or truncated)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_evaluation(
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        controller_name=str(args.controller),
        checkpoint_path=args.checkpoint,
        max_steps_override=args.max_steps,
        output_csv_path=args.output_csv,
        seed_override=args.seed,
    )
    print(
        f"controller={result['controller']} env_id={result['env_id']} layout={result['layout']} "
        f"episode_return={result['episode_return']:.3f} steps={result['steps']} "
        f"done_reason={result['done_reason']} csv={result['output_csv']}"
    )


if __name__ == "__main__":
    main()

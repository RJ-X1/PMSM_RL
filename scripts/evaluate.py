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
from utils.config import (
    apply_train_action_smoothness_override,
    apply_train_reward_override,
    parse_env_config,
    parse_pi_config,
    parse_train_config,
)
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_PI_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    build_rl_agent,
    make_env_build_config,
    resolve_agent_name,
)
from utils.run_layout import ensure_run_layout, make_run_layout
from utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DDPG or PI baseline and export trajectory")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--pi-config", type=Path, default=DEFAULT_PI_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-tag",
        choices=("best", "latest"),
        default="best",
        help="Preferred RL checkpoint alias when --checkpoint is not provided.",
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


def _default_checkpoint_path(train_cfg: Any, *, tag: str = "best") -> Path:
    layout = ensure_run_layout(make_run_layout(train_cfg.run_name, train_cfg.output_dir))
    if str(tag) == "best":
        best_path = layout.checkpoints_dir / "checkpoint_best.pt"
        if best_path.exists():
            return best_path
    return layout.checkpoints_dir / "checkpoint_latest.pt"


def run_evaluation(
    *,
    env_config_path: Path,
    train_config_path: Path,
    pi_config_path: Path,
    controller_name: str,
    checkpoint_path: Path | None,
    checkpoint_tag: str,
    max_steps_override: int | None,
    output_csv_path: Path | None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Run one evaluation episode and export the trajectory CSV."""
    env_cfg = parse_env_config(env_config_path)
    train_cfg = parse_train_config(train_config_path)
    env_cfg = apply_train_reward_override(env_cfg, train_cfg)
    env_cfg = apply_train_action_smoothness_override(env_cfg, train_cfg)
    pi_cfg = parse_pi_config(pi_config_path)
    eval_seed = int(seed_override if seed_override is not None else env_cfg.seed)
    set_seed(eval_seed)

    env = make_eval_env(make_env_build_config(env_cfg, seed_override=eval_seed))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    action_low = float(env.action_space.low.min())
    action_high = float(env.action_space.high.max())
    default_eval_steps = getattr(train_cfg, "max_steps_per_episode", None)
    if default_eval_steps is None:
        default_eval_steps = getattr(getattr(env_cfg, "environment", None), "episode_steps", 500)
    max_steps = int(max_steps_override if max_steps_override is not None else default_eval_steps)

    obs, _info = env.reset(seed=eval_seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    signal_fieldnames = list(parse_flat_observation(obs, spec=spec).keys())
    base_env = getattr(env, "unwrapped", env)
    reward_term_names = list(getattr(base_env, "reward_term_names", []))
    reward_mode_name = getattr(base_env, "reward_mode", "")
    action_smoothness_enabled = bool(getattr(base_env, "action_smoothness_enabled", False))
    action_smoothness_weight = float(getattr(base_env, "action_smoothness_weight", 0.0))
    missing_trace_columns = [
        name
        for name in required_eval_trace_columns(layout=spec.layout)
        if name not in signal_fieldnames and name not in spec.action_names
    ]
    if missing_trace_columns:
        raise KeyError(f"Evaluation export is missing required trace columns: {missing_trace_columns}")

    if controller_name == "rl":
        agent = build_rl_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            train_cfg=train_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        resolved_checkpoint = checkpoint_path or _default_checkpoint_path(train_cfg, tag=checkpoint_tag)
        agent.load_checkpoint(resolved_checkpoint)

        def policy_fn(policy_obs: np.ndarray) -> np.ndarray:
            return agent.select_action(policy_obs, add_noise=False)

    else:
        resolved_checkpoint = checkpoint_path
        motor_params = env_cfg.motor.to_motor_params() if bool(getattr(env_cfg, "use_custom_env", False)) else None
        pi = PICurrentController(
            action_dim=act_dim,
            signal_names=spec.signal_names,
            config=pi_cfg.to_controller_config(),
            motor_params=motor_params,
        )
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
        "reward_mode",
        "action_smoothness_enabled",
        "action_smoothness_weight",
    ]
    fieldnames += signal_fieldnames + spec.action_names + reward_term_names

    with final_output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(max_steps):
            action = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(act_dim)
            action = np.clip(action, action_low, action_high)
            if not np.isfinite(action).all():
                raise FloatingPointError(
                    f"{controller_name} produced non-finite action at step {step}: {action}"
                )

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
                "reward_mode": str(reward_mode_name),
                "action_smoothness_enabled": int(action_smoothness_enabled),
                "action_smoothness_weight": float(action_smoothness_weight),
            }
            row.update(parse_flat_observation(obs, spec=spec))
            row.update({name: float(value) for name, value in zip(spec.action_names, action)})
            if reward_term_names:
                reward_terms = _info.get("reward_terms", {}) if isinstance(_info, dict) else {}
                row.update({name: float(reward_terms.get(name, 0.0)) for name in reward_term_names})
            writer.writerow(row)

            obs = next_obs
            if done:
                break

    return {
        "controller": controller_name,
        "agent_name": resolve_agent_name(train_cfg) if controller_name == "rl" else "pi",
        "env_id": env_cfg.env_id,
        "layout": spec.layout,
        "episode_return": float(cum_reward),
        "steps": int(step + 1),
        "done_reason": final_done_reason or "not_done",
        "output_csv": final_output_csv,
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_tag": None if controller_name != "rl" else str(checkpoint_tag),
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
        pi_config_path=args.pi_config,
        controller_name=str(args.controller),
        checkpoint_path=args.checkpoint,
        checkpoint_tag=str(args.checkpoint_tag),
        max_steps_override=args.max_steps,
        output_csv_path=args.output_csv,
        seed_override=args.seed,
    )
    print(
        f"controller={result['controller']} agent={result['agent_name']} "
        f"env_id={result['env_id']} layout={result['layout']} "
        f"episode_return={result['episode_return']:.3f} steps={result['steps']} "
        f"done_reason={result['done_reason']} checkpoint={result['checkpoint_path']} "
        f"csv={result['output_csv']}"
    )


if __name__ == "__main__":
    main()

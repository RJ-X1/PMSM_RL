"""Smoke-inspect the custom gun-servo position environment."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.make_env import make_eval_env
from envs.obs_parser import build_observation_spec, parse_flat_observation
from scripts.core.evaluate import _make_cascade_servo_controller
from utils.config import parse_env_config
from utils.experiment_factory import make_env_build_config
from utils.seed import set_seed


REQUIRED_INFO_KEYS = (
    "theta_ref_deg",
    "theta_L_deg",
    "theta_meas_deg",
    "omega_L_deg_s",
    "omega_cmd_deg_s",
    "action_raw",
    "action_safe",
    "Te_Nm",
    "T_out_Nm",
    "disturbance_torque_Nm",
    "constraint_violation",
    "done_reason",
    "action_type",
    "flag_U_safe",
    "flag_E_safe",
    "flag_X_safe",
    "sigma_safe",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-steps", type=int, default=100)
    parser.add_argument("--pid-steps", type=int, default=300)
    return parser


def _rollout_random(env: Any, *, steps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    rewards: list[float] = []
    done_reason = ""
    for _ in range(int(steps)):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise FloatingPointError("random rollout produced non-finite observation or reward")
        if terminated or truncated:
            done_reason = str(info.get("done_reason", "done"))
            break
    return {
        "steps": len(rewards),
        "return": float(np.sum(rewards)),
        "done_reason": done_reason or "not_done",
        "last_info": info,
    }


def _rollout_pid(env: Any, *, steps: int, seed: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=str(info.get("env_id", "")), env=env)
    controller = _make_cascade_servo_controller(spec=spec, env=env)
    rewards: list[float] = []
    done_reason = ""
    max_abs_error_deg = 0.0
    for _ in range(int(steps)):
        action = controller.compute_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        max_abs_error_deg = max(max_abs_error_deg, abs(float(info.get("e_theta_deg", 0.0))))
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise FloatingPointError("PID rollout produced non-finite observation or reward")
        if terminated or truncated:
            done_reason = str(info.get("done_reason", "done"))
            break
    return {
        "steps": len(rewards),
        "return": float(np.sum(rewards)),
        "max_abs_error_deg": float(max_abs_error_deg),
        "done_reason": done_reason or "not_done",
        "last_info": info,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    env_cfg = parse_env_config(args.env_config)
    seed = int(args.seed if args.seed is not None else env_cfg.seed)
    set_seed(seed)
    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed))
    obs, info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    named_obs = parse_flat_observation(obs, spec=spec)
    missing_info = [key for key in REQUIRED_INFO_KEYS if key not in info]
    if missing_info:
        raise KeyError(f"Missing required gun-servo info keys: {missing_info}")
    if not np.isfinite(obs).all():
        raise FloatingPointError("reset observation contains non-finite values")

    random_summary = _rollout_random(env, steps=args.random_steps, seed=seed + 101)
    pid_summary = _rollout_pid(env, steps=args.pid_steps, seed=seed + 202)
    summary = {
        "env_id": env_cfg.env_id,
        "layout": spec.layout,
        "obs_dim": int(env.observation_space.shape[0]),
        "act_dim": int(env.action_space.shape[0]),
        "action_low": np.asarray(env.action_space.low, dtype=float).reshape(-1).tolist(),
        "action_high": np.asarray(env.action_space.high, dtype=float).reshape(-1).tolist(),
        "signal_names": spec.signal_names,
        "action_names": spec.action_names,
        "action_type": str(info.get("action_type", "")),
        "named_observation_preview": dict(named_obs),
        "reset_info_keys": sorted(info.keys()),
        "random_rollout": {
            key: value for key, value in random_summary.items() if key != "last_info"
        },
        "pid_rollout": {
            key: value for key, value in pid_summary.items() if key != "last_info"
        },
        "pid_last_theta_ref_deg": float(pid_summary["last_info"].get("theta_ref_deg", 0.0)),
        "pid_last_theta_L_deg": float(pid_summary["last_info"].get("theta_L_deg", 0.0)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

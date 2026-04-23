"""Quick smoke test for the PI baseline on the PMSM current-control env."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np

from baselines.pi_current_controller import PICurrentController
from envs.make_env import make_eval_env
from envs.obs_parser import build_observation_spec
from utils.config import parse_env_config, parse_pi_config
from utils.experiment_factory import DEFAULT_ENV_CONFIG_PATH, DEFAULT_PI_CONFIG_PATH, make_env_build_config
from utils.seed import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the PI current controller")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--pi-config", type=Path, default=DEFAULT_PI_CONFIG_PATH)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    env_cfg = parse_env_config(args.env_config)
    pi_cfg = parse_pi_config(args.pi_config)
    seed = int(args.seed if args.seed is not None else env_cfg.seed)
    set_seed(seed)

    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed))
    obs, _info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    motor_params = env_cfg.motor.to_motor_params() if bool(getattr(env_cfg, "use_custom_env", False)) else None

    controller = PICurrentController(
        action_dim=int(env.action_space.shape[0]),
        signal_names=spec.signal_names,
        config=pi_cfg.to_controller_config(),
        motor_params=motor_params,
    )
    controller.reset()

    first_action = controller.compute_action(obs)
    first_action_ok = (
        tuple(first_action.shape) == tuple(env.action_space.shape) and np.isfinite(first_action).all()
    )

    steps_run = 0
    last_action = first_action
    terminated = False
    truncated = False
    action_all_finite = bool(np.isfinite(first_action).all())

    for _step in range(int(args.max_steps)):
        action = controller.compute_action(obs)
        action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
        if not np.isfinite(action).all():
            action_all_finite = False
            last_action = action
            break

        obs, _reward, terminated, truncated, _info = env.step(action)
        last_action = action
        steps_run += 1
        if bool(terminated or truncated):
            break

    print(f"env_id={env_cfg.env_id}")
    print(f"use_custom_env={bool(getattr(env_cfg, 'use_custom_env', False))}")
    print(f"obs_dim={int(env.observation_space.shape[0])}")
    print(f"act_dim={int(env.action_space.shape[0])}")
    print(f"pi_reset_ok=True")
    print(f"first_action_shape={tuple(first_action.shape)}")
    print(f"first_action_finite={bool(np.isfinite(first_action).all())}")
    print(f"first_action={first_action.astype(float).tolist()}")
    print(f"first_action_ok={bool(first_action_ok)}")
    print(f"steps_run={steps_run}")
    print(f"action_all_finite={bool(action_all_finite)}")
    print(f"terminated={bool(terminated)} truncated={bool(truncated)}")
    print(f"last_action={np.asarray(last_action, dtype=np.float32).astype(float).tolist()}")


if __name__ == "__main__":
    main()

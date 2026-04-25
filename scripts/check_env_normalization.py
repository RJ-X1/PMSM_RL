"""Lightweight normalization check for the custom PMSM current-control env."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np

from envs.make_env import make_eval_env
from envs.obs_parser import build_observation_spec, parse_flat_observation
from utils.config import parse_env_config
from utils.experiment_factory import DEFAULT_ENV_CONFIG_PATH, make_env_build_config
from utils.seed import set_seed


SAMPLE_ACTIONS = (
    np.asarray([0.0, 0.0], dtype=np.float32),
    np.asarray([0.5, -0.5], dtype=np.float32),
    np.asarray([1.0, 1.0], dtype=np.float32),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check custom PMSM env normalization and action mapping")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    env_cfg = parse_env_config(args.env_config)
    seed = int(env_cfg.seed)
    set_seed(seed)

    env = make_eval_env(
        make_env_build_config(
            env_cfg,
            seed_override=seed,
            apply_domain_randomization=False,
        )
    )
    base_env = getattr(env, "unwrapped", env)

    obs, _info = env.reset(seed=seed)
    spec = build_observation_spec(env_id=env_cfg.env_id, env=env)
    named_obs = parse_flat_observation(obs, spec=spec)
    scales = getattr(base_env, "observation_normalization_scales", {})
    action_bounds = getattr(base_env, "action_normalization_bounds", None)
    voltage_limit = float(getattr(base_env.motor_params, "Umax", 0.0))

    print(f"env_id: {env_cfg.env_id}")
    print(f"use_custom_env: {bool(getattr(env_cfg, 'use_custom_env', False))}")
    print(f"observation_dim: {int(env.observation_space.shape[0])}")
    print(f"action_dim: {int(env.action_space.shape[0])}")
    print(f"observation_layout: {spec.layout}")
    print(f"signal_names: {spec.signal_names}")
    print(f"action_names: {spec.action_names}")
    print(f"observation_normalization_scales: {scales}")
    print(f"action_normalization_bounds: {action_bounds}")
    print(f"reset_observation_all_finite: {bool(np.isfinite(obs).all())}")
    print("example_normalized_observation:")
    print(json.dumps(named_obs, indent=2, ensure_ascii=False))
    print("sample_action_checks:")

    for action_idx, action in enumerate(SAMPLE_ACTIONS):
        env.reset(seed=seed + action_idx)
        physical_u_dq = base_env.denormalize_action(action)
        saturated_u_dq = base_env.action_to_voltage_dq(action)
        next_obs, reward, terminated, truncated, _step_info = env.step(action)
        named_next_obs = parse_flat_observation(next_obs, spec=spec)
        prev_u_norm = np.asarray(
            [named_next_obs["prev_u_d"], named_next_obs["prev_u_q"]],
            dtype=np.float64,
        )
        expected_prev_u_norm = saturated_u_dq / max(voltage_limit, 1e-6)
        voltage_within_bounds = bool(np.linalg.norm(saturated_u_dq) <= voltage_limit + 1e-9)
        prev_u_matches = bool(
            np.allclose(prev_u_norm, expected_prev_u_norm, rtol=1e-6, atol=1e-6)
        )

        print(f"- action {action.tolist()}:")
        print(f"  physical_u_dq_before_saturation: {physical_u_dq.astype(float).tolist()}")
        print(f"  physical_u_dq_after_saturation: {saturated_u_dq.astype(float).tolist()}")
        print(
            f"  voltage_within_bounds: {voltage_within_bounds} "
            f"(|u_dq|={float(np.linalg.norm(saturated_u_dq)):.6f}, Umax={voltage_limit:.6f})"
        )
        print(f"  next_observation_all_finite: {bool(np.isfinite(next_obs).all())}")
        print(
            f"  step_status: reward={float(reward):.6f} "
            f"terminated={bool(terminated)} truncated={bool(truncated)}"
        )
        print(f"  normalized_prev_u_dq: {prev_u_norm.astype(float).tolist()}")
        print(f"  expected_normalized_prev_u_dq: {expected_prev_u_norm.astype(float).tolist()}")
        print(f"  prev_u_matches_expected: {prev_u_matches}")

    env.close()


if __name__ == "__main__":
    main()

"""Inspect the live GEM environment and print observation/action semantics.

This script is intentionally lightweight: it does not assume one fixed GEM API,
but it tries to surface the most relevant information needed before changing the
PI baseline or observation parsing logic.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Any

import numpy as np

from envs.make_env import _make_raw_env, make_eval_env
from envs.obs_parser import build_observation_spec, introspect_env, parse_flat_observation
from utils.config import parse_env_config
from utils.experiment_factory import DEFAULT_ENV_CONFIG_PATH, make_env_build_config
from utils.seed import set_seed



def _summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _summarize_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_summarize_value(v) for v in value]
    arr = np.asarray(value)
    return {
        "type": type(value).__name__,
        "shape": list(arr.shape),
        "preview": arr.reshape(-1)[: min(8, arr.size)].astype(float).tolist(),
    }



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect PMSM GEM environment layout")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--as-json", action="store_true", help="Print machine-readable JSON summary")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    env_cfg = parse_env_config(args.env_config)
    set_seed(int(env_cfg.seed))

    build_cfg = make_env_build_config(env_cfg)
    raw_env = _make_raw_env(build_cfg)
    raw_obs, raw_info = raw_env.reset(seed=int(env_cfg.seed))

    wrapped_env = make_eval_env(build_cfg)
    flat_obs, flat_info = wrapped_env.reset(seed=int(env_cfg.seed))

    obs_dim = int(wrapped_env.observation_space.shape[0])
    act_dim = int(wrapped_env.action_space.shape[0])
    spec = build_observation_spec(env_id=env_cfg.env_id, env=wrapped_env)
    env_details = introspect_env(raw_env)
    first_values = np.asarray(flat_obs).reshape(-1)[: min(12, obs_dim)].astype(float).tolist()
    preview_action = np.zeros((act_dim,), dtype=np.float32)
    next_obs, step_reward, terminated, truncated, step_info = wrapped_env.step(preview_action)
    named_obs = parse_flat_observation(flat_obs, spec=spec)
    next_named_obs = parse_flat_observation(next_obs, spec=spec)

    summary = {
        "env_id": env_cfg.env_id,
        "use_custom_env": bool(getattr(env_cfg, "use_custom_env", False)),
        "raw_env_class": type(raw_env).__name__,
        "wrapped_env_class": type(wrapped_env).__name__,
        "observation_space": str(wrapped_env.observation_space),
        "action_space": str(wrapped_env.action_space),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "action_low": np.asarray(wrapped_env.action_space.low).reshape(-1).astype(float).tolist(),
        "action_high": np.asarray(wrapped_env.action_space.high).reshape(-1).astype(float).tolist(),
        "reset_observation_shape": list(np.asarray(flat_obs).shape),
        "first_observation_values": first_values,
        "raw_observation_structure": _summarize_value(raw_obs),
        "raw_info_keys": sorted(list(raw_info.keys())),
        "wrapped_info_keys": sorted(list(flat_info.keys())),
        "introspected_state_names": env_details.state_names,
        "introspected_reference_names": env_details.reference_names,
        "inferred_layout": spec.layout,
        "signal_names": spec.signal_names,
        "action_names": spec.action_names,
        "named_observation_preview": named_obs,
        "step_action_preview": preview_action.astype(float).tolist(),
        "step_reward": float(step_reward),
        "step_terminated": bool(terminated),
        "step_truncated": bool(truncated),
        "step_info_keys": sorted(list(step_info.keys())),
        "step_all_finite": bool(np.isfinite(next_obs).all() and np.isfinite(step_reward)),
        "named_step_observation_preview": next_named_obs,
    }

    if args.as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print(f"env_id: {summary['env_id']}")
    print(f"use_custom_env: {summary['use_custom_env']}")
    print(f"raw_env_class: {summary['raw_env_class']}")
    print(f"wrapped_env_class: {summary['wrapped_env_class']}")
    print(f"observation dim: {summary['obs_dim']}")
    print(f"action dim: {summary['act_dim']}")
    print(f"observation_space: {summary['observation_space']}")
    print(f"action_space: {summary['action_space']}")
    print(f"reset_observation_shape: {summary['reset_observation_shape']}")
    print(f"first_observation_values: {summary['first_observation_values']}")
    print("raw_observation_structure:")
    print(json.dumps(summary["raw_observation_structure"], indent=2, ensure_ascii=False))
    print(f"action_low: {summary['action_low']}")
    print(f"action_high: {summary['action_high']}")
    print(f"raw_info_keys: {summary['raw_info_keys']}")
    print(f"wrapped_info_keys: {summary['wrapped_info_keys']}")
    print(f"introspected_state_names: {summary['introspected_state_names']}")
    print(f"introspected_reference_names: {summary['introspected_reference_names']}")
    print(f"inferred_layout: {summary['inferred_layout']}")
    print(f"signal_names: {summary['signal_names']}")
    print(f"action_names: {summary['action_names']}")
    print("named_observation_preview:")
    print(json.dumps(summary["named_observation_preview"], indent=2, ensure_ascii=False))
    print(f"step_action_preview: {summary['step_action_preview']}")
    print(
        f"step_reward: {summary['step_reward']:.6f} "
        f"terminated={summary['step_terminated']} truncated={summary['step_truncated']} "
        f"all_finite={summary['step_all_finite']}"
    )
    print(f"step_info_keys: {summary['step_info_keys']}")
    print("named_step_observation_preview:")
    print(json.dumps(summary["named_step_observation_preview"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

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

from envs.make_env import EnvBuildConfig, _make_raw_gem_env, make_eval_env
from envs.obs_parser import build_observation_spec, introspect_env, parse_flat_observation
from utils.config import parse_env_config
from utils.experiment_factory import DEFAULT_ENV_CONFIG_PATH
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

    build_cfg = EnvBuildConfig(env_id=env_cfg.env_id, seed=env_cfg.seed, gem_kwargs=dict(env_cfg.gem_kwargs))
    raw_env = _make_raw_gem_env(build_cfg)
    raw_obs, raw_info = raw_env.reset(seed=int(env_cfg.seed))

    wrapped_env = make_eval_env(build_cfg)
    flat_obs, flat_info = wrapped_env.reset(seed=int(env_cfg.seed))

    obs_dim = int(wrapped_env.observation_space.shape[0])
    spec = build_observation_spec(env_id=env_cfg.env_id, env=wrapped_env)
    env_details = introspect_env(raw_env)
    first_values = np.asarray(flat_obs).reshape(-1)[: min(12, obs_dim)].astype(float).tolist()

    summary = {
        "env_id": env_cfg.env_id,
        "raw_env_class": type(raw_env).__name__,
        "wrapped_env_class": type(wrapped_env).__name__,
        "observation_space": str(wrapped_env.observation_space),
        "action_space": str(wrapped_env.action_space),
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
        "inferred_signal_names": spec.signal_names,
        "inferred_action_names": spec.action_names,
        "named_observation_preview": parse_flat_observation(flat_obs, spec=spec),
    }

    if args.as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print(f"env_id: {summary['env_id']}")
    print(f"raw_env_class: {summary['raw_env_class']}")
    print(f"wrapped_env_class: {summary['wrapped_env_class']}")
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
    print(f"inferred_signal_names: {summary['inferred_signal_names']}")
    print(f"inferred_action_names: {summary['inferred_action_names']}")
    print("named_observation_preview:")
    print(json.dumps(summary["named_observation_preview"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Lightweight sanity checks for PMSM reward variants."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.make_env import make_eval_env
from utils.config import parse_env_config
from utils.experiment_factory import make_env_build_config


BASELINE_ENV_CONFIG = Path("configs/env/pmsm_cc_train_external_step_randomtime.yaml")
CONSTRAINT_ENV_CONFIG = Path(
    "configs/env/pmsm_cc_train_external_step_randomtime_constraint_reward.yaml"
)
PHYSICS_ENV_CONFIG = Path(
    "configs/env/pmsm_cc_train_external_step_randomtime_physics_reward_v2.yaml"
)
REQUIRED_COMPONENTS = {
    "reward_tracking",
    "reward_voltage",
    "reward_smoothness",
    "reward_saturation",
    "reward_prestep",
    "reward_total",
}
REQUIRED_PHYSICS_COMPONENTS = {
    "reward_tracking",
    "reward_saturation_margin",
    "reward_smoothness",
    "reward_prestep_current",
    "reward_ff_residual",
    "reward_convergence",
    "reward_total",
    "voltage_utilization",
    "action_delta_norm",
}


def _make_env(config_path: Path):
    cfg = parse_env_config(config_path)
    return make_eval_env(make_env_build_config(cfg, apply_domain_randomization=False))


def _check_finite_step(env, action: np.ndarray) -> dict:
    obs, reward, terminated, truncated, info = env.step(action)
    if not np.isfinite(obs).all():
        raise AssertionError("Environment produced a non-finite observation")
    if not np.isfinite(float(reward)):
        raise AssertionError("Environment produced a non-finite reward")
    if bool(terminated or truncated):
        raise AssertionError(f"Unexpected early episode end: {info}")
    return info


def _exercise_env(
    config_path: Path,
    *,
    required_components: set[str] | None = None,
) -> str:
    env = _make_env(config_path)
    try:
        obs, info = env.reset(seed=123)
        if not np.isfinite(obs).all():
            raise AssertionError("Environment reset produced a non-finite observation")
        mode = str(info.get("reward_mode", ""))
        actions = (
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.15, -0.10], dtype=np.float32),
            np.asarray([0.85, 0.85], dtype=np.float32),
            np.asarray([-0.25, 0.20], dtype=np.float32),
        )
        last_info = info
        for action in actions:
            last_info = _check_finite_step(env, action)

        if required_components:
            base_env = getattr(env, "unwrapped", env)
            reward_term_names = set(getattr(base_env, "reward_term_names", []))
            components = last_info.get("reward_components", {})
            terms = last_info.get("reward_terms", {})
            missing = sorted(
                name
                for name in required_components
                if name not in components and name not in terms and name not in reward_term_names
            )
            if missing:
                raise AssertionError(f"Missing reward component fields: {missing}")
            for name in ("voltage_utilization", "action_delta_norm"):
                if name in required_components:
                    value = terms.get(name, components.get(name, np.nan))
                    if not np.isfinite(float(value)):
                        raise AssertionError(f"Non-finite {name}: {value}")
        return mode
    finally:
        env.close()


def main() -> None:
    baseline_mode = _exercise_env(BASELINE_ENV_CONFIG)
    constraint_mode = _exercise_env(
        CONSTRAINT_ENV_CONFIG,
        required_components=REQUIRED_COMPONENTS,
    )
    physics_mode = _exercise_env(
        PHYSICS_ENV_CONFIG,
        required_components=REQUIRED_PHYSICS_COMPONENTS,
    )
    print(f"baseline_env={BASELINE_ENV_CONFIG} reward_mode={baseline_mode} ok")
    print(f"constraint_env={CONSTRAINT_ENV_CONFIG} reward_mode={constraint_mode} ok")
    print(f"physics_env={PHYSICS_ENV_CONFIG} reward_mode={physics_mode} ok")
    print("reward component sanity check passed")


if __name__ == "__main__":
    main()

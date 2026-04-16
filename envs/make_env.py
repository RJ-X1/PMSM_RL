"""PMSM current-control environment builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym

from envs.wrappers import apply_common_wrappers, make_eval_wrappers, make_train_wrappers

Mode = Literal["train", "eval"]


@dataclass(slots=True)
class EnvBuildConfig:
    """Config placeholder for building PMSM environments."""

    env_id: str = "Cont-CC-PMSM-v0"
    seed: int | None = None
    gem_kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve_env_id(env_id: str) -> str:
    """Resolve a known PMSM current-control env id from the registry."""
    candidates = [env_id, "Cont-CC-PMSM-v0", "DqCont-CC-PMSM-v0"]
    for candidate in candidates:
        if candidate in gym.registry:
            return candidate
    raise ValueError(
        "No supported PMSM current-control env id found in Gym registry. "
        "Checked: " + ", ".join(candidates)
    )


def _make_raw_gem_env(config: EnvBuildConfig) -> gym.Env:
    """Create raw GEM env with minimal assumptions.

    TODO: If future GEM versions change init kwargs or reset/step signatures,
    inspect installed docs/package and update this function.
    """
    import gym_electric_motor as gem

    resolved = _resolve_env_id(config.env_id)
    return gem.make(resolved, **dict(config.gem_kwargs))


def make_env(config: EnvBuildConfig, mode: Mode = "train") -> gym.Env:
    """Build wrapped environment for training or evaluation."""
    env = _make_raw_gem_env(config)
    wrappers = make_train_wrappers() if mode == "train" else make_eval_wrappers()
    env = apply_common_wrappers(env, wrappers)
    if config.seed is not None:
        env.reset(seed=int(config.seed))
    return env


def make_train_env(config: EnvBuildConfig) -> gym.Env:
    """Convenience builder for train-mode environment."""
    return make_env(config=config, mode="train")


def make_eval_env(config: EnvBuildConfig) -> gym.Env:
    """Convenience builder for eval-mode environment."""
    return make_env(config=config, mode="eval")

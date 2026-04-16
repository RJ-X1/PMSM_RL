"""Environment wrappers for PMSM RL experiments."""

from __future__ import annotations

from typing import Any, Iterable

import gymnasium as gym
import numpy as np


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten tuple/dict observations into a 1D Box-friendly vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        sample_obs, _ = self.env.reset()
        flat = self._flatten(sample_obs)

        # 这里只要求 shape 正确，值允许任意有限 float32
        low = np.full(flat.shape, -np.inf, dtype=np.float32)
        high = np.full(flat.shape, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: Any) -> np.ndarray:
        """Convert raw GEM observation to a flat numpy vector."""
        obs = self._flatten(observation)

        # 强制 shape 一致
        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"Flattened observation shape mismatch: got {obs.shape}, "
                f"expected {self.observation_space.shape}"
            )

        # 调试时如果有异常值，打印一次
        if not np.isfinite(obs).all():
            bad_idx = np.where(~np.isfinite(obs))[0].tolist()
            print(f"[FlattenObservationWrapper] non-finite obs at indices {bad_idx}: {obs}")

        return obs

    def _flatten(self, observation: Any) -> np.ndarray:
        parts: list[np.ndarray] = []
        self._collect(observation, parts)
        if not parts:
            return np.zeros((0,), dtype=np.float32)

        obs = np.concatenate(parts).astype(np.float32, copy=False)

        # 把 NaN / Inf 处理掉，避免 passive_env_checker 警告
        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        ).astype(np.float32, copy=False)

        return obs

    def _collect(self, value: Any, out: list[np.ndarray]) -> None:
        if isinstance(value, dict):
            for key in sorted(value.keys()):
                self._collect(value[key], out)
            return
        if isinstance(value, tuple):
            for item in value:
                self._collect(item, out)
            return
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out.append(arr)


class DqPlaceholderWrapper(gym.Wrapper):
    """Placeholder wrapper for future dq-axis preprocessing.

    TODO: Add dq-axis transforms/normalization once control conventions are fixed.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)


def apply_common_wrappers(env: gym.Env, extra_wrappers: Iterable[type[gym.Wrapper]] | None = None) -> gym.Env:
    """Apply stable wrapper stack shared by training and evaluation."""
    wrapped = FlattenObservationWrapper(env)
    for wrapper_cls in extra_wrappers or []:
        wrapped = wrapper_cls(wrapped)
    return wrapped


def make_train_wrappers() -> tuple[type[gym.Wrapper], ...]:
    """Return wrapper placeholders specific to training."""
    # TODO: Add exploration/state augmentation wrappers when needed.
    return (DqPlaceholderWrapper,)


def make_eval_wrappers() -> tuple[type[gym.Wrapper], ...]:
    """Return wrapper placeholders specific to evaluation."""
    # TODO: Add evaluation-only wrappers (e.g., metric taps) when needed.
    return (DqPlaceholderWrapper,)


def make_train_env(env_id: str, gem_kwargs: dict[str, Any] | None = None) -> gym.Env:
    """Backward-compatible train builder; prefer `envs.make_env.make_train_env`."""
    from envs.make_env import EnvBuildConfig, make_train_env as _make_train_env

    return _make_train_env(EnvBuildConfig(env_id=env_id, gem_kwargs=dict(gem_kwargs or {})))


def make_eval_env(env_id: str, gem_kwargs: dict[str, Any] | None = None) -> gym.Env:
    """Backward-compatible eval builder; prefer `envs.make_env.make_eval_env`."""
    from envs.make_env import EnvBuildConfig, make_eval_env as _make_eval_env

    return _make_eval_env(EnvBuildConfig(env_id=env_id, gem_kwargs=dict(gem_kwargs or {})))

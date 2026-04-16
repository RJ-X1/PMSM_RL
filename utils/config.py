"""Configuration helpers and typed placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvConfig:
    """Environment configuration placeholder."""

    env_id: str = 'Cont-CC-PMSM-v0'
    seed: int = 0
    gem_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainConfig:
    """Training configuration placeholder for DDPG experiments."""

    run_name: str = '_old_ddpg_pmsm_cc'
    output_dir: str = 'outputs'
    device: str = 'cpu'
    gamma: float = 0.99
    polyak: float = 0.995
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    replay_capacity: int = 100_000
    batch_size: int = 128
    warmup_steps: int = 2_000
    exploration_noise: float = 0.1
    max_episodes: int = 50
    max_steps_per_episode: int = 500
    eval_every_episodes: int = 10
    eval_episodes: int = 3
    log_every_steps: int = 100
    save_every_episodes: int = 25
    hidden_dim: int = 256


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML as a dictionary."""
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f'Expected mapping at root of {p}, got {type(data)}')
    return data


def parse_env_config(path: str | Path) -> EnvConfig:
    """Parse YAML into :class:`EnvConfig`."""
    data = load_yaml(path)
    return EnvConfig(**data)


def parse_train_config(path: str | Path) -> TrainConfig:
    """Parse YAML into :class:`TrainConfig`."""
    data = load_yaml(path)
    return TrainConfig(**data)

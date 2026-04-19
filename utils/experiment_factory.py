"""Shared experiment construction helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.ddpg_agent import DDPGAgent, DDPGHyperParams
from envs.make_env import EnvBuildConfig


DEFAULT_ENV_CONFIG_PATH = Path("configs/env/pmsm_cc.yaml")
DEFAULT_TRAIN_CONFIG_PATH = Path("configs/train/ddpg_main.yaml")
DEFAULT_EVAL_CONFIG_PATH = Path("configs/eval/default.yaml")
DEFAULT_PI_CONFIG_PATH = Path("configs/pi/pi_default.yaml")
DEFAULT_AGENT_NAME = "ddpg"
DEFAULT_TASK_NAME = "pmsm_current_control"


def make_env_build_config(env_cfg: Any, *, seed_override: int | None = None) -> EnvBuildConfig:
    """Convert parsed env config into an EnvBuildConfig."""
    seed = int(seed_override if seed_override is not None else env_cfg.seed)
    use_custom_env = bool(getattr(env_cfg, "use_custom_env", False))
    custom_env_kwargs = {}
    if use_custom_env and hasattr(env_cfg, "to_custom_env_kwargs"):
        custom_env_kwargs = dict(env_cfg.to_custom_env_kwargs())
    return EnvBuildConfig(
        env_id=str(env_cfg.env_id),
        seed=seed,
        use_custom_env=use_custom_env,
        custom_env_kwargs=custom_env_kwargs,
        gem_kwargs=dict(env_cfg.gem_kwargs),
    )


def build_agent_hparams(train_cfg: Any) -> DDPGHyperParams:
    """Map TrainConfig fields into DDPG hyperparameters."""
    tau = 1.0 - float(train_cfg.polyak)
    tau = min(max(tau, 0.0), 1.0)
    return DDPGHyperParams(
        gamma=float(train_cfg.gamma),
        tau=tau,
        actor_lr=float(train_cfg.lr_actor),
        critic_lr=float(train_cfg.lr_critic),
        hidden_dim=int(train_cfg.hidden_dim),
        policy_noise_std=float(train_cfg.exploration_noise),
    )


def build_ddpg_agent(
    *,
    obs_dim: int,
    act_dim: int,
    train_cfg: Any,
    action_low: float,
    action_high: float,
) -> DDPGAgent:
    """Construct a DDPG agent without coupling train/eval modules."""
    return DDPGAgent(
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        device=train_cfg.device,
        hparams=build_agent_hparams(train_cfg),
        action_low=float(action_low),
        action_high=float(action_high),
    )

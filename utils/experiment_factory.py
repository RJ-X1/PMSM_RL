"""Shared experiment construction helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.ddpg_agent import DDPGAgent, DDPGHyperParams
from agents.td3_agent import TD3Agent, TD3HyperParams
from envs.make_env import EnvBuildConfig


DEFAULT_ENV_CONFIG_PATH = Path("configs/env/pmsm_cc.yaml")
DEFAULT_TRAIN_CONFIG_PATH = Path("configs/train/td3_main.yaml")
DEFAULT_EVAL_CONFIG_PATH = Path("configs/eval/default.yaml")
DEFAULT_PI_CONFIG_PATH = Path("configs/pi/pi_default.yaml")
DEFAULT_AGENT_NAME = "td3"
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
    tau = min(max(float(train_cfg.tau), 0.0), 1.0)
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


def build_td3_hparams(train_cfg: Any) -> TD3HyperParams:
    """Map TrainConfig fields into TD3 hyperparameters."""
    return TD3HyperParams(
        gamma=float(train_cfg.gamma),
        tau=float(train_cfg.tau),
        actor_lr=float(train_cfg.lr_actor),
        critic_lr=float(train_cfg.lr_critic),
        hidden_dim=int(train_cfg.hidden_dim),
        exploration_noise=float(train_cfg.exploration_noise),
        target_policy_noise=float(train_cfg.target_policy_noise),
        target_noise_clip=float(train_cfg.target_noise_clip),
        policy_delay=int(train_cfg.policy_delay),
    )


def build_td3_agent(
    *,
    obs_dim: int,
    act_dim: int,
    train_cfg: Any,
    action_low: float,
    action_high: float,
) -> TD3Agent:
    """Construct a TD3 agent without coupling train/eval modules."""
    return TD3Agent(
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        device=train_cfg.device,
        hparams=build_td3_hparams(train_cfg),
        action_low=float(action_low),
        action_high=float(action_high),
    )


def resolve_algo_name(train_cfg: Any) -> str:
    """Return the configured RL algorithm name."""
    return str(getattr(train_cfg, "algo", "ddpg")).strip().lower()


def resolve_agent_name(train_cfg: Any) -> str:
    """Return the user-facing agent name for metadata/logging."""
    algo = resolve_algo_name(train_cfg)
    if algo in {"td3", "ddpg"}:
        return algo
    return DEFAULT_AGENT_NAME


def build_rl_agent(
    *,
    obs_dim: int,
    act_dim: int,
    train_cfg: Any,
    action_low: float,
    action_high: float,
) -> DDPGAgent | TD3Agent:
    """Construct the configured RL agent while preserving DDPG compatibility."""
    algo = resolve_algo_name(train_cfg)
    if algo == "td3":
        return build_td3_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            train_cfg=train_cfg,
            action_low=action_low,
            action_high=action_high,
        )
    if algo == "ddpg":
        return build_ddpg_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            train_cfg=train_cfg,
            action_low=action_low,
            action_high=action_high,
        )
    raise ValueError(f"Unsupported RL algorithm: {algo}")

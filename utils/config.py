"""Configuration helpers and typed placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from envs.motor_model import PMSMEnvParams, PMSMMotorParams


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{name}' to be a mapping, got {type(value)}")
    return dict(value)


def _as_float_pair(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return float(default[0]), float(default[1])
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-value range, got {value}")
        lo = float(value[0])
        hi = float(value[1])
        return (lo, hi) if lo <= hi else (hi, lo)
    raise TypeError(f"Expected range value to be a list/tuple, got {type(value)}")


@dataclass(slots=True)
class MotorConfig:
    """Motor and inverter parameters for the custom PMSM model."""

    p: float = 4.0
    Rs: float = 0.35
    Ld: float = 2.8e-3
    Lq: float = 3.2e-3
    psi_f: float = 0.095
    J: float = 1.2e-3
    B: float = 1.5e-4
    Vdc: float = 300.0
    Umax: float = 173.0
    Imax: float = 20.0
    Ts: float = 1e-4

    def to_motor_params(self) -> PMSMMotorParams:
        """Convert config values into the runtime motor-parameter dataclass."""
        return PMSMMotorParams(
            p=float(self.p),
            Rs=float(self.Rs),
            Ld=float(self.Ld),
            Lq=float(self.Lq),
            psi_f=float(self.psi_f),
            J=float(self.J),
            B=float(self.B),
            Vdc=float(self.Vdc),
            Umax=float(self.Umax),
            Imax=float(self.Imax),
            Ts=float(self.Ts),
        )


@dataclass(slots=True)
class CustomEnvSettings:
    """Episode length and reset-domain settings for the custom environment."""

    episode_steps: int = 2_000
    init_i_d_range: tuple[float, float] = (0.0, 0.0)
    init_i_q_range: tuple[float, float] = (0.0, 0.0)
    init_omega_m_range: tuple[float, float] = (0.0, 0.0)
    load_torque: float = 0.0
    load_torque_range: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class ReferenceConfig:
    """Reference-current settings for the custom PMSM environment."""

    ref_i_d: float = 0.0
    ref_i_q: float = 10.0
    randomize_on_reset: bool = False
    ref_i_d_range: tuple[float, float] = (0.0, 0.0)
    ref_i_q_range: tuple[float, float] = (10.0, 10.0)


@dataclass(slots=True)
class NoiseConfig:
    """Noise magnitudes injected into the custom environment."""

    observation_noise_std: float = 0.0
    process_noise_std: float = 0.0


@dataclass(slots=True)
class TerminationConfig:
    """Safety and numerical termination settings."""

    terminate_on_overcurrent: bool = True
    current_limit: float | None = None
    current_limit_factor: float = 1.20
    terminate_on_nonfinite: bool = True

    def to_env_params(self, *, episode_steps: int) -> PMSMEnvParams:
        """Convert config values into runtime environment settings."""
        return PMSMEnvParams(
            episode_steps=int(episode_steps),
            current_limit=None if self.current_limit is None else float(self.current_limit),
            current_limit_factor=float(self.current_limit_factor),
            terminate_on_overcurrent=bool(self.terminate_on_overcurrent),
            terminate_on_nonfinite=bool(self.terminate_on_nonfinite),
        )


@dataclass(slots=True)
class EnvConfig:
    """Environment configuration supporting both GEM and custom PMSM modes."""

    env_id: str = "Cont-CC-PMSM-v0"
    seed: int = 0
    gem_kwargs: dict[str, Any] = field(default_factory=dict)
    use_custom_env: bool = False
    motor: MotorConfig = field(default_factory=MotorConfig)
    environment: CustomEnvSettings = field(default_factory=CustomEnvSettings)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)

    def to_custom_env_kwargs(self) -> dict[str, Any]:
        """Build kwargs for :class:`envs.pmsm_current_env.PMSMCurrentControlEnv`."""
        return {
            "env_id": str(self.env_id),
            "motor_params": self.motor.to_motor_params(),
            "env_params": self.termination.to_env_params(
                episode_steps=int(self.environment.episode_steps)
            ),
            "ref_i_d": float(self.reference.ref_i_d),
            "ref_i_q": float(self.reference.ref_i_q),
            "randomize_reference": bool(self.reference.randomize_on_reset),
            "ref_i_d_range": tuple(self.reference.ref_i_d_range),
            "ref_i_q_range": tuple(self.reference.ref_i_q_range),
            "init_i_d_range": tuple(self.environment.init_i_d_range),
            "init_i_q_range": tuple(self.environment.init_i_q_range),
            "init_omega_m_range": tuple(self.environment.init_omega_m_range),
            "load_torque": float(self.environment.load_torque),
            "load_torque_range": tuple(self.environment.load_torque_range),
            "observation_noise_std": float(self.noise.observation_noise_std),
            "process_noise_std": float(self.noise.process_noise_std),
        }


@dataclass(slots=True)
class TrainConfig:
    """Training configuration placeholder for DDPG experiments."""

    run_name: str = "_old_ddpg_pmsm_cc"
    output_dir: str = "outputs"
    device: str = "cpu"
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
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at root of {p}, got {type(data)}")
    return data


def _parse_motor_config(data: dict[str, Any]) -> MotorConfig:
    section = _as_mapping(data.get("motor"), name="motor")
    return MotorConfig(
        p=float(section.get("p", MotorConfig.p)),
        Rs=float(section.get("Rs", MotorConfig.Rs)),
        Ld=float(section.get("Ld", MotorConfig.Ld)),
        Lq=float(section.get("Lq", MotorConfig.Lq)),
        psi_f=float(section.get("psi_f", MotorConfig.psi_f)),
        J=float(section.get("J", MotorConfig.J)),
        B=float(section.get("B", MotorConfig.B)),
        Vdc=float(section.get("Vdc", MotorConfig.Vdc)),
        Umax=float(section.get("Umax", MotorConfig.Umax)),
        Imax=float(section.get("Imax", MotorConfig.Imax)),
        Ts=float(section.get("Ts", MotorConfig.Ts)),
    )


def _parse_environment_settings(data: dict[str, Any]) -> CustomEnvSettings:
    section = _as_mapping(data.get("environment"), name="environment")
    default_load_torque = float(section.get("load_torque", CustomEnvSettings.load_torque))
    return CustomEnvSettings(
        episode_steps=int(section.get("episode_steps", CustomEnvSettings.episode_steps)),
        init_i_d_range=_as_float_pair(
            section.get("init_i_d_range"),
            default=CustomEnvSettings.init_i_d_range,
        ),
        init_i_q_range=_as_float_pair(
            section.get("init_i_q_range"),
            default=CustomEnvSettings.init_i_q_range,
        ),
        init_omega_m_range=_as_float_pair(
            section.get("init_omega_m_range"),
            default=CustomEnvSettings.init_omega_m_range,
        ),
        load_torque=default_load_torque,
        load_torque_range=_as_float_pair(
            section.get("load_torque_range"),
            default=(default_load_torque, default_load_torque),
        ),
    )


def _parse_reference_config(data: dict[str, Any]) -> ReferenceConfig:
    section = _as_mapping(data.get("reference"), name="reference")
    ref_i_d = float(section.get("ref_i_d", section.get("i_d", ReferenceConfig.ref_i_d)))
    ref_i_q = float(section.get("ref_i_q", section.get("i_q", ReferenceConfig.ref_i_q)))
    return ReferenceConfig(
        ref_i_d=ref_i_d,
        ref_i_q=ref_i_q,
        randomize_on_reset=bool(section.get("randomize_on_reset", ReferenceConfig.randomize_on_reset)),
        ref_i_d_range=_as_float_pair(
            section.get("ref_i_d_range"),
            default=(ref_i_d, ref_i_d),
        ),
        ref_i_q_range=_as_float_pair(
            section.get("ref_i_q_range"),
            default=(ref_i_q, ref_i_q),
        ),
    )


def _parse_noise_config(data: dict[str, Any]) -> NoiseConfig:
    section = _as_mapping(data.get("noise"), name="noise")
    return NoiseConfig(
        observation_noise_std=float(
            section.get("observation_noise_std", NoiseConfig.observation_noise_std)
        ),
        process_noise_std=float(section.get("process_noise_std", NoiseConfig.process_noise_std)),
    )


def _parse_termination_config(data: dict[str, Any]) -> TerminationConfig:
    section = _as_mapping(data.get("termination"), name="termination")
    current_limit = section.get("current_limit", TerminationConfig.current_limit)
    return TerminationConfig(
        terminate_on_overcurrent=bool(
            section.get("terminate_on_overcurrent", TerminationConfig.terminate_on_overcurrent)
        ),
        current_limit=None if current_limit is None else float(current_limit),
        current_limit_factor=float(
            section.get("current_limit_factor", TerminationConfig.current_limit_factor)
        ),
        terminate_on_nonfinite=bool(
            section.get("terminate_on_nonfinite", TerminationConfig.terminate_on_nonfinite)
        ),
    )


def parse_env_config(path: str | Path) -> EnvConfig:
    """Parse YAML into :class:`EnvConfig` with backward-compatible defaults."""
    data = load_yaml(path)
    gem_kwargs = _as_mapping(data.get("gem_kwargs"), name="gem_kwargs")
    return EnvConfig(
        env_id=str(data.get("env_id", EnvConfig.env_id)),
        seed=int(data.get("seed", EnvConfig.seed)),
        gem_kwargs=gem_kwargs,
        use_custom_env=bool(data.get("use_custom_env", False)),
        motor=_parse_motor_config(data),
        environment=_parse_environment_settings(data),
        reference=_parse_reference_config(data),
        noise=_parse_noise_config(data),
        termination=_parse_termination_config(data),
    )


def parse_train_config(path: str | Path) -> TrainConfig:
    """Parse YAML into :class:`TrainConfig`."""
    data = load_yaml(path)
    return TrainConfig(**data)

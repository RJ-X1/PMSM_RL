"""Configuration helpers and typed placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
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


def _as_int_pair(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return int(default[0]), int(default[1])
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-value integer range, got {value}")
        lo = int(value[0])
        hi = int(value[1])
        return (lo, hi) if lo <= hi else (hi, lo)
    raise TypeError(f"Expected integer range value to be a list/tuple, got {type(value)}")


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
class RewardConfig:
    """Reward-mode settings for the custom PMSM environment."""

    mode: str = "vanilla_td3_reward"
    w_ed: float = 0.40
    w_eq: float = 0.40
    w_u: float = 0.08
    w_da: float = 0.07
    w_lim: float = 0.05


@dataclass(slots=True)
class ActionSmoothnessConfig:
    """Explicit action-smoothness regularization settings."""

    enabled: bool = True
    weight: float = 0.07


@dataclass(slots=True)
class DomainRandomizationConfig:
    """Training-time domain-randomization settings for the custom PMSM env."""

    enabled: bool = False
    rs_scale_range: tuple[float, float] = (1.0, 1.0)
    ld_scale_range: tuple[float, float] = (1.0, 1.0)
    lq_scale_range: tuple[float, float] = (1.0, 1.0)
    psi_f_scale_range: tuple[float, float] = (1.0, 1.0)
    j_scale_range: tuple[float, float] = (1.0, 1.0)
    vdc_scale_range: tuple[float, float] = (1.0, 1.0)
    sigma_i_range: tuple[float, float] = (0.0, 0.0)
    sigma_omega_rpm_range: tuple[float, float] = (0.0, 0.0)
    load_torque_range: tuple[float, float] = (0.0, 0.0)
    load_change_interval_steps_range: tuple[int, int] = (0, 0)

    def to_env_kwargs(self) -> dict[str, Any]:
        """Convert config into env-friendly keyword arguments."""
        sigma_omega_rad_range = tuple(
            float(value) * (2.0 * math.pi / 60.0) for value in self.sigma_omega_rpm_range
        )
        return {
            "enabled": bool(self.enabled),
            "rs_scale_range": tuple(self.rs_scale_range),
            "ld_scale_range": tuple(self.ld_scale_range),
            "lq_scale_range": tuple(self.lq_scale_range),
            "psi_f_scale_range": tuple(self.psi_f_scale_range),
            "j_scale_range": tuple(self.j_scale_range),
            "vdc_scale_range": tuple(self.vdc_scale_range),
            "sigma_i_range": tuple(self.sigma_i_range),
            "sigma_omega_range": sigma_omega_rad_range,
            "sigma_omega_rpm_range": tuple(self.sigma_omega_rpm_range),
            "load_torque_range": tuple(self.load_torque_range),
            "load_change_interval_steps_range": tuple(self.load_change_interval_steps_range),
        }


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
    reward: RewardConfig = field(default_factory=RewardConfig)
    action_smoothness: ActionSmoothnessConfig = field(default_factory=ActionSmoothnessConfig)
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)

    def to_custom_env_kwargs(self, *, apply_domain_randomization: bool | None = None) -> dict[str, Any]:
        """Build kwargs for :class:`envs.pmsm_current_env.PMSMCurrentControlEnv`."""
        effective_apply_dr = (
            bool(self.domain_randomization.enabled)
            if apply_domain_randomization is None
            else bool(apply_domain_randomization)
        )
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
            "reward_mode": str(self.reward.mode),
            "reward_w_ed": float(self.reward.w_ed),
            "reward_w_eq": float(self.reward.w_eq),
            "reward_w_u": float(self.reward.w_u),
            "reward_w_da": float(self.reward.w_da),
            "reward_w_lim": float(self.reward.w_lim),
            "action_smoothness_enabled": bool(self.action_smoothness.enabled),
            "action_smoothness_weight": float(self.action_smoothness.weight),
            "domain_randomization": self.domain_randomization.to_env_kwargs(),
            "apply_domain_randomization": effective_apply_dr,
        }


@dataclass(slots=True)
class PIConfig:
    """PI-controller configuration for dq current-control baselines."""

    kp_d: float = 6.0
    ki_d: float = 800.0
    kp_q: float = 6.0
    ki_q: float = 800.0
    integrator_limit_d: float = 120.0
    integrator_limit_q: float = 120.0
    voltage_limit: float | None = None
    action_limit: float = 1.0
    anti_windup: str = "conditional_integration"
    use_resistance_compensation: bool = True
    use_decoupling: bool = True
    use_back_emf_compensation: bool = True
    epsilon_scale: float = float(3.141592653589793)

    def to_controller_config(self) -> "PIControllerConfig":
        """Convert this config into the runtime PI-controller dataclass."""
        from baselines.pi_current_controller import PIControllerConfig

        return PIControllerConfig(
            kp_d=float(self.kp_d),
            ki_d=float(self.ki_d),
            kp_q=float(self.kp_q),
            ki_q=float(self.ki_q),
            integrator_limit_d=float(self.integrator_limit_d),
            integrator_limit_q=float(self.integrator_limit_q),
            voltage_limit=None if self.voltage_limit is None else float(self.voltage_limit),
            action_limit=float(self.action_limit),
            anti_windup=str(self.anti_windup),
            use_resistance_compensation=bool(self.use_resistance_compensation),
            use_decoupling=bool(self.use_decoupling),
            use_back_emf_compensation=bool(self.use_back_emf_compensation),
            epsilon_scale=float(self.epsilon_scale),
        )


@dataclass(slots=True)
class TrainConfig:
    """Training configuration supporting legacy DDPG and new TD3 runs."""

    algo: str = "ddpg"
    run_name: str = "_old_ddpg_pmsm_cc"
    output_dir: str = "outputs"
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    polyak: float | None = None
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    replay_capacity: int = 100_000
    batch_size: int = 128
    warmup_steps: int = 2_000
    update_after: int = 0
    total_steps: int | None = None
    max_episodes: int | None = 50
    max_steps_per_episode: int | None = None
    hidden_dim: int = 256
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    policy_delay: int = 2
    eval_every_steps: int | None = None
    eval_every_episodes: int | None = 10
    eval_episodes: int = 3
    log_every_steps: int = 100
    save_every_steps: int | None = None
    save_every_episodes: int | None = 25
    reward: RewardConfig | None = None
    action_smoothness: ActionSmoothnessConfig | None = None
    domain_randomization: DomainRandomizationConfig | None = None


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
    defaults = MotorConfig()
    section = _as_mapping(data.get("motor"), name="motor")
    return MotorConfig(
        p=float(section.get("p", defaults.p)),
        Rs=float(section.get("Rs", defaults.Rs)),
        Ld=float(section.get("Ld", defaults.Ld)),
        Lq=float(section.get("Lq", defaults.Lq)),
        psi_f=float(section.get("psi_f", defaults.psi_f)),
        J=float(section.get("J", defaults.J)),
        B=float(section.get("B", defaults.B)),
        Vdc=float(section.get("Vdc", defaults.Vdc)),
        Umax=float(section.get("Umax", defaults.Umax)),
        Imax=float(section.get("Imax", defaults.Imax)),
        Ts=float(section.get("Ts", defaults.Ts)),
    )


def _parse_environment_settings(data: dict[str, Any]) -> CustomEnvSettings:
    defaults = CustomEnvSettings()
    section = _as_mapping(data.get("environment"), name="environment")
    default_load_torque = float(section.get("load_torque", defaults.load_torque))
    return CustomEnvSettings(
        episode_steps=int(section.get("episode_steps", defaults.episode_steps)),
        init_i_d_range=_as_float_pair(
            section.get("init_i_d_range"),
            default=defaults.init_i_d_range,
        ),
        init_i_q_range=_as_float_pair(
            section.get("init_i_q_range"),
            default=defaults.init_i_q_range,
        ),
        init_omega_m_range=_as_float_pair(
            section.get("init_omega_m_range"),
            default=defaults.init_omega_m_range,
        ),
        load_torque=default_load_torque,
        load_torque_range=_as_float_pair(
            section.get("load_torque_range"),
            default=(default_load_torque, default_load_torque),
        ),
    )


def _parse_reference_config(data: dict[str, Any]) -> ReferenceConfig:
    defaults = ReferenceConfig()
    section = _as_mapping(data.get("reference"), name="reference")
    ref_i_d = float(section.get("ref_i_d", section.get("i_d", defaults.ref_i_d)))
    ref_i_q = float(section.get("ref_i_q", section.get("i_q", defaults.ref_i_q)))
    return ReferenceConfig(
        ref_i_d=ref_i_d,
        ref_i_q=ref_i_q,
        randomize_on_reset=bool(section.get("randomize_on_reset", defaults.randomize_on_reset)),
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
    defaults = NoiseConfig()
    section = _as_mapping(data.get("noise"), name="noise")
    return NoiseConfig(
        observation_noise_std=float(
            section.get("observation_noise_std", defaults.observation_noise_std)
        ),
        process_noise_std=float(section.get("process_noise_std", defaults.process_noise_std)),
    )


def _parse_termination_config(data: dict[str, Any]) -> TerminationConfig:
    defaults = TerminationConfig()
    section = _as_mapping(data.get("termination"), name="termination")
    current_limit = section.get("current_limit", defaults.current_limit)
    return TerminationConfig(
        terminate_on_overcurrent=bool(
            section.get("terminate_on_overcurrent", defaults.terminate_on_overcurrent)
        ),
        current_limit=None if current_limit is None else float(current_limit),
        current_limit_factor=float(
            section.get("current_limit_factor", defaults.current_limit_factor)
        ),
        terminate_on_nonfinite=bool(
            section.get("terminate_on_nonfinite", defaults.terminate_on_nonfinite)
        ),
    )


def _build_reward_config(section: Mapping[str, Any], defaults: RewardConfig) -> RewardConfig:
    return RewardConfig(
        mode=str(section.get("mode", defaults.mode)).strip().lower(),
        w_ed=float(section.get("w_ed", defaults.w_ed)),
        w_eq=float(section.get("w_eq", defaults.w_eq)),
        w_u=float(section.get("w_u", defaults.w_u)),
        w_da=float(section.get("w_da", defaults.w_da)),
        w_lim=float(section.get("w_lim", defaults.w_lim)),
    )


def _parse_env_reward_config(data: dict[str, Any]) -> RewardConfig:
    defaults = RewardConfig()
    section = _as_mapping(data.get("reward"), name="reward")
    return _build_reward_config(section, defaults)


def _parse_train_reward_config(data: dict[str, Any]) -> RewardConfig | None:
    if "reward" not in data:
        return None
    defaults = RewardConfig()
    section = _as_mapping(data.get("reward"), name="reward")
    return _build_reward_config(section, defaults)


def _build_action_smoothness_config(
    section: Mapping[str, Any],
    *,
    default_enabled: bool,
    default_weight: float,
) -> ActionSmoothnessConfig:
    return ActionSmoothnessConfig(
        enabled=bool(section.get("enabled", default_enabled)),
        weight=float(section.get("weight", default_weight)),
    )


def _parse_env_action_smoothness_config(
    data: dict[str, Any],
    *,
    reward_cfg: RewardConfig,
) -> ActionSmoothnessConfig:
    section = _as_mapping(data.get("action_smoothness"), name="action_smoothness")
    return _build_action_smoothness_config(
        section,
        default_enabled=True,
        default_weight=float(reward_cfg.w_da),
    )


def _parse_train_action_smoothness_config(
    data: dict[str, Any],
    *,
    reward_cfg: RewardConfig | None,
) -> ActionSmoothnessConfig | None:
    if "action_smoothness" not in data:
        return None
    section = _as_mapping(data.get("action_smoothness"), name="action_smoothness")
    default_weight = float(reward_cfg.w_da) if reward_cfg is not None else float(ActionSmoothnessConfig().weight)
    return _build_action_smoothness_config(
        section,
        default_enabled=True,
        default_weight=default_weight,
    )


def _build_domain_randomization_config(
    section: Mapping[str, Any],
    defaults: DomainRandomizationConfig,
) -> DomainRandomizationConfig:
    return DomainRandomizationConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        rs_scale_range=_as_float_pair(
            section.get("rs_scale_range"),
            default=defaults.rs_scale_range,
        ),
        ld_scale_range=_as_float_pair(
            section.get("ld_scale_range"),
            default=defaults.ld_scale_range,
        ),
        lq_scale_range=_as_float_pair(
            section.get("lq_scale_range"),
            default=defaults.lq_scale_range,
        ),
        psi_f_scale_range=_as_float_pair(
            section.get("psi_f_scale_range"),
            default=defaults.psi_f_scale_range,
        ),
        j_scale_range=_as_float_pair(
            section.get("j_scale_range"),
            default=defaults.j_scale_range,
        ),
        vdc_scale_range=_as_float_pair(
            section.get("vdc_scale_range"),
            default=defaults.vdc_scale_range,
        ),
        sigma_i_range=_as_float_pair(
            section.get("sigma_i_range"),
            default=defaults.sigma_i_range,
        ),
        sigma_omega_rpm_range=_as_float_pair(
            section.get("sigma_omega_rpm_range"),
            default=defaults.sigma_omega_rpm_range,
        ),
        load_torque_range=_as_float_pair(
            section.get("load_torque_range"),
            default=defaults.load_torque_range,
        ),
        load_change_interval_steps_range=_as_int_pair(
            section.get("load_change_interval_steps_range"),
            default=defaults.load_change_interval_steps_range,
        ),
    )


def _parse_env_domain_randomization_config(data: dict[str, Any]) -> DomainRandomizationConfig:
    defaults = DomainRandomizationConfig()
    section = _as_mapping(data.get("domain_randomization"), name="domain_randomization")
    return _build_domain_randomization_config(section, defaults)


def _parse_train_domain_randomization_config(
    data: dict[str, Any],
) -> DomainRandomizationConfig | None:
    if "domain_randomization" not in data:
        return None
    defaults = DomainRandomizationConfig()
    section = _as_mapping(data.get("domain_randomization"), name="domain_randomization")
    return _build_domain_randomization_config(section, defaults)


def _parse_pi_config(data: dict[str, Any]) -> PIConfig:
    defaults = PIConfig()
    return PIConfig(
        kp_d=float(data.get("kp_d", defaults.kp_d)),
        ki_d=float(data.get("ki_d", defaults.ki_d)),
        kp_q=float(data.get("kp_q", defaults.kp_q)),
        ki_q=float(data.get("ki_q", defaults.ki_q)),
        integrator_limit_d=float(data.get("integrator_limit_d", defaults.integrator_limit_d)),
        integrator_limit_q=float(data.get("integrator_limit_q", defaults.integrator_limit_q)),
        voltage_limit=None if data.get("voltage_limit", defaults.voltage_limit) is None else float(data["voltage_limit"]),
        action_limit=float(data.get("action_limit", defaults.action_limit)),
        anti_windup=str(data.get("anti_windup", defaults.anti_windup)),
        use_resistance_compensation=bool(
            data.get("use_resistance_compensation", defaults.use_resistance_compensation)
        ),
        use_decoupling=bool(data.get("use_decoupling", defaults.use_decoupling)),
        use_back_emf_compensation=bool(
            data.get("use_back_emf_compensation", defaults.use_back_emf_compensation)
        ),
        epsilon_scale=float(data.get("epsilon_scale", defaults.epsilon_scale)),
    )


def _parse_train_config(data: dict[str, Any]) -> TrainConfig:
    defaults = TrainConfig()
    tau_value = data.get("tau")
    polyak_value = data.get("polyak")
    total_steps_value = data.get("total_steps", defaults.total_steps)
    max_episodes_value = data.get("max_episodes", defaults.max_episodes)
    max_steps_per_episode_value = data.get("max_steps_per_episode", defaults.max_steps_per_episode)
    eval_every_steps_value = data.get("eval_every_steps", defaults.eval_every_steps)
    eval_every_episodes_value = data.get("eval_every_episodes", defaults.eval_every_episodes)
    save_every_steps_value = data.get("save_every_steps", defaults.save_every_steps)
    save_every_episodes_value = data.get("save_every_episodes", defaults.save_every_episodes)
    if tau_value is None and polyak_value is not None:
        tau_value = 1.0 - float(polyak_value)
    train_reward = _parse_train_reward_config(data)

    return TrainConfig(
        algo=str(data.get("algo", defaults.algo)).lower(),
        run_name=str(data.get("run_name", defaults.run_name)),
        output_dir=str(data.get("output_dir", defaults.output_dir)),
        device=str(data.get("device", defaults.device)),
        gamma=float(data.get("gamma", defaults.gamma)),
        tau=float(tau_value if tau_value is not None else defaults.tau),
        polyak=None if polyak_value is None else float(polyak_value),
        lr_actor=float(data.get("lr_actor", defaults.lr_actor)),
        lr_critic=float(data.get("lr_critic", defaults.lr_critic)),
        replay_capacity=int(data.get("replay_capacity", defaults.replay_capacity)),
        batch_size=int(data.get("batch_size", defaults.batch_size)),
        warmup_steps=int(data.get("warmup_steps", defaults.warmup_steps)),
        update_after=int(data.get("update_after", defaults.update_after)),
        total_steps=None if total_steps_value is None else int(total_steps_value),
        max_episodes=None if max_episodes_value is None else int(max_episodes_value),
        max_steps_per_episode=None if max_steps_per_episode_value is None else int(max_steps_per_episode_value),
        hidden_dim=int(data.get("hidden_dim", defaults.hidden_dim)),
        exploration_noise=float(data.get("exploration_noise", defaults.exploration_noise)),
        target_policy_noise=float(data.get("target_policy_noise", defaults.target_policy_noise)),
        target_noise_clip=float(data.get("target_noise_clip", defaults.target_noise_clip)),
        policy_delay=int(data.get("policy_delay", defaults.policy_delay)),
        eval_every_steps=None if eval_every_steps_value is None else int(eval_every_steps_value),
        eval_every_episodes=None if eval_every_episodes_value is None else int(eval_every_episodes_value),
        eval_episodes=int(data.get("eval_episodes", defaults.eval_episodes)),
        log_every_steps=int(data.get("log_every_steps", defaults.log_every_steps)),
        save_every_steps=None if save_every_steps_value is None else int(save_every_steps_value),
        save_every_episodes=None if save_every_episodes_value is None else int(save_every_episodes_value),
        reward=train_reward,
        action_smoothness=_parse_train_action_smoothness_config(
            data,
            reward_cfg=train_reward,
        ),
        domain_randomization=_parse_train_domain_randomization_config(data),
    )


def parse_env_config(path: str | Path) -> EnvConfig:
    """Parse YAML into :class:`EnvConfig` with backward-compatible defaults."""
    data = load_yaml(path)
    defaults = EnvConfig()
    gem_kwargs = _as_mapping(data.get("gem_kwargs"), name="gem_kwargs")
    reward_cfg = _parse_env_reward_config(data)
    return EnvConfig(
        env_id=str(data.get("env_id", defaults.env_id)),
        seed=int(data.get("seed", defaults.seed)),
        gem_kwargs=gem_kwargs,
        use_custom_env=bool(data.get("use_custom_env", defaults.use_custom_env)),
        motor=_parse_motor_config(data),
        environment=_parse_environment_settings(data),
        reference=_parse_reference_config(data),
        noise=_parse_noise_config(data),
        termination=_parse_termination_config(data),
        reward=reward_cfg,
        action_smoothness=_parse_env_action_smoothness_config(
            data,
            reward_cfg=reward_cfg,
        ),
        domain_randomization=_parse_env_domain_randomization_config(data),
    )


def parse_train_config(path: str | Path) -> TrainConfig:
    """Parse YAML into :class:`TrainConfig`."""
    data = load_yaml(path)
    return _parse_train_config(data)


def parse_pi_config(path: str | Path) -> PIConfig:
    """Parse YAML into :class:`PIConfig`."""
    data = load_yaml(path)
    return _parse_pi_config(data)


def apply_train_reward_override(env_cfg: EnvConfig, train_cfg: TrainConfig) -> EnvConfig:
    """Return an env config with any custom-env reward override applied."""
    reward_override = getattr(train_cfg, "reward", None)
    if reward_override is None:
        return env_cfg
    return replace(env_cfg, reward=reward_override)


def apply_train_domain_randomization_override(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
) -> EnvConfig:
    """Return an env config with any training-only DR override applied."""
    dr_override = getattr(train_cfg, "domain_randomization", None)
    if dr_override is None:
        return env_cfg
    return replace(env_cfg, domain_randomization=dr_override)


def apply_train_action_smoothness_override(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
) -> EnvConfig:
    """Return an env config with any explicit action-smoothness override applied."""
    smoothness_override = getattr(train_cfg, "action_smoothness", None)
    if smoothness_override is None:
        return env_cfg
    return replace(env_cfg, action_smoothness=smoothness_override)

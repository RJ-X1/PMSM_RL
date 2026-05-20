"""Parameter helpers and sensor utilities for gun-servo simulations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


DEG_TO_RAD = float(np.pi / 180.0)
RAD_TO_DEG = float(180.0 / np.pi)
RPM_TO_RAD_S = float(2.0 * np.pi / 60.0)


def _first(config: dict[str, Any], names: Iterable[str], default: Any) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


def _deg(value: Any, default: float) -> float:
    return float(_first(dict(value or {}), ("value",), default)) * DEG_TO_RAD


def deg_to_rad(value: float) -> float:
    return float(value) * DEG_TO_RAD


def rpm_to_rad_s(value: float) -> float:
    return float(value) * RPM_TO_RAD_S


def arcmin_to_rad(value: float) -> float:
    return deg_to_rad(float(value) / 60.0)


def arcsec_to_rad(value: float) -> float:
    return deg_to_rad(float(value) / 3600.0)


def degree_range_to_rad(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    raw = default if value is None else value
    lo, hi = float(raw[0]), float(raw[1])
    if lo > hi:
        lo, hi = hi, lo
    return deg_to_rad(lo), deg_to_rad(hi)


@dataclass(slots=True)
class MotorSpec:
    """PMSM parameters used by the equivalent inner-loop model."""

    pole_pairs: float = 4.0
    rated_torque_nm: float = 50.0
    max_torque_nm: float = 135.0
    rated_speed_rad_s: float = rpm_to_rad_s(1500.0)
    max_speed_rad_s: float = rpm_to_rad_s(1800.0)
    rated_current_a: float = 15.4
    max_current_a: float = 43.2
    kt_nm_per_a: float = 3.24
    rs_ohm: float = 0.480
    ld_h: float = 8.08e-3
    lq_h: float = 8.08e-3
    jm_kg_m2: float = 7.5e-3

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MotorSpec":
        defaults = cls()
        pole_pairs = _first(config, ("pole_pairs", "p"), defaults.pole_pairs)
        kt = _first(config, ("kt_nm_per_a", "Kt"), defaults.kt_nm_per_a)
        psi_f = config.get("psi_f")
        if "kt_nm_per_a" not in config and "Kt" not in config and psi_f is not None:
            kt = 1.5 * float(pole_pairs) * float(psi_f)
        return cls(
            pole_pairs=float(pole_pairs),
            rated_torque_nm=float(_first(config, ("rated_torque_nm",), defaults.rated_torque_nm)),
            max_torque_nm=float(_first(config, ("max_torque_nm",), defaults.max_torque_nm)),
            rated_speed_rad_s=rpm_to_rad_s(
                float(_first(config, ("rated_speed_rpm",), defaults.rated_speed_rad_s / RPM_TO_RAD_S))
            ),
            max_speed_rad_s=rpm_to_rad_s(
                float(_first(config, ("max_speed_rpm",), defaults.max_speed_rad_s / RPM_TO_RAD_S))
            ),
            rated_current_a=float(_first(config, ("rated_current_a", "I_rated"), defaults.rated_current_a)),
            max_current_a=float(_first(config, ("max_current_a", "Imax"), defaults.max_current_a)),
            kt_nm_per_a=float(kt),
            rs_ohm=float(_first(config, ("rs_ohm", "Rs"), defaults.rs_ohm)),
            ld_h=float(_first(config, ("ld_h", "Ld"), defaults.ld_h)),
            lq_h=float(_first(config, ("lq_h", "Lq"), defaults.lq_h)),
            jm_kg_m2=float(_first(config, ("jm_kg_m2", "Jm", "J"), defaults.jm_kg_m2)),
        )


@dataclass(slots=True)
class ServoDriveSpec:
    """Equivalent speed-mode drive settings."""

    current_limit_a: float = 43.2
    speed_loop_gain_hz: float = 25.0
    speed_loop_integral_time_s: float = 31.83e-3
    torque_filter_time_s: float = 0.79e-3

    @classmethod
    def from_config(cls, config: dict[str, Any], *, motor: MotorSpec) -> "ServoDriveSpec":
        defaults = cls(current_limit_a=float(motor.max_current_a))
        ti_ms = _first(config, ("speed_loop_integral_time_ms",), defaults.speed_loop_integral_time_s * 1000.0)
        torque_filter_ms = _first(config, ("torque_filter_time_ms",), defaults.torque_filter_time_s * 1000.0)
        return cls(
            current_limit_a=float(_first(config, ("current_limit_a", "iq_limit"), defaults.current_limit_a)),
            speed_loop_gain_hz=float(
                _first(config, ("speed_loop_gain_hz", "bandwidth_hz"), defaults.speed_loop_gain_hz)
            ),
            speed_loop_integral_time_s=max(float(ti_ms) * 1e-3, 1e-6),
            torque_filter_time_s=max(float(torque_filter_ms) * 1e-3, 0.0),
        )


@dataclass(slots=True)
class GearboxSpec:
    """Reduction gear parameters and output torque constraints."""

    ratio: float = 50.0
    efficiency_nominal: float = 0.70
    backlash_rad: float = arcmin_to_rad(2.0)
    torsional_stiffness_nm_per_rad: float = 11.6e4
    torsional_damping_nms_per_rad: float = 0.0
    output_torque_limit_nm: float = 1800.0
    buckling_torque_nm: float = 3600.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GearboxSpec":
        defaults = cls()
        backlash_rad = _first(config, ("backlash_rad",), None)
        if backlash_rad is None:
            backlash_rad = arcmin_to_rad(float(_first(config, ("backlash_arcmin",), 2.0)))
        return cls(
            ratio=float(_first(config, ("ratio", "gear_ratio"), defaults.ratio)),
            efficiency_nominal=float(
                _first(config, ("efficiency_nominal", "efficiency"), defaults.efficiency_nominal)
            ),
            backlash_rad=float(backlash_rad),
            torsional_stiffness_nm_per_rad=float(
                _first(
                    config,
                    ("torsional_stiffness_nm_per_rad", "torsional_stiffness"),
                    defaults.torsional_stiffness_nm_per_rad,
                )
            ),
            torsional_damping_nms_per_rad=float(
                _first(
                    config,
                    ("torsional_damping_nms_per_rad", "torsional_damping"),
                    defaults.torsional_damping_nms_per_rad,
                )
            ),
            output_torque_limit_nm=float(
                _first(config, ("output_torque_limit_nm",), defaults.output_torque_limit_nm)
            ),
            buckling_torque_nm=float(_first(config, ("buckling_torque_nm",), defaults.buckling_torque_nm)),
        )

    def motor_to_load_torque(self, motor_torque_nm: float) -> tuple[float, bool]:
        raw = float(self.efficiency_nominal) * float(self.ratio) * float(motor_torque_nm)
        limited = float(np.clip(raw, -float(self.output_torque_limit_nm), float(self.output_torque_limit_nm)))
        return limited, bool(abs(raw - limited) > 1e-9)


@dataclass(slots=True)
class LoadSpec:
    """Load-side dynamics and constraints."""

    theta_range_rad: tuple[float, float] = (-30.0 * DEG_TO_RAD, 30.0 * DEG_TO_RAD)
    omega_limit_rad_s: float = 30.0 * DEG_TO_RAD
    alpha_limit_rad_s2: float = 150.0 * DEG_TO_RAD
    inertia_kg_m2: float = 300.0
    viscous_damping_nms_per_rad: float = 40.0
    coulomb_friction_nm: float = 50.0
    gravity_torque_coeff_nm: float = 300.0
    gravity_phase_rad: float = 0.0
    disturbance_torque_nm: float = 0.0
    disturbance_step_nm: float = 0.0
    disturbance_step_time_s: float = 0.5
    disturbance_noise_std_nm: float = 0.0
    friction_eps_rad_s: float = 1e-3

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LoadSpec":
        defaults = cls()
        theta_range = degree_range_to_rad(
            _first(config, ("theta_range_deg",), None),
            tuple(value * RAD_TO_DEG for value in defaults.theta_range_rad),
        )
        return cls(
            theta_range_rad=theta_range,
            omega_limit_rad_s=deg_to_rad(
                float(_first(config, ("omega_limit_deg_s",), defaults.omega_limit_rad_s * RAD_TO_DEG))
            ),
            alpha_limit_rad_s2=deg_to_rad(
                float(_first(config, ("alpha_limit_deg_s2",), defaults.alpha_limit_rad_s2 * RAD_TO_DEG))
            ),
            inertia_kg_m2=float(_first(config, ("inertia_kg_m2", "J_load"), defaults.inertia_kg_m2)),
            viscous_damping_nms_per_rad=float(
                _first(config, ("viscous_damping_nms_per_rad", "B_load"), defaults.viscous_damping_nms_per_rad)
            ),
            coulomb_friction_nm=float(
                _first(config, ("coulomb_friction_nm", "coulomb_friction"), defaults.coulomb_friction_nm)
            ),
            gravity_torque_coeff_nm=float(
                _first(
                    config,
                    ("gravity_torque_coeff_nm", "gravity_torque_coeff"),
                    defaults.gravity_torque_coeff_nm,
                )
            ),
            gravity_phase_rad=float(_first(config, ("gravity_phase_rad", "gravity_phase"), defaults.gravity_phase_rad)),
            disturbance_torque_nm=float(
                _first(config, ("disturbance_torque_nm", "disturbance_torque"), defaults.disturbance_torque_nm)
            ),
            disturbance_step_nm=float(_first(config, ("disturbance_step_nm",), defaults.disturbance_step_nm)),
            disturbance_step_time_s=float(
                _first(config, ("disturbance_step_time_s",), defaults.disturbance_step_time_s)
            ),
            disturbance_noise_std_nm=float(
                _first(config, ("disturbance_noise_std_nm",), defaults.disturbance_noise_std_nm)
            ),
            friction_eps_rad_s=max(
                float(_first(config, ("friction_eps_rad_s",), defaults.friction_eps_rad_s)),
                1e-9,
            ),
        )


@dataclass(slots=True)
class EncoderSpec:
    """Load encoder model."""

    resolution_bits: int = 20
    accuracy_rad: float = arcsec_to_rad(2.0)
    noise_std_rad: float = 1e-5
    delay_steps: int = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EncoderSpec":
        defaults = cls()
        accuracy_rad = _first(config, ("accuracy_rad",), None)
        if accuracy_rad is None:
            accuracy_rad = arcsec_to_rad(float(_first(config, ("accuracy_arcsec",), 2.0)))
        return cls(
            resolution_bits=int(_first(config, ("resolution_bits",), defaults.resolution_bits)),
            accuracy_rad=float(accuracy_rad),
            noise_std_rad=float(_first(config, ("noise_std_rad",), defaults.noise_std_rad)),
            delay_steps=max(0, int(_first(config, ("delay_steps",), defaults.delay_steps))),
        )


class LoadEncoder:
    """Quantized delayed load-side encoder."""

    def __init__(self, spec: EncoderSpec | None = None) -> None:
        self.spec = spec or EncoderSpec()
        self._buffer: deque[tuple[float, float]] = deque(maxlen=int(self.spec.delay_steps) + 1)

    def reset(self, *, theta: float, omega: float) -> None:
        self._buffer.clear()
        for _ in range(int(self.spec.delay_steps) + 1):
            self._buffer.append((float(theta), float(omega)))

    def measure(self, *, theta: float, omega: float, rng: np.random.Generator) -> tuple[float, float]:
        self._buffer.append((float(theta), float(omega)))
        delayed_theta, delayed_omega = self._buffer[0]
        bits = max(1, int(self.spec.resolution_bits))
        quantum = float(2.0 * np.pi / (2**bits))
        theta_meas = float(np.round(float(delayed_theta) / quantum) * quantum)
        total_noise_std = float(np.hypot(float(self.spec.noise_std_rad), float(self.spec.accuracy_rad) / 3.0))
        if total_noise_std > 0.0:
            theta_meas += float(rng.normal(0.0, total_noise_std))
        return float(theta_meas), float(delayed_omega)


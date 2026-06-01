"""Reference trajectories for gun-servo position outer-loop experiments.

These trajectories support the project shift from PMSM current-loop benchmarks
to gun/fire-control servo position tracking simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrajectoryPoint:
    """Reference position, velocity feedforward, and acceleration feedforward."""

    theta_ref: float
    omega_ff: float
    alpha_ff: float


def _as_range(values: tuple[float, float] | list[float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if values is None:
        return default
    lo, hi = float(values[0]), float(values[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _optional_range(values: tuple[float, float] | list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    return _as_range(values, (0.0, 0.0))


class GunServoTrajectoryGenerator:
    """Small trajectory generator for single-axis gun servo tracking.

    All public outputs are in radians, radians per second, and radians per
    second squared.  Config files may still use degree-based values for human
    readability; conversion happens before construction.
    """

    VALID_PROFILES = {"step", "trapezoid", "sinusoidal", "random_step"}

    def __init__(
        self,
        *,
        profile: str = "step",
        theta_range: tuple[float, float] = (-0.35, 0.35),
        theta_initial_range: tuple[float, float] | None = None,
        theta_final_range: tuple[float, float] | None = None,
        max_speed: float = 0.7,
        max_accel: float = 2.0,
        step_time: float = 0.2,
        sine_frequency_hz: float = 0.25,
        step_time_range: tuple[float, float] | None = None,
        max_speed_range: tuple[float, float] | None = None,
        max_accel_range: tuple[float, float] | None = None,
        sine_amplitude_range: tuple[float, float] | None = None,
        sine_frequency_hz_range: tuple[float, float] | None = None,
        sine_phase_range: tuple[float, float] | None = None,
        randomize_on_reset: bool = False,
    ) -> None:
        self.profile = str(profile).strip().lower()
        if self.profile not in self.VALID_PROFILES:
            raise ValueError(f"Unsupported gun-servo trajectory profile: {profile}")
        self.theta_range = _as_range(theta_range, (-0.35, 0.35))
        self.theta_initial_range = None if theta_initial_range is None else _as_range(theta_initial_range, theta_range)
        self.theta_final_range = None if theta_final_range is None else _as_range(theta_final_range, theta_range)
        self.max_speed = max(float(max_speed), 1e-6)
        self.max_accel = max(float(max_accel), 1e-6)
        self.step_time = max(float(step_time), 0.0)
        self.sine_frequency_hz = max(float(sine_frequency_hz), 1e-6)
        self.step_time_range = _optional_range(step_time_range)
        self.max_speed_range = _optional_range(max_speed_range)
        self.max_accel_range = _optional_range(max_accel_range)
        self.sine_amplitude_range = _optional_range(sine_amplitude_range)
        self.sine_frequency_hz_range = _optional_range(sine_frequency_hz_range)
        self.sine_phase_range = _optional_range(sine_phase_range)
        self.randomize_on_reset = bool(randomize_on_reset)
        self.theta_initial = 0.0
        self.theta_final = float(self.theta_range[1])
        self.active_step_time = self.step_time
        self.active_max_speed = self.max_speed
        self.active_max_accel = self.max_accel
        self.active_sine_amplitude = 0.5 * (self.theta_range[1] - self.theta_range[0])
        self.active_sine_frequency_hz = self.sine_frequency_hz
        self.active_sine_phase = 0.0
        self._rng = np.random.default_rng(0)

    def _sample_range(self, value_range: tuple[float, float] | None, default: float) -> float:
        if value_range is None:
            return float(default)
        lo, hi = value_range
        if lo == hi:
            return float(lo)
        return float(self._rng.uniform(lo, hi))

    def reset(self, *, rng: np.random.Generator | None = None) -> None:
        """Sample any episode-specific reference settings."""
        if rng is not None:
            self._rng = rng
        self.active_step_time = max(self._sample_range(self.step_time_range, self.step_time), 0.0)
        self.active_max_speed = max(self._sample_range(self.max_speed_range, self.max_speed), 1e-6)
        self.active_max_accel = max(self._sample_range(self.max_accel_range, self.max_accel), 1e-6)
        default_amp = 0.5 * (self.theta_range[1] - self.theta_range[0])
        self.active_sine_amplitude = max(self._sample_range(self.sine_amplitude_range, default_amp), 0.0)
        self.active_sine_frequency_hz = max(
            self._sample_range(self.sine_frequency_hz_range, self.sine_frequency_hz),
            1e-6,
        )
        self.active_sine_phase = self._sample_range(self.sine_phase_range, 0.0)
        lo, hi = self.theta_range
        if self.theta_initial_range is not None or self.theta_final_range is not None:
            init_lo, init_hi = self.theta_initial_range or self.theta_range
            final_lo, final_hi = self.theta_final_range or self.theta_range
            self.theta_initial = float(self._rng.uniform(init_lo, init_hi))
            self.theta_final = float(self._rng.uniform(final_lo, final_hi))
            if abs(self.theta_final - self.theta_initial) < np.deg2rad(1.0):
                self.theta_final = float(final_hi if self.theta_initial < (final_lo + final_hi) * 0.5 else final_lo)
        elif self.randomize_on_reset or self.profile == "random_step":
            self.theta_initial = float(self._rng.uniform(lo, hi))
            self.theta_final = float(self._rng.uniform(lo, hi))
            if abs(self.theta_final - self.theta_initial) < np.deg2rad(1.0):
                self.theta_final = float(hi if self.theta_initial < (lo + hi) * 0.5 else lo)
        else:
            self.theta_initial = 0.0 if lo <= 0.0 <= hi else float(lo)
            self.theta_final = float(hi)

    def _step(self, t: float) -> TrajectoryPoint:
        theta = self.theta_initial if float(t) < self.active_step_time else self.theta_final
        return TrajectoryPoint(theta_ref=float(theta), omega_ff=0.0, alpha_ff=0.0)

    def _sinusoidal(self, t: float) -> TrajectoryPoint:
        lo, hi = self.theta_range
        bias = 0.5 * (lo + hi)
        amp = min(float(self.active_sine_amplitude), 0.5 * (hi - lo))
        omega = 2.0 * np.pi * self.active_sine_frequency_hz
        phase_t = omega * float(t) + float(self.active_sine_phase)
        theta = bias + amp * np.sin(phase_t)
        omega_ff = amp * omega * np.cos(phase_t)
        alpha_ff = -amp * omega**2 * np.sin(phase_t)
        return TrajectoryPoint(float(theta), float(omega_ff), float(alpha_ff))

    def _trapezoid(self, t: float) -> TrajectoryPoint:
        start = self.theta_initial
        target = self.theta_final
        distance = float(target - start)
        sign = 1.0 if distance >= 0.0 else -1.0
        distance_abs = abs(distance)
        if distance_abs <= 1e-12:
            return TrajectoryPoint(float(target), 0.0, 0.0)

        t = max(float(t) - self.active_step_time, 0.0)
        a = self.active_max_accel
        v = self.active_max_speed
        t_accel = v / a
        d_accel = 0.5 * a * t_accel**2
        if 2.0 * d_accel >= distance_abs:
            t_accel = np.sqrt(distance_abs / a)
            t_flat = 0.0
            v_peak = a * t_accel
        else:
            t_flat = (distance_abs - 2.0 * d_accel) / v
            v_peak = v
        t_total = 2.0 * t_accel + t_flat

        if t <= 0.0:
            pos = 0.0
            vel = 0.0
            acc = 0.0
        elif t < t_accel:
            pos = 0.5 * a * t**2
            vel = a * t
            acc = a
        elif t < t_accel + t_flat:
            tau = t - t_accel
            pos = 0.5 * a * t_accel**2 + v_peak * tau
            vel = v_peak
            acc = 0.0
        elif t < t_total:
            tau = t - t_accel - t_flat
            pos = 0.5 * a * t_accel**2 + v_peak * t_flat + v_peak * tau - 0.5 * a * tau**2
            vel = v_peak - a * tau
            acc = -a
        else:
            pos = distance_abs
            vel = 0.0
            acc = 0.0

        return TrajectoryPoint(
            theta_ref=float(start + sign * pos),
            omega_ff=float(sign * vel),
            alpha_ff=float(sign * acc),
        )

    def sample(self, t: float) -> TrajectoryPoint:
        """Return the trajectory point at time ``t``."""
        if self.profile in {"step", "random_step"}:
            return self._step(t)
        if self.profile == "trapezoid":
            return self._trapezoid(t)
        if self.profile == "sinusoidal":
            return self._sinusoidal(t)
        raise ValueError(f"Unsupported profile: {self.profile}")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GunServoTrajectoryGenerator":
        """Build from an env config dictionary using degree-based values."""
        theta_deg = _as_range(config.get("theta_range_deg"), (-10.0, 10.0))
        theta_initial_deg = config.get("theta_initial_range_deg")
        theta_final_deg = config.get("theta_final_range_deg")
        theta_initial_range = None
        theta_final_range = None
        if theta_initial_deg is not None:
            theta_initial_deg = _as_range(theta_initial_deg, theta_deg)
            theta_initial_range = (float(np.deg2rad(theta_initial_deg[0])), float(np.deg2rad(theta_initial_deg[1])))
        if theta_final_deg is not None:
            theta_final_deg = _as_range(theta_final_deg, theta_deg)
            theta_final_range = (float(np.deg2rad(theta_final_deg[0])), float(np.deg2rad(theta_final_deg[1])))
        return cls(
            profile=str(config.get("profile", "step")),
            theta_range=(float(np.deg2rad(theta_deg[0])), float(np.deg2rad(theta_deg[1]))),
            theta_initial_range=theta_initial_range,
            theta_final_range=theta_final_range,
            max_speed=float(np.deg2rad(float(config.get("max_speed_deg_s", 40.0)))),
            max_accel=float(np.deg2rad(float(config.get("max_accel_deg_s2", 120.0)))),
            step_time=float(config.get("step_time_s", 0.2)),
            sine_frequency_hz=float(config.get("sine_frequency_hz", 0.25)),
            step_time_range=_optional_range(config.get("step_time_s_range")),
            max_speed_range=(
                None
                if config.get("max_speed_deg_s_range") is None
                else tuple(float(np.deg2rad(value)) for value in _as_range(config.get("max_speed_deg_s_range"), (40.0, 40.0)))
            ),
            max_accel_range=(
                None
                if config.get("max_accel_deg_s2_range") is None
                else tuple(float(np.deg2rad(value)) for value in _as_range(config.get("max_accel_deg_s2_range"), (120.0, 120.0)))
            ),
            sine_amplitude_range=(
                None
                if config.get("sine_amplitude_deg_range") is None
                else tuple(float(np.deg2rad(value)) for value in _as_range(config.get("sine_amplitude_deg_range"), (5.0, 5.0)))
            ),
            sine_frequency_hz_range=_optional_range(config.get("sine_frequency_hz_range")),
            sine_phase_range=_optional_range(config.get("sine_phase_range_rad")),
            randomize_on_reset=bool(config.get("randomize_on_reset", False)),
        )

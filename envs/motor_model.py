"""Lightweight dq-axis PMSM motor model utilities.

The goal of this module is to keep the custom current-control environment easy
to read and easy to extend. It intentionally models only the states required by
the paper-style workflow: dq currents and mechanical speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class PMSMMotorParams:
    """Physical and converter limits used by the PMSM dq model."""

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

    def __post_init__(self) -> None:
        positive_fields = {
            "p": self.p,
            "Ld": self.Ld,
            "Lq": self.Lq,
            "J": self.J,
            "Vdc": self.Vdc,
            "Umax": self.Umax,
            "Imax": self.Imax,
            "Ts": self.Ts,
        }
        non_negative_fields = {
            "Rs": self.Rs,
            "psi_f": self.psi_f,
            "B": self.B,
        }
        for name, value in positive_fields.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(slots=True)
class PMSMEnvParams:
    """Environment-level limits that sit on top of the motor model."""

    episode_steps: int = 2_000
    current_limit: float | None = None
    current_limit_factor: float = 1.20
    terminate_on_overcurrent: bool = True
    terminate_on_nonfinite: bool = True

    def __post_init__(self) -> None:
        if int(self.episode_steps) <= 0:
            raise ValueError(f"episode_steps must be positive, got {self.episode_steps}")
        if float(self.current_limit_factor) <= 0.0:
            raise ValueError(
                f"current_limit_factor must be positive, got {self.current_limit_factor}"
            )
        if self.current_limit is not None and float(self.current_limit) <= 0.0:
            raise ValueError(f"current_limit must be positive when provided, got {self.current_limit}")

    def resolved_current_limit(self, motor_params: PMSMMotorParams) -> float:
        """Return the absolute current-magnitude safety threshold."""
        if self.current_limit is not None:
            return float(self.current_limit)
        return float(self.current_limit_factor) * float(motor_params.Imax)


@dataclass(slots=True)
class PMSMState:
    """Minimal continuous-time PMSM state in dq coordinates."""

    i_d: float = 0.0
    i_q: float = 0.0
    omega_m: float = 0.0

    def as_array(self, *, dtype: np.dtype | type[np.floating] = np.float64) -> np.ndarray:
        """Return state as a dense vector."""
        return np.asarray([self.i_d, self.i_q, self.omega_m], dtype=dtype)

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "PMSMState":
        """Construct state from a flat iterable."""
        i_d, i_q, omega_m = np.asarray(list(values), dtype=np.float64).reshape(3)
        return cls(i_d=float(i_d), i_q=float(i_q), omega_m=float(omega_m))


def electrical_speed(state: PMSMState, params: PMSMMotorParams) -> float:
    """Return electrical speed from mechanical speed and pole-pair count."""
    return float(params.p) * float(state.omega_m)


def electromagnetic_torque(state: PMSMState, params: PMSMMotorParams) -> float:
    """Return electromagnetic torque for the current dq state."""
    i_d = float(state.i_d)
    i_q = float(state.i_q)
    return 1.5 * float(params.p) * (
        float(params.psi_f) * i_q + (float(params.Ld) - float(params.Lq)) * i_d * i_q
    )


def dq_derivatives(
    state: PMSMState,
    u_dq: Iterable[float],
    load_torque: float,
    params: PMSMMotorParams,
) -> np.ndarray:
    """Return continuous-time dq current and speed derivatives.

    PMSM equations in the rotor-synchronous dq frame:

    u_d = R_s i_d + L_d di_d/dt - omega_e L_q i_q
    u_q = R_s i_q + L_q di_q/dt + omega_e (L_d i_d + psi_f)
    J domega_m/dt = T_e - T_L - B omega_m
    """

    u_d, u_q = np.asarray(list(u_dq), dtype=np.float64).reshape(2)
    omega_e = electrical_speed(state, params)

    di_d = (u_d - float(params.Rs) * float(state.i_d) + omega_e * float(params.Lq) * float(state.i_q)) / float(
        params.Ld
    )
    di_q = (
        u_q
        - float(params.Rs) * float(state.i_q)
        - omega_e * (float(params.Ld) * float(state.i_d) + float(params.psi_f))
    ) / float(params.Lq)

    torque_e = electromagnetic_torque(state, params)
    domega_m = (
        torque_e - float(load_torque) - float(params.B) * float(state.omega_m)
    ) / float(params.J)
    return np.asarray([di_d, di_q, domega_m], dtype=np.float64)


def clip_voltage_vector(u_dq: Iterable[float], limit: float) -> np.ndarray:
    """Saturate a dq voltage vector to the requested magnitude."""
    vec = np.asarray(list(u_dq), dtype=np.float64).reshape(2)
    magnitude = float(np.linalg.norm(vec))
    if magnitude <= float(limit) or magnitude == 0.0:
        return vec
    return vec * (float(limit) / magnitude)


def rk4_step(
    state: PMSMState,
    u_dq: Iterable[float],
    load_torque: float,
    params: PMSMMotorParams,
    dt: float | None = None,
) -> PMSMState:
    """Advance the PMSM state with a fourth-order Runge-Kutta step."""

    dt_value = float(params.Ts if dt is None else dt)
    y0 = state.as_array(dtype=np.float64)
    u_vec = clip_voltage_vector(u_dq, limit=float(params.Umax))
    load_torque_value = float(load_torque)

    def _f(y: np.ndarray) -> np.ndarray:
        return dq_derivatives(
            PMSMState.from_array(y),
            u_dq=u_vec,
            load_torque=load_torque_value,
            params=params,
        )

    k1 = _f(y0)
    k2 = _f(y0 + 0.5 * dt_value * k1)
    k3 = _f(y0 + 0.5 * dt_value * k2)
    k4 = _f(y0 + dt_value * k3)
    y_next = y0 + (dt_value / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if not np.isfinite(y_next).all():
        raise FloatingPointError(f"RK4 step produced non-finite state: {y_next}")
    return PMSMState.from_array(y_next)

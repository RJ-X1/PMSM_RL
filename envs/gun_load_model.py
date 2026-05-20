"""Single-axis gun load dynamics for position outer-loop RL experiments.

The project is shifting from PMSM dq current-control experiments toward
gun/fire-control servo tracking simulations.  This module deliberately starts
with a compact single-inertia load model while keeping the parameters that a
future two-inertia flexible transmission model will need.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(slots=True)
class GunLoadParams:
    """Load-side equivalent parameters for a geared elevation axis."""

    gear_ratio: float = 120.0
    efficiency: float = 0.85
    J_load: float = 85.0
    B_load: float = 12.0
    coulomb_friction: float = 3.0
    gravity_torque_coeff: float = 45.0
    gravity_phase: float = 0.0
    backlash_rad: float = 0.0
    torsional_stiffness: float = 0.0
    torsional_damping: float = 0.0


@dataclass(slots=True)
class GunLoadState:
    """Load-side state in radians and radians per second."""

    theta: float = 0.0
    omega: float = 0.0


class SingleInertiaGunLoad:
    """Minimal single-inertia gun load model.

    Dynamics:
        J_eq * theta_ddot =
            T_out
            - B_eq * theta_dot
            - T_c * sign(theta_dot)
            - K_g * sin(theta + phi_g)
            - T_dist

    where T_out = gear_ratio * efficiency * T_e.
    """

    def __init__(self, params: GunLoadParams | None = None) -> None:
        self.params = replace(params or GunLoadParams())

    def motor_to_load_torque(self, motor_torque: float) -> float:
        """Map motor electromagnetic torque to load-side output torque."""
        return float(self.params.gear_ratio) * float(self.params.efficiency) * float(motor_torque)

    def gravity_torque(self, theta: float) -> float:
        """Gravity imbalance torque at the load side."""
        return float(self.params.gravity_torque_coeff) * float(
            np.sin(float(theta) + float(self.params.gravity_phase))
        )

    def coulomb_torque(self, omega: float) -> float:
        """Smooth Coulomb friction sign to avoid numerical chatter near zero."""
        omega_smooth = 1e-3
        return float(self.params.coulomb_friction) * float(np.tanh(float(omega) / omega_smooth))

    def acceleration(
        self,
        state: GunLoadState,
        *,
        motor_torque: float,
        disturbance_torque: float = 0.0,
    ) -> float:
        """Return load-side angular acceleration."""
        j_eq = max(float(self.params.J_load), 1e-9)
        t_out = self.motor_to_load_torque(motor_torque)
        damping = float(self.params.B_load) * float(state.omega)
        friction = self.coulomb_torque(float(state.omega))
        gravity = self.gravity_torque(float(state.theta))
        net = t_out - damping - friction - gravity - float(disturbance_torque)
        return float(net / j_eq)

    def step(
        self,
        state: GunLoadState,
        *,
        motor_torque: float,
        disturbance_torque: float = 0.0,
        dt: float,
    ) -> GunLoadState:
        """Advance the single-inertia load by one semi-implicit Euler step."""
        alpha = self.acceleration(
            state,
            motor_torque=motor_torque,
            disturbance_torque=disturbance_torque,
        )
        omega_next = float(state.omega) + float(alpha) * float(dt)
        theta_next = float(state.theta) + omega_next * float(dt)
        return GunLoadState(theta=theta_next, omega=omega_next)

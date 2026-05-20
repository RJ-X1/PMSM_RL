"""Position PID outer-loop baseline for gun servo tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PIDPositionControllerConfig:
    """PID gains and command limits for position outer-loop control."""

    kp: float = 8.0
    ki: float = 0.0
    kd: float = 1.2
    integrator_limit: float = 0.6
    max_delta_omega: float = float(np.deg2rad(50.0))


class PIDPositionController:
    """Position PID controller that outputs a speed-command correction."""

    def __init__(self, config: PIDPositionControllerConfig | None = None) -> None:
        self.config = config or PIDPositionControllerConfig()
        self.integrator = 0.0
        self.prev_error: float | None = None

    def reset(self) -> None:
        self.integrator = 0.0
        self.prev_error = None

    def compute_delta_omega(self, *, e_theta: float, e_omega: float, dt: float) -> float:
        error = float(e_theta)
        self.integrator += error * float(dt)
        self.integrator = float(
            np.clip(
                self.integrator,
                -float(self.config.integrator_limit),
                float(self.config.integrator_limit),
            )
        )
        derivative = float(e_omega)
        if self.prev_error is not None and float(dt) > 0.0:
            derivative = 0.5 * derivative + 0.5 * ((error - self.prev_error) / float(dt))
        self.prev_error = error
        raw = (
            float(self.config.kp) * error
            + float(self.config.ki) * self.integrator
            + float(self.config.kd) * derivative
        )
        return float(np.clip(raw, -float(self.config.max_delta_omega), float(self.config.max_delta_omega)))

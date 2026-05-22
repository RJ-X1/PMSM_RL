"""Speed PI controller for gun-servo cascade baselines and inner-loop simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PISpeedControllerConfig:
    """PI gains and q-axis current limits for the speed loop."""

    kp: float = 18.0
    ki: float = 80.0
    integrator_limit: float = 8.0
    iq_limit: float = 12.0


class PISpeedController:
    """Conventional speed PI loop: speed error -> q-axis current command."""

    def __init__(self, config: PISpeedControllerConfig | None = None) -> None:
        self.config = config or PISpeedControllerConfig()
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def compute_iq_command_raw(self, *, omega_cmd: float, omega_meas: float, dt: float) -> tuple[float, float, bool]:
        error = float(omega_cmd) - float(omega_meas)
        self.integrator += float(self.config.ki) * error * float(dt)
        self.integrator = float(
            np.clip(self.integrator, -float(self.config.integrator_limit), float(self.config.integrator_limit))
        )
        raw_iq = float(self.config.kp) * error + self.integrator
        iq_cmd = float(np.clip(raw_iq, -float(self.config.iq_limit), float(self.config.iq_limit)))
        return raw_iq, iq_cmd, bool(abs(raw_iq - iq_cmd) > 1e-9)

    def compute_iq_command(self, *, omega_cmd: float, omega_meas: float, dt: float) -> tuple[float, bool]:
        _raw_iq, iq_cmd, saturated = self.compute_iq_command_raw(
            omega_cmd=omega_cmd,
            omega_meas=omega_meas,
            dt=dt,
        )
        return iq_cmd, saturated

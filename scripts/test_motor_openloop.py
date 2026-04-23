"""Open-loop PMSM dq current verification under fixed mechanical speed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envs.motor_model import PMSMMotorParams, PMSMState, clip_voltage_vector, dq_derivatives


RPM_TO_RAD_PER_SEC = 2.0 * np.pi / 60.0
FIXED_SPEED_RPM = 1000.0
FIXED_SPEED_RAD_PER_SEC = FIXED_SPEED_RPM * RPM_TO_RAD_PER_SEC
LOAD_TORQUE = 0.0
NUM_STEPS = 2_000
REASONABLE_CURRENT_LIMIT_FACTOR = 5.0


@dataclass(frozen=True, slots=True)
class OpenLoopCase:
    """One constant-voltage verification case."""

    name: str
    u_d: float
    u_q: float


def rk4_step_fixed_speed(
    state: PMSMState,
    u_dq: np.ndarray,
    *,
    fixed_omega_m: float,
    load_torque: float,
    params: PMSMMotorParams,
) -> PMSMState:
    """Advance only the electrical states while holding omega_m fixed."""

    dt_value = float(params.Ts)
    y0 = state.as_array(dtype=np.float64)
    y0[2] = float(fixed_omega_m)
    u_vec = clip_voltage_vector(u_dq, limit=float(params.Umax))

    def _f(y: np.ndarray) -> np.ndarray:
        y_eval = np.asarray(y, dtype=np.float64).copy()
        y_eval[2] = float(fixed_omega_m)
        deriv = dq_derivatives(
            PMSMState.from_array(y_eval),
            u_dq=u_vec,
            load_torque=float(load_torque),
            params=params,
        )
        deriv[2] = 0.0
        return deriv

    k1 = _f(y0)
    k2 = _f(y0 + 0.5 * dt_value * k1)
    k3 = _f(y0 + 0.5 * dt_value * k2)
    k4 = _f(y0 + dt_value * k3)
    y_next = y0 + (dt_value / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    y_next[2] = float(fixed_omega_m)
    return PMSMState.from_array(y_next)


def simulate_case(
    case: OpenLoopCase,
    *,
    params: PMSMMotorParams,
    output_dir: Path,
) -> None:
    state = PMSMState(i_d=0.0, i_q=0.0, omega_m=FIXED_SPEED_RAD_PER_SEC)
    u_dq = np.asarray([case.u_d, case.u_q], dtype=np.float64)

    states = np.zeros((NUM_STEPS + 1, 3), dtype=np.float64)
    states[0] = state.as_array(dtype=np.float64)

    for step_idx in range(NUM_STEPS):
        state = rk4_step_fixed_speed(
            state,
            u_dq=u_dq,
            fixed_omega_m=FIXED_SPEED_RAD_PER_SEC,
            load_torque=LOAD_TORQUE,
            params=params,
        )
        states[step_idx + 1] = state.as_array(dtype=np.float64)

    time = np.arange(NUM_STEPS + 1, dtype=np.float64) * float(params.Ts)
    current_mag = np.linalg.norm(states[:, :2], axis=1)
    reasonable_limit = REASONABLE_CURRENT_LIMIT_FACTOR * float(params.Imax)
    finite_ok = bool(np.isfinite(states).all())
    current_ok = bool(np.isfinite(current_mag).all() and np.max(current_mag) <= reasonable_limit)

    print(f"[{case.name}] finite states: {finite_ok}")
    print(
        f"[{case.name}] max current magnitude: {np.max(current_mag):.3f} A "
        f"(sanity limit {reasonable_limit:.3f} A): {current_ok}"
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time, states[:, 0], label="i_d")
    ax.plot(time, states[:, 1], label="i_q")
    ax.set_title(
        f"{case.name}: open-loop dq currents at {FIXED_SPEED_RPM:.0f} rpm\n"
        f"u_d={case.u_d:.1f} V, u_q={case.u_q:.1f} V"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"{case.name.lower().replace(' ', '_')}_currents.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[{case.name}] saved figure: {output_path}")


def main() -> None:
    params = PMSMMotorParams()
    output_dir = ROOT / "outputs" / "model_verification" / "step1_openloop"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = (
        OpenLoopCase(name="Case A", u_d=0.0, u_q=20.0),
        OpenLoopCase(name="Case B", u_d=-20.0, u_q=0.0),
    )

    print(f"Fixed speed: {FIXED_SPEED_RPM:.1f} rpm = {FIXED_SPEED_RAD_PER_SEC:.6f} rad/s")
    print(f"Load torque: {LOAD_TORQUE:.1f} Nm")
    print(f"Simulation steps: {NUM_STEPS}")

    for case in cases:
        simulate_case(case, params=params, output_dir=output_dir)


if __name__ == "__main__":
    main()

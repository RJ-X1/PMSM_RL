"""Closed-loop PMSM PI current-loop verification under fixed speed."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from baselines.pi_current_controller import (
    CUSTOM_OBS_OMEGA_BASE_RAD_PER_SEC,
    PICurrentController,
    PIControllerConfig,
)
from envs.motor_model import PMSMMotorParams, PMSMState, clip_voltage_vector, dq_derivatives


RPM_TO_RAD_PER_SEC = 2.0 * np.pi / 60.0
FIXED_SPEED_RPM = 1000.0
FIXED_SPEED_RAD_PER_SEC = FIXED_SPEED_RPM * RPM_TO_RAD_PER_SEC
LOAD_TORQUE = 1.0
NUM_STEPS = 2_000
STEP_TIME_SEC = 0.02
INITIAL_REFERENCE_DQ = np.asarray([0.0, 0.0], dtype=np.float64)
STEP_REFERENCE_DQ = np.asarray([-3.0, 10.0], dtype=np.float64)
CONTROLLER_SIGNAL_NAMES = ("i_d", "i_q", "ref_i_d", "ref_i_q", "omega_m")


def build_controller_observation(
    state: PMSMState,
    ref_dq: np.ndarray,
    params: PMSMMotorParams,
) -> np.ndarray:
    """Build the normalized custom-env observation expected by the PI baseline."""

    return np.asarray(
        [
            float(state.i_d) / float(params.Imax),
            float(state.i_q) / float(params.Imax),
            float(ref_dq[0]) / float(params.Imax),
            float(ref_dq[1]) / float(params.Imax),
            float(FIXED_SPEED_RAD_PER_SEC) / float(CUSTOM_OBS_OMEGA_BASE_RAD_PER_SEC),
        ],
        dtype=np.float32,
    )


def rk4_step_fixed_speed(
    state: PMSMState,
    u_dq: np.ndarray,
    *,
    fixed_omega_m: float,
    load_torque: float,
    params: PMSMMotorParams,
) -> PMSMState:
    """Advance dq currents while holding the mechanical speed fixed."""

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


def build_reference_trace(params: PMSMMotorParams) -> tuple[np.ndarray, int]:
    """Build the delayed dq current step reference used for PI verification."""

    step_index = int(round(float(STEP_TIME_SEC) / float(params.Ts)))
    step_index = int(np.clip(step_index, 0, NUM_STEPS))
    refs = np.tile(INITIAL_REFERENCE_DQ, (NUM_STEPS + 1, 1))
    refs[step_index:, :] = STEP_REFERENCE_DQ
    return refs, step_index


def compute_axis_metrics(
    *,
    time: np.ndarray,
    response: np.ndarray,
    reference: np.ndarray,
    step_index: int,
) -> dict[str, str]:
    """Compute lightweight step-response metrics for one current axis."""

    metrics: dict[str, str] = {}
    final_ref = float(reference[-1])
    initial_ref = float(reference[step_index - 1] if step_index > 0 else reference[0])
    step_amplitude = final_ref - initial_ref
    post_response = np.asarray(response[step_index:], dtype=np.float64)
    post_time = np.asarray(time[step_index:], dtype=np.float64)

    if post_response.size == 0:
        return {
            "overshoot": "not available (no samples after the step)",
            "settling_time": "not available (no samples after the step)",
            "steady_state_error": "not available (no samples after the step)",
        }

    if abs(step_amplitude) <= 1e-12:
        metrics["overshoot"] = "not available (zero step amplitude)"
    else:
        excursion_direction = np.sign(step_amplitude)
        overshoot_abs = max(
            0.0,
            float(np.max(excursion_direction * (post_response - final_ref))),
        )
        overshoot_pct = 100.0 * overshoot_abs / abs(step_amplitude)
        metrics["overshoot"] = f"{overshoot_abs:.6f} A ({overshoot_pct:.2f}% of step)"

    settling_band = 0.02 * abs(final_ref)
    if settling_band <= 1e-12:
        metrics["settling_time"] = "not available (final reference too close to zero for a 2% band)"
    else:
        within_band = np.abs(post_response - final_ref) <= settling_band
        settled_from_here = np.logical_and.accumulate(within_band[::-1])[::-1]
        settling_candidates = np.flatnonzero(settled_from_here)
        if settling_candidates.size == 0:
            metrics["settling_time"] = "not settled within the simulation window"
        else:
            settling_time = float(post_time[settling_candidates[0]] - time[step_index])
            metrics["settling_time"] = f"{settling_time:.6f} s"

    steady_state_error = float(response[-1] - final_ref)
    metrics["steady_state_error"] = (
        f"{steady_state_error:.6f} A "
        f"(abs {abs(steady_state_error):.6f} A)"
    )
    return metrics


def simulate_pi_current_loop() -> tuple[np.ndarray, np.ndarray, np.ndarray, PMSMMotorParams, int]:
    params = PMSMMotorParams()
    controller = PICurrentController(
        action_dim=2,
        signal_names=CONTROLLER_SIGNAL_NAMES,
        config=PIControllerConfig(),
        motor_params=params,
    )
    controller.reset()

    state = PMSMState(i_d=0.0, i_q=0.0, omega_m=FIXED_SPEED_RAD_PER_SEC)
    states = np.zeros((NUM_STEPS + 1, 2), dtype=np.float64)
    refs, step_index = build_reference_trace(params)
    voltages = np.zeros((NUM_STEPS + 1, 2), dtype=np.float64)
    states[0] = [state.i_d, state.i_q]

    for step_idx in range(NUM_STEPS):
        ref_dq = refs[step_idx]

        obs = build_controller_observation(state, ref_dq, params)
        u_dq = controller.compute_voltage_dq(obs)
        voltages[step_idx] = u_dq

        state = rk4_step_fixed_speed(
            state,
            u_dq=u_dq,
            fixed_omega_m=FIXED_SPEED_RAD_PER_SEC,
            load_torque=LOAD_TORQUE,
            params=params,
        )
        states[step_idx + 1] = [state.i_d, state.i_q]

    voltages[-1] = voltages[-2]
    return states, refs, voltages, params, step_index


def save_plot(
    *,
    time: np.ndarray,
    states: np.ndarray,
    refs: np.ndarray,
    voltages: np.ndarray,
    step_time: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].step(time, refs[:, 0], where="post", label="i_d*", linewidth=1.5)
    axes[0].plot(time, states[:, 0], label="i_d", linewidth=1.5)
    axes[0].axvline(step_time, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="step time")
    axes[0].set_ylabel("d-axis current [A]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].step(time, refs[:, 1], where="post", label="i_q*", linewidth=1.5)
    axes[1].plot(time, states[:, 1], label="i_q", linewidth=1.5)
    axes[1].axvline(step_time, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel("q-axis current [A]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time, voltages[:, 0], label="u_d", linewidth=1.5)
    axes[2].plot(time, voltages[:, 1], label="u_q", linewidth=1.5)
    axes[2].axvline(step_time, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Voltage [V]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        "PI current-loop verification at fixed speed\n"
        f"omega_m={FIXED_SPEED_RPM:.0f} rpm, load torque={LOAD_TORQUE:.1f} Nm"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    states, refs, voltages, params, step_index = simulate_pi_current_loop()
    time = np.arange(NUM_STEPS + 1, dtype=np.float64) * float(params.Ts)
    step_time = float(time[step_index])

    output_dir = ROOT / "outputs" / "model_verification" / "pi_current_loop"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pi_current_loop_step_response.png"

    save_plot(
        time=time,
        states=states,
        refs=refs,
        voltages=voltages,
        step_time=step_time,
        output_path=output_path,
    )

    d_metrics = compute_axis_metrics(
        time=time,
        response=states[:, 0],
        reference=refs[:, 0],
        step_index=step_index,
    )
    q_metrics = compute_axis_metrics(
        time=time,
        response=states[:, 1],
        reference=refs[:, 1],
        step_index=step_index,
    )

    print(f"Fixed speed: {FIXED_SPEED_RPM:.1f} rpm = {FIXED_SPEED_RAD_PER_SEC:.6f} rad/s")
    print(f"Fixed load torque: {LOAD_TORQUE:.1f} Nm")
    print(f"Simulation steps: {NUM_STEPS}")
    print(f"Step time: {step_time:.6f} s (step index {step_index})")
    print("d-axis metrics:")
    print(f"  overshoot: {d_metrics['overshoot']}")
    print(f"  settling time: {d_metrics['settling_time']}")
    print(f"  steady-state error: {d_metrics['steady_state_error']}")
    print("q-axis metrics:")
    print(f"  overshoot: {q_metrics['overshoot']}")
    print(f"  settling time: {q_metrics['settling_time']}")
    print(f"  steady-state error: {q_metrics['steady_state_error']}")
    print(f"Final currents: i_d={states[-1, 0]:.3f} A, i_q={states[-1, 1]:.3f} A")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()

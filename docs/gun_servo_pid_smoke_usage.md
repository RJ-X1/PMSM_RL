# Gun Servo PID Smoke Usage

## Purpose

This smoke check validates the classical PID-PI-FOC closed loop before TD3-PI
training. It exercises the `Custom-GunServo-Position-v0` environment, gun load
model, gearbox model, encoder feedback, PID position outer loop, speed/current
execution chain, CSV export, plots, and basic evaluation metrics.

The goal is not to train TD3 yet. The goal is to prove that the simulation and
baseline evaluation pipeline run end to end with deterministic, inspectable
outputs.

## v1 and v2

The first smoke config, `configs/eval/gun_servo_pid_smoke.yaml`, is kept as the
initial pipeline check. Its results showed two useful boundary findings:

- `C2_20deg_step` hit `omega_limit_violation` because the instantaneous 20 deg
  step drove the load speed past the 30 deg/s limit.
- `C4_sine_tracking` hit `omega_limit_violation` because the original 10 deg,
  0.5 Hz sine needs about 31.4 deg/s before controller error is added.

The v2 config, `configs/eval/gun_servo_pid_smoke_v2.yaml`, separates formal
baseline cases from aggressive safety-boundary cases. The main baseline cases
use conservative position PID gains `kp=8.0, ki=0.0, kd=2.0`, which reduce the
C1 final error below 0.02 deg without changing the RL framework.

## Run

From the repository root:

```bash
python scripts/inspect_env.py --env-config configs/env/gun_servo_position.yaml

python scripts/diagnostics/run_gun_servo_pid_smoke.py \
  --env-config configs/env/gun_servo_position.yaml \
  --eval-config configs/eval/gun_servo_pid_smoke_v2.yaml
```

On Windows PowerShell, the second command can be written as:

```powershell
python scripts/diagnostics/run_gun_servo_pid_smoke.py `
  --env-config configs/env/gun_servo_position.yaml `
  --eval-config configs/eval/gun_servo_pid_smoke_v2.yaml
```

To reproduce the original v1 pipeline smoke, replace the eval config with
`configs/eval/gun_servo_pid_smoke.yaml`.

## Outputs

The v2 run writes to:

```text
outputs/runs/gun_servo_pid_smoke_v2/
  eval/
    C1_10deg_step_pid.csv
    C2a_20deg_step_safe_pid.csv
    C3_trapezoid_tracking_pid.csv
    C4a_sine_tracking_safe_pid.csv
    C5a_disturbance_after_settling_pid.csv
    C2b_20deg_step_aggressive_pid.csv
    C4_sine_tracking_aggressive_pid.csv
  summaries/
    pid_smoke_summary.csv
  figures/
    C1_10deg_step_position.png
    C1_10deg_step_error.png
    C1_10deg_step_speed_command.png
    C1_10deg_step_torque.png
    C1_10deg_step_safety_flags.png
    ...
  configs/
    gun_servo_position.yaml
    gun_servo_pid_smoke_v2.yaml
    gun_servo_test_conditions.yaml
  metadata.json
```

Each trajectory CSV includes reference/load position, speed command, motor/load
torque, q-axis current, safety flags, reward, and done reason. The summary CSV
contains tracking error metrics, settling/recovery estimates, torque usage,
action variation, safety flag counts, first violation time/type, final state,
episode return, and termination reason.

Every scenario emits position, error, speed command, torque, and safety flag
figures. `C5a_disturbance_after_settling` also emits
`C5a_disturbance_after_settling_disturbance_zoom.png`, showing the disturbance
window with position error and load torque.

## Baseline Cases

Use these v2 primary scenarios for formal PID-vs-RL comparisons:

- `C1_10deg_step`: 10 deg step with conservative PID.
- `C2a_20deg_step_safe`: 20 deg target with a speed-limited trapezoid reference.
- `C3_trapezoid_tracking`: nominal trapezoid tracking.
- `C4a_sine_tracking_safe`: 5 deg, 0.1 Hz sine tracking over 10 s.
- `C5a_disturbance_after_settling`: 150 Nm disturbance at 2.5 s after the 10 deg step has settled.

Use these as safety-boundary diagnostics, not as primary baseline metrics:

- `C2b_20deg_step_aggressive`: instantaneous 20 deg step.
- `C4_sine_tracking_aggressive`: original 10 deg, 0.5 Hz sine.

## Healthy Run Checks

A normal smoke run should satisfy these basic checks:

- `scripts/inspect_env.py` prints finite observation/action dimensions and a
  finite PID rollout summary.
- All five scenario CSV files are created under `eval/`.
- `summaries/pid_smoke_summary.csv` has one row for each C1-C5 scenario.
- Position plots and error plots are created for every scenario.
- `done_reason=episode_limit` means the episode ran for the requested duration.
  A state-limit reason such as `omega_limit_violation` means the current PID
  baseline and scenario hit a physical safety boundary; inspect it as a
  controller/scenario finding, not as a CSV or plotting failure.
- `rmse_theta_deg`, `mae_theta_deg`, torque, current, and episode return should
  be finite.
- For v2 primary scenarios, `include_in_baseline=1` and `done_reason` should be
  `episode_limit`.
- For aggressive scenarios, `include_in_baseline=0`; `first_violation_time_s`
  and `first_violation_type` document the safety boundary that was reached.

Some safety flags can be nonzero during aggressive step commands because the
environment applies speed, acceleration, current, or torque protection. Treat
those counts as diagnostic signals rather than an immediate failure.

## Next Step: TD3-PI Smoke Training

After the v2 safe PID smoke is stable, move to a short TD3-PI smoke training run
on the safe cases first, then gradually add aggressive and randomized cases:

```bash
python scripts/train.py --env-config configs/env/gun_servo_position.yaml --train-config configs/train/td3_gun_position_smoke.yaml
```

Then evaluate the trained or residual controller against the same C1-C5
conditions and compare against `outputs/runs/gun_servo_pid_smoke_v2/` as the
classical baseline.

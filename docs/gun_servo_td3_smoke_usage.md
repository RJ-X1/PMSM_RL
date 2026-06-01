# Gun Servo TD3 Smoke Usage

## Purpose

This stage verifies the TD3-PI position outer-loop training pipeline for
`Custom-GunServo-Position-v0`. TD3 outputs one normalized load-side speed
correction action in `[-1, 1]`; the environment still owns the safe speed
command mapping, speed PI/equivalent execution chain, current and torque limits,
gearbox, gun load dynamics, and encoder feedback.

The goal is pipeline health, not beating the tuned PID/PD baseline yet.

## Relation To PID Smoke v2

PID smoke v2 is the classical baseline and safety reference. Its primary
scenarios run to `episode_limit`, while aggressive scenarios intentionally hit
speed limits. The first TD3 smoke trains only on `C1_10deg_step` so that replay,
updates, checkpoints, and evaluation can be debugged before adding multi-case
tracking, disturbances, or safety-constrained exploration.

## Train

```bash
python scripts/train.py \
  --env-config configs/env/gun_servo_position.yaml \
  --train-config configs/train/gun_servo_td3_smoke.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position.yaml `
  --train-config configs/train/gun_servo_td3_smoke.yaml
```

The config writes to `outputs/runs/gun_servo_td3_smoke_c1/`.

## Evaluate

The standard evaluate entrypoint can batch the scenarios listed in the eval
config and write trajectory CSV files:

```bash
python scripts/evaluate.py \
  --env-config configs/env/gun_servo_position.yaml \
  --eval-config configs/eval/gun_servo_td3_smoke_eval.yaml \
  --controller rl \
  --checkpoint outputs/runs/gun_servo_td3_smoke_c1/checkpoints/best.pt
```

For summary CSVs, figures, return curve, and PID v2 comparison:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py \
  --checkpoint outputs/runs/gun_servo_td3_smoke_c1/checkpoints/best.pt \
  --env-config configs/env/gun_servo_position.yaml \
  --eval-config configs/eval/gun_servo_td3_smoke_eval.yaml
```

`best.pt` is an alias for `checkpoint_best.pt`; `latest.pt` is an alias for
`checkpoint_latest.pt`.

## Outputs

```text
outputs/runs/gun_servo_td3_smoke_c1/
  checkpoints/
    best.pt
    latest.pt
    checkpoint_best.pt
    checkpoint_latest.pt
    checkpoint_step_*.pt
  eval/
    C1_10deg_step_td3.csv
    C2a_20deg_step_safe_td3.csv
    C3_trapezoid_tracking_td3.csv
  summaries/
    td3_smoke_summary.csv
    td3_vs_pid_v2_summary.csv
  figures/
    C1_10deg_step_td3_position.png
    C1_10deg_step_td3_error.png
    td3_return_curve.png
  train_log.csv
  metadata.json
  configs/
```

## Healthy Training Checks

- `train_log.csv` has episode rows with finite returns and actor/critic losses.
- Replay buffer size grows and checkpoints are saved at the configured interval.
- `best.pt` and `latest.pt` exist.
- Evaluation CSVs contain finite `theta_ref_deg`, `theta_L_deg`, `reward`, and
  `done_reason`.
- TD3 may perform worse than PID in this stage. That is expected because this is
  short, single-scenario vanilla TD3 with unconstrained exploration.

If TD3 hits `omega_limit_violation`, treat it as a training safety finding. The
next tuning targets are smaller exploration noise, action warmup scaling, a
PID-imitation or residual warm start, reward terms that penalize speed boundary
approach, and then SC-TD3-PI safety-constrained training.

## Next Steps

1. SC-TD3-PI: add safety-aware exploration and constraint penalties/barriers.
2. TD3-PI residual warm start: learn residual speed corrections around the PID
   baseline rather than a fully free position outer loop.
3. MC-SC-TD3-PI: train over safe multi-condition scenarios, then gradually add
   aggressive and randomized cases.

## TD3 Smoke v3

The smoke sequence is now:

- v1 connected the TD3 training/evaluation pipeline, but the learned controller
  was unsafe on the harder checks and could hit `omega_limit_violation`.
- v2 found a safe checkpoint for C1a/C1/C2a/C3, but the speed command and safe
  action were too conservative, especially for `C1_10deg_step`.
- v3 keeps the same isolated TD3 smoke lane while increasing the load-side
  action scale to 15 deg/s, keeping the 30 deg/s load-speed safety limit, and
  adding v3-only reward terms for tracking progress, speed-limit margin,
  saturation, and unsafe terminal penalties.

Training:

```bash
python scripts/train.py \
  --env-config configs/env/gun_servo_position_td3_smoke_v3.yaml \
  --train-config configs/train/gun_servo_td3_smoke_v3.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position_td3_smoke_v3.yaml `
  --train-config configs/train/gun_servo_td3_smoke_v3.yaml
```

Evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py \
  --checkpoint outputs/runs/gun_servo_td3_smoke_v3_c1/checkpoints/checkpoint_best.pt \
  --env-config configs/env/gun_servo_position_td3_smoke_v3.yaml \
  --eval-config configs/eval/gun_servo_td3_smoke_v3_eval.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py `
  --checkpoint outputs/runs/gun_servo_td3_smoke_v3_c1/checkpoints/checkpoint_best.pt `
  --env-config configs/env/gun_servo_position_td3_smoke_v3.yaml `
  --eval-config configs/eval/gun_servo_td3_smoke_v3_eval.yaml
```

Success criteria:

- `C1a_5deg_step` and `C1_10deg_step` reach `episode_limit`.
- No evaluated scenario ends with `omega_limit_violation`.
- C1a and C1 RMSE improve relative to TD3 smoke v2.
- Evaluation CSVs, summary CSVs, comparison CSVs, figures, and return curve are
  generated under `outputs/runs/gun_servo_td3_smoke_v3_c1/`.

If v3 is still worse than PID, interpret it as a vanilla-TD3 baseline limit, not
a failure of the environment connection. The next fix is to move to SC-TD3-PI or
TD3 safety-constrained training, preferably with a PID/residual warm start so the
policy learns safe corrective speed commands instead of discovering the entire
outer loop from scratch.

## Residual TD3-PI Smoke v4

Residual TD3 v4 keeps the PID/PD position outer loop as the stabilizing command
source and lets TD3 learn only a small load-side speed correction:

```text
omega_L_cmd_raw = omega_L_cmd_baseline + delta_omega_L_TD3
```

The composed command still goes through the same speed limit, acceleration/rate
limit, smoothing, actuator limits, and state-safety checks as direct TD3. The v4
config uses the PID smoke v2 gains and a 3 deg/s residual action budget.

Training:

```bash
python scripts/train.py \
  --env-config configs/env/gun_servo_position_residual_td3_v4.yaml \
  --train-config configs/train/gun_servo_residual_td3_smoke_v4.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position_residual_td3_v4.yaml `
  --train-config configs/train/gun_servo_residual_td3_smoke_v4.yaml
```

Evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py \
  --checkpoint outputs/runs/gun_servo_residual_td3_smoke_v4/checkpoints/checkpoint_best.pt \
  --env-config configs/env/gun_servo_position_residual_td3_v4.yaml \
  --eval-config configs/eval/gun_servo_residual_td3_smoke_v4_eval.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py `
  --checkpoint outputs/runs/gun_servo_residual_td3_smoke_v4/checkpoints/checkpoint_best.pt `
  --env-config configs/env/gun_servo_position_residual_td3_v4.yaml `
  --eval-config configs/eval/gun_servo_residual_td3_smoke_v4_eval.yaml
```

Success criteria:

- C1a and C1 reach `episode_limit`.
- No evaluated scenario ends with `omega_limit_violation`.
- C1a and C1 RMSE improve relative to TD3 smoke v3.
- Command-layer safety intervention is materially lower than direct TD3 v3 on
  the main C1 step.
- CSVs, summaries, comparison tables, plots, and return curve are generated.

If v4 still underperforms PID, the likely cause is that the residual learner is
being trained from sparse scalar feedback around an already competent baseline.
The next fix is to add residual-specific shaping or imitation-style targets,
then move to SC-TD3-PI with explicit constraint costs.

## SC-Residual TD3-PI Smoke v5 and v5.1

The later smoke sequence is:

- v4: residual TD3 became structurally healthy by keeping the PID/PD outer loop
  as the stabilizing command and learning only a residual speed correction.
- v5: safety-constrained residual TD3 became the main safe residual baseline.
  All primary scenarios reached `episode_limit`, with no
  `omega_limit_violation` and no hard safety violation.
- v5.1: reduced `flag_U_safe_count`, but did not improve C1a/C1 and slightly
  degraded C3. Treat it as an over-conservative side branch, not the main
  version for the next stage.

v5.1 keeps the same physical speed, current, torque, gearbox, and state safety
limits as v5. The tuning raises the residual action scale modestly to 4 deg/s,
eases residual/action penalties, keeps `flag_U` penalties active, and increases
progress reward weight.

Training:

```bash
python scripts/train.py --env-config configs/env/gun_servo_position_sc_residual_td3_v51.yaml --train-config configs/train/gun_servo_sc_residual_td3_smoke_v51.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position_sc_residual_td3_v51.yaml `
  --train-config configs/train/gun_servo_sc_residual_td3_smoke_v51.yaml
```

Evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_sc_residual_td3_smoke_v51/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_sc_residual_td3_v51.yaml --eval-config configs/eval/gun_servo_sc_residual_td3_smoke_v51_eval.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py `
  --checkpoint outputs/runs/gun_servo_sc_residual_td3_smoke_v51/checkpoints/checkpoint_best.pt `
  --env-config configs/env/gun_servo_position_sc_residual_td3_v51.yaml `
  --eval-config configs/eval/gun_servo_sc_residual_td3_smoke_v51_eval.yaml
```

Success criteria:

- C1a and C1 reach `episode_limit`.
- No evaluated scenario ends with `omega_limit_violation`.
- No hard safety violation occurs: `flag_E_safe_count`, `flag_X_safe_count`,
  and `sigma_safe_count` stay at 0.
- C1a/C1 RMSE is no worse than v5 and preferably close to or better than v4.
- C2a/C3 do not degrade significantly compared with v5.
- Total `flag_U_safe_count` remains lower than or close to v5, and clearly
  lower than v4.

Because v5.1 did not improve the primary tracking tradeoff, the next stage uses
v5 as the base method rather than v5.1.

## MC-SC-Residual TD3 Smoke v6

v6 moves from single/small-step residual tuning to multi-condition smoke
training while keeping the v5 safety-constrained residual structure. Training
uses episode-level uniform sampling, so each episode is drawn from one of:

- `C1a_5deg_step`
- `C1_10deg_step`
- `C2a_20deg_step_safe`
- `C3_trapezoid_tracking`

C4 sine and C5 disturbance scenarios are intentionally excluded from this first
multi-condition smoke stage.

Training:

```bash
python scripts/train.py --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v6.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_smoke_v6.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v6.yaml `
  --train-config configs/train/gun_servo_mc_sc_residual_td3_smoke_v6.yaml
```

Select the multi-condition safe checkpoint before final reporting:

```bash
python scripts/diagnostics/select_gun_servo_mc_safe_checkpoint.py --run-dir outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v6 --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v6.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_smoke_v6_eval.yaml
```

Evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v6/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v6.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_smoke_v6_eval.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py `
  --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v6/checkpoints/checkpoint_best.pt `
  --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v6.yaml `
  --eval-config configs/eval/gun_servo_mc_sc_residual_td3_smoke_v6_eval.yaml
```

Success criteria:

- Training and evaluation complete successfully.
- C1a, C1, C2a, and C3 all reach `episode_limit`.
- No `omega_limit_violation` occurs.
- No hard safety violation occurs.
- Average C1a/C1/C2a/C3 RMSE is comparable to or better than v5, with no severe
  C1a/C1 regression.
- `flag_U_safe_count` remains reasonable and preferably close to or lower than
  v5.
- CSVs, summaries, comparison tables, figures, and the v6 conclusion report are
  generated under `outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v6/`.

After v6, proceed to C4/C5 and disturbance training only if the multi-condition
checkpoint preserves v5-level safety and does not materially degrade the
average primary-scenario RMSE. Otherwise, tune v6 again before widening the
scenario set.

## MC-SC-Residual TD3 Smoke v7

v6 succeeded as the first multi-condition step/trapezoid smoke stage: all four
primary scenarios reached `episode_limit`, with no `omega_limit_violation` and
no hard safety violation. v7 keeps the v6 SC-Residual-TD3 structure and extends
the smoke set with safe sine tracking and disturbance rejection after settling.

Training uses the current trainer's episode-level uniform sampling over:

- `C1a_5deg_step`
- `C1_10deg_step`
- `C2a_20deg_step_safe`
- `C3_trapezoid_tracking`
- `C4a_sine_tracking_safe`
- `C5a_disturbance_after_settling`

The originally desired 15/20/15/20/15/15 weighted mix is left for a later
trainer change; this smoke run avoids rewriting the training framework.

Training:

```bash
python scripts/train.py --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v7.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_smoke_v7.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v7.yaml `
  --train-config configs/train/gun_servo_mc_sc_residual_td3_smoke_v7.yaml
```

Select the safe checkpoint with the v7 multi-condition score:

```bash
python scripts/diagnostics/select_gun_servo_mc_safe_checkpoint.py --run-dir outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7 --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v7.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_smoke_v7_eval.yaml --recovery-beta 0.05 --unsafe-gamma 1000
```

Evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v7.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_smoke_v7_eval.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py `
  --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/checkpoints/checkpoint_best.pt `
  --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v7.yaml `
  --eval-config configs/eval/gun_servo_mc_sc_residual_td3_smoke_v7_eval.yaml
```

Success criteria:

- Training and evaluation complete successfully.
- C1a, C1, C2a, C3, C4a, and C5a all reach `episode_limit`.
- No `omega_limit_violation` occurs.
- No hard safety violation occurs.
- C4a sine tracking is stable and does not drift.
- C5a `recovery_time_s` is finite and reasonable.
- C1a/C1 do not degrade severely compared with v6.
- CSVs, summaries, comparison tables, figures, and the v7 conclusion report are
  generated under `outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/`.

After v7, proceed to formal MC-SC-Residual-TD3 training with randomization if
the six-scenario smoke checkpoint remains safe, keeps C1a/C1 close to v6, and
shows finite C5a recovery. Otherwise, tune v7 again before widening into the
paper-scale randomized experiment.

## MC-SC-Residual TD3 Randomized v8

The fixed-scenario sequence is now:

- v6: multi-condition fixed step/trapezoid training.
- v7: added fixed safe sine tracking and disturbance-after-settling scenarios.
- v8: randomized multi-condition training with fixed evaluation plus Monte
  Carlo randomized evaluation.

v8 keeps the v7 SC-Residual-TD3 controller, PID/PD stabilizing baseline, 30
deg/s physical load-speed limit, and current/torque/gearbox/state safety
limits. Randomized training samples safe step, trapezoid, sine, and
disturbance-after-settling families at episode boundaries. The first v8 pass
uses moderate plant/load randomization ranges before expanding toward the full
paper-scale ranges.

Training:

```bash
python scripts/train.py --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8.yaml
```

PowerShell:

```powershell
python scripts/train.py `
  --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml `
  --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8.yaml
```

Select the fixed-scenario safe checkpoint:

```bash
python scripts/diagnostics/select_gun_servo_mc_safe_checkpoint.py --run-dir outputs/runs/gun_servo_mc_sc_residual_td3_random_v8 --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_fixed.yaml --alpha 0.25 --recovery-beta 0.0 --disturbance-beta 0.05 --unsafe-gamma 1000
```

Fixed evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_fixed.yaml
```

Monte Carlo evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_mc_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_mc.yaml
```

Success criteria:

- Training reaches 600000 steps and writes regular 20000-step checkpoints.
- Fixed C1a, C1, C2a, C3, C4a, and C5a all reach `episode_limit`.
- No fixed or Monte Carlo episode ends with `omega_limit_violation`.
- No hard safety violation occurs in fixed evaluation.
- Monte Carlo success rate is high, with finite aggregate RMSE and disturbance
  recovery metrics.
- `checkpoint_best.pt` is selected from fixed safe checkpoints using mean RMSE,
  normalized `flag_U_safe_count`, disturbance peak error, and unsafe penalty.

After v8, proceed to final larger-step, multi-seed randomized training only if
fixed safety is preserved and Monte Carlo robustness is acceptable. Otherwise,
tune the randomization ranges or reward/safety weights before increasing the
training scale.

## MC-SC-Residual TD3 Randomized v8 Full Multi-Seed

The first randomized v8 run was useful but partial: it stopped at 420000 steps
without stderr output, traceback, violation rows, or trainer early-stop evidence.
The most likely cause was an external interruption. The current trainer supports
warm-start loading from a policy checkpoint, but not clean full-run resume with
global step, replay buffer, episode index, and append-mode training log state, so
seed 0 is rerun as a clean full run rather than resumed.

`v8_full_multi_seed` keeps the v8 algorithm and environment unchanged and runs
three 600000-step seeds for formal reporting:

```bash
python scripts/train.py --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8_full_seed0.yaml
python scripts/train.py --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8_full_seed1.yaml
python scripts/train.py --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8_full_seed2.yaml
```

Select each seed's fixed-scenario safe checkpoint before final evaluation:

```bash
python scripts/diagnostics/select_gun_servo_mc_safe_checkpoint.py --run-dir outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0 --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed0_eval_fixed.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8_full_seed0.yaml --alpha 0.25 --recovery-beta 0.05 --disturbance-beta 0.05 --unsafe-gamma 1000
python scripts/diagnostics/select_gun_servo_mc_safe_checkpoint.py --run-dir outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1 --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed1_eval_fixed.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8_full_seed1.yaml --alpha 0.25 --recovery-beta 0.05 --disturbance-beta 0.05 --unsafe-gamma 1000
python scripts/diagnostics/select_gun_servo_mc_safe_checkpoint.py --run-dir outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2 --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed2_eval_fixed.yaml --train-config configs/train/gun_servo_mc_sc_residual_td3_random_v8_full_seed2.yaml --alpha 0.25 --recovery-beta 0.05 --disturbance-beta 0.05 --unsafe-gamma 1000
```

Fixed evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed0_eval_fixed.yaml
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed1_eval_fixed.yaml
python scripts/diagnostics/run_gun_servo_td3_smoke_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed2_eval_fixed.yaml
```

Monte Carlo evaluation:

```bash
python scripts/diagnostics/run_gun_servo_td3_mc_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed0_eval_mc.yaml
python scripts/diagnostics/run_gun_servo_td3_mc_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed1_eval_mc.yaml
python scripts/diagnostics/run_gun_servo_td3_mc_eval.py --checkpoint outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2/checkpoints/checkpoint_best.pt --env-config configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_full_seed2_eval_mc.yaml
```

Aggregate reporting:

```bash
python scripts/diagnostics/aggregate_gun_servo_v8_full_multiseed.py
```

Aggregate outputs are written under
`outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_aggregate/`,
including `summaries/fixed_eval_mean_std.csv`,
`summaries/mc_eval_mean_std.csv`,
`summaries/v8_full_vs_pid_v2_mean_std.csv`,
`summaries/v8_full_vs_v7_mean_std.csv`,
`summaries/v8_full_result_conclusion.md`, and the multi-seed figures under
`figures/`.

Interpretation guidance:

- Treat `checkpoint_best.pt` as the fixed-scenario safe selected checkpoint, not
  necessarily the latest 600000-step policy.
- The full v8 run is acceptable for robustness reporting only when all seeds
  complete 600000 steps, fixed and Monte Carlo evaluations complete for every
  seed, success rate remains high, and omega/hard-safety violation counts stay
  at zero.
- If the selector repeatedly chooses early checkpoints, report that as a policy
  drift signal and tune the randomized reward/safety tradeoff before claiming
  longer training improves the controller.

## Classical Robust Baseline: SMC-PI

SMC-PI adds a classical robust-control comparison point between PID/PD-PI and
the TD3-family methods. Like PID and RL, SMC is only the position outer loop: it
emits a load-side speed correction, then the existing environment applies the
same speed command safety processing, speed PI loop, actuator limits, PMSM,
gearbox, gun-load dynamics, and encoder feedback.

SMC differs from PID by using a sliding surface,
`s = e_omega + lambda_smc * e_theta`, with a saturated switching term to reduce
chattering. It differs from RL because its command is a fixed analytical
controller rather than a learned policy or residual policy.

Evaluation:

```bash
python scripts/diagnostics/run_gun_servo_smc_baseline.py --env-config configs/env/gun_servo_position.yaml --eval-config configs/eval/gun_servo_smc_baseline_eval.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_smc_baseline.py `
  --env-config configs/env/gun_servo_position.yaml `
  --eval-config configs/eval/gun_servo_smc_baseline_eval.yaml
```

Outputs are written under `outputs/runs/gun_servo_smc_baseline/`:

- `eval/`: per-scenario SMC trajectory CSVs.
- `summaries/smc_baseline_summary.csv`: fixed six-scenario SMC metrics.
- `summaries/smc_vs_pid_v2_summary.csv`: fixed comparison with PID v2.
- `summaries/smc_vs_v8_full_summary.csv`: fixed comparison with v8 full
  multi-seed aggregate results.
- `summaries/smc_baseline_conclusion.md`: controller parameters, safety
  outcome, and paper recommendation.
- `figures/`: position, error, speed-command, safety-flag, and C5a disturbance
  zoom plots.

Use SMC-PI in the final method comparison as the classical robust baseline:
PID/PD-PI remains the engineering baseline, SMC-PI is the analytical robust
baseline, and TD3-PI through MC-SC-Residual-TD3-PI are the learning-based
methods.

## SMC-PI Monte Carlo Evaluation

After fixed-scenario SMC-PI succeeds, evaluate it on the same randomized
Monte Carlo scenario families used by randomized v8:

```bash
python scripts/diagnostics/run_gun_servo_smc_mc_eval.py --env-config configs/env/gun_servo_position.yaml --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_mc.yaml
```

PowerShell:

```powershell
python scripts/diagnostics/run_gun_servo_smc_mc_eval.py `
  --env-config configs/env/gun_servo_position.yaml `
  --eval-config configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_mc.yaml
```

The SMC MC evaluator uses the v8 MC scenario-family weights and applies the v8
randomized domain ranges from the eval config's referenced v8 env config. Its
default run is 100 episodes with fixed seeds for reproducibility.

Outputs:

- `outputs/runs/gun_servo_smc_baseline/summaries/smc_mc_eval_summary.csv`
- `outputs/runs/gun_servo_smc_baseline/summaries/smc_mc_eval_episode_metrics.csv`
- `outputs/runs/gun_servo_smc_baseline/summaries/smc_mc_vs_v8_full_mc_summary.csv`
- `outputs/runs/gun_servo_smc_baseline/figures/smc_mc_rmse_boxplot.png`
- `outputs/runs/gun_servo_smc_baseline/figures/smc_mc_scenario_rmse_boxplot.png`
- `outputs/runs/gun_servo_smc_baseline/figures/smc_mc_violation_counts.png`
- `outputs/runs/gun_servo_smc_baseline/figures/smc_mc_recovery_time_boxplot.png`

Interpret fixed and Monte Carlo results separately. Fixed scenarios are
deterministic tracking and disturbance probes, so they expose best-case tracking
accuracy and method-to-method behavior on known references. Monte Carlo results
measure robustness under randomized reference families, plant/load parameters,
disturbances, encoder noise, gear efficiency, and DC-bus scaling. A method can
look worse on fixed RMSE but still be important if it has strong randomized
success, zero hard violations, and stable disturbance recovery.

## Final Method Comparison

Build the final comparison bundle after PID, SMC, and RL summaries exist:

```bash
python scripts/diagnostics/build_gun_servo_final_comparison.py
```

Outputs:

- `outputs/runs/final_method_comparison/summaries/fixed_six_scenario_comparison.csv`
- `outputs/runs/final_method_comparison/summaries/monte_carlo_comparison.csv`
- `outputs/runs/final_method_comparison/summaries/final_comparison_conclusion.md`
- `outputs/runs/final_method_comparison/figures/fixed_rmse_by_method_and_scenario.png`
- `outputs/runs/final_method_comparison/figures/fixed_flag_U_by_method_and_scenario.png`
- `outputs/runs/final_method_comparison/figures/monte_carlo_rmse_comparison.png`
- `outputs/runs/final_method_comparison/figures/monte_carlo_success_rate_comparison.png`
- `outputs/runs/final_method_comparison/figures/monte_carlo_violation_counts_comparison.png`

For paper reporting, use PID/PD-PI as the engineering baseline, SMC-PI as the
classical robust-control baseline, and MC-SC-Residual-TD3-PI as the proposed
learning method. The current results show classical controllers remain very
strong on fixed tracking RMSE, while randomized MC-SC-Residual-TD3-PI supplies
the multi-seed learned-robustness evidence needed for the proposed method.

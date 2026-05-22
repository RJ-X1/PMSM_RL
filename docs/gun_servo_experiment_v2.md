# Gun Servo RL-PI Experiment V2

This project version keeps RL in the position outer loop. The actor emits one
normalized load-speed correction action, and the environment maps it through:

`a_raw -> a_clip -> U_safe -> omega_L_cmd_safe -> omega_m_ref -> speed PI -> i_q -> T_e -> T_out -> load`

## Controllers

- `pid_pi_foc`: position PID plus speed/current PI execution layer.
- `td3_pi`: TD3 position actor with only basic action clipping and actuator hard limits.
- `sc_td3_pi`: TD3 position actor with U/E/X safety enabled.
- `mc_sc_td3_pi`: `sc_td3_pi` plus training-time multi-condition randomization and scalar condition code `m_k`.
- `smc_pi_foc`: reserved second-round robust baseline placeholder.

## Smoke Commands

```powershell
python -m compileall agents baselines envs scripts utils
python scripts/diagnostics/inspect_gun_servo_env.py --env-config configs/env/gun_servo_position.yaml

python scripts/evaluate.py `
  --env-config configs/env/gun_servo_position.yaml `
  --eval-config configs/eval/gun_servo_eval.yaml `
  --controller pid_pi_foc `
  --scenario step_10deg `
  --max-steps 1000 `
  --output-csv outputs/smoke/pid_step_10deg.csv

python scripts/train.py `
  --env-config configs/env/gun_servo_position.yaml `
  --train-config configs/train/td3_gun_position_smoke.yaml `
  --controller sc_td3_pi `
  --run-name sc_td3_pi_smoke `
  --total-steps 1000

python scripts/evaluate.py `
  --env-config configs/env/gun_servo_position.yaml `
  --eval-config configs/eval/gun_servo_eval.yaml `
  --controller sc_td3_pi `
  --scenario step_10deg `
  --checkpoint outputs/runs/sc_td3_pi_smoke/checkpoints/checkpoint_latest.pt `
  --max-steps 1000 `
  --output-csv outputs/smoke/sc_td3_pi_step_10deg.csv

python scripts/analysis/summarize_gun_position_eval.py `
  --input-glob "outputs/smoke/*.csv" `
  --output-csv outputs/smoke/gun_position_summary.csv
```

# PMSM RL current control

Lightweight research project for PMSM current-control reinforcement learning experiments using gym-electric-motor.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Current status

- Canonical CLI entrypoints live under `scripts/`, with root `train.py` and `evaluate.py` kept as backward-compatible wrappers.
- Run artifacts are organized under `outputs/runs/<run_name>/`.
- Flat GEM observations are exported with named columns via `envs/obs_parser.py`.
- Checkpoint sweep summaries and plotting scripts are available for validation-first workflows.

## Repository layout

```text
configs/
  env/
  train/
  eval/
docs/
outputs/
  runs/
  paper/
scripts/
utils/
```

Run layout:

```text
outputs/runs/<run_name>/
  checkpoints/
  eval/
  summaries/
  figures/
  configs/
  train_log.csv
  metadata.json
```

## Useful commands

```powershell
python scripts/inspect_env.py
python train.py --train-config configs/train/ddpg_smoke.yaml
python evaluate.py --train-config configs/train/ddpg_main.yaml --controller rl
python evaluate.py --train-config configs/train/ddpg_main.yaml --controller pi
python scripts/batch_eval_checkpoints.py outputs/runs/ddpg_smoke/checkpoints --train-config configs/train/ddpg_smoke.yaml --max-steps 100 --num-repeats 3
python scripts/plot_training_curves.py outputs/runs/ddpg_smoke/train_log.csv
python scripts/plot_checkpoint_sweep.py outputs/runs/ddpg_smoke/summaries/aggregate_summary.csv
python scripts/plot_episode_trace.py outputs/runs/ddpg_smoke/eval/eval_rl.csv
```

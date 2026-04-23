# PMSM RL current control

Lightweight research project for PMSM current-control reinforcement learning experiments using gym-electric-motor.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Current status

- Canonical CLI entrypoints live under `scripts/`, with new grouped utilities for `analysis/` and `plotting/`.
- Run artifacts are organized under `outputs/runs/<run_name>/`.
- Flat GEM observations are exported with named columns via `envs/obs_parser.py`.
- Checkpoint sweep summaries and paper-style plotting commands are available for validation-first workflows.

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
  analysis/
  plotting/
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

Top-level compatible entrypoints:

```powershell
python scripts/inspect_env.py
python scripts/train.py --train-config configs/train/td3_smoke.yaml
python scripts/evaluate.py --train-config configs/train/cdr_td3_full_main.yaml --controller rl
python scripts/evaluate.py --train-config configs/train/cdr_td3_full_main.yaml --controller pi
python scripts/batch_eval_checkpoints.py outputs/runs/cdr_td3_full_main/checkpoints --train-config configs/train/cdr_td3_full_main.yaml --max-steps 200 --num-repeats 3
```

The core implementations now live under `scripts/core`, `scripts/diagnostics`, `scripts/analysis`, and `scripts/plotting`. The top-level `scripts/*.py` entrypoints above are kept as backward-compatible wrappers for older README commands and local habits.

Paper-ready metric aggregation:

```powershell
python scripts/analysis/summarize_eval_batch.py --glob "outputs/runs/**/eval/*.csv" --output outputs/paper/eval_summary.csv
python scripts/diagnostics/summarize_eval_single.py outputs/runs/cdr_td3_full_main/eval/eval_rl.csv
```

Paper-style figures from one plotting toolkit:

```powershell
python scripts/plotting/plot_episode_trace.py outputs/runs/cdr_td3_full_main/eval/eval_rl.csv
python scripts/plotting/plot_training_curves.py outputs/runs/cdr_td3_full_main/train_log.csv
python scripts/plotting/plot_checkpoint_sweep.py outputs/runs/cdr_td3_full_main/summaries/aggregate_summary.csv
python scripts/plotting/paper_figures.py training_dashboard outputs/runs/cdr_td3_full_main/train_log.csv --output outputs/paper/training_dashboard.png
python scripts/plotting/paper_figures.py nominal_waveforms --csv outputs/runs/cdr_td3_full_main/eval/eval_rl.csv --output outputs/paper/nominal_waveforms.png
python scripts/plotting/paper_figures.py event_response --csv outputs/runs/cdr_td3_full_main/eval/eval_rl.csv --events 20 60 100 --labels "load step" "speed step" "recover" --output outputs/paper/event_response.png
python scripts/plotting/paper_figures.py performance_heatmap --csv outputs/paper/eval_summary.csv --metric rmse --output outputs/paper/performance_heatmap.png
python scripts/plotting/paper_figures.py pareto_front --csv outputs/paper/eval_summary.csv --x rmse --y control_energy --annotate --output outputs/paper/pareto_front.png
python scripts/plotting/paper_figures.py robustness_distribution --csv outputs/paper/eval_summary.csv --metric rmse --output outputs/paper/robustness_distribution.png
python scripts/plotting/paper_figures.py ablation_waterfall --csv outputs/paper/ablation.csv --lower-better --output outputs/paper/ablation_waterfall.png
```

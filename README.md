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

Core workflows:

```powershell
python scripts/inspect_env.py
python scripts/train.py --train-config configs/train/ddpg_smoke.yaml
python scripts/evaluate.py --train-config configs/train/ddpg_main.yaml --controller rl
python scripts/evaluate.py --train-config configs/train/ddpg_main.yaml --controller pi
python scripts/batch_eval_checkpoints.py outputs/runs/ddpg_smoke/checkpoints --train-config configs/train/ddpg_smoke.yaml --max-steps 100 --num-repeats 3
```

Paper-ready metric aggregation:

```powershell
python scripts/analysis/summarize_eval.py --glob "outputs/runs/**/eval/*.csv" --output outputs/paper/eval_summary.csv
```

Paper-style figures from one plotting toolkit:

```powershell
python scripts/plotting/paper_figures.py training_dashboard outputs/runs/ddpg_smoke/train_log.csv --output outputs/paper/training_dashboard.png
python scripts/plotting/paper_figures.py nominal_waveforms --csv outputs/runs/ddpg_smoke/eval/eval_rl.csv --output outputs/paper/nominal_waveforms.png
python scripts/plotting/paper_figures.py event_response --csv outputs/runs/ddpg_smoke/eval/eval_rl.csv --events 20 60 100 --labels "load step" "speed step" "recover" --output outputs/paper/event_response.png
python scripts/plotting/paper_figures.py performance_heatmap --csv outputs/paper/eval_summary.csv --metric rmse --output outputs/paper/performance_heatmap.png
python scripts/plotting/paper_figures.py pareto_front --csv outputs/paper/eval_summary.csv --x rmse --y control_energy --annotate --output outputs/paper/pareto_front.png
python scripts/plotting/paper_figures.py robustness_distribution --csv outputs/paper/eval_summary.csv --metric rmse --output outputs/paper/robustness_distribution.png
python scripts/plotting/paper_figures.py ablation_waterfall --csv outputs/paper/ablation.csv --lower-better --output outputs/paper/ablation_waterfall.png
```

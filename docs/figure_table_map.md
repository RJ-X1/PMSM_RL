# Figure/Table Map

## Figures

- Training curves: `scripts/plot_training_curves.py`
- Checkpoint sweep comparison: `scripts/plot_checkpoint_sweep.py`
- Episode trace: `scripts/plot_episode_trace.py`
- Failure case trace: `scripts/plot_failure_case.py`

Generated paper-ready assets should be copied or exported into:

- `outputs/paper/figures/`
- `outputs/paper/manifests/`

## Tables

Suggested summary tables should be written under:

- `outputs/paper/tables/`

Checkpoint sweep summaries can be sourced from:

- `outputs/runs/<run_name>/summaries/per_run_summary.csv`
- `outputs/runs/<run_name>/summaries/aggregate_summary.csv`

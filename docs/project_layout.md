# Project Layout

## Source structure

- `agents/`: RL model and replay-buffer implementation
- `baselines/`: classical controllers used as reference baselines
- `envs/`: environment construction and observation semantics
- `utils/`: shared config, run-layout, metrics, and experiment helpers
- `scripts/`: canonical CLI entrypoints and plotting tools

## Config structure

- `configs/env/`: environment configs
- `configs/train/`: training configs
- `configs/eval/`: evaluation defaults

Backward-compatible config copies remain at the old top-level paths where needed.

## Output structure

- `outputs/runs/`: experiment run artifacts
- `outputs/paper/`: paper-facing figures, tables, and provenance manifests

Use `outputs/runs/<run_name>/` for executable experiment artifacts.
Use `outputs/paper/` only for curated paper outputs.

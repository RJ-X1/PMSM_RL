# Experiment Protocol

## Validation-first workflow

1. Inspect the environment and observation semantics.
2. Validate PI baseline behavior.
3. Run smoke training and verify checkpoint save/load.
4. Batch-evaluate checkpoints before selecting a candidate.
5. Only after diagnostics are stable, expand training scale.

## Standard run assets

Each run under `outputs/runs/<run_name>/` should contain:

- `checkpoints/`
- `eval/`
- `summaries/`
- `figures/`
- `configs/`
- `train_log.csv`
- `metadata.json`

## Current policy

- Do not change the DDPG algorithm without an explicit experiment decision.
- Keep PI and environment semantics stable while validating RL behavior.

# Metric Definitions

## Episode-level metrics

- `steps`: number of environment steps written to an evaluation CSV
- `done`: `1` if the episode terminated or truncated before the step budget, else `0`
- `done_reason`: best-effort textual reason for termination
- `episode_return`: final cumulative reward

## Tracking metrics

- `mae_i_d`: mean absolute error between `i_d` and `ref_i_d`
- `mae_i_q`: mean absolute error between `i_q` and `ref_i_q`
- `mae_all`: average absolute error across all detected tracking pairs
- `rmse_all`: root mean squared error across all detected tracking pairs

## Repeated-checkpoint metrics

- `done_count`: number of repeated evaluations ending with `done = 1`
- `success_count`: number of repeated evaluations ending with `done = 0`
- `success_rate`: `success_count / num_runs`
- `mean_mae_all`: average `mae_all` across repeated evaluations
- `mean_rmse_all`: average `rmse_all` across repeated evaluations

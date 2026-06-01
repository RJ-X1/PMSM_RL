# Cleanup Action Report

Generated after archive/cache cleanup: 2026-06-01 20:09:35 +08:00

No final experiment folders or checkpoint files were deleted. Old unrelated experiment folders were moved to `outputs/archive_old_experiments/`. Editor metadata was only listed, not deleted.

## 1. Folders Kept
- envs
- baselines
- utils
- scripts
- configs
- docs
- outputs/runs/gun_servo_pid_smoke_v2
- outputs/runs/gun_servo_pid_mc_baseline
- outputs/runs/gun_servo_smc_baseline
- outputs/runs/gun_servo_residual_td3_smoke_v4
- outputs/runs/gun_servo_sc_residual_td3_smoke_v5
- outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v6
- outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7
- outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0
- outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1
- outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2
- outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_aggregate
- outputs/runs/final_method_comparison
- outputs/paper_ready_results
- outputs/figure_audit
- outputs/experiment_closeout
- outputs/runs/gun_servo_td3_smoke_c1
- outputs/runs/gun_servo_td3_smoke_v2_c1
- outputs/runs/gun_servo_td3_smoke_v3_c1
- outputs/runs/gun_servo_td3_smoke_c1_quickcheck
- outputs/runs/gun_servo_td3_smoke_v2_quickcheck

## 2. Folders Archived
| source | target | status | size MB |
|---|---|---|---:|
| outputs/runs/sc_td3_pi_smoke | outputs/archive_old_experiments/sc_td3_pi_smoke | archived | 2.578 |
| outputs/runs/td3_constraint_reward_200k_seed0 | outputs/archive_old_experiments/td3_constraint_reward_200k_seed0 | archived | 35.277 |
| outputs/runs/td3_gun_position | outputs/archive_old_experiments/td3_gun_position | archived | 7.188 |
| outputs/runs/td3_gun_position_smoke | outputs/archive_old_experiments/td3_gun_position_smoke | archived | 3.435 |
| outputs/runs/td3_main_2m_external_step_randomtime_seed0 | outputs/archive_old_experiments/td3_main_2m_external_step_randomtime_seed0 | archived | 101.107 |
| outputs/runs/td3_main_500k_external_step_randomtime_seed0_low_actor_lr | outputs/archive_old_experiments/td3_main_500k_external_step_randomtime_seed0_low_actor_lr | archived | 56.604 |
| outputs/runs/td3_main_500k_external_step_randomtime_seed1_low_actor_lr | outputs/archive_old_experiments/td3_main_500k_external_step_randomtime_seed1_low_actor_lr | archived | 56.684 |
| outputs/runs/td3_main_500k_external_step_randomtime_seed2_low_actor_lr | outputs/archive_old_experiments/td3_main_500k_external_step_randomtime_seed2_low_actor_lr | archived | 56.611 |
| outputs/runs/td3_physics_reward_v2_200k_seed0 | outputs/archive_old_experiments/td3_physics_reward_v2_200k_seed0 | archived | 35.633 |
| outputs/runs/td3_pi_smoke | outputs/archive_old_experiments/td3_pi_smoke | archived | 2.578 |
| outputs/runs/td3_pretrain_100k_external_train_test1_eval_seed0 | outputs/archive_old_experiments/td3_pretrain_100k_external_train_test1_eval_seed0 | archived | 20.694 |
| outputs/runs/td3_pretrain_100k_test1_eval_seed0 | outputs/archive_old_experiments/td3_pretrain_100k_test1_eval_seed0 | archived | 22.263 |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed0 | outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed0 | archived | 56.444 |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1 | outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1 | archived | 56.626 |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2 | outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2 | archived | 56.65 |
| outputs/runs/td3_pretrain_500k_external_step_test1_eval_seed0 | outputs/archive_old_experiments/td3_pretrain_500k_external_step_test1_eval_seed0 | archived | 41.611 |
| outputs/runs/td3_pretrain_500k_seed0 | outputs/archive_old_experiments/td3_pretrain_500k_seed0 | archived | 54.105 |
| outputs/runs/td3_smoke | outputs/archive_old_experiments/td3_smoke | archived | 36.111 |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_low_actor_lr | outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_low_actor_lr | archived | 35.202 |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_nodecay | outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_nodecay | archived | 36.868 |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_noisedecay | outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_noisedecay | archived | 35.286 |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed1_low_actor_lr | outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed1_low_actor_lr | archived | 35.282 |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed2_low_actor_lr | outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed2_low_actor_lr | archived | 35.282 |

## 3. Files Deleted As Cache/Temp
| path | type | status | size MB | reason |
|---|---|---|---:|---|
| agents/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| agents/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| agents/__pycache__/ddpg.cpython-313.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| agents/__pycache__/ddpg_agent.cpython-310.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| agents/__pycache__/ddpg_agent.cpython-313.pyc | file | deleted | 0.011 | obvious cache/temp/log file |
| agents/__pycache__/networks.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| agents/__pycache__/networks.cpython-313.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| agents/__pycache__/replay_buffer.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| agents/__pycache__/replay_buffer.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| agents/__pycache__/td3_agent.cpython-310.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| agents/__pycache__/td3_agent.cpython-313.pyc | file | deleted | 0.013 | obvious cache/temp/log file |
| baselines/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| baselines/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| baselines/__pycache__/cascade_servo_controller.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| baselines/__pycache__/cascade_servo_controller.cpython-313.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| baselines/__pycache__/pi_current_control.cpython-313.pyc | file | deleted | 0.002 | obvious cache/temp/log file |
| baselines/__pycache__/pi_current_controller.cpython-310.pyc | file | deleted | 0.011 | obvious cache/temp/log file |
| baselines/__pycache__/pi_current_controller.cpython-313.pyc | file | deleted | 0.02 | obvious cache/temp/log file |
| baselines/__pycache__/pi_speed_controller.cpython-310.pyc | file | deleted | 0.002 | obvious cache/temp/log file |
| baselines/__pycache__/pi_speed_controller.cpython-313.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| baselines/__pycache__/pid_position_controller.cpython-310.pyc | file | deleted | 0.002 | obvious cache/temp/log file |
| baselines/__pycache__/pid_position_controller.cpython-313.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| baselines/__pycache__/smc_position_controller.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| envs/__pycache__/__init__.cpython-310.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| envs/__pycache__/__init__.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| envs/__pycache__/gem_factory.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| envs/__pycache__/gun_load_model.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| envs/__pycache__/gun_load_model.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| envs/__pycache__/gun_servo_components.cpython-310.pyc | file | deleted | 0.01 | obvious cache/temp/log file |
| envs/__pycache__/gun_servo_components.cpython-313.pyc | file | deleted | 0.016 | obvious cache/temp/log file |
| envs/__pycache__/gun_servo_position_env.cpython-310.pyc | file | deleted | 0.028 | obvious cache/temp/log file |
| envs/__pycache__/gun_servo_position_env.cpython-313.pyc | file | deleted | 0.066 | obvious cache/temp/log file |
| envs/__pycache__/make_env.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| envs/__pycache__/make_env.cpython-313.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| envs/__pycache__/motor_model.cpython-310.pyc | file | deleted | 0.007 | obvious cache/temp/log file |
| envs/__pycache__/motor_model.cpython-313.pyc | file | deleted | 0.011 | obvious cache/temp/log file |
| envs/__pycache__/obs_parser.cpython-310.pyc | file | deleted | 0.012 | obvious cache/temp/log file |
| envs/__pycache__/obs_parser.cpython-313.pyc | file | deleted | 0.018 | obvious cache/temp/log file |
| envs/__pycache__/pmsm_current_env.cpython-310.pyc | file | deleted | 0.027 | obvious cache/temp/log file |
| envs/__pycache__/pmsm_current_env.cpython-313.pyc | file | deleted | 0.056 | obvious cache/temp/log file |
| envs/__pycache__/reference_generators.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| envs/__pycache__/reference_generators.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| envs/__pycache__/trajectory_generator.cpython-310.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| envs/__pycache__/trajectory_generator.cpython-313.pyc | file | deleted | 0.015 | obvious cache/temp/log file |
| envs/__pycache__/wrappers.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| envs/__pycache__/wrappers.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8/train.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8/train_stderr.log | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8/train_stdout.log | file | deleted | 0.055 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0/train.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0/train_stderr.log | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed0/train_stdout.log | file | deleted | 0.083 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1/train.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1/train_stderr.log | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed1/train_stdout.log | file | deleted | 0.084 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2/train.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2/train_stderr.log | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_random_v8_full_seed2/train_stdout.log | file | deleted | 0.083 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/final_eval.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/selection.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/gun_servo_mc_sc_residual_td3_smoke_v7/training.pid | file | deleted | 0 | obvious cache/temp/log file |
| outputs/runs/td3_constraint_reward_200k_seed0/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_constraint_reward_200k_seed0/train_stdout.log | file | skipped_missing | 0.075 | obvious cache/temp/log file |
| outputs/runs/td3_physics_reward_v2_200k_seed0/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_physics_reward_v2_200k_seed0/train_stdout.log | file | skipped_missing | 0.15 | obvious cache/temp/log file |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1/train_stdout.log | file | skipped_missing | 0.099 | obvious cache/temp/log file |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2/train_stdout.log | file | skipped_missing | 0.096 | obvious cache/temp/log file |
| outputs/runs/td3_pretrain_500k_seed0/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_pretrain_500k_seed0/train_stdout.log | file | skipped_missing | 0.092 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_low_actor_lr/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_low_actor_lr/train_stdout.log | file | skipped_missing | 0.078 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_nodecay/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_nodecay/train_stdout.log | file | skipped_missing | 0.074 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_noisedecay/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed0_noisedecay/train_stdout.log | file | skipped_missing | 0.073 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed1_low_actor_lr/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed1_low_actor_lr/train_stdout.log | file | skipped_missing | 0.076 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed2_low_actor_lr/train_stderr.log | file | skipped_missing | 0 | obvious cache/temp/log file |
| outputs/runs/td3_stability_200k_external_step_randomtime_seed2_low_actor_lr/train_stdout.log | file | skipped_missing | 0.145 | obvious cache/temp/log file |
| scripts/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/__pycache__/batch_eval_checkpoints.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/__pycache__/batch_eval_checkpoints.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| scripts/__pycache__/check_env_normalization.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| scripts/__pycache__/check_env_normalization.cpython-313.pyc | file | deleted | 0.007 | obvious cache/temp/log file |
| scripts/__pycache__/check_experiment_conditions.cpython-310.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| scripts/__pycache__/check_experiment_conditions.cpython-313.pyc | file | deleted | 0.013 | obvious cache/temp/log file |
| scripts/__pycache__/check_metrics_and_plots.cpython-310.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| scripts/__pycache__/check_metrics_and_plots.cpython-313.pyc | file | deleted | 0.013 | obvious cache/temp/log file |
| scripts/__pycache__/check_reference_profile.cpython-310.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| scripts/__pycache__/check_reference_profile.cpython-313.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| scripts/__pycache__/evaluate.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/__pycache__/evaluate.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| scripts/__pycache__/inspect_env.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/__pycache__/inspect_env.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| scripts/__pycache__/test_motor_openloop.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| scripts/__pycache__/test_motor_openloop.cpython-313.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| scripts/__pycache__/test_pi_current_loop.cpython-310.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| scripts/__pycache__/test_pi_current_loop.cpython-313.pyc | file | deleted | 0.013 | obvious cache/temp/log file |
| scripts/__pycache__/train.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/__pycache__/train.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/batch_eval_checkpoints.cpython-310.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/batch_eval_checkpoints.cpython-313.pyc | file | deleted | 0.009 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/summarize_eval_batch.cpython-310.pyc | file | deleted | 0.007 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/summarize_eval_batch.cpython-313.pyc | file | deleted | 0.012 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/summarize_gun_position_eval.cpython-310.pyc | file | deleted | 0.007 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/summarize_gun_position_eval.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/test1_pre_post_metrics.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| scripts/analysis/__pycache__/test1_pre_post_metrics.cpython-313.pyc | file | deleted | 0.007 | obvious cache/temp/log file |
| scripts/core/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/core/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/core/__pycache__/evaluate.cpython-310.pyc | file | deleted | 0.024 | obvious cache/temp/log file |
| scripts/core/__pycache__/evaluate.cpython-313.pyc | file | deleted | 0.041 | obvious cache/temp/log file |
| scripts/core/__pycache__/train.cpython-310.pyc | file | deleted | 0.023 | obvious cache/temp/log file |
| scripts/core/__pycache__/train.cpython-313.pyc | file | deleted | 0.057 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/aggregate_gun_servo_v8_full_multiseed.cpython-313.pyc | file | deleted | 0.034 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/build_gun_servo_final_comparison.cpython-313.pyc | file | deleted | 0.03 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/build_paper_ready_results.cpython-313.pyc | file | deleted | 0.044 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/check_pi_residual_interface.cpython-310.pyc | file | deleted | 0.015 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/check_pi_residual_interface.cpython-313.pyc | file | deleted | 0.025 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/check_reward_components.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/check_reward_components.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/checkpoint_sweep_multiseed_500k.cpython-310.pyc | file | deleted | 0.011 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/checkpoint_sweep_multiseed_500k.cpython-313.pyc | file | deleted | 0.019 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/checkpoint_sweep_test1.cpython-310.pyc | file | deleted | 0.012 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/checkpoint_sweep_test1.cpython-313.pyc | file | deleted | 0.022 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/final_figure_completeness_audit.cpython-313.pyc | file | deleted | 0.071 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/inspect_env.cpython-310.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/inspect_env.cpython-313.pyc | file | deleted | 0.01 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/inspect_gun_servo_env.cpython-310.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/inspect_gun_servo_env.cpython-313.pyc | file | deleted | 0.008 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_pid_mc_eval.cpython-313.pyc | file | deleted | 0.022 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_pid_smoke.cpython-313.pyc | file | deleted | 0.042 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_smc_baseline.cpython-313.pyc | file | deleted | 0.033 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_smc_mc_eval.cpython-313.pyc | file | deleted | 0.033 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_td3_mc_eval.cpython-313.pyc | file | deleted | 0.034 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_td3_smoke_eval.cpython-310.pyc | file | deleted | 0.02 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/run_gun_servo_td3_smoke_eval.cpython-313.pyc | file | deleted | 0.037 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/select_gun_servo_mc_safe_checkpoint.cpython-313.pyc | file | deleted | 0.019 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/smoke_test_pi.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/smoke_test_pi.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/summarize_eval_single.cpython-310.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| scripts/diagnostics/__pycache__/summarize_eval_single.cpython-313.pyc | file | deleted | 0.002 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/__init__.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/__init__.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/paper_figures.cpython-310.pyc | file | deleted | 0.02 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/paper_figures.cpython-313.pyc | file | deleted | 0.035 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_checkpoint_sweep.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_checkpoint_sweep.cpython-313.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_episode_trace.cpython-310.pyc | file | deleted | 0.011 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_episode_trace.cpython-313.pyc | file | deleted | 0.017 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_failure_case.cpython-310.pyc | file | deleted | 0.002 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_failure_case.cpython-313.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_gun_position_trace.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_gun_position_trace.cpython-313.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_gun_servo_comparison.cpython-310.pyc | file | deleted | 0.002 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_gun_servo_comparison.cpython-313.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_training_curves.cpython-310.pyc | file | deleted | 0.003 | obvious cache/temp/log file |
| scripts/plotting/__pycache__/plot_training_curves.cpython-313.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| utils/__pycache__/__init__.cpython-310.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| utils/__pycache__/__init__.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| utils/__pycache__/checkpoint.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| utils/__pycache__/checkpoint.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| utils/__pycache__/config.cpython-310.pyc | file | deleted | 0.03 | obvious cache/temp/log file |
| utils/__pycache__/config.cpython-313.pyc | file | deleted | 0.056 | obvious cache/temp/log file |
| utils/__pycache__/experiment_factory.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| utils/__pycache__/experiment_factory.cpython-313.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| utils/__pycache__/io.cpython-310.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| utils/__pycache__/io.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| utils/__pycache__/logger.cpython-310.pyc | file | deleted | 0 | obvious cache/temp/log file |
| utils/__pycache__/logger.cpython-313.pyc | file | deleted | 0 | obvious cache/temp/log file |
| utils/__pycache__/metrics.cpython-310.pyc | file | deleted | 0.013 | obvious cache/temp/log file |
| utils/__pycache__/metrics.cpython-313.pyc | file | deleted | 0.024 | obvious cache/temp/log file |
| utils/__pycache__/residual_control.cpython-310.pyc | file | deleted | 0.006 | obvious cache/temp/log file |
| utils/__pycache__/residual_control.cpython-313.pyc | file | deleted | 0.009 | obvious cache/temp/log file |
| utils/__pycache__/run_layout.cpython-310.pyc | file | deleted | 0.004 | obvious cache/temp/log file |
| utils/__pycache__/run_layout.cpython-313.pyc | file | deleted | 0.005 | obvious cache/temp/log file |
| utils/__pycache__/seed.cpython-310.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| utils/__pycache__/seed.cpython-313.pyc | file | deleted | 0.001 | obvious cache/temp/log file |
| __pycache__ | directory | deleted | 0.035 | Python __pycache__ directory |
| agents/__pycache__ | directory | deleted | 1.601 | Python __pycache__ directory |
| baselines/__pycache__ | directory | deleted | 0.859 | Python __pycache__ directory |
| configs/__pycache__ | directory | deleted | 0 | Python __pycache__ directory |
| envs/__pycache__ | directory | deleted | 7.007 | Python __pycache__ directory |
| scripts/__pycache__ | directory | deleted | 0.149 | Python __pycache__ directory |
| scripts/analysis/__pycache__ | directory | deleted | 0.174 | Python __pycache__ directory |
| scripts/core/__pycache__ | directory | deleted | 0.329 | Python __pycache__ directory |
| scripts/diagnostics/__pycache__ | directory | deleted | 0.611 | Python __pycache__ directory |
| scripts/plotting/__pycache__ | directory | deleted | 0.151 | Python __pycache__ directory |
| utils/__pycache__ | directory | deleted | 4.321 | Python __pycache__ directory |


### Archive-Path Cache/Temp Logs Deleted After Move

| path | type | status | size MB | reason |
|---|---|---|---:|---|
| outputs/archive_old_experiments/td3_constraint_reward_200k_seed0/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_constraint_reward_200k_seed0/train_stdout.log | file | deleted | 0.075 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_physics_reward_v2_200k_seed0/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_physics_reward_v2_200k_seed0/train_stdout.log | file | deleted | 0.15 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed1/train_stdout.log | file | deleted | 0.099 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_pretrain_500k_external_step_randomtime_test1_eval_seed2/train_stdout.log | file | deleted | 0.096 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_pretrain_500k_seed0/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_pretrain_500k_seed0/train_stdout.log | file | deleted | 0.092 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_low_actor_lr/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_low_actor_lr/train_stdout.log | file | deleted | 0.078 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_nodecay/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_nodecay/train_stdout.log | file | deleted | 0.074 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_noisedecay/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed0_noisedecay/train_stdout.log | file | deleted | 0.073 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed1_low_actor_lr/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed1_low_actor_lr/train_stdout.log | file | deleted | 0.076 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed2_low_actor_lr/train_stderr.log | file | deleted | 0 | archive-path train stdout/stderr temp log listed before move |
| outputs/archive_old_experiments/td3_stability_200k_external_step_randomtime_seed2_low_actor_lr/train_stdout.log | file | deleted | 0.145 | archive-path train stdout/stderr temp log listed before move |
## 4. Editor Metadata Still Requiring Confirmation
- .idea (0.046 MB): editor metadata; requires explicit confirmation
- .cursor (0.01 MB): editor metadata; requires explicit confirmation

## 5. Project Size Before And After
- Total workspace size before: 13148.674 MB
- Total workspace size after: 13132.236 MB
- Active workspace size before excluding archive folder: 13148.674 MB
- Active workspace size after excluding archive folder: 12253.076 MB
- Note: archiving moves files inside the workspace, so total workspace size changes only by deleted cache/temp bytes.

## 6. Critical File Verification
| path | exists |
|---|---|
| configs/env/gun_servo_position_mc_sc_residual_td3_v8_rand.yaml | True |
| configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_fixed.yaml | True |
| configs/eval/gun_servo_mc_sc_residual_td3_random_v8_eval_mc.yaml | True |
| scripts/train.py | True |
| scripts/diagnostics/run_gun_servo_td3_smoke_eval.py | True |
| scripts/diagnostics/run_gun_servo_pid_mc_eval.py | True |
| scripts/diagnostics/run_gun_servo_smc_mc_eval.py | True |
| outputs/paper_ready_results | True |
| outputs/runs/final_method_comparison | True |


Requested keep path status: `outputs/experiment_closeout` is not present in the workspace; it was not moved or deleted by this cleanup.
## 7. Cleanup Safety
- Cleanup is safe with respect to the requested critical current-project paths.
- Current gun-servo final run folders, paper-ready results, figure audit outputs, and Direct TD3 candidate folders were kept in place.
- No `.pt` files were deleted from final folders.

## 8. Next Recommended Action
- Review `cleanup_reports/delete_cache_temp_list.md` editor metadata entries and confirm separately if `.idea/`, `.cursor/`, or `.vscode/` should be removed.
- Optionally back up `outputs/archive_old_experiments/` outside the repo before any future deletion.

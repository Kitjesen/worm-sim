# Flat High-Speed Random-Policy Progress

Date: 2026-05-29

This note tracks the first PPO residual run after switching the deployable V6
contract to `cmaes_tri_anchor_residual_v1` and `high_speed_directional_v1`.

## Training

| Chunk | Device | Start steps | End steps | Best eval reward | Best eval step | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | CUDA | 0 | 114,688 | 1791.32 | 75,000 | CUDA available, but SB3 warned MLP-PPO has poor GPU utilization; about 333 it/s. |
| 2 | CPU | 114,688 | >200,000 | 1835.17 | 154,688 | CPU continuation was faster locally; about 362 it/s. |

Run directory: `runs/worm_v6_ppo_flat_random/`

The stale low-speed run artifacts were archived to:
`runs/worm_v6_ppo_flat_random/incompatible_timing_archive/20260529_092238/`

## Fixed-Blend Video Metrics

All rows use `cmd_vel=0.25 m/s`, 5 s rollouts, and the corrected deployable
CMA-ES phase-clock prior. PPO rows use the flat/random residual policy.

| Policy | Mode | Cmd yaw | Speed (mm/s) | Drift (mm) | Root yaw delta (rad) | Metrics |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Prior only | worm | 0.0 | 33.35 | 1.9 | 0.000 | `record/v6/paper_results/eval_flat_worm_cmaes_prior_20260529.json` |
| PPO 115k best | worm | 0.0 | 38.73 | 2.1 | -0.001 | `record/v6/paper_results/eval_flat_worm_ppo_highspeed_random_115k_best_20260529.json` |
| PPO 213k best | worm | 0.0 | 40.62 | 3.4 | 0.001 | `record/v6/paper_results/eval_flat_worm_ppo_highspeed_random_213k_best_20260529.json` |
| Prior only | mixed | 0.0 | 139.16 | 56.0 | 0.000 | `record/v6/paper_results/eval_flat_mixed_cmaes_prior_20260529.json` |
| PPO 115k best | mixed | 0.0 | 161.80 | 132.8 | -0.233 | `record/v6/paper_results/eval_flat_mixed_ppo_highspeed_random_115k_best_20260529.json` |
| PPO 213k best | mixed | 0.0 | 143.82 | 119.4 | -0.402 | `record/v6/paper_results/eval_flat_mixed_ppo_highspeed_random_213k_best_20260529.json` |
| Prior only | snake | 0.0 | 68.09 | 80.3 | 0.000 | `record/v6/paper_results/eval_flat_snake_cmaes_prior_20260529.json` |
| PPO 115k best | snake | 0.0 | 62.84 | 52.0 | -0.442 | `record/v6/paper_results/eval_flat_snake_ppo_highspeed_random_115k_best_20260529.json` |
| PPO 213k best | snake | 0.0 | 65.58 | 57.3 | -0.460 | `record/v6/paper_results/eval_flat_snake_ppo_highspeed_random_213k_best_20260529.json` |
| PPO 213k best | mixed | +1.0 | 154.24 | 257.2 | 0.104 | `record/v6/paper_results/eval_flat_mixed_left_ppo_highspeed_random_213k_best_20260529.json` |
| PPO 213k best | mixed | -1.0 | 143.79 | 77.5 | -0.375 | `record/v6/paper_results/eval_flat_mixed_right_ppo_highspeed_random_213k_best_20260529.json` |

## Interpretation

The corrected PPO residual is beginning to improve useful straight-line
locomotion: worm mode improved over the prior in both PPO checkpoints, and the
115k mixed rollout exceeded the corrected prior. The 213k best eval reward is
higher than the 115k run, but single-video fixed-blend mixed speed is noisy and
needs more seeds or a robust eval table before making a paper claim.

Yaw/directional control is not solved yet. The left/right yaw responses are now
measurable, but still asymmetric and too weak for the final all-directional
claim.

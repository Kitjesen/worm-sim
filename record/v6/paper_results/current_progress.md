# Worm V6 Deployable Multimodal Progress

Generated on 2026-05-28.

## Current Status

- Goal is not complete yet.
- Deployable observation contract is implemented: policy observations use command, joint encoders, previous action, per-segment IMU gravity/gyro, and phase clock.
- Forbidden policy observations are audited out: base linear velocity, global pose/yaw, and MuJoCo freejoint truth are not policy inputs.
- Actuator contract is now explicit and enforced:
  - slide targets: `[-0.05, 0.0] m`
  - yaw targets: `[-1.57, 1.57] rad`
  - peristaltic actuation period: `1.0 s`
- Flat fixed-mode policies have been retrained under the current deployable observation and actuator contract:
  - `flat/worm`: 1,015,808 PPO steps
  - `flat/snake`: 1,015,808 PPO steps
  - `flat/mixed`: 1,015,808 PPO steps
- `flat/random` continuous `gait_blend` policy is currently retraining under the same contract; last checked while running, it had produced checkpoints past 160,000 steps.

## Effect So Far

- Flat fixed modes now have current-contract model artifacts and can be compared without relying on the old unrealistic actuator mapping.
- `flat/mixed` did improve during the second training chunk: the best evaluation checkpoint was updated near 976k steps, but the evaluation reward remains more variable than the fixed modes.
- Current cross-terrain summary files still contain stale sand/slope and old random-policy data. Treat those as exploratory until sand/slope/random are retrained and the eval/scan stage is rerun.

## Viewable Evidence

- `record/v6/videos/gait_modes_comparison_1s_720p.mp4`
- `record/v6/videos/worm_v6_worm.mp4`
- `record/v6/videos/worm_v6_snake.mp4`
- `record/v6/videos/worm_v6_combined.mp4`
- `record/v6/videos/gait_comparison_1280x720.mp4`
- `record/v6/videos/eval_flat_random.mp4`
- `record/v6/videos/eval_sand_random.mp4`
- `record/v6/videos/eval_slope_random.mp4`
- `record/v6/trajectory/`
- `record/v6/paper_results/results_index.md`
- `record/v6/paper_results/completion_audit.md`

Large 4K videos are intentionally not committed to Git because GitHub rejects files over 100 MB without LFS.

## Remaining Work

- Finish retraining `flat/random`.
- Retrain sand and slope policies under the current actuator contract.
- Rerun deploy/export, fixed-mode eval, robust eval, gait_blend scan, summary, and audit.
- Collect real flat/sand/slope hardware logs with video references.
- Rebuild the final paper figures after all current-contract evals are complete.

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
- `flat/random` continuous `gait_blend` policy has also reached the formal threshold:
  - `flat/random`: 1,015,808 PPO steps
- The flat random deploy bundle has been exported:
  - `record/v6/deploy_bundles/flat_random/policy_actor.pt`
  - `record/v6/deploy_bundles/flat_random/deploy_config.json`
- Flat fixed-mode nominal and robust evals have been regenerated under the current contract.
- Flat random `gait_blend` scan has been regenerated under the current contract.
- `sand/worm` current-contract retraining has started:
  - old incompatible artifacts were archived by the training entry point
  - new `training_config.json` includes `control_timing` and `actuator_contract_fingerprint`
  - current progress: `114,688 / 1,000,000` PPO steps

## Effect So Far

- Flat fixed modes now have current-contract model artifacts and can be compared without relying on the old unrealistic actuator mapping.
- `flat/mixed` did improve during the second training chunk: the best evaluation checkpoint was updated near 976k steps, but the evaluation reward remains more variable than the fixed modes.
- Flat `random` blend scan result:
  - best current blend: `gait_blend=0.75`
  - speed: `24.832 mm/s`
  - success: `1.0`
  - comparison: `gait_blend=1.0` gives `24.116 mm/s`; `gait_blend=0.5` gives `20.799 mm/s`; `gait_blend=0.0` gives `1.876 mm/s`.
- Flat robust fixed-mode result:
  - `worm`: `13.770 mm/s`, success `1.0`
  - `snake`: `-14.457 mm/s`, success `0.0`
  - `mixed`: `31.325 mm/s`, success `1.0`
- Current cross-terrain summary files now mark stale sand/slope artifacts as `stale` and leave their metric cells blank. They are not counted as current paper evidence.
- `sand/worm` has moved from "stale contract" to "current contract but below formal threshold"; it should continue with `--resume-partial` until it reaches 1M steps before eval/scan/deploy are counted.
- Hardware deploy preflight currently passes for `flat/random`, but fails for sand/slope because their deploy bundles are not current-contract bundles yet.

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
- `record/v6/paper_results/summary.md`
- `record/v6/paper_results/paper_claims.md`
- `record/v6/deploy_bundles/flat_random/`
- `runs/worm_v6_ppo_sand_worm/` current model/config/result files

Large 4K videos are intentionally not committed to Git because GitHub rejects files over 100 MB without LFS.

## Remaining Work

- Retrain sand and slope policies under the current actuator contract.
- Rerun deploy/export, fixed-mode eval, robust eval, gait_blend scan, summary, and audit.
- Collect real flat/sand/slope hardware logs with video references.
- Rebuild the final paper figures after all current-contract evals are complete.

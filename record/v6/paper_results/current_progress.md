# Worm V6 Deployable Multimodal Progress

Generated on 2026-05-29.

## Current Status

- Goal is not complete yet.
- The current main line is `src/v6`, not the legacy `src/v3` wrappers.
- Deployable observation contract is implemented: policy observations use command, joint encoders, previous action, per-segment IMU gravity/gyro, and phase clock.
- Forbidden policy observations are audited out: base linear velocity, global pose/yaw, and MuJoCo freejoint truth are not policy inputs.
- Actuator contract is explicit and enforced:
  - slide targets: `[-0.05, 0.0] m`
  - yaw targets: `[-1.57, 1.57] rad`
  - peristaltic actuation period: `1.0 s`
- The current training/evaluation line is stricter than the older 1M-step artifacts:
  - reward contract: `forward_progress_v3`
  - action adapter: `gait_prior_residual_v1`
  - residual exploration: `low_noise_residual_v1`
  - fixed eval command: `cmd_vel=0.025 m/s`, `cmd_yaw=0`, no command resampling
- Old flat/sand/slope 1M-step artifacts are now treated as stale when they do not match the current reward, action-adapter, actuator, timing, or residual-exploration contract.

## Current Training Evidence

- Fresh `flat/worm` low-noise residual PPO run:
  - current progress: `65,536 / 1,000,000` PPO steps
  - run directory: `runs/worm_v6_ppo_flat_worm/`
  - best model: `runs/worm_v6_ppo_flat_worm/best_model.zip`
  - training result: `runs/worm_v6_ppo_flat_worm/training_result.json`
- Training behavior improved after lowering residual exploration:
  - previous high-noise residual training showed episode rewards near `-3700` and short terminated episodes
  - current low-noise run shows episode rewards near `-750` with full `1000`-step episodes during the 65k chunk
- Fixed-command robust 10 s eval of the current best `flat/worm` checkpoint:
  - distance: `153.4 mm`
  - speed: `15.34 mm/s`
  - lateral drift: `3.1 mm`
  - success: `1.0`
  - metrics: `runs/worm_v6_ppo_flat_worm/eval_metrics_reward_v3_prior_lownoise_65k_best.json`
- Low-resolution video rollout of the same checkpoint:
  - duration: `5.0 s`
  - distance: `110.1 mm`
  - speed: `22.0 mm/s`
  - termination: `false`
  - video: `record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_65k_best_20260529.mp4`
  - metrics: `runs/worm_v6_ppo_flat_worm/eval_metrics_reward_v3_prior_lownoise_65k_best_video.json`

## Current Audit Result

`python src/v6/paper_status_v6.py --refresh-audit` reports:

- `complete: False`
- next training target: `flat/worm`
- training artifacts present: `11/12`
- current formal training threshold met: `0/12`

The audit still requires:

- 12 current-contract PPO model artifacts at the formal threshold
- 9 fixed-mode eval JSON files
- 9 robust fixed-mode eval JSON files
- 3 continuous `gait_blend` scan result files
- 3 deployable random-policy bundles
- hardware deploy preflight report
- flat/sand/slope hardware logs with video references

## Viewable Evidence

- `record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_65k_best_20260529.mp4`
- `record/v6/trajectory/`
- `record/v6/paper_results/results_index.md`
- `record/v6/paper_results/completion_audit.md`

Large 4K videos are intentionally not committed to Git because GitHub rejects files over 100 MB without LFS.

## Remaining Work

- Continue `flat/worm` from `65,536` to `1,000,000` current-contract steps.
- Retrain flat snake/mixed/random and all sand/slope modes under the current reward/action/exploration contracts.
- Rerun deploy/export, fixed-mode eval, robust eval, `gait_blend` scan, summary, and audit.
- Collect real flat/sand/slope hardware logs with video references.
- Rebuild final paper figures after all current-contract evals are complete.

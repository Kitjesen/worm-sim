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
- The current training/evaluation line has been corrected against the stronger 4K CMA-ES comparison video:
  - reward contract: `high_speed_directional_v1`
  - action adapter: `cmaes_tri_anchor_residual_v1`
  - residual policy scale: `0.35`
  - command range: `cmd_vel in [0.0, 0.25] m/s`, `cmd_yaw in [-1.0, 1.0] rad/s`
  - gait anchors: peristaltic at `gait_blend=0.0`, full combined at `0.5`, serpentine at `1.0`
- The 4K comparison video is an open-loop CMA-ES baseline, not a PPO result. Its strongest flat full-combined gait is `247.97 mm/s` from `runs/cmaes_speed_full/best_gait.json`.
- Old PPO artifacts trained under `forward_progress_v3` and `cmd_vel=0.025 m/s` are stale for paper claims because they optimized a `25 mm/s` task.

## Current Training Evidence

- Old `flat/worm` low-noise residual PPO run:
  - status: finished to the 1M-step target, but stale under the new high-speed contract
  - run directory: `runs/worm_v6_ppo_flat_worm/`
  - best model: `runs/worm_v6_ppo_flat_worm/best_model.zip`
  - training result: `runs/worm_v6_ppo_flat_worm/training_result.json`
- What the old run taught us:
  - previous high-noise residual training showed episode rewards near `-3700` and short terminated episodes
  - lowering residual exploration reduced instability
  - the remaining speed ceiling came from the wrong reward target, not from an intrinsic robot limit
- Fixed-command robust 10 s eval of the current best `flat/worm` checkpoint:
  - distance: `153.4 mm`
  - speed: `15.34 mm/s`
  - lateral drift: `3.1 mm`
  - success: `1.0`
  - metrics: `runs/worm_v6_ppo_flat_worm/eval_metrics_reward_v3_prior_lownoise_65k_best.json`
- Fixed-command robust 10 s eval of the current final `flat/worm` checkpoint:
  - distance: `186.2 mm`
  - speed: `18.62 mm/s`
  - lateral drift: `17.2 mm`
  - success: `1.0`
  - metrics: `runs/worm_v6_ppo_flat_worm/eval_metrics_reward_v3_prior_lownoise_278k_final.json`
- Low-resolution video rollout of the same checkpoint:
  - duration: `5.0 s`
  - distance: `110.1 mm`
  - speed: `22.0 mm/s`
  - termination: `false`
  - video: `record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_65k_best_20260529.mp4`
  - metrics: `runs/worm_v6_ppo_flat_worm/eval_metrics_reward_v3_prior_lownoise_65k_best_video.json`
- Low-resolution video rollout of the final checkpoint:
  - duration: `5.0 s`
  - distance: `121.7 mm`
  - speed: `24.34 mm/s`
  - lateral drift: `1.34 mm`
  - termination: `false`
  - video: `record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_278k_final_20260529.mp4`
  - metrics: `runs/worm_v6_ppo_flat_worm/eval_metrics_reward_v3_prior_lownoise_278k_final_video.json`
- Corrected zero-residual `flat/worm` CMA-ES-anchor prior rollout:
  - duration: `5.0 s`
  - command: `cmd_vel=0.25 m/s`, `cmd_yaw=0`
  - distance: `166.8 mm`
  - speed: `33.35 mm/s`
  - lateral drift: `1.9 mm`
  - termination: `false`
  - video: `record/v6/videos/eval_flat_worm_cmaes_prior_20260529.mp4`
  - metrics: `record/v6/paper_results/eval_flat_worm_cmaes_prior_20260529.json`
- Corrected zero-residual `flat/mixed` CMA-ES-anchor prior rollout:
  - duration: `5.0 s`
  - command: `cmd_vel=0.25 m/s`, `cmd_yaw=0`
  - distance: `695.8 mm`
  - speed: `139.16 mm/s`
  - lateral drift: `56.0 mm`
  - termination: `false`
  - video: `record/v6/videos/eval_flat_mixed_cmaes_prior_20260529.mp4`
  - metrics: `record/v6/paper_results/eval_flat_mixed_cmaes_prior_20260529.json`
  - interpretation: this is the corrected deployable prior baseline; PPO still needs to be retrained as a residual on top of it.
- Corrected zero-residual `flat/snake` CMA-ES-anchor prior rollout:
  - duration: `5.0 s`
  - command: `cmd_vel=0.25 m/s`, `cmd_yaw=0`
  - distance: `340.4 mm`
  - speed: `68.09 mm/s`
  - lateral drift: `80.3 mm`
  - termination: `false`
  - video: `record/v6/videos/eval_flat_snake_cmaes_prior_20260529.mp4`
  - metrics: `record/v6/paper_results/eval_flat_snake_cmaes_prior_20260529.json`
- Interpretation: the corrected deployable prior is materially better than the old low-speed PPO videos, but it still does not reproduce the raw 4K `247.97 mm/s` full-combined CMA-ES baseline because V6 now projects anchors through the deployable 1 s phase clock instead of hidden simulator time.

## Current Audit Result

`python src/v6/paper_status_v6.py --refresh-audit` reports:

- `complete: False`
- next training target: fresh high-speed flat random/mixed residual PPO
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

- `record/v6/videos/eval_flat_worm_cmaes_prior_20260529.mp4`
- `record/v6/videos/eval_flat_mixed_cmaes_prior_20260529.mp4`
- `record/v6/videos/eval_flat_snake_cmaes_prior_20260529.mp4`
- `record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_65k_best_20260529.mp4`
- `record/v6/trajectory/`
- `record/v6/paper_results/results_index.md`
- `record/v6/paper_results/completion_audit.md`

Large 4K videos are intentionally not committed to Git because GitHub rejects files over 100 MB without LFS.

## Remaining Work

- Train fresh flat random/mixed residual PPO under `high_speed_directional_v1`.
- Retrain flat worm/snake and all sand/slope modes under the current reward/action contracts.
- Rerun deploy/export, fixed-mode eval, robust eval, `gait_blend` scan, summary, and audit.
- Collect real flat/sand/slope hardware logs with video references.
- Rebuild final paper figures after all current-contract evals are complete.

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
  - reward contract: `high_speed_directional_v3`
  - action adapter: `cmaes_tri_anchor_residual_v1`
  - residual policy scale: `0.35`
  - command range: `cmd_vel in [0.0, 0.25] m/s`, `cmd_yaw in [-0.5, 0.5] rad/s`
  - gait anchors: peristaltic at `gait_blend=0.0`, full combined at `0.5`, serpentine at `1.0`
  - balanced best-eval schedule: `gait_blend in {0.0, 0.5, 1.0}` x `cmd_yaw in {-0.5, 0.0, 0.5}`
- The 4K comparison video is an open-loop CMA-ES baseline, not a PPO result. Its strongest flat full-combined gait is `247.97 mm/s` from `runs/cmaes_speed_full/best_gait.json`.
- Old PPO artifacts trained under `forward_progress_v3`, low-speed commands, or older reward contracts are stale for paper claims.

## Why The Current PPO Looks Worse Than The 4K Video

The 4K video shows the raw single-objective CMA-ES full-combined gait. It optimizes one open-loop gait for speed and does not need to respond to `gait_blend`, yaw commands, or deployable policy observations.

The current PPO policy is a residual controller on top of that prior. It must use only deployable observations, support worm/mixed/snake commands, and respond to left/straight/right yaw commands. That broader task is not solved yet. The current 300k v3 policy preserves some forward motion, but the directional command response is still biased toward negative yaw.

## Current Training Evidence

- Run directory: `runs/worm_v6_ppo_flat_random/`
- Current run status: about `311k / 1M` timesteps under `high_speed_directional_v3`
- Best balanced eval reward: `1946.40` at `194,688` steps
- Best-eval schedule fingerprint: `runs/worm_v6_ppo_flat_random/best_eval_summary.json`
- Earlier incompatible artifacts were archived under `runs/worm_v6_ppo_flat_random/incompatible_timing_archive/`

## Fixed-Blend Video Metrics

All rows use `cmd_vel=0.25 m/s` and 5 s rollouts. PPO rows use the flat/random residual policy.

| Policy | Mode | Cmd yaw | Speed (mm/s) | Drift (mm) | Root yaw delta (rad) | Metrics |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CMA-ES prior only | worm | 0.0 | 33.35 | 1.9 | 0.000 | `record/v6/paper_results/eval_flat_worm_cmaes_prior_20260529.json` |
| CMA-ES prior only | mixed | 0.0 | 139.16 | 56.0 | 0.000 | `record/v6/paper_results/eval_flat_mixed_cmaes_prior_20260529.json` |
| CMA-ES prior only | snake | 0.0 | 68.09 | 80.3 | 0.000 | `record/v6/paper_results/eval_flat_snake_cmaes_prior_20260529.json` |
| PPO v3 300k best | worm | 0.0 | 38.32 | 3.5 | 0.002 | `record/v6/paper_results/eval_flat_worm_ppo_highspeed_random_v3_300k_balanced_best_20260529.json` |
| PPO v3 300k best | mixed | 0.0 | 133.48 | 93.6 | -0.270 | `record/v6/paper_results/eval_flat_mixed_ppo_highspeed_random_v3_300k_balanced_best_20260529.json` |
| PPO v3 300k best | snake | 0.0 | 62.89 | 119.1 | -0.385 | `record/v6/paper_results/eval_flat_snake_ppo_highspeed_random_v3_300k_balanced_best_20260529.json` |
| PPO v3 300k best | mixed | +0.5 | 147.03 | 139.6 | -0.516 | `record/v6/paper_results/eval_flat_mixed_left_ppo_highspeed_random_v3_300k_balanced_best_20260529.json` |
| PPO v3 300k best | mixed | -0.5 | 146.92 | 47.8 | -0.383 | `record/v6/paper_results/eval_flat_mixed_right_ppo_highspeed_random_v3_300k_balanced_best_20260529.json` |

Interpretation:

- Worm mode has a small improvement over the corrected deployable prior.
- Mixed and snake mode are not better than the prior on the current 300k v3 checkpoint.
- The raw 4K full-combined CMA-ES gait remains much faster at `247.97 mm/s`.
- Directional control is not solved: positive yaw command still produces negative yaw in the current mixed-mode video.

## What Was Fixed

- Best-model selection no longer evaluates only straight-line motion; it now evaluates all three blends and left/straight/right yaw commands.
- The yaw command range was reduced from `[-1.0, 1.0]` to `[-0.5, 0.5] rad/s` after measuring the practical turning authority of the prior/residual actuator setup.
- The reward now includes signed yaw alignment, so correct yaw sign is explicitly rewarded and wrong yaw sign is penalized.
- Lateral drift penalty is reduced during commanded turning so normal turn arcs are not treated as straight-line drift.

## Remaining Failure

The policy is still stuck in a negative-yaw basin. The reward contract now distinguishes the correct yaw sign, but the 300k learned policy has not separated left and right commands. This is a training/control failure, not a video-recording failure.

Likely next technical steps:

- continue v3 to the 1M formal target with sign-aware eval metrics;
- add a hard directional success metric instead of relying only on mean eval reward;
- consider curriculum training: straight first, then isolated left/right, then mixed random;
- consider residual limits or prior mirroring so the policy can more easily produce symmetric yaw corrections.

## Current Audit Result

The formal paper goal still requires:

- 12 current-contract PPO model artifacts at the formal threshold;
- fixed-mode eval JSON files for flat/sand/slope and worm/mixed/snake;
- robust fixed-mode eval JSON files;
- continuous `gait_blend` scan results;
- deployable random-policy bundles;
- hardware deploy preflight report;
- flat/sand/slope hardware logs with video references.

## Viewable Evidence

- `record/v6/videos/gait_comparison_4k.mp4`
- `record/v6/videos/eval_flat_worm_cmaes_prior_20260529.mp4`
- `record/v6/videos/eval_flat_mixed_cmaes_prior_20260529.mp4`
- `record/v6/videos/eval_flat_snake_cmaes_prior_20260529.mp4`
- `record/v6/videos/eval_flat_worm_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4`
- `record/v6/videos/eval_flat_mixed_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4`
- `record/v6/videos/eval_flat_snake_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4`
- `record/v6/videos/eval_flat_mixed_left_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4`
- `record/v6/videos/eval_flat_mixed_right_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4`

Large 4K videos should use Git LFS or external release assets before being pushed to GitHub.

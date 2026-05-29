# Flat Directional Reward V3 Progress

Date: 2026-05-29

This note records why the current PPO videos do not yet match
`record/v6/videos/gait_comparison_4k.mp4`.

## Baseline

The 4K comparison video is the raw open-loop CMA-ES gait-comparison baseline.
The strongest flat full-combined gait is stored in
`runs/cmaes_speed_full/best_gait.json`:

- speed: `247.97 mm/s`
- duration: `10.0 s`
- displacement: `2479.71 mm`
- optimizer budget: `300` generations, `4800` evaluations

That controller optimizes one open-loop speed gait. It is not a deployable PPO
policy and does not need to follow yaw commands.

## PPO V3 Setup

The current learned line is the flat/random PPO residual policy:

- run directory: `runs/worm_v6_ppo_flat_random/`
- reward contract: `high_speed_directional_v3`
- action adapter: `cmaes_tri_anchor_residual_v1`
- command range: `cmd_vel in [0.0, 0.25] m/s`, `cmd_yaw in [-0.5, 0.5] rad/s`
- best eval schedule: `gait_blend in {0.0, 0.5, 1.0}` x `cmd_yaw in {-0.5, 0.0, 0.5}`
- current training amount: about `311k / 1M`
- best balanced eval reward: `1946.40` at `194,688` steps
- next best-model selection contract: `directional_sign_gate_v1`

## Superseding Auto-Gated ABI

After the deployment review, the current main line has moved beyond this V3
setup:

- reward contract: `omni_auto_gate_v4`
- action adapter: `cmaes_tri_anchor_auto_gate_v2`
- command range: `cmd_vx in [-0.25, 0.25] m/s`, `cmd_vy in [-0.15, 0.15] m/s`, `cmd_yaw in [-0.5, 0.5] rad/s`
- policy action: 11 residual motor commands plus one learned gait gate
- policy observation no longer includes externally commanded `gait_blend`

The metrics below remain useful as failure diagnosis, but they are stale for
formal paper claims and must be rerun under the new auto-gated contract.

## 300k Video Metrics

| Policy | Mode | Cmd yaw | Speed (mm/s) | Drift (mm) | Root yaw delta (rad) |
| --- | --- | ---: | ---: | ---: | ---: |
| CMA-ES prior only | worm | 0.0 | 33.35 | 1.9 | 0.000 |
| CMA-ES prior only | mixed | 0.0 | 139.16 | 56.0 | 0.000 |
| CMA-ES prior only | snake | 0.0 | 68.09 | 80.3 | 0.000 |
| PPO v3 300k best | worm | 0.0 | 38.32 | 3.5 | 0.002 |
| PPO v3 300k best | mixed | 0.0 | 133.48 | 93.6 | -0.270 |
| PPO v3 300k best | snake | 0.0 | 62.89 | 119.1 | -0.385 |
| PPO v3 300k best | mixed | +0.5 | 147.03 | 139.6 | -0.516 |
| PPO v3 300k best | mixed | -0.5 | 146.92 | 47.8 | -0.383 |

## Conclusion

The current PPO policy is not yet better than the CMA-ES comparison video. It
adds the right deployable interfaces and a balanced objective, but the learned
policy is still biased toward negative yaw. The clearest failure is the
`cmd_yaw=+0.5` mixed-mode rollout: it should turn left/positive yaw, but the
recorded root yaw delta is `-0.516 rad`.

This should be treated as a failed directional-learning checkpoint, not as a
paper result.

The current gate would reject this checkpoint: straight mixed motion has
`straight_drift`, the left-yaw command has `wrong_sign`, and only the right-yaw
command has the correct sign.

## Next Technical Fixes

- Continue v3 training to the formal 1M target only if sign-aware per-case
  metrics improve.
- Add directional pass/fail metrics to checkpoint selection: positive command
  must produce positive yaw, negative command must produce negative yaw.
- Train a curriculum with isolated straight, left, and right command phases
  before full random blend/yaw training.
- Consider symmetric residual regularization or mirrored yaw-prior augmentation
  so left/right actions are easier to learn.

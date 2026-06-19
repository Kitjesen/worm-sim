# Worm V6 V34 Training Log

Date: 2026-05-31

## Purpose

Continue training after the V29 visual baseline because the current policy is
not yet a good continuous omnidirectional tracker.

V34 deliberately resumes from V29 best instead of V33 best, because V30-V33 did
not beat V29 and V33 still had one wrong-yaw case.

## Training Setup

- Run label: `flat_random_continuous_tracking_v34_low_lr_from_v29best`
- Resume checkpoint:
  `runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip`
- Terrain: `flat`
- Gait mode: `random`
- Curriculum: `continuous_omni`
- Additional training requested: `60,000` steps
- Start timestep from resumed checkpoint: `260,000`
- Completed timestep: `325,536`
- Learning rate: `1e-4`
- Actor: `512-256-128`
- Critic: `512-256-128`
- Observation/action ABI unchanged: 80D obs, 12D action
- Contract resume was allowed because the deployable ABI stayed fixed while the
  reward/curriculum research contract moved to V18.

## Held-Out Directional Selection

| Policy | selection score | planar RMSE m/s | yaw RMSE rad/s | planar success | yaw success | wrong yaw | straight violations | stationary violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V29 best | `-1006644.12` | `0.1070` | `0.1729` | `0.7647` | `0.7647` | `0` | `3` | `4` |
| V34 best | `-1007782.83` | `0.1095` | `0.1825` | `0.7647` | `0.8235` | `1` | `2` | `4` |

V34 improves the held-out yaw-success rate and reduces one straight violation,
but it does not beat V29 overall and reintroduces one wrong-yaw case.

## 35-Command Scan

Artifact directory:
`record/current/flat_omni_v34_low_lr_scan`

Files:

- `scan_35_commands_best.json`
- `scan_35_commands_best.csv`

Scan summary for V34 best:

| Metric | Value |
| --- | ---: |
| commands | `35` |
| planar RMSE m/s | `0.1635` |
| yaw RMSE rad/s | `0.1128` |
| planar sign rate | `0.84` |
| yaw sign rate | `1.00` |
| zero-command mean speed m/s | `0.000048` |
| yaw-only mean planar speed m/s | `0.1046` |
| yaw-only max planar speed m/s | `0.1119` |

## Decision

V34 is a diagnostic continuation, not the new accepted policy. V29 remains the
current best viewable flat baseline. The next training change should not be
more of the same low-learning-rate fine-tuning; it should directly change the
task structure for lateral/yaw separation, for example:

1. add a yaw-only phase where translation is heavily penalized before mixed
   commands are sampled;
2. add a lateral-only phase where forward drift is clipped by a stronger
   success/failure gate;
3. keep axial forward/reverse worm-like actuation visible through the learned
   gait gate;
4. only return to full continuous `vx/vy/yaw` mixing after the pure primitives
   pass fixed-command thresholds.

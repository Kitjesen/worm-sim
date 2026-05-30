# Worm V6 V30-V33 Omni Repair Log

Date: 2026-05-31

## Purpose

After recording the V29 HD direction videos, four short flat repair attempts
were run to target the remaining continuous-omni blockers:

- lateral commands still move mostly forward;
- yaw-only commands turn but translate forward;
- `forward_yaw_left` is weak or can fall into the wrong yaw sign;
- the formal `tracking_gate_passed` and `direction_gate_passed` metrics remain
  false.

The deployable ABI was kept fixed throughout:

- observation: 80D deployable sensors and commands;
- action: 12D, `11D residual motor action + 1D learned latent gait gate`;
- actor: `512-256-128`;
- critic: `512-256-128`;
- terrain: flat only.

## Result Summary

V29 remains the current best viewable flat policy. None of V30-V33 beat its
held-out selection score or solved the lateral/yaw-only drift.

| Version | Run | Best step | Selection score | Planar RMSE | Yaw RMSE | Wrong yaw | Straight violations | Stationary violations | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V29 | `worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate` | `260000` | `-1006644.12` | `0.107` | `0.173` | `0` | `3` | `4` | current best |
| V30 | `worm_v6_ppo_flat_random_continuous_tracking_v30_yaw_lateral_repair_from_v29best` | `270000` | `-1007737.07` | `0.108` | `0.181` | `1` | `2` | `4` | rejected |
| V31 | `worm_v6_ppo_flat_random_continuous_tracking_v31_mixed_yaw_prior_from_v29best` | `270000` | `-1009336.90` | `0.111` | `0.216` | `2` | `2` | `4` | rejected |
| V32 | `worm_v6_ppo_flat_random_continuous_tracking_v32_mixed_yaw_sampling_from_v29best` | `280000` | `-1007879.03` | `0.111` | `0.174` | `0` | `3` | `4` | diagnostic only |
| V33 | `worm_v6_ppo_flat_random_continuous_tracking_v33_low_lr_from_v32best` | `310000` | `-1008438.39` | `0.106` | `0.182` | `1` | `3` | `4` | rejected |

## What Changed

### V30

Code intent:

- reward contract bumped to `omni_directional_offaxis_yaw_v17`;
- lateral penalty and yaw-only stationary penalty were increased;
- pure lateral and pure yaw commands were oversampled in `continuous_omni`.

Outcome:

- `straight_violation_count` improved from `3` to `2`;
- `yaw_success_rate` improved numerically, but `forward_yaw_left` became
  wrong-sign;
- yaw-only and lateral forward drift remained.

### V31

Code intent:

- experimental mixed planar+yaw prior blend in the action adapter;
- pure translation and pure yaw priors were blended for mixed `vx+yaw`
  commands.

Outcome:

- worsened both `forward_yaw_left` and `forward_yaw_right`;
- `wrong_yaw_sign_count` increased to `2`;
- this adapter change was reverted and is not the default code path.

### V32

Code intent:

- keep the stable adapter v23;
- reward contract bumped to `omni_directional_offaxis_yaw_v18`;
- add evaluator-targeted mixed-yaw samples to `continuous_omni`, especially
  `forward+yaw_left/right` and `reverse+yaw_left/right`.

Outcome:

- best checkpoint recovered `wrong_yaw_sign_count=0`;
- still worse than V29 on selection score;
- `forward_yaw_left` remained only a weak turn, not a tracked turn;
- yaw-only stationary drift remained at four violations.

### V33

Code intent:

- add `--learning-rate` to `train_v6.py`;
- continue from V32 best with `1e-4` instead of the default `3e-4`.

Outcome:

- lower learning rate did not solve drift;
- `wrong_yaw_sign_count` returned to `1`;
- no acceptance metric passed beyond the previous V29/V32 level.

## Key Case Diagnostics

Current V29 best still has these representative failures:

| Case | Measured response | Status |
| --- | --- | --- |
| `lateral_left` | `vx=0.108`, `vy=0.033`, `yaw=0.078` | correct sign, but forward off-axis dominates |
| `lateral_right` | `vx=0.106`, `vy=-0.030`, `yaw=-0.080` | borderline lateral sign, forward off-axis dominates |
| `yaw_left` | `vx=0.087`, `vy=0.042`, `yaw=0.377` | correct yaw sign, but translates |
| `yaw_right` | `vx=0.103`, `vy=-0.035`, `yaw=-0.421` | correct yaw sign, but translates |
| `forward_yaw_left` | `vx=0.071`, `vy=-0.018`, `yaw=-0.001` | weak turn |

## Current Decision

Do not promote V30, V31, V32, or V33 as the current policy. V29 remains the
current viewable flat candidate and should still be described as a
`six-direction primitive controller / weak omnidirectional prototype`.

The useful code changes retained after this audit are:

- `continuous_omni_mixed_yaw_repair_sampling` in reward contract v18;
- `--learning-rate` in `train_v6.py` for controlled repair fine-tuning.

The mixed planar+yaw action-adapter blend tested in V31 was reverted.

## Next Technical Target

The next real blocker is not yaw sign separation; it is coupling:

1. yaw-only commands need in-place turn reward/actuation that reduces slide
   forward thrust without losing yaw rate;
2. lateral commands need a lateral-specific prior or residual target that does
   not reuse the forward-thrust body mode;
3. model selection should reject checkpoints with
   `yaw_only_mean_planar_speed_m_s > 0.05` and strong lateral forward drift,
   even if yaw sign rate looks good.

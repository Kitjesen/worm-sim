# Worm V6 Deployable Multimodal Progress

Updated on 2026-05-31.

## Current Status

- Latest flat candidate: V29
  `runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate`.
  It keeps the 80D observation and 12D action ABI, uses actor/critic
  `512-256-128`, and runs the V23 speed-adaptive gait gate plus V16 reward
  contract.
- V29 restores full-speed forward/reverse thresholds and keeps yaw signs
  separated, but it is still not a complete continuous omnidirectional
  tracker. Lateral commands have large forward off-axis motion, and yaw-only
  commands translate while turning.
- Latest V29 artifacts are under
  `record/v6/omni_v29_speed_gate_videos`, including six videos, a 3x2
  comparison video, per-segment trajectory CSV/plots, and a 35-command scan.
- Latest HD visual artifacts are under
  `record/current/flat_omni_v29_hd`: six H.264 `1920x1080`
  command videos and 3x2 comparison videos at `3840x1440` and `3840x2160`.
  These videos keep spring-steel strip rendering and the head speed overlay
  enabled.
- Detailed V29 implementation and metrics are in
  `docs/omni_v29_training_log.md`.

- Flat-only directional gate now has a current accepted candidate. The overall
  paper goal is still incomplete because robust flat tests, ablations, deploy
  bundles, hardware logs, and sand/slope transfer are not done.
- The current main line is `src/v6`, not the legacy `src/v3` wrappers.
- Deployable observation contract is implemented: policy observations use
  body-frame `vx/vy/yaw` command, joint encoders, previous action,
  per-segment IMU gravity/gyro, and phase clock.
- Policy action remains 12D: 11 residual motor actions plus one learned latent
  gait gate.
- Forbidden policy observations are audited out: base linear velocity, global
  pose/yaw, and MuJoCo freejoint truth are not policy inputs.
- Actuator contract is explicit and enforced:
  - slide targets: `[-0.05, 0.0] m`
  - yaw targets: `[-1.57, 1.57] rad`
  - peristaltic actuation period: `1.0 s`
- Current contract line:
  - reward contract: `omni_directional_offaxis_yaw_v16`
  - action adapter: `cmaes_tri_anchor_auto_gate_directional_v23`
  - best-selection gate: `omni_tracking_scan_v3`
  - command range: `cmd_vx in [-0.25, 0.25] m/s`,
    `cmd_vy in [-0.15, 0.15] m/s`, `cmd_yaw in [-0.5, 0.5] rad/s`
  - `gait_blend` is learned from the 12th policy action, not passed as a policy
    observation command.

Old PPO artifacts trained under `forward_progress_v3`, low-speed commands, or
older reward/action contracts are stale for paper claims.

## Latest Flat Evidence

The current source model for V12 flat evidence is:

```text
runs/worm_v6_ppo_flat_random_heading_omni_v11_from_hhbest/best_model.zip
```

The V12 directional-gate summary is:

```text
runs/worm_v6_ppo_flat_random_heading_omni_v11_from_hhbest/v12_directional_eval_summary.json
```

It is accepted for flat:

| Metric | Value | Required | Status |
| --- | ---: | ---: | --- |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `planar_success_rate` | `1.00` | `>= 0.85` | pass |
| `yaw_success_rate` | `0.875` | `>= 0.70` | pass |
| `straight_violation_count` | `1` | `<= 1` | pass |

The fixed 6 s artifact set is:

```text
record/v6/omni_v12_heading_prior_artifacts
```

It reaches all six simple fixed-command thresholds:

| Command | Measured response | Required | Status |
| --- | --- | --- | --- |
| forward | `body_vx = 0.1886 m/s` | `> 0.12` | pass |
| reverse | `body_vx = -0.0767 m/s` | `< -0.03` | pass |
| lateral_left | `body_vy = 0.0376 m/s` | `> 0.03` | pass |
| lateral_right | `body_vy = -0.0393 m/s` | `< -0.03` | pass |
| yaw_left | `yaw_rate = 0.0324 rad/s` | `> 0.03` | pass |
| yaw_right | `yaw_rate = -0.0379 rad/s` | `< -0.03` | pass |

The six videos, fixed eval JSON files, trajectory CSVs, trajectory plots, and
3x2 comparison video are under `record/v6/omni_v12_heading_prior_artifacts`.

## Latest Continuous-Command Evidence

The current V13 continuous-command run is:

```text
runs/worm_v6_ppo_flat_random_continuous_omni_v13_from_v12best
```

V13 adds command-magnitude action scaling and the `continuous_omni` curriculum.
It keeps the deployable ABI fixed: 80D observation and 12D policy action.

The restored-default 35-command scan is:

```text
record/v6/omni_v13_command_scan/flat_random_v13_restored_default_command_scan.json
```

| Metric | Value | Interpretation |
| --- | ---: | --- |
| `planar_rmse_m_s` | `0.1628` | Still too high for arbitrary velocity tracking. |
| `yaw_rmse_rad_s` | `0.3189` | Yaw magnitude tracking is weak. |
| `planar_sign_rate` | `0.84` | Some mixed planar commands still have wrong/weak sign. |
| `yaw_sign_rate` | `1.00` | Yaw sign is now consistently separated on the scan. |
| `zero_command_mean_speed_m_s` | `0.000105` | Stop command is effectively stopped in simulation. |

The V13 six-command video set is under:

```text
record/v6/omni_v13_continuous_artifacts
```

The generated comparison video is:

```text
record/v6/omni_v13_continuous_artifacts/gait_comparison_v13_flat_6cmd.mp4
```

The updated steel-strip/head-speed comparison video is:

```text
record/v6/omni_v13_steel_speed_6cmd/gait_comparison_v13_flat_6cmd_steel_speed.mp4
```

This video uses visual-only MuJoCo scene geoms for the spring-steel strips and
overlays average head-frame speed near the head. The steel-strip shape now
shares the same procedural geometry model as the standalone preview, with about
50 mm pre-bend compression plus up to another 50 mm active compression.

An experimental V14 continuous-vector prior blend was tested and rejected as
the default for now. It worsened the 35-command scan after short training
(`planar_rmse_m_s ~= 0.189`, `planar_sign_rate = 0.76`), so V13 remains the
current accepted default action adapter.

## What Was Fixed

- Best-model selection now has an explicit hard gate instead of selecting only
  by mean reward.
- Progress checkpoints can be saved when the selection score improves, even if
  the formal gate has not passed yet.
- Resume compatibility now allows experimental selection-contract changes while
  keeping ABI/reward/action compatibility checks.
- V9 reward evaluates zero-yaw translation in the reset/start body axes and
  adds integrated yaw-drift penalty.
- V10 action adapter raises lateral prior authority and pure right-yaw
  authority while keeping the same deployable 80D observation and 12D action.
- V12 action adapter adds command-conditioned zero-yaw heading trims while
  keeping the same deployable 80D observation and 12D action:
  - zero-yaw forward/reverse receive axial phase offsets;
  - zero-yaw lateral commands receive a small yaw trim;
  - all transforms depend only on command `obs[0:3]` and phase clock.
- V13 action adapter adds command-magnitude activity scaling so zero and
  low-speed commands can reduce motor amplitude without changing the action ABI.
- A `continuous_omni` command curriculum and 35-command scan tool now exist for
  evaluating arbitrary `vx/vy/yaw` command tracking instead of only six
  primitive directions.
- Video recording now seeds the base environment deterministically before
  VecNormalize normalization, so fixed-command artifacts match the requested
  seed more closely.
- The low-overhead video recorder now includes spring-steel visualization and a
  head speed overlay by default. Both can be disabled with recorder flags when
  a faster raw render is needed.

## Remaining Failure

The main blocker is no longer basic signed direction control. The current
limitations are quality and generalization:

- lateral commands still carry large off-axis forward motion;
- yaw-only left/right still translate while turning;
- arbitrary continuous speed tracking is not solved yet; V13 can stop at zero
  command, but it does not accurately match intermediate `vx/vy/yaw` magnitudes;
- model selection still over-emphasizes primitive direction gates; the next
  training gate should include held-out command-scan RMSE/sign/off-axis metrics;
- robust flat tests and fixed-gate ablations are not complete;
- sand/slope training is still paused until the flat V12 result is repeated
  under the formal pipeline.

The model-selection code has now been upgraded to
`omni_tracking_scan_v3`: stop, slow, mixed planar, and half-rate yaw commands
are part of the held-out best-model schedule, and best-model score penalizes
planar velocity RMSE, yaw-rate RMSE, off-axis speed, and zero-command drift.
The first continuation run under this contract is V15:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v15_from_v13best
```

V15 resumed from the V13 best checkpoint at `507680` timesteps and reached
`622368` total timesteps. The best checkpoint selected by
`omni_tracking_scan_v3` is at `537680` timesteps:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v15_from_v13best/best_model.zip
```

The V15 held-out selection result is:

| Metric | Value | Required | Status |
| --- | ---: | ---: | --- |
| `tracking_gate_passed` | `false` | `true` | fail |
| `direction_gate_passed` | `false` | `true` | fail |
| `planar_velocity_rmse_m_s` | `0.0940` | `<= 0.10` | pass |
| `yaw_rate_rmse_rad_s` | `0.2453` | `<= 0.20` | fail |
| `mean_off_axis_speed_m_s` | `0.0381` | `<= 0.08` | pass |
| `zero_command_mean_speed_m_s` | `0.000019` | `<= 0.02` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `straight_violation_count` | `3` | `<= 1` | fail |

The independent 35-command scan for V15 best is:

```text
record/v6/omni_v15_tracking_scan/flat_random_v15_best_command_scan.json
```

| Metric | Value | Interpretation |
| --- | ---: | --- |
| `num_commands` | `35` | Full scan was generated. |
| `planar_rmse_m_s` | `0.1604` | Still too high for continuous tracking. |
| `yaw_rmse_rad_s` | `0.3205` | Yaw magnitude tracking is still weak. |
| `planar_sign_rate` | `0.80` | Mixed planar commands still fail or weaken. |
| `yaw_sign_rate` | `1.00` | Yaw left/right sign remains separated. |
| `zero_command_mean_speed_m_s` | `0.000009` | Stop command remains effectively stopped. |

The V15 final checkpoint was not better than V15 best on the scan
(`planar_rmse_m_s=0.1632`, `yaw_rmse_rad_s=0.3333`), so the current V15
artifact set should reference the best checkpoint rather than the final one.

The V15 fixed 6 s artifact set is:

```text
record/v6/omni_v15_tracking_artifacts
```

It contains six MP4 videos, the 3x2 comparison video, eval JSON files,
per-segment trajectory CSVs, and per-segment trajectory plots. The comparison
video is a non-empty `1920x720`, 10 fps, 60-frame MP4 with visual spring-steel
strips and the head-speed overlay enabled.

The six fixed-command video measurements from the trajectory CSVs are:

| Command | Measured response | Required | Status |
| --- | --- | --- | --- |
| forward | `avg_vx = 0.1744 m/s` | `> 0.12` | pass |
| reverse | `avg_vx = -0.0714 m/s` | `< -0.03` | pass |
| lateral_left | `avg_vy = 0.0331 m/s` | `> 0.03` | pass |
| lateral_right | `avg_vy = -0.0334 m/s` | `< -0.03` | pass |
| yaw_left | `yaw_rate = 0.0303 rad/s` | `> 0.03` | pass |
| yaw_right | `yaw_rate = -0.0428 rad/s` | `< -0.03` | pass |

These six primitive commands still pass the simple threshold check, but the
same videos show the remaining limitation: reverse and lateral commands retain
large forward/lateral coupling, and yaw-only commands are near the minimum
threshold rather than accurately tracking the requested `0.5 rad/s`.

## V16 Yaw-Prior Repair Attempt

V16 keeps the real-robot interface unchanged:

- observation remains 80D;
- policy action remains 12D: 11 residual motor actions plus one learned gait
  gate;
- motor targets remain the same 6 slide plus 5 yaw actuator contract.

The V16 code change only increases yaw-only prior authority and enriches the
`continuous_omni` sampler:

- `YAW_ONLY_PRIOR_SCALE_FLOOR = 0.30`
- `YAW_ONLY_SLIDE_PRIOR_SCALE = 0.80`
- `YAW_ONLY_YAW_PRIOR_SCALE = 5.00`
- `YAW_RIGHT_ONLY_YAW_PRIOR_SCALE = 5.00`
- extra yaw-only and reverse/mixed zero-yaw samples in `continuous_omni`

The V16 run is:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v16_yawprior_from_v15best
```

It resumed from the V15 best checkpoint at `537680` timesteps and reached
`652368` total timesteps. The best checkpoint selected by
`omni_tracking_scan_v3` is at `597680` timesteps:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v16_yawprior_from_v15best/best_model.zip
```

The V16 held-out selection result is:

| Metric | Value | Required | Status |
| --- | ---: | ---: | --- |
| `tracking_gate_passed` | `false` | `true` | fail |
| `direction_gate_passed` | `false` | `true` | fail |
| `planar_velocity_rmse_m_s` | `0.0976` | `<= 0.10` | pass |
| `yaw_rate_rmse_rad_s` | `0.2285` | `<= 0.20` | fail |
| `mean_off_axis_speed_m_s` | `0.0366` | `<= 0.08` | pass |
| `zero_command_mean_speed_m_s` | `0.000021` | `<= 0.02` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `straight_violation_count` | `2` | `<= 1` | fail |

Compared with V15, yaw RMSE improved on the held-out selection set
(`0.2453 -> 0.2285`) and on the independent 35-command scan
(`0.3205 -> 0.2868`), but the flat continuous-tracking goal still does not
pass.

The independent 35-command scan for V16 best is:

```text
record/v6/omni_v16_tracking_scan/flat_random_v16_best_command_scan.json
```

| Metric | V15 best | V16 best | Interpretation |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1604` | `0.1621` | No planar improvement. |
| `yaw_rmse_rad_s` | `0.3205` | `0.2868` | Yaw magnitude improved but still too weak. |
| `planar_sign_rate` | `0.80` | `0.80` | Mixed planar signs are still weak. |
| `yaw_sign_rate` | `1.00` | `1.00` | Yaw sign separation is preserved. |
| `zero_command_mean_speed_m_s` | `0.000009` | `0.000026` | Stop is still effectively stopped. |

The V16 fixed 6 s artifact set is:

```text
record/v6/omni_v16_tracking_artifacts
```

The comparison video is a non-empty `1920x720`, 10 fps, 60-frame MP4 with
visual spring-steel strips and the head-speed overlay enabled.

The six fixed-command video measurements from the trajectory CSVs are:

| Command | Measured response | Required | Status |
| --- | --- | --- | --- |
| forward | `avg_vx = 0.1766 m/s` | `> 0.12` | pass |
| reverse | `avg_vx = -0.0799 m/s` | `< -0.03` | pass |
| lateral_left | `avg_vy = 0.0256 m/s` | `> 0.03` | fail |
| lateral_right | `avg_vy = -0.0430 m/s` | `< -0.03` | pass |
| yaw_left | `yaw_rate = 0.0689 rad/s` | `> 0.03` | pass |
| yaw_right | `yaw_rate = -0.0815 rad/s` | `< -0.03` | pass |

V16 therefore improves yaw visibility and yaw-rate magnitude, but it is not the
accepted controller: lateral-left weakened below the simple fixed threshold,
and yaw-only commands translate forward while turning. The current paper wording
must remain "weak omnidirectional prototype / six-direction primitive
controller", not "continuous body-frame command tracking".

## V29 HD Videos And V30-V34 Repair Attempts

The current best viewable flat candidate is V29:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip
```

HD videos were recorded before the next repair attempts:

```text
record/current/flat_omni_v29_hd
```

This directory contains six `1920x1080`, 25 fps, 6 s direction videos, plus
`gait_comparison_3x2_3840x1440.mp4` and
`gait_comparison_3x2_4k_uhd.mp4`. Visual spring-steel strips and the head-speed
overlay are enabled.

Longer 15 s direction videos were then recorded for easier visual inspection:

```text
record/current/flat_omni_v29_long15
```

This directory contains six `1920x1080`, 25 fps, 15 s direction videos, a
`3840x1440` 3x2 comparison video, and a `3840x2160` UHD comparison video. The
recorder now accumulates yaw over time, so long yaw videos are not affected by
final-angle wraparound.

The HD rollouts confirm that forward/reverse and yaw signs are visible, but
also show the current failure mode:

| Case | `body_vx` m/s | `body_vy` m/s | `yaw_rate` rad/s | Interpretation |
| --- | ---: | ---: | ---: | --- |
| forward | `0.1499` | `0.0222` | `-0.0009` | usable axial motion |
| reverse | `-0.0809` | `0.0394` | `-0.0037` | usable axial motion with drift |
| lateral_left | `0.1044` | `0.0352` | `0.0757` | lateral sign exists, forward off-axis dominates |
| lateral_right | `0.1025` | `-0.0158` | `-0.0470` | right lateral is weak |
| yaw_left | `0.0952` | `0.0396` | `0.3909` | yaw sign strong, but translates |
| yaw_right | `0.0924` | `-0.0365` | `-0.4023` | yaw sign strong, but translates |

Five short flat repair attempts were then run:

| Version | Best step | Selection score | Wrong yaw | Verdict |
| --- | ---: | ---: | ---: | --- |
| V29 | `260000` | `-1006644.12` | `0` | current best |
| V30 | `270000` | `-1007737.07` | `1` | rejected |
| V31 | `270000` | `-1009336.90` | `2` | rejected; mixed planar+yaw action prior reverted |
| V32 | `280000` | `-1007879.03` | `0` | diagnostic only |
| V33 | `310000` | `-1008438.39` | `1` | rejected |
| V34 | `310000` | `-1007782.83` | `1` | rejected; low-LR V29 continuation |

The retained code changes are:

- reward/curriculum contract `omni_directional_offaxis_yaw_v18`, which adds
  targeted mixed-yaw repair sampling;
- `train_v6.py --learning-rate`, so repair fine-tuning can use lower PPO step
  sizes.

V34's 35-command scan reports `planar_rmse_m_s=0.1635`,
`yaw_rmse_rad_s=0.1128`, `planar_sign_rate=0.84`, `yaw_sign_rate=1.00`, and
yaw-only planar drift about `0.1046 m/s`.

None of V30-V34 is accepted as the current policy. The next technical target is
to reduce lateral forward off-axis speed and yaw-only planar drift without
reintroducing yaw-sign errors.

## Viewable Evidence

- `record/v6/omni_v12_heading_prior_artifacts/gait_comparison_v12_flat_6cmd.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/forward.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/reverse.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/lateral_left.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/lateral_right.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/yaw_left.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/yaw_right.mp4`
- `record/v6/omni_v13_continuous_artifacts/gait_comparison_v13_flat_6cmd.mp4`
- `record/v6/omni_v13_steel_speed_6cmd/gait_comparison_v13_flat_6cmd_steel_speed.mp4`
- `record/v6/omni_v15_tracking_artifacts/gait_comparison_v15_flat_6cmd_steel_speed.mp4`
- `record/v6/omni_v15_tracking_artifacts/forward.mp4`
- `record/v6/omni_v15_tracking_artifacts/reverse.mp4`
- `record/v6/omni_v15_tracking_artifacts/lateral_left.mp4`
- `record/v6/omni_v15_tracking_artifacts/lateral_right.mp4`
- `record/v6/omni_v15_tracking_artifacts/yaw_left.mp4`
- `record/v6/omni_v15_tracking_artifacts/yaw_right.mp4`
- `record/v6/omni_v16_tracking_artifacts/gait_comparison_v16_flat_6cmd_steel_speed.mp4`
- `record/v6/omni_v16_tracking_artifacts/forward.mp4`
- `record/v6/omni_v16_tracking_artifacts/reverse.mp4`
- `record/v6/omni_v16_tracking_artifacts/lateral_left.mp4`
- `record/v6/omni_v16_tracking_artifacts/lateral_right.mp4`
- `record/v6/omni_v16_tracking_artifacts/yaw_left.mp4`
- `record/v6/omni_v16_tracking_artifacts/yaw_right.mp4`
- `record/current/flat_omni_v29_hd/gait_comparison_3x2_4k_uhd.mp4`
- `record/current/flat_omni_v29_hd/forward_1080p.mp4`
- `record/current/flat_omni_v29_hd/reverse_1080p.mp4`
- `record/current/flat_omni_v29_hd/lateral_left_1080p.mp4`
- `record/current/flat_omni_v29_hd/lateral_right_1080p.mp4`
- `record/current/flat_omni_v29_hd/yaw_left_1080p.mp4`
- `record/current/flat_omni_v29_hd/yaw_right_1080p.mp4`
- `record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_4k_uhd.mp4`
- `record/current/flat_omni_v29_long15/forward_15s_1080p.mp4`
- `record/current/flat_omni_v29_long15/reverse_15s_1080p.mp4`
- `record/current/flat_omni_v29_long15/lateral_left_15s_1080p.mp4`
- `record/current/flat_omni_v29_long15/lateral_right_15s_1080p.mp4`
- `record/current/flat_omni_v29_long15/yaw_left_15s_1080p.mp4`
- `record/current/flat_omni_v29_long15/yaw_right_15s_1080p.mp4`

Large 4K videos should use Git LFS or external release assets before being
pushed to GitHub.

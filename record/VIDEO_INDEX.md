# Worm Video Index

Date: 2026-05-31

This file is the canonical entry point for current robot videos. Current
paper-facing videos are kept under `record/current/` instead of a code-version
folder such as `record/v6/`.

## Current Videos To Show

Use these videos when demonstrating the current flat policy. They are from the
V29 policy:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip
```

Primary long-duration video directory:

```text
record/current/flat_omni_v29_long15
```

Short HD video directory:

```text
record/current/flat_omni_v29_hd
```

| Purpose | File | Resolution | Duration | Notes |
| --- | --- | ---: | ---: | --- |
| Long six-direction comparison | `record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_4k_uhd.mp4` | `3840x2160` | `15 s` | Best single file to show first |
| Long wide six-direction comparison | `record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_3840x1440.mp4` | `3840x1440` | `15 s` | Less vertical padding |
| Long forward | `record/current/flat_omni_v29_long15/forward_15s_1080p.mp4` | `1920x1080` | `15 s` | Shows long-run drift |
| Long reverse | `record/current/flat_omni_v29_long15/reverse_15s_1080p.mp4` | `1920x1080` | `15 s` | Shows long-run drift |
| Long lateral left | `record/current/flat_omni_v29_long15/lateral_left_15s_1080p.mp4` | `1920x1080` | `15 s` | Correct primitive but high forward off-axis |
| Long lateral right | `record/current/flat_omni_v29_long15/lateral_right_15s_1080p.mp4` | `1920x1080` | `15 s` | Weak/right contaminated by forward motion |
| Long yaw left | `record/current/flat_omni_v29_long15/yaw_left_15s_1080p.mp4` | `1920x1080` | `15 s` | Correct yaw sign, translates |
| Long yaw right | `record/current/flat_omni_v29_long15/yaw_right_15s_1080p.mp4` | `1920x1080` | `15 s` | Correct yaw sign, translates |

The long-duration directory also contains:

- `*_15s_1080p.json`: rollout metrics for each video;
- `*_15s_trajectory.csv`: head and segment trajectories;
- `*_15s_trajectory.png`: trajectory/time plots;
- `gait_comparison_*_mid.png`: thumbnail frames for quick preview.

The shorter 6 s HD set remains available in `record/current/flat_omni_v29_hd/`.

## Latest Diagnostic Videos

V35 is not the accepted policy, but it is the latest long-duration diagnostic
after fixing the yaw-only prior scaling path:

```text
record/current/flat_omni_v35_axis_sep_long20
```

| Purpose | File | Resolution | Duration | Notes |
| --- | --- | ---: | ---: | --- |
| V35 six-direction comparison | `record/current/flat_omni_v35_axis_sep_long20/gait_comparison_3x2_20s_4k_uhd.mp4` | `3840x2160` | `20 s` | Shows the latest diagnostic behavior |
| V35 wide six-direction comparison | `record/current/flat_omni_v35_axis_sep_long20/gait_comparison_3x2_20s_3840x1440.mp4` | `3840x1440` | `20 s` | Less vertical padding |
| V35 forward | `record/current/flat_omni_v35_axis_sep_long20/forward_20s_1080p.mp4` | `1920x1080` | `20 s` | Forward remains usable |
| V35 reverse | `record/current/flat_omni_v35_axis_sep_long20/reverse_20s_1080p.mp4` | `1920x1080` | `20 s` | Reverse sign works, drift remains |
| V35 lateral left | `record/current/flat_omni_v35_axis_sep_long20/lateral_left_20s_1080p.mp4` | `1920x1080` | `20 s` | Still contaminated by axial motion |
| V35 lateral right | `record/current/flat_omni_v35_axis_sep_long20/lateral_right_20s_1080p.mp4` | `1920x1080` | `20 s` | Still contaminated by axial motion |
| V35 yaw left | `record/current/flat_omni_v35_axis_sep_long20/yaw_left_20s_1080p.mp4` | `1920x1080` | `20 s` | Yaw sign works, turn rate weak |
| V35 yaw right | `record/current/flat_omni_v35_axis_sep_long20/yaw_right_20s_1080p.mp4` | `1920x1080` | `20 s` | Yaw sign works, turn rate weak |

V37 is the latest long-duration diagnostic after moving dominant lateral
commands back to the worm-centered lateral prior. It is not accepted as a
completed continuous tracker because lateral speed remains below the fixed
thresholds:

```text
record/current/flat_omni_v37_lateral_wormcenter_long30
```

| Purpose | File | Resolution | Duration | Notes |
| --- | --- | ---: | ---: | --- |
| V37 six-direction comparison | `record/current/flat_omni_v37_lateral_wormcenter_long30/gait_comparison_3x2_30s_4k_uhd.mp4` | `3840x2160` | `30 s` | Latest long diagnostic video |
| V37 wide six-direction comparison | `record/current/flat_omni_v37_lateral_wormcenter_long30/gait_comparison_3x2_30s_3840x1440.mp4` | `3840x1440` | `30 s` | Less vertical padding |
| V37 forward | `record/current/flat_omni_v37_lateral_wormcenter_long30/forward_30s_1080p.mp4` | `1920x1080` | `30 s` | Forward works but drifts over long horizon |
| V37 reverse | `record/current/flat_omni_v37_lateral_wormcenter_long30/reverse_30s_1080p.mp4` | `1920x1080` | `30 s` | Reverse sign works, drift remains |
| V37 lateral left | `record/current/flat_omni_v37_lateral_wormcenter_long30/lateral_left_30s_1080p.mp4` | `1920x1080` | `30 s` | Lateral displacement visible, speed still too low |
| V37 lateral right | `record/current/flat_omni_v37_lateral_wormcenter_long30/lateral_right_30s_1080p.mp4` | `1920x1080` | `30 s` | Right lateral remains too weak |
| V37 yaw left | `record/current/flat_omni_v37_lateral_wormcenter_long30/yaw_left_30s_1080p.mp4` | `1920x1080` | `30 s` | Yaw sign works, turn rate weak |
| V37 yaw right | `record/current/flat_omni_v37_lateral_wormcenter_long30/yaw_right_30s_1080p.mp4` | `1920x1080` | `30 s` | Yaw sign works, turn rate weak |

## Current Interpretation

The V29 videos should be described as:

```text
six-direction primitive controller / weak omnidirectional prototype
```

Do not describe them as final continuous `vx/vy/yaw` tracking. The videos show:

- forward and reverse motion are usable;
- left/right yaw signs are separated;
- lateral commands still contain large forward off-axis motion;
- yaw-only commands still translate while turning.

## Historical Diagnostic Video Directories

These are useful for comparison and debugging, but they are not the current
paper-facing policy:

| Directory | Use |
| --- | --- |
| `record/current/flat_omni_v37_lateral_wormcenter_long30` | Latest V37 diagnostic long videos; not accepted policy |
| `record/current/flat_omni_v35_axis_sep_long20` | Latest V35 diagnostic long videos; not accepted policy |
| `record/v6/omni_v29_speed_gate_videos` | Lower-resolution V29 fixed-command videos and 35-command scan artifacts |
| `record/current/flat_omni_v29_hd` | Short 6 s HD V29 videos |
| `record/v6/omni_v16_tracking_artifacts` | Older yaw-prior repair attempt |
| `record/v6/omni_v15_tracking_artifacts` | Older continuous-tracking attempt |
| `record/v6/omni_v13_steel_speed_6cmd` | Earlier steel-strip and speed-overlay smoke videos |
| `record/v6/omni_v12_heading_prior_artifacts` | Earlier accepted six-direction directional-gate candidate |
| `record/v6/videos` | Mixed legacy eval videos; do not use as the primary entry point |

## Related Logs

- `docs/omni_v29_hd_recording_log.md`
- `docs/omni_v29_long15_recording_log.md`
- `docs/omni_v29_training_log.md`
- `docs/omni_v34_training_log.md`
- `docs/omni_v35_v36_axis_separation_log.md`
- `docs/omni_v37_v38_lateral_repair_log.md`
- `docs/omni_v30_v33_repair_log.md`
- `record/v6/paper_results/current_progress.md`

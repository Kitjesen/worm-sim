# Worm Video Index

Date: 2026-06-01

This file is the canonical entry point for current robot videos. Current
paper-facing videos are kept under `record/current/` instead of a code-version
folder such as `record/v6/`.

## Current Videos To Show

Use these videos when demonstrating the current flat policy. They are from the
V71 best checkpoint with the V37 slow-right action adapter:

```text
runs/worm_v6_ppo_flat_random_v71_server_v37_slow_right_prior_lowlr_from_v70best_np2/best_model.zip
```

Primary current video directory:

```text
record/current/flat_omni_v71_server_v37_hd20
```

All videos in this set use visual spring-steel strips and the head-speed
overlay.

| Purpose | File | Resolution | Duration | Notes |
| --- | --- | ---: | ---: | --- |
| V71 six-direction comparison | `record/current/flat_omni_v71_server_v37_hd20/gait_comparison_3x2_20s_3840x1440.mp4` | `3840x1440` | `20 s` | Best single file to show first |
| V71 continuous speed sweep | `record/current/flat_omni_v71_server_v37_hd20/continuous_sweep_24s_1080p.mp4` | `1920x1080` | `24 s` | Shows changing command input |
| V71 forward | `record/current/flat_omni_v71_server_v37_hd20/forward_20s_1080p.mp4` | `1920x1080` | `20 s` | `66.06 mm/s` forward in the manifest |
| V71 reverse | `record/current/flat_omni_v71_server_v37_hd20/reverse_20s_1080p.mp4` | `1920x1080` | `20 s` | `-53.53 mm/s` reverse in the manifest |
| V71 lateral left | `record/current/flat_omni_v71_server_v37_hd20/lateral_left_20s_1080p.mp4` | `1920x1080` | `20 s` | `31.95 mm/s` command-direction speed, mean `body_vy=+0.053 m/s` |
| V71 lateral right | `record/current/flat_omni_v71_server_v37_hd20/lateral_right_20s_1080p.mp4` | `1920x1080` | `20 s` | `37.34 mm/s` command-direction speed, mean `body_vy=-0.118 m/s` |
| V71 yaw left | `record/current/flat_omni_v71_server_v37_hd20/yaw_left_20s_1080p.mp4` | `1920x1080` | `20 s` | `0.047 rad/s`, high gait-gate blend |
| V71 yaw right | `record/current/flat_omni_v71_server_v37_hd20/yaw_right_20s_1080p.mp4` | `1920x1080` | `20 s` | `-0.047 rad/s`, high gait-gate blend |

The current directory also contains:

- `*_metrics.json`: rollout metrics for each video;
- `*_trajectory.csv`: head and segment trajectories;
- `*_trajectory.png`: trajectory/time plots;
- `*_telemetry.csv` and `*_telemetry.png`: commanded/measured velocity,
  gait-gate/blend, prior/residual/action heatmaps;
- `video_manifest.md` and `video_manifest.json`: video inventory and summary
  metrics.

The V71 videos are evidence for the current flat simulation checkpoint only.
They do not prove sand/slope transfer or hardware deployment.

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

The V71 videos should be described as:

```text
flat strict-scan accepted simulation checkpoint with six fixed commands and a continuous command sweep
```

Do not describe them as sand/slope or hardware results. The videos show:

- the current V71 best checkpoint keeps zero wrong planar/yaw signs on the
  strict 35-command flat scan;
- lateral-left/right now have the intended signs in the strict scan and
  telemetry;
- yaw-only commands have the intended signs and high gait-gate blend;
- a continuous changing-command sweep has been recorded, but robustness and
  sand/slope transfer are still open.

## Historical Diagnostic Video Directories

These are useful for comparison and debugging, but they are not the current
paper-facing policy:

| Directory | Use |
| --- | --- |
| `record/current/flat_omni_v71_server_v37_hd20` | Current V71 HD videos with steel strips, speed overlay, trajectories, and telemetry |
| `record/current/flat_omni_v37_lateral_wormcenter_long30` | Latest V37 diagnostic long videos; not accepted policy |
| `record/current/flat_omni_v35_axis_sep_long20` | Latest V35 diagnostic long videos; not accepted policy |
| `record/current/flat_omni_v29_long15` | Earlier V29 long videos |
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
- `docs/omni_v70_v71_server_training_log.md`
- `record/v6/paper_results/current_progress.md`

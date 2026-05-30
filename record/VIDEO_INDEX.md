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
- `docs/omni_v30_v33_repair_log.md`
- `record/v6/paper_results/current_progress.md`

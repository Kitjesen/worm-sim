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

Video directory:

```text
record/current/flat_omni_v29_hd
```

| Purpose | File | Resolution | Duration | Notes |
| --- | --- | ---: | ---: | --- |
| Six-direction comparison | `record/current/flat_omni_v29_hd/gait_comparison_3x2_4k_uhd.mp4` | `3840x2160` | `6 s` | Best single file to show first |
| Wide six-direction comparison | `record/current/flat_omni_v29_hd/gait_comparison_3x2_3840x1440.mp4` | `3840x1440` | `6 s` | Less vertical padding |
| Forward | `record/current/flat_omni_v29_hd/forward_1080p.mp4` | `1920x1080` | `6 s` | Passes forward threshold |
| Reverse | `record/current/flat_omni_v29_hd/reverse_1080p.mp4` | `1920x1080` | `6 s` | Passes reverse threshold |
| Lateral left | `record/current/flat_omni_v29_hd/lateral_left_1080p.mp4` | `1920x1080` | `6 s` | Correct sign, high forward off-axis |
| Lateral right | `record/current/flat_omni_v29_hd/lateral_right_1080p.mp4` | `1920x1080` | `6 s` | Weak/right borderline |
| Yaw left | `record/current/flat_omni_v29_hd/yaw_left_1080p.mp4` | `1920x1080` | `6 s` | Correct yaw sign, translates |
| Yaw right | `record/current/flat_omni_v29_hd/yaw_right_1080p.mp4` | `1920x1080` | `6 s` | Correct yaw sign, translates |

The same directory also contains:

- `*_1080p.json`: rollout metrics for each video;
- `*_trajectory.csv`: head and segment trajectories;
- `*_trajectory.png`: trajectory/time plots;
- `gait_comparison_*_mid.png`: thumbnail frames for quick preview.

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
| `record/v6/omni_v16_tracking_artifacts` | Older yaw-prior repair attempt |
| `record/v6/omni_v15_tracking_artifacts` | Older continuous-tracking attempt |
| `record/v6/omni_v13_steel_speed_6cmd` | Earlier steel-strip and speed-overlay smoke videos |
| `record/v6/omni_v12_heading_prior_artifacts` | Earlier accepted six-direction directional-gate candidate |
| `record/v6/videos` | Mixed legacy eval videos; do not use as the primary entry point |

## Related Logs

- `docs/omni_v29_hd_recording_log.md`
- `docs/omni_v29_training_log.md`
- `docs/omni_v30_v33_repair_log.md`
- `record/v6/paper_results/current_progress.md`

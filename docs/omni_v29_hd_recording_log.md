# Worm V6 V29 HD Direction Recording Log

Date: 2026-05-31

## Purpose

Record the current best flat V29 policy in higher resolution before continuing
the next control fix. The goal is visual evidence, not a new training claim.

Policy:
`runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip`

Recording settings:

- Terrain: `flat`
- Gait mode: `random`
- Duration: `6.0 s`
- Single-command videos: `1920x1080`, `25 fps`, H.264
- Comparison videos: `3840x1440` wide grid and `3840x2160` UHD padded grid
- Visual spring-steel strips: enabled
- Head speed overlay: enabled
- Observation/action ABI unchanged: 80D obs, 12D action

## Recorded Commands

| Case | Command `(vx, vy, yaw)` | Video |
| --- | --- | --- |
| forward | `(0.25, 0.00, 0.00)` | `record/v6/omni_v29_speed_gate_videos_hd1080/forward_1080p.mp4` |
| reverse | `(-0.25, 0.00, 0.00)` | `record/v6/omni_v29_speed_gate_videos_hd1080/reverse_1080p.mp4` |
| lateral_left | `(0.00, 0.15, 0.00)` | `record/v6/omni_v29_speed_gate_videos_hd1080/lateral_left_1080p.mp4` |
| lateral_right | `(0.00, -0.15, 0.00)` | `record/v6/omni_v29_speed_gate_videos_hd1080/lateral_right_1080p.mp4` |
| yaw_left | `(0.00, 0.00, 0.50)` | `record/v6/omni_v29_speed_gate_videos_hd1080/yaw_left_1080p.mp4` |
| yaw_right | `(0.00, 0.00, -0.50)` | `record/v6/omni_v29_speed_gate_videos_hd1080/yaw_right_1080p.mp4` |

Comparison videos:

- `record/v6/omni_v29_speed_gate_videos_hd1080/gait_comparison_3x2_3840x1440.mp4`
- `record/v6/omni_v29_speed_gate_videos_hd1080/gait_comparison_3x2_4k_uhd.mp4`

## Verification

| File | Codec | Resolution | FPS | Duration | Frames |
| --- | --- | ---: | ---: | ---: | ---: |
| `forward_1080p.mp4` | H.264 | `1920x1080` | `25` | `6.0 s` | `150` |
| `reverse_1080p.mp4` | H.264 | `1920x1080` | `25` | `6.0 s` | `150` |
| `lateral_left_1080p.mp4` | H.264 | `1920x1080` | `25` | `6.0 s` | `150` |
| `lateral_right_1080p.mp4` | H.264 | `1920x1080` | `25` | `6.0 s` | `150` |
| `yaw_left_1080p.mp4` | H.264 | `1920x1080` | `25` | `6.0 s` | `150` |
| `yaw_right_1080p.mp4` | H.264 | `1920x1080` | `25` | `6.0 s` | `150` |
| `gait_comparison_3x2_4k_uhd.mp4` | H.264 | `3840x2160` | `25` | `6.0 s` | `150` |

## Measured Response From HD Rollouts

The values below are measured from the same 6 s HD rollouts. `body_vx` is the
initial body forward axis, `body_vy` is the initial lateral axis, and yaw is
root yaw-rate over the rollout.

| Case | `body_vx` m/s | `body_vy` m/s | `yaw_rate` rad/s | Visual status |
| --- | ---: | ---: | ---: | --- |
| forward | `0.1499` | `0.0222` | `-0.0009` | forward works; small lateral drift |
| reverse | `-0.0809` | `0.0394` | `-0.0037` | reverse works; lateral drift remains |
| lateral_left | `0.1044` | `0.0352` | `0.0757` | left sign exists, but forward off-axis dominates |
| lateral_right | `0.1025` | `-0.0158` | `-0.0470` | right sign is weak and below fixed threshold |
| yaw_left | `0.0952` | `0.0396` | `0.3909` | yaw sign strong, but translates forward |
| yaw_right | `0.0924` | `-0.0365` | `-0.4023` | yaw sign strong, but translates forward |

## Interpretation

The HD recordings confirm the current V29 status:

- The robot visibly uses the spring-steel strip rendering and head speed
  overlay.
- Axial forward/reverse primitives are usable.
- Left/right yaw signs are separated.
- Continuous omnidirectional tracking is still not solved.

The next control fix should target:

1. Reduce yaw-only translation while preserving yaw-rate authority.
2. Reduce lateral-command forward off-axis speed.
3. Recover stronger right-lateral `vy < -0.03 m/s`.
4. Keep visible axial peristaltic actuation for slow forward/reverse commands.

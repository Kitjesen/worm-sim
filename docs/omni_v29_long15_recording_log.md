# Worm V6 V29 15 s Direction Recording Log

Date: 2026-05-31

## Purpose

Record each current flat V29 direction for a longer visual inspection window.
This is still a V29 visual baseline, not a new training claim.

Policy:
`runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip`

Artifact directory:
`record/current/flat_omni_v29_long15`

## Recording Settings

- Terrain: `flat`
- Gait mode: `random`
- Duration: `15.0 s`
- Single-command videos: `1920x1080`, `25 fps`, H.264
- Comparison videos: `3840x1440` wide grid and `3840x2160` UHD padded grid
- Visual spring-steel strips: enabled
- Head speed overlay: enabled
- Observation/action ABI unchanged: 80D obs, 12D action

## Recorded Commands

| Case | Command `(vx, vy, yaw)` | Video |
| --- | --- | --- |
| forward | `(0.25, 0.00, 0.00)` | `record/current/flat_omni_v29_long15/forward_15s_1080p.mp4` |
| reverse | `(-0.25, 0.00, 0.00)` | `record/current/flat_omni_v29_long15/reverse_15s_1080p.mp4` |
| lateral_left | `(0.00, 0.15, 0.00)` | `record/current/flat_omni_v29_long15/lateral_left_15s_1080p.mp4` |
| lateral_right | `(0.00, -0.15, 0.00)` | `record/current/flat_omni_v29_long15/lateral_right_15s_1080p.mp4` |
| yaw_left | `(0.00, 0.00, 0.50)` | `record/current/flat_omni_v29_long15/yaw_left_15s_1080p.mp4` |
| yaw_right | `(0.00, 0.00, -0.50)` | `record/current/flat_omni_v29_long15/yaw_right_15s_1080p.mp4` |

Comparison videos:

- `record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_3840x1440.mp4`
- `record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_4k_uhd.mp4`

## Verification

The UHD comparison video was checked with `ffprobe`:

| File | Codec | Resolution | FPS | Duration | Frames |
| --- | --- | ---: | ---: | ---: | ---: |
| `gait_comparison_3x2_15s_4k_uhd.mp4` | H.264 | `3840x2160` | `25` | `15.0 s` | `375` |

Each single-command video records `375` frames at `25 fps` for `15.0 s`.

## Measured Response From 15 s Rollouts

Yaw-rate is now computed from accumulated per-step yaw deltas, so long yaw
videos no longer suffer from final-angle wraparound at `+-pi`.

| Case | forward speed mm/s | lateral drift mm | yaw rate rad/s | Interpretation |
| --- | ---: | ---: | ---: | --- |
| forward | `139.24` | `352.4` | `-0.021` | forward remains usable, drift is visible over long duration |
| reverse | `-67.78` | `1063.5` | `0.036` | reverse sign works, but lateral drift grows |
| lateral_left | `42.88` | `1043.2` | `-0.061` | lateral command still leaks into forward/off-axis motion |
| lateral_right | `55.97` | `890.9` | `0.066` | right lateral remains weak and contaminated by forward motion |
| yaw_left | `46.56` | `167.3` | `0.275` | yaw sign works, but yaw-only still translates |
| yaw_right | `29.87` | `366.7` | `-0.331` | yaw sign works, but yaw-only still translates |

## Conclusion

The longer videos make the same limitation clearer than the 6 s clips:
V29 is a six-direction primitive controller, not a completed continuous
`vx/vy/yaw` tracker. The next control work should reduce lateral off-axis
motion and yaw-only translation before claiming full omnidirectional tracking.

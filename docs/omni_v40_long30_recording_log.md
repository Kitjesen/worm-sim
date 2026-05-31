# V40 30 s Recording Log

Date: 2026-05-31

These videos show the current V40 final policy. V40 is not an accepted
continuous omnidirectional tracker; the videos are diagnostic evidence, not a
paper claim.

Output folder:

```text
record/current/flat_omni_v40_v21_long30
```

Recording settings:

- `1920x1080`, 25 fps, 30 s per command;
- visual spring-steel strips enabled;
- head speed overlay enabled;
- trajectory CSV/PNG generated for every command;
- gait gate and prior/residual/applied action telemetry CSV/PNG generated for
  every command;
- comparison videos generated at `3840x1440` and padded `3840x2160`.

## Video Files

| Case | Video | Trajectory | Telemetry |
| --- | --- | --- | --- |
| forward | `forward_30s_1080p.mp4` | `forward_30s_trajectory.png` | `forward_30s_telemetry.png` |
| reverse | `reverse_30s_1080p.mp4` | `reverse_30s_trajectory.png` | `reverse_30s_telemetry.png` |
| lateral_left | `lateral_left_30s_1080p.mp4` | `lateral_left_30s_trajectory.png` | `lateral_left_30s_telemetry.png` |
| lateral_right | `lateral_right_30s_1080p.mp4` | `lateral_right_30s_trajectory.png` | `lateral_right_30s_telemetry.png` |
| yaw_left | `yaw_left_30s_1080p.mp4` | `yaw_left_30s_trajectory.png` | `yaw_left_30s_telemetry.png` |
| yaw_right | `yaw_right_30s_1080p.mp4` | `yaw_right_30s_trajectory.png` | `yaw_right_30s_telemetry.png` |

Comparison videos:

```text
record/current/flat_omni_v40_v21_long30/gait_comparison_3x2_30s_3840x1440.mp4
record/current/flat_omni_v40_v21_long30/gait_comparison_3x2_30s_4k_uhd.mp4
```

The 4K comparison video verifies as:

```text
3840x2160, 25 fps, 30.0 s, 750 frames
```

## Measured 30 s Rollout Summary

The values below are from each video's JSON sidecar. `speed_mm_s` is the
legacy forward-axis speed field, so lateral videos should be interpreted mainly
from the trajectory and the formal scan.

| Case | forward speed mm/s | lateral drift mm | yaw rate rad/s | mean deployed gait blend |
| --- | ---: | ---: | ---: | ---: |
| forward | `119.32` | `1383.0` | `+0.032` | `0.472` |
| reverse | `-34.01` | `1895.4` | `+0.048` | `0.481` |
| lateral_left | `13.44` | `293.4` | `+0.027` | `0.015` |
| lateral_right | `6.77` | `118.1` | `+0.006` | `0.002` |
| yaw_left | `10.92` | `179.3` | `+0.086` | `0.828` |
| yaw_right | `11.87` | `150.4` | `-0.084` | `0.808` |

Interpretation:

- the learned/deployed gait gate is separating modes:
  - axial commands stay near mixed/axial values around `0.47-0.48`;
  - lateral commands are near the worm-centered lateral target around `0.0`;
  - yaw commands are near serpentine/yaw values around `0.81-0.83`;
- the visual mode separation is present, but lateral translation remains weak;
- the video evidence matches the 35-command scan: V40 is still a weak
  six-direction prototype, not a solved continuous tracker.

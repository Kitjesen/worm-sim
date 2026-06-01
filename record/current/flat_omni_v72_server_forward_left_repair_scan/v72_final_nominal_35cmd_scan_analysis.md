# Worm V6 command scan strict analysis

Verdict: **PASS**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Evaluation condition

- condition: `nominal`
- sensor_noise: `{'encoder_pos_noise_std': 0.0, 'encoder_vel_noise_std': 0.0, 'imu_gravity_noise_std': 0.0, 'imu_gyro_noise_std': 0.0}`
- action_delay_steps: `0`
- action_saturation: `1.0`

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0613` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0192` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0312` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0280` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0314` | `0.0535` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0336` | `0.0792` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0283` | `0.0215` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0730` | `0.0484` | 0 | 2 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0728` | `0.1115` | 0 | 0 | 2 | 0 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5838` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4247` | `1.1034` | `0.0470` | `1.0974` | `8.5479` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0182` | `0.9716` | `0.0470` | `0.9768` | `8.1101` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9011` | `0.6108` | `0.0155` | `0.6164` | `1.0723` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3562` | `1.0144` | `0.0995` | `1.0159` | `8.6636` | `0.2500` | `0.4401` |
| `mixed_vx_yaw` | `0.4439` | `1.1052` | `0.0926` | `1.0990` | `8.6084` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.012`, `-0.096`, `-0.146`) | `0.1302` | `0.0463` | `1.75` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.063`, `-0.042`, `-0.074`) | `0.1229` | `0.0740` | `1.65` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.062`, `0.041`, `0.001`) | `0.1223` | `0.0013` | `1.50` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.033`, `0.079`, `0.134`) | `0.1039` | `0.0339` | `1.11` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.054`, `-0.012`, `0.024`) | `0.0988` | `0.0239` | `0.99` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.048`, `-0.085`, `0.001`) | `0.0981` | `0.0008` | `0.96` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.027`, `0.050`, `0.095`) | `0.0811` | `0.0954` | `0.89` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.048`, `-0.002`, `0.018`) | `0.0893` | `0.0179` | `0.81` |

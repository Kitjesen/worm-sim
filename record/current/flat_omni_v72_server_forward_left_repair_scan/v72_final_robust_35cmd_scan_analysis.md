# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Evaluation condition

- condition: `robust_sensor_delay_sat_v1`
- sensor_noise: `{'encoder_pos_noise_std': 0.01, 'encoder_vel_noise_std': 0.02, 'imu_gravity_noise_std': 0.01, 'imu_gyro_noise_std': 0.01}`
- action_delay_steps: `1`
- action_saturation: `0.9`

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0643` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0199` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0240` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0277` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0298` | `0.0719` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0529` | `0.0620` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0279` | `0.0222` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0744` | `0.0483` | 0 | 1 | 4 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0534` | `0.1103` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5105` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4416` | `1.1275` | `0.0307` | `1.1000` | `8.6612` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0192` | `0.9716` | `0.0279` | `0.9710` | `8.0377` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8539` | `0.6108` | `0.0093` | `0.6106` | `1.1255` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3661` | `1.0228` | `0.0602` | `1.0204` | `8.7213` | `0.2500` | `0.4729` |
| `mixed_vx_yaw` | `0.4765` | `1.1033` | `0.0564` | `1.0989` | `8.6812` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.070`, `0.046`, `0.076`) | `0.1246` | `0.0760` | `1.70` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.038`, `-0.017`, `0.013`) | `0.1111` | `0.0125` | `1.24` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.032`, `0.006`, `0.007`) | `0.1075` | `0.0072` | `1.16` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.022`, `-0.145`, `-0.036`) | `0.1005` | `0.0361` | `1.04` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.053`, `-0.002`, `0.070`) | `0.0471` | `0.1695` | `0.94` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.057`, `-0.004`, `-0.054`) | `0.0906` | `0.0538` | `0.89` |
| `mixed_vx_vy` | (`-0.100`, `0.037`, `0.000`) | (`-0.054`, `-0.024`, `-0.082`) | `0.0769` | `0.0819` | `0.76` |
| `pure_vy` | (`0.000`, `-0.075`, `0.000`) | (`-0.017`, `-0.157`, `-0.027`) | `0.0834` | `0.0271` | `0.71` |

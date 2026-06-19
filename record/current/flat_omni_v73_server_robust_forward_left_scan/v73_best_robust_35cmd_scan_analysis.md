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
| `planar_rmse_m_s` | `0.0629` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0208` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0268` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0288` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0303` | `0.0613` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0415` | `0.0523` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0290` | `0.0233` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0743` | `0.0620` | 0 | 1 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0580` | `0.1116` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5260` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4334` | `1.1224` | `0.0339` | `1.0922` | `8.7369` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0173` | `0.9716` | `0.0309` | `0.9710` | `8.0562` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8652` | `0.6108` | `0.0104` | `0.6112` | `1.1285` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3638` | `1.0227` | `0.0681` | `1.0194` | `8.6877` | `0.2500` | `0.4495` |
| `mixed_vx_yaw` | `0.4679` | `1.1046` | `0.0632` | `1.0973` | `8.6555` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.047`, `-0.046`, `-0.113`) | `0.1321` | `0.1134` | `2.07` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.035`, `0.014`, `0.021`) | `0.1044` | `0.0208` | `1.10` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.033`, `-0.007`, `-0.031`) | `0.0674` | `0.1315` | `0.89` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.044`, `-0.004`, `0.041`) | `0.0907` | `0.0410` | `0.87` |
| `mixed_vx_vy` | (`-0.100`, `0.037`, `0.000`) | (`-0.054`, `-0.034`, `-0.074`) | `0.0846` | `0.0742` | `0.85` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.033`, `-0.097`, `-0.006`) | `0.0862` | `0.0055` | `0.74` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.033`, `0.046`, `0.102`) | `0.0813` | `0.0020` | `0.66` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.037`, `0.011`, `0.058`) | `0.0173` | `0.1575` | `0.65` |

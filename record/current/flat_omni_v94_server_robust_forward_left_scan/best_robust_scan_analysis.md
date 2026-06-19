# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Evaluation condition

- condition: `robust`
- sensor_noise: `{'encoder_pos_noise_std': 0.01, 'encoder_vel_noise_std': 0.02, 'imu_gravity_noise_std': 0.01, 'imu_gyro_noise_std': 0.01}`
- action_delay_steps: `1`
- action_saturation: `0.9`

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0570` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0219` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0288` | `0.0472` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0337` | `0.0712` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0224` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0677` | `0.0462` | 0 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0486` | `0.1106` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5653` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4287` | `1.1111` | `0.0421` | `1.0794` | `8.6092` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0169` | `0.9716` | `0.0391` | `0.9717` | `8.0781` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8884` | `0.6108` | `0.0127` | `0.6128` | `1.0882` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3603` | `1.0193` | `0.0843` | `1.0154` | `8.6579` | `0.2500` | `0.4378` |
| `mixed_vx_yaw` | `0.4583` | `1.1071` | `0.0788` | `1.0957` | `8.5609` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.034`, `0.006`, `0.011`) | `0.1091` | `0.0113` | `1.19` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.040`, `0.007`, `0.039`) | `0.1015` | `0.0394` | `1.07` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.035`, `-0.116`, `-0.036`) | `0.0947` | `0.0363` | `0.93` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.053`, `0.009`, `0.090`) | `0.0804` | `0.0902` | `0.85` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.039`, `0.006`, `0.033`) | `0.0610` | `0.1329` | `0.81` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.060`, `-0.000`, `-0.044`) | `0.0853` | `0.0443` | `0.78` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.055`, `-0.018`, `-0.069`) | `0.0725` | `0.0685` | `0.64` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `0.100`) | (`0.039`, `-0.019`, `-0.051`) | `0.0223` | `0.1512` | `0.62` |

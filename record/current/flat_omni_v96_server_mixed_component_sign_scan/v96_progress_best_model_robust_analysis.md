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
| `planar_rmse_m_s` | `0.0595` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0237` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `8` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0285` | `0.0341` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0457` | `0.0629` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0225` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0693` | `0.0585` | 1 | 1 | 2 | 0 | 1 | 8 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5674` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4301` | `1.1123` | `0.0425` | `1.0802` | `8.5543` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0169` | `0.9716` | `0.0393` | `0.9719` | `8.0856` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8900` | `0.6108` | `0.0129` | `0.6129` | `1.0907` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3601` | `1.0194` | `0.0854` | `1.0154` | `8.6472` | `0.2500` | `0.4438` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.062`, `-0.029`, `-0.073`) | `0.1107` | `0.0729` | `1.36` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.059`, `-0.080`, `0.016`) | `0.1094` | `0.0159` | `1.20` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.035`, `0.028`, `0.047`) | `0.0972` | `0.0473` | `1.00` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.055`, `0.012`, `-0.026`) | `0.0980` | `0.0256` | `0.98` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.051`, `-0.003`, `0.017`) | `0.0866` | `0.0174` | `0.76` |
| `mixed_vx_vy` | (`0.100`, `0.037`, `0.000`) | (`0.078`, `0.027`, `0.159`) | `0.0243` | `0.1586` | `0.69` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.062`, `0.013`, `0.067`) | `0.0723` | `0.0665` | `0.63` |
| `pure_vy` | (`0.000`, `-0.075`, `0.000`) | (`0.002`, `-0.149`, `-0.032`) | `0.0739` | `0.0318` | `0.57` |

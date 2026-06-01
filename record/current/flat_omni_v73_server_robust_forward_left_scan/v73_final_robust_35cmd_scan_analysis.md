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
| `planar_rmse_m_s` | `0.0584` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0213` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0226` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0287` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0418` | `0.0493` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0394` | `0.0637` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0290` | `0.0238` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0671` | `0.0503` | 1 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0532` | `0.1119` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5331` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4321` | `1.1211` | `0.0350` | `1.0902` | `8.6830` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0172` | `0.9716` | `0.0327` | `0.9710` | `8.0440` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8696` | `0.6108` | `0.0108` | `0.6115` | `1.1212` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3634` | `1.0219` | `0.0713` | `1.0182` | `8.7063` | `0.2500` | `0.4501` |
| `mixed_vx_yaw` | `0.4660` | `1.1062` | `0.0653` | `1.0983` | `8.6754` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.054`, `-0.074`, `0.028`) | `0.1036` | `0.0284` | `1.09` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.032`, `0.013`, `0.032`) | `0.1026` | `0.0319` | `1.08` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.047`, `0.032`, `0.086`) | `0.0321` | `0.1864` | `0.97` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.060`, `-0.002`, `-0.048`) | `0.0873` | `0.0476` | `0.82` |
| `mixed_vx_vy` | (`-0.050`, `0.075`, `0.000`) | (`-0.033`, `-0.013`, `0.006`) | `0.0893` | `0.0062` | `0.80` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.042`, `0.009`, `0.024`) | `0.0590` | `0.1245` | `0.74` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.064`, `0.020`, `0.100`) | `0.0660` | `0.1005` | `0.69` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.049`, `0.031`, `0.002`) | `0.0592` | `0.0982` | `0.59` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

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
| `planar_rmse_m_s` | `0.0741` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0219` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0160` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `7` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0302` | `0.0174` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0337` | `0.0778` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0245` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0898` | `0.0395` | 1 | 0 | 5 | 0 | 0 | 7 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5946` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4265` | `1.1009` | `0.0482` | `1.0938` | `8.4809` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0203` | `0.9716` | `0.0465` | `0.9766` | `8.0787` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9102` | `0.6108` | `0.0156` | `0.6164` | `1.0735` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.2312` | `0.6181` | `0.1341` | `0.6494` | `6.9426` | `0.2500` | `0.3908` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.015`, `0.016`, `0.048`) | `0.1245` | `0.0478` | `1.61` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.007`, `-0.003`, `-0.011`) | `0.1210` | `0.0105` | `1.47` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.063`, `-0.042`, `0.014`) | `0.1172` | `0.0142` | `1.38` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.005`, `-0.007`, `-0.021`) | `0.1169` | `0.0205` | `1.38` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.021`, `0.010`, `0.062`) | `0.1022` | `0.0616` | `1.14` |
| `mixed_vx_vy` | (`0.100`, `0.037`, `0.000`) | (`0.016`, `-0.008`, `0.062`) | `0.0958` | `0.0618` | `1.01` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.028`, `0.016`, `0.037`) | `0.0978` | `0.0373` | `0.99` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.015`, `-0.002`, `-0.007`) | `0.0925` | `0.0074` | `0.86` |

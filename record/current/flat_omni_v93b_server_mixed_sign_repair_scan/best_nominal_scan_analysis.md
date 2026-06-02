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
| `planar_rmse_m_s` | `0.0541` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0217` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0308` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0288` | `0.0132` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0278` | `0.0812` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0242` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0645` | `0.0592` | 1 | 1 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0745` | `0.0877` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5943` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4264` | `1.1008` | `0.0486` | `1.0936` | `8.5378` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0204` | `0.9716` | `0.0463` | `0.9766` | `8.0703` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9100` | `0.6108` | `0.0156` | `0.6164` | `1.0752` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3594` | `1.0121` | `0.0990` | `1.0122` | `8.6401` | `0.2500` | `0.4309` |
| `mixed_vx_yaw` | `0.4442` | `1.1023` | `0.0950` | `1.0944` | `8.5401` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.017`, `-0.097`, `-0.139`) | `0.1276` | `0.0394` | `1.67` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.076`, `0.032`, `0.129`) | `0.1100` | `0.1292` | `1.63` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.029`, `0.089`, `0.150`) | `0.1134` | `0.0497` | `1.35` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.051`, `-0.081`, `0.013`) | `0.1015` | `0.0127` | `1.03` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.025`, `0.058`, `0.107`) | `0.0773` | `0.1069` | `0.88` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.057`, `-0.002`, `0.046`) | `0.0429` | `0.1456` | `0.71` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.061`, `-0.010`, `-0.040`) | `0.0763` | `0.0403` | `0.62` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.066`, `0.014`, `0.065`) | `0.0697` | `0.0654` | `0.59` |

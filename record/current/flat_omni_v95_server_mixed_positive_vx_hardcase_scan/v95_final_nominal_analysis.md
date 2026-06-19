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
| `planar_rmse_m_s` | `0.0572` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0218` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0333` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `11` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0216` | `0.0477` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0275` | `0.0852` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0243` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0693` | `0.0589` | 0 | 0 | 2 | 0 | 0 | 11 |
| `mixed_vx_yaw` | 6 | `0.0785` | `0.1121` | 0 | 1 | 2 | 0 | 2 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5943` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4254` | `1.1010` | `0.0484` | `1.0939` | `8.4878` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0206` | `0.9716` | `0.0464` | `0.9768` | `8.1305` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9099` | `0.6108` | `0.0156` | `0.6164` | `1.0762` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3582` | `1.0121` | `0.0994` | `1.0121` | `8.6276` | `0.2500` | `0.4263` |
| `mixed_vx_yaw` | `0.4444` | `1.1022` | `0.0948` | `1.0941` | `8.5460` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.017`, `0.101`, `0.170`) | `0.1308` | `0.0701` | `1.83` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.015`, `-0.094`, `-0.142`) | `0.1270` | `0.0418` | `1.66` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.055`, `0.016`, `0.062`) | `0.1013` | `0.0622` | `1.12` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.066`, `0.024`, `0.003`) | `0.1049` | `0.0034` | `1.10` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.042`, `-0.082`, `-0.002`) | `0.0919` | `0.0018` | `0.84` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.069`, `0.007`, `0.070`) | `0.0316` | `0.1705` | `0.83` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.027`, `0.052`, `0.085`) | `0.0802` | `0.0851` | `0.82` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.066`, `0.003`, `0.074`) | `0.0801` | `0.0743` | `0.78` |

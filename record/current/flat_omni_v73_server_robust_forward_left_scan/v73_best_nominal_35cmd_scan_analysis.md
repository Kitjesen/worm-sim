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
| `planar_rmse_m_s` | `0.0605` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0206` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0322` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0288` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0258` | `0.0393` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0351` | `0.0787` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0290` | `0.0230` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0724` | `0.0484` | 1 | 1 | 3 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0733` | `0.1088` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5889` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4258` | `1.1026` | `0.0474` | `1.0963` | `8.5150` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0193` | `0.9716` | `0.0472` | `0.9768` | `8.1124` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9055` | `0.6108` | `0.0155` | `0.6164` | `1.0742` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3582` | `1.0141` | `0.0997` | `1.0157` | `8.6202` | `0.2500` | `0.4376` |
| `mixed_vx_yaw` | `0.4456` | `1.1053` | `0.0931` | `1.0987` | `8.5781` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.013`, `-0.100`, `-0.141`) | `0.1325` | `0.0409` | `1.80` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.061`, `0.046`, `0.022`) | `0.1276` | `0.0224` | `1.64` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.034`, `0.086`, `0.137`) | `0.1080` | `0.0373` | `1.20` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.052`, `-0.022`, `-0.007`) | `0.1087` | `0.0066` | `1.18` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.052`, `-0.087`, `0.011`) | `0.1030` | `0.0114` | `1.06` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.032`, `0.048`, `0.090`) | `0.0863` | `0.0899` | `0.95` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.060`, `0.050`, `0.020`) | `0.0957` | `0.0202` | `0.93` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.064`, `-0.006`, `0.061`) | `0.0364` | `0.1608` | `0.78` |

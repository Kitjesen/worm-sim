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
| `planar_rmse_m_s` | `0.0609` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0189` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0318` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0286` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0323` | `0.0298` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0399` | `0.0731` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0211` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0717` | `0.0563` | 0 | 1 | 2 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.0752` | `0.1204` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5921` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4264` | `1.1039` | `0.0478` | `1.0968` | `8.5102` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0201` | `0.9716` | `0.0461` | `0.9765` | `8.0816` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9096` | `0.6108` | `0.0155` | `0.6164` | `1.0754` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3855` | `1.0129` | `0.0992` | `1.0135` | `8.6098` | `0.2500` | `0.4394` |
| `mixed_vx_yaw` | `0.4465` | `1.1039` | `0.0941` | `1.0961` | `8.5702` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.059`, `0.063`, `0.040`) | `0.1440` | `0.0395` | `2.11` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.021`, `0.094`, `0.164`) | `0.1223` | `0.0636` | `1.60` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.021`, `-0.086`, `-0.128`) | `0.1168` | `0.0280` | `1.38` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.058`, `0.022`, `0.088`) | `0.1057` | `0.0881` | `1.31` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.050`, `-0.008`, `0.064`) | `0.0970` | `0.0637` | `1.04` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.047`, `-0.095`, `-0.006`) | `0.0991` | `0.0056` | `0.98` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.062`, `0.006`, `0.074`) | `0.0386` | `0.1740` | `0.91` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.025`, `0.056`, `0.106`) | `0.0773` | `0.1057` | `0.88` |

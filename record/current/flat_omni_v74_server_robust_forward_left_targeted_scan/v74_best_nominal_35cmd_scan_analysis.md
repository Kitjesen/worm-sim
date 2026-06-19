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
| `planar_rmse_m_s` | `0.0553` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0192` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0324` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0279` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0004` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0301` | `0.0435` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0344` | `0.0760` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0282` | `0.0215` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0652` | `0.0536` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.0839` | `0.1022` | 0 | 2 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5907` | `0.0000` | `0.0000` | `0.0000` | `0.0001` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4265` | `1.1020` | `0.0477` | `1.0953` | `8.5474` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0194` | `0.9716` | `0.0469` | `0.9768` | `8.0647` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9072` | `0.6108` | `0.0156` | `0.6164` | `1.0701` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3585` | `1.0143` | `0.0994` | `1.0153` | `8.6467` | `0.2500` | `0.4501` |
| `mixed_vx_yaw` | `0.4467` | `1.1056` | `0.0934` | `1.0984` | `8.5633` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.001`, `-0.111`, `-0.165`) | `0.1489` | `0.0651` | `2.32` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.023`, `0.102`, `0.159`) | `0.1280` | `0.0590` | `1.72` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.062`, `0.014`, `-0.012`) | `0.0971` | `0.0124` | `0.95` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.042`, `-0.060`, `0.022`) | `0.0936` | `0.0216` | `0.89` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.024`, `0.064`, `0.108`) | `0.0744` | `0.1077` | `0.84` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.050`, `-0.008`, `0.032`) | `0.0841` | `0.0322` | `0.73` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.055`, `0.013`, `0.100`) | `0.0674` | `0.1000` | `0.70` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.059`, `-0.013`, `0.044`) | `0.0428` | `0.1443` | `0.70` |

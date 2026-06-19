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
| `planar_rmse_m_s` | `0.0569` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0200` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0245` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0302` | `0.0221` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0410` | `0.0516` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0224` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0664` | `0.0429` | 0 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0570` | `0.1175` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5654` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4300` | `1.1125` | `0.0418` | `1.0805` | `8.5320` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0165` | `0.9716` | `0.0387` | `0.9718` | `8.1261` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8885` | `0.6108` | `0.0128` | `0.6128` | `1.0886` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3607` | `1.0194` | `0.0843` | `1.0155` | `8.6719` | `0.2500` | `0.4479` |
| `mixed_vx_yaw` | `0.4573` | `1.1073` | `0.0785` | `1.0963` | `8.5901` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.063`, `-0.020`, `-0.068`) | `0.1023` | `0.0684` | `1.16` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.032`, `0.015`, `0.022`) | `0.1012` | `0.0221` | `1.04` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.040`, `0.030`, `0.085`) | `0.0316` | `0.1854` | `0.96` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.050`, `-0.000`, `0.073`) | `0.0902` | `0.0726` | `0.94` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.042`, `-0.089`, `0.010`) | `0.0929` | `0.0097` | `0.87` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.047`, `0.012`, `0.042`) | `0.0542` | `0.1423` | `0.80` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.062`, `0.000`, `-0.048`) | `0.0843` | `0.0478` | `0.77` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.057`, `0.008`, `0.033`) | `0.0793` | `0.0326` | `0.65` |

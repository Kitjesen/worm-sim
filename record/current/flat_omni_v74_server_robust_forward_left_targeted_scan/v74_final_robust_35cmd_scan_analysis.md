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
| `planar_rmse_m_s` | `0.0592` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0202` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0243` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0284` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0004` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0349` | `0.0510` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0464` | `0.0532` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0287` | `0.0226` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0680` | `0.0519` | 0 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0528` | `0.1118` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5528` | `0.0000` | `0.0000` | `0.0000` | `0.0001` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4308` | `1.1155` | `0.0369` | `1.0844` | `8.6141` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0180` | `0.9716` | `0.0347` | `0.9718` | `8.1041` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8751` | `0.6108` | `0.0112` | `0.6118` | `1.1000` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3604` | `1.0228` | `0.0752` | `1.0187` | `8.7084` | `0.2500` | `0.4357` |
| `mixed_vx_yaw` | `0.4652` | `1.1064` | `0.0701` | `1.0975` | `8.6072` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.059`, `-0.017`, `-0.072`) | `0.1010` | `0.0715` | `1.15` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.035`, `0.015`, `0.034`) | `0.1045` | `0.0341` | `1.12` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.037`, `-0.115`, `-0.026`) | `0.0953` | `0.0260` | `0.92` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.044`, `0.001`, `-0.035`) | `0.0560` | `0.1352` | `0.77` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.053`, `-0.003`, `0.025`) | `0.0864` | `0.0251` | `0.76` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.057`, `-0.005`, `-0.061`) | `0.0816` | `0.0612` | `0.76` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.032`, `0.046`, `0.104`) | `0.0825` | `0.0042` | `0.68` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.044`, `0.005`, `0.063`) | `0.0077` | `0.1627` | `0.67` |

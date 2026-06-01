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
| `planar_rmse_m_s` | `0.0632` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0199` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0257` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0277` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0287` | `0.0386` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0494` | `0.0565` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0279` | `0.0222` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0736` | `0.0500` | 0 | 2 | 4 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0587` | `0.1097` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5101` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4427` | `1.1284` | `0.0307` | `1.1014` | `8.5838` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0189` | `0.9716` | `0.0278` | `0.9709` | `8.0664` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8536` | `0.6108` | `0.0093` | `0.6106` | `1.1252` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3662` | `1.0232` | `0.0603` | `1.0209` | `8.7278` | `0.2500` | `0.4590` |
| `mixed_vx_yaw` | `0.4773` | `1.1031` | `0.0565` | `1.0985` | `8.6598` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.048`, `-0.029`, `-0.086`) | `0.1161` | `0.0865` | `1.54` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.057`, `0.030`, `0.056`) | `0.1136` | `0.0558` | `1.37` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.035`, `0.001`, `0.009`) | `0.1127` | `0.0091` | `1.27` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.056`, `0.017`, `-0.043`) | `0.1020` | `0.0430` | `1.09` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.038`, `-0.030`, `-0.051`) | `0.0685` | `0.1509` | `1.04` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.050`, `-0.005`, `0.045`) | `0.0946` | `0.0455` | `0.95` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.030`, `-0.119`, `-0.034`) | `0.0912` | `0.0337` | `0.86` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.029`, `0.050`, `0.115`) | `0.0864` | `0.0147` | `0.75` |

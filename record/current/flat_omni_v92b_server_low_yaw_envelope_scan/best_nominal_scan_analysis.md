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
| `planar_rmse_m_s` | `0.0584` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0191` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0331` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0289` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0333` | `0.0402` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0306` | `0.0832` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0290` | `0.0214` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0694` | `0.0563` | 0 | 1 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0788` | `0.1057` | 0 | 1 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5934` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4267` | `1.1018` | `0.0483` | `1.0948` | `8.4805` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0205` | `0.9716` | `0.0461` | `0.9768` | `8.0911` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9102` | `0.6108` | `0.0155` | `0.6163` | `1.0807` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3864` | `1.0129` | `0.0985` | `1.0134` | `8.6615` | `0.2500` | `0.4399` |
| `mixed_vx_yaw` | `0.4463` | `1.1036` | `0.0943` | `1.0962` | `8.5278` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.014`, `0.104`, `0.179`) | `0.1351` | `0.0792` | `1.98` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.021`, `-0.093`, `-0.129`) | `0.1221` | `0.0289` | `1.51` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.066`, `0.026`, `-0.017`) | `0.1071` | `0.0168` | `1.15` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.065`, `0.015`, `0.073`) | `0.0969` | `0.0725` | `1.07` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.056`, `-0.015`, `0.048`) | `0.1002` | `0.0475` | `1.06` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.016`, `-0.146`, `-0.030`) | `0.0974` | `0.0302` | `0.97` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.021`, `0.060`, `0.113`) | `0.0724` | `0.1128` | `0.84` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.060`, `0.044`, `0.012`) | `0.0911` | `0.0115` | `0.83` |

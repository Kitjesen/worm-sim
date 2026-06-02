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
| `planar_rmse_m_s` | `0.0532` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0192` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0311` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0289` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0282` | `0.0213` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0317` | `0.0822` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0291` | `0.0215` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0631` | `0.0515` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.0729` | `0.1207` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5930` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4267` | `1.1012` | `0.0487` | `1.0941` | `8.5067` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0205` | `0.9716` | `0.0459` | `0.9768` | `8.1037` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9100` | `0.6108` | `0.0155` | `0.6163` | `1.0819` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3863` | `1.0128` | `0.0984` | `1.0130` | `8.6277` | `0.2500` | `0.4394` |
| `mixed_vx_yaw` | `0.4463` | `1.1033` | `0.0940` | `1.0957` | `8.6244` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.028`, `0.097`, `0.147`) | `0.1209` | `0.0471` | `1.52` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.023`, `-0.085`, `-0.133`) | `0.1143` | `0.0327` | `1.33` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.065`, `0.013`, `0.078`) | `0.0378` | `0.1780` | `0.93` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.044`, `-0.061`, `0.022`) | `0.0948` | `0.0224` | `0.91` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.058`, `0.015`, `0.087`) | `0.0170` | `0.1873` | `0.91` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.024`, `0.060`, `0.111`) | `0.0759` | `0.1107` | `0.88` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.052`, `0.002`, `0.037`) | `0.0878` | `0.0374` | `0.81` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.068`, `-0.001`, `-0.036`) | `0.0805` | `0.0361` | `0.68` |

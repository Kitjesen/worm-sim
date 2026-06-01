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
| `planar_rmse_m_s` | `0.0560` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0203` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0194` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0284` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0004` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0280` | `0.0695` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0437` | `0.0448` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0287` | `0.0227` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0651` | `0.0460` | 0 | 0 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0452` | `0.1217` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5502` | `0.0000` | `0.0000` | `0.0000` | `0.0001` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4317` | `1.1180` | `0.0364` | `1.0869` | `8.6050` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0179` | `0.9716` | `0.0340` | `0.9715` | `8.1409` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8734` | `0.6108` | `0.0111` | `0.6117` | `1.1022` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3612` | `1.0222` | `0.0747` | `1.0181` | `8.6996` | `0.2500` | `0.4370` |
| `mixed_vx_yaw` | `0.4658` | `1.1054` | `0.0696` | `1.0970` | `8.6762` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.033`, `0.018`, `0.017`) | `0.1004` | `0.0165` | `1.02` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.050`, `0.006`, `0.063`) | `0.0505` | `0.1627` | `0.92` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.023`, `-0.125`, `-0.016`) | `0.0881` | `0.0164` | `0.78` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.053`, `0.009`, `0.061`) | `0.0816` | `0.0605` | `0.76` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.051`, `-0.012`, `-0.053`) | `0.0799` | `0.0534` | `0.71` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.045`, `-0.001`, `0.066`) | `0.0046` | `0.1662` | `0.69` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.046`, `-0.016`, `-0.040`) | `0.0802` | `0.0402` | `0.68` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.054`, `-0.004`, `-0.018`) | `0.0463` | `0.1182` | `0.56` |

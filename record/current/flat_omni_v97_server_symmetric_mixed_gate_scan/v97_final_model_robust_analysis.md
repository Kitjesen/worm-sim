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
| `planar_rmse_m_s` | `0.0844` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0185` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `3` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `8` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0379` | `0.0460` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0411` | `0.0601` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0224` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.1017` | `0.0303` | 2 | 0 | 9 | 0 | 2 | 8 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5683` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4296` | `1.1114` | `0.0429` | `1.0795` | `8.4854` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0164` | `0.9716` | `0.0397` | `0.9718` | `8.0551` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8905` | `0.6108` | `0.0130` | `0.6129` | `1.0898` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.2239` | `0.6184` | `0.1072` | `0.6385` | `7.6086` | `0.2500` | `0.4334` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.091`, `-0.012`, `0.046`) | `0.1545` | `0.0458` | `2.44` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`-0.001`, `0.005`, `0.002`) | `0.1294` | `0.0023` | `1.67` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.003`, `-0.002`, `-0.017`) | `0.1209` | `0.0171` | `1.47` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.002`, `0.007`, `0.010`) | `0.1197` | `0.0102` | `1.43` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.005`, `0.008`, `0.015`) | `0.1160` | `0.0155` | `1.35` |
| `mixed_vx_vy` | (`-0.050`, `-0.075`, `0.000`) | (`-0.094`, `0.020`, `0.079`) | `0.1043` | `0.0789` | `1.24` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.010`, `0.015`, `0.023`) | `0.1046` | `0.0235` | `1.11` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.018`, `-0.004`, `-0.002`) | `0.1043` | `0.0017` | `1.09` |

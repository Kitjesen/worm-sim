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
| `planar_rmse_m_s` | `0.0592` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0192` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0340` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0279` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0004` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0414` | `0.0494` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0293` | `0.0855` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0282` | `0.0215` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0696` | `0.0491` | 0 | 0 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0686` | `0.1042` | 0 | 0 | 1 | 0 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5916` | `0.0000` | `0.0000` | `0.0000` | `0.0001` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4274` | `1.1022` | `0.0473` | `1.0960` | `8.4962` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0199` | `0.9716` | `0.0469` | `0.9768` | `8.0835` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9077` | `0.6108` | `0.0156` | `0.6164` | `1.0691` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3586` | `1.0141` | `0.0992` | `1.0151` | `8.6649` | `0.2500` | `0.4405` |
| `mixed_vx_yaw` | `0.4459` | `1.1055` | `0.0940` | `1.0978` | `8.5694` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.018`, `-0.095`, `-0.138`) | `0.1253` | `0.0377` | `1.61` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.066`, `0.024`, `-0.009`) | `0.1049` | `0.0088` | `1.10` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.062`, `0.019`, `0.074`) | `0.0422` | `0.1742` | `0.94` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.059`, `0.049`, `0.020`) | `0.0961` | `0.0202` | `0.93` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.075`, `-0.015`, `-0.042`) | `0.0932` | `0.0418` | `0.91` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.039`, `0.071`, `0.125`) | `0.0933` | `0.0247` | `0.89` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.028`, `0.057`, `0.094`) | `0.0805` | `0.0945` | `0.87` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.053`, `-0.002`, `0.040`) | `0.0902` | `0.0403` | `0.85` |

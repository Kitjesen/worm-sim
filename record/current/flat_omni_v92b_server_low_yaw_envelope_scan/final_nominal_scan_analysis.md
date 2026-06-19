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
| `planar_rmse_m_s` | `0.0545` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0184` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0311` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0281` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0277` | `0.0446` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0269` | `0.0833` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0283` | `0.0205` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0653` | `0.0546` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.0867` | `0.1037` | 0 | 2 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5936` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4264` | `1.1011` | `0.0478` | `1.0941` | `8.4788` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0206` | `0.9716` | `0.0464` | `0.9766` | `8.1112` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9095` | `0.6108` | `0.0155` | `0.6163` | `1.0736` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3851` | `1.0117` | `0.0986` | `1.0118` | `8.6090` | `0.2500` | `0.4341` |
| `mixed_vx_yaw` | `0.4448` | `1.1027` | `0.0942` | `1.0948` | `8.5990` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.002`, `0.114`, `0.200`) | `0.1509` | `0.1001` | `2.53` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.010`, `-0.102`, `-0.148`) | `0.1357` | `0.0481` | `1.90` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.067`, `0.018`, `-0.016`) | `0.0988` | `0.0156` | `0.98` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.056`, `0.010`, `0.045`) | `0.0958` | `0.0446` | `0.97` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.028`, `0.052`, `0.102`) | `0.0810` | `0.1016` | `0.91` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.051`, `-0.011`, `0.048`) | `0.0503` | `0.1479` | `0.80` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.056`, `0.009`, `0.070`) | `0.0796` | `0.0697` | `0.76` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.029`, `-0.094`, `-0.015`) | `0.0813` | `0.0148` | `0.67` |

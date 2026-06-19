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
| `planar_rmse_m_s` | `0.0571` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0216` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0318` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0295` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0374` | `0.0363` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0310` | `0.0773` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0242` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0671` | `0.0505` | 1 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0771` | `0.1151` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5946` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4266` | `1.1009` | `0.0484` | `1.0939` | `8.4568` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0206` | `0.9716` | `0.0463` | `0.9767` | `8.0812` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9100` | `0.6108` | `0.0156` | `0.6164` | `1.0741` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3590` | `1.0116` | `0.1000` | `1.0117` | `8.6508` | `0.2500` | `0.4327` |
| `mixed_vx_yaw` | `0.4448` | `1.1032` | `0.0944` | `1.0952` | `8.5744` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.011`, `-0.098`, `-0.149`) | `0.1320` | `0.0494` | `1.80` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.024`, `0.091`, `0.155`) | `0.1189` | `0.0547` | `1.49` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.064`, `0.024`, `0.014`) | `0.1058` | `0.0138` | `1.12` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.051`, `-0.082`, `-0.005`) | `0.1016` | `0.0047` | `1.03` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.057`, `-0.003`, `0.076`) | `0.0433` | `0.1758` | `0.96` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.053`, `0.008`, `0.035`) | `0.0951` | `0.0345` | `0.93` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.025`, `0.060`, `0.103`) | `0.0761` | `0.1032` | `0.85` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.060`, `0.003`, `0.046`) | `0.0825` | `0.0456` | `0.73` |

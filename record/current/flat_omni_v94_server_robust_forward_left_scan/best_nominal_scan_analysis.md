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
| `planar_rmse_m_s` | `0.0547` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0217` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0292` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0251` | `0.0301` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0292` | `0.0747` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0243` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0656` | `0.0491` | 0 | 1 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0738` | `0.1093` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5937` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4263` | `1.1008` | `0.0478` | `1.0937` | `8.5438` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0204` | `0.9716` | `0.0463` | `0.9764` | `8.0760` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9094` | `0.6108` | `0.0155` | `0.6163` | `1.0768` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3585` | `1.0120` | `0.0993` | `1.0118` | `8.6235` | `0.2500` | `0.4584` |
| `mixed_vx_yaw` | `0.4452` | `1.1026` | `0.0942` | `1.0945` | `8.6000` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.024`, `0.098`, `0.158`) | `0.1243` | `0.0580` | `1.63` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.064`, `0.039`, `0.015`) | `0.1193` | `0.0151` | `1.43` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.023`, `-0.084`, `-0.130`) | `0.1139` | `0.0298` | `1.32` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.059`, `0.006`, `0.074`) | `0.0909` | `0.0737` | `0.96` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.054`, `0.021`, `0.089`) | `0.0217` | `0.1895` | `0.94` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.023`, `0.058`, `0.114`) | `0.0749` | `0.1136` | `0.88` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.032`, `-0.118`, `-0.028`) | `0.0929` | `0.0284` | `0.88` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.059`, `0.013`, `0.048`) | `0.0427` | `0.1482` | `0.73` |

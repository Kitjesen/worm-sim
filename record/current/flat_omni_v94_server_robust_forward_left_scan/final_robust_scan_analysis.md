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
| `planar_rmse_m_s` | `0.0553` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0201` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0286` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0308` | `0.0466` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0407` | `0.0675` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0225` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0642` | `0.0525` | 0 | 0 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0512` | `0.1147` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5666` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4305` | `1.1114` | `0.0417` | `1.0799` | `8.5223` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0170` | `0.9716` | `0.0391` | `0.9719` | `8.0947` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8894` | `0.6108` | `0.0129` | `0.6129` | `1.0918` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3602` | `1.0198` | `0.0853` | `1.0156` | `8.6739` | `0.2500` | `0.4529` |
| `mixed_vx_yaw` | `0.4564` | `1.1082` | `0.0795` | `1.0969` | `8.6175` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.034`, `0.016`, `0.033`) | `0.1032` | `0.0333` | `1.09` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.038`, `-0.117`, `-0.008`) | `0.0977` | `0.0078` | `0.96` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.049`, `0.002`, `0.043`) | `0.0925` | `0.0432` | `0.90` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.050`, `0.003`, `0.041`) | `0.0506` | `0.1414` | `0.76` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.037`, `0.010`, `0.069`) | `0.0158` | `0.1687` | `0.74` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.061`, `0.001`, `-0.040`) | `0.0833` | `0.0402` | `0.73` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.049`, `-0.036`, `-0.109`) | `0.0638` | `0.1093` | `0.71` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.048`, `0.021`, `0.059`) | `0.0749` | `0.0590` | `0.65` |

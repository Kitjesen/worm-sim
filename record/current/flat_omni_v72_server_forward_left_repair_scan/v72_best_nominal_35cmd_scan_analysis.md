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
| `planar_rmse_m_s` | `0.0556` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0194` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0280` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0282` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0328` | `0.0247` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0424` | `0.0765` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0285` | `0.0217` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0641` | `0.0475` | 0 | 0 | 0 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0732` | `0.1067` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5825` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4227` | `1.1016` | `0.0477` | `1.0955` | `8.5849` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0180` | `0.9716` | `0.0472` | `0.9764` | `8.0705` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9002` | `0.6108` | `0.0154` | `0.6163` | `1.0709` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3562` | `1.0147` | `0.0996` | `1.0162` | `8.6370` | `0.2500` | `0.4413` |
| `mixed_vx_yaw` | `0.4438` | `1.1047` | `0.0927` | `1.0985` | `8.5930` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.017`, `-0.090`, `-0.130`) | `0.1223` | `0.0296` | `1.52` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.033`, `0.086`, `0.139`) | `0.1089` | `0.0391` | `1.22` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.049`, `-0.007`, `0.052`) | `0.0963` | `0.0525` | `1.00` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.039`, `0.003`, `0.022`) | `0.0988` | `0.0222` | `0.99` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.032`, `0.048`, `0.090`) | `0.0861` | `0.0901` | `0.95` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.061`, `0.014`, `0.071`) | `0.0410` | `0.1709` | `0.90` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.037`, `-0.096`, `-0.014`) | `0.0899` | `0.0140` | `0.81` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.049`, `0.014`, `0.066`) | `0.0727` | `0.0658` | `0.64` |

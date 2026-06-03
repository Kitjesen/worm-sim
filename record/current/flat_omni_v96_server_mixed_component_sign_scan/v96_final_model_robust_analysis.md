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
| `planar_rmse_m_s` | `0.0575` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0200` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0258` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `8` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0335` | `0.0591` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0341` | `0.0707` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0224` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0678` | `0.0484` | 0 | 0 | 1 | 0 | 1 | 8 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5679` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4312` | `1.1121` | `0.0419` | `1.0805` | `8.5336` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0168` | `0.9716` | `0.0395` | `0.9721` | `8.0822` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8904` | `0.6108` | `0.0130` | `0.6129` | `1.0918` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3606` | `1.0194` | `0.0855` | `1.0153` | `8.6566` | `0.2500` | `0.4534` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.042`, `0.013`, `0.021`) | `0.1056` | `0.0207` | `1.13` |
| `mixed_vx_vy` | (`-0.100`, `0.037`, `0.000`) | (`-0.053`, `-0.036`, `-0.102`) | `0.0871` | `0.1017` | `1.02` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.057`, `-0.014`, `0.023`) | `0.0984` | `0.0232` | `0.98` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.031`, `0.030`, `0.054`) | `0.0927` | `0.0540` | `0.93` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.021`, `-0.132`, `-0.042`) | `0.0907` | `0.0420` | `0.87` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.063`, `-0.003`, `-0.045`) | `0.0860` | `0.0455` | `0.79` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.058`, `-0.034`, `-0.088`) | `0.0585` | `0.0884` | `0.54` |
| `pure_vx` | (`0.100`, `0.000`, `0.000`) | (`0.090`, `0.047`, `0.108`) | `0.0477` | `0.1082` | `0.52` |

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
| `planar_rmse_m_s` | `0.0566` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0206` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0318` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0288` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0370` | `0.0216` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0257` | `0.0779` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0290` | `0.0230` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0671` | `0.0485` | 0 | 1 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0748` | `0.1103` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5904` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4262` | `1.1016` | `0.0469` | `1.0955` | `8.5327` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0196` | `0.9716` | `0.0472` | `0.9766` | `8.0881` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9067` | `0.6108` | `0.0156` | `0.6164` | `1.0743` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3590` | `1.0141` | `0.0991` | `1.0157` | `8.6403` | `0.2500` | `0.4481` |
| `mixed_vx_yaw` | `0.4463` | `1.1051` | `0.0926` | `1.0988` | `8.5655` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.022`, `0.098`, `0.160`) | `0.1254` | `0.0600` | `1.66` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.021`, `-0.092`, `-0.129`) | `0.1206` | `0.0294` | `1.48` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.062`, `0.031`, `0.003`) | `0.1126` | `0.0027` | `1.27` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.064`, `0.001`, `0.083`) | `0.0359` | `0.1830` | `0.97` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.051`, `-0.001`, `0.043`) | `0.0903` | `0.0427` | `0.86` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.035`, `-0.110`, `-0.013`) | `0.0923` | `0.0133` | `0.86` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.026`, `0.062`, `0.103`) | `0.0767` | `0.1033` | `0.86` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.056`, `-0.004`, `0.052`) | `0.0830` | `0.0521` | `0.76` |

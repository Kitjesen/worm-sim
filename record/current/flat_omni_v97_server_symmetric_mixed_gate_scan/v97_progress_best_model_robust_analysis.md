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
| `planar_rmse_m_s` | `0.0825` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0135` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `2` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `7` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0296` | `0.0563` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0393` | `0.0578` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0224` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.1001` | `0.0253` | 2 | 0 | 6 | 0 | 1 | 7 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5676` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4299` | `1.1119` | `0.0425` | `1.0799` | `8.5145` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0170` | `0.9716` | `0.0393` | `0.9720` | `8.1172` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8901` | `0.6108` | `0.0130` | `0.6129` | `1.0912` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.2243` | `0.6184` | `0.1064` | `0.6379` | `7.6120` | `0.2500` | `0.4259` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.105`, `0.014`, `0.078`) | `0.1788` | `0.0782` | `3.35` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`-0.003`, `-0.006`, `-0.002`) | `0.1246` | `0.0022` | `1.55` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.005`, `0.001`, `-0.006`) | `0.1212` | `0.0063` | `1.47` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.001`, `0.013`, `0.038`) | `0.1166` | `0.0377` | `1.40` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.005`, `0.008`, `0.011`) | `0.1160` | `0.0110` | `1.35` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.016`, `-0.014`, `-0.018`) | `0.1109` | `0.0183` | `1.24` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.004`, `-0.009`, `-0.013`) | `0.0999` | `0.0134` | `1.00` |
| `mixed_vx_vy` | (`0.100`, `0.037`, `0.000`) | (`0.008`, `0.005`, `0.037`) | `0.0974` | `0.0367` | `0.98` |

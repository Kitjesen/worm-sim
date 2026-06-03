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
| `planar_rmse_m_s` | `0.0616` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0250` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `10` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0343` | `0.0466` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0488` | `0.0562` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0224` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0710` | `0.0504` | 0 | 1 | 2 | 0 | 1 | 10 |
| `mixed_vx_yaw` | 6 | `0.0478` | `0.1190` | 0 | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5666` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4298` | `1.1125` | `0.0420` | `1.0805` | `8.5758` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0166` | `0.9716` | `0.0390` | `0.9718` | `8.0981` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8895` | `0.6108` | `0.0129` | `0.6129` | `1.0882` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3608` | `1.0191` | `0.0853` | `1.0150` | `8.6372` | `0.2500` | `0.4466` |
| `mixed_vx_yaw` | `0.4563` | `1.1061` | `0.0800` | `1.0950` | `8.5701` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.061`, `-0.034`, `-0.088`) | `0.1157` | `0.0882` | `1.53` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.058`, `-0.020`, `0.007`) | `0.1041` | `0.0067` | `1.08` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.021`, `-0.142`, `-0.051`) | `0.0973` | `0.0509` | `1.01` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.033`, `0.024`, `0.046`) | `0.0973` | `0.0464` | `1.00` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.051`, `-0.002`, `0.042`) | `0.0877` | `0.0423` | `0.81` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `0.100`) | (`0.044`, `-0.028`, `-0.066`) | `0.0287` | `0.1664` | `0.77` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.061`, `0.005`, `0.034`) | `0.0397` | `0.1340` | `0.61` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.055`, `-0.014`, `-0.020`) | `0.0475` | `0.1203` | `0.59` |

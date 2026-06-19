# Worm V6 command scan strict analysis

Verdict: **FAIL**

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
| `planar_rmse_m_s` | `0.0595` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0218` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0324` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `9` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0324` | `0.0335` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0300` | `0.0759` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0244` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0710` | `0.0602` | 0 | 1 | 2 | 0 | 1 | 9 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5947` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4261` | `1.1016` | `0.0483` | `1.0942` | `8.4885` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0208` | `0.9716` | `0.0464` | `0.9767` | `8.1101` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9103` | `0.6108` | `0.0156` | `0.6164` | `1.0748` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3590` | `1.0122` | `0.0990` | `1.0125` | `8.6085` | `0.2500` | `0.4325` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.063`, `0.027`, `0.084`) | `0.1087` | `0.0838` | `1.36` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.069`, `0.024`, `-0.002`) | `0.1038` | `0.0023` | `1.08` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.042`, `-0.114`, `-0.018`) | `0.0998` | `0.0177` | `1.00` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.031`, `0.052`, `0.104`) | `0.0841` | `0.1042` | `0.98` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.064`, `0.049`, `0.023`) | `0.0935` | `0.0227` | `0.89` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.064`, `0.004`, `0.086`) | `0.0796` | `0.0859` | `0.82` |
| `mixed_vx_vy` | (`0.100`, `0.037`, `0.000`) | (`0.047`, `-0.023`, `0.039`) | `0.0801` | `0.0389` | `0.68` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.065`, `0.014`, `0.102`) | `0.0625` | `0.1021` | `0.65` |

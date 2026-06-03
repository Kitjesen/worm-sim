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
| `planar_rmse_m_s` | `0.0775` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0218` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0177` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `wrong_mixed_component_sign_count` | `8` | `0` | fail |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed | mixed comp sign fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0455` | `0.0326` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0252` | `0.0809` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0244` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0934` | `0.0427` | 1 | 0 | 4 | 0 | 0 | 8 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5945` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4263` | `1.1005` | `0.0486` | `1.0934` | `8.5475` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0203` | `0.9716` | `0.0468` | `0.9764` | `8.1178` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9100` | `0.6108` | `0.0156` | `0.6164` | `1.0736` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.2309` | `0.6181` | `0.1343` | `0.6495` | `6.9569` | `0.2500` | `0.3925` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.087`, `0.010`, `0.081`) | `0.1610` | `0.0808` | `2.76` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.016`, `0.017`, `0.050`) | `0.1246` | `0.0495` | `1.61` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.007`, `-0.004`, `-0.014`) | `0.1222` | `0.0142` | `1.50` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.005`, `-0.004`, `-0.014`) | `0.1186` | `0.0143` | `1.41` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.023`, `0.014`, `0.067`) | `0.0986` | `0.0668` | `1.08` |
| `mixed_vx_vy` | (`0.100`, `0.037`, `0.000`) | (`0.015`, `-0.009`, `0.049`) | `0.0966` | `0.0493` | `0.99` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.024`, `0.020`, `0.042`) | `0.0920` | `0.0418` | `0.89` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.015`, `0.000`, `-0.002`) | `0.0932` | `0.0023` | `0.87` |

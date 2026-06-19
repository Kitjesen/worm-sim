# Worm V6 command scan strict analysis

Verdict: **PASS**

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
| `planar_rmse_m_s` | `0.0610` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0203` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0264` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0286` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0341` | `0.0279` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0399` | `0.0643` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0289` | `0.0227` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0716` | `0.0515` | 0 | 1 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0490` | `0.1306` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5634` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4296` | `1.1130` | `0.0416` | `1.0811` | `8.5463` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0179` | `0.9716` | `0.0392` | `0.9722` | `8.0958` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8910` | `0.6108` | `0.0128` | `0.6128` | `1.0746` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3873` | `1.0195` | `0.0843` | `1.0155` | `8.6737` | `0.2500` | `0.4459` |
| `mixed_vx_yaw` | `0.4595` | `1.1081` | `0.0797` | `1.0972` | `8.5772` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.057`, `-0.032`, `-0.081`) | `0.1151` | `0.0810` | `1.49` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.042`, `0.007`, `0.089`) | `0.0581` | `0.1894` | `1.24` |
| `mixed_vx_vy` | (`-0.100`, `0.037`, `0.000`) | (`-0.054`, `-0.036`, `-0.095`) | `0.0873` | `0.0954` | `0.99` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `0.100`) | (`0.047`, `-0.025`, `-0.076`) | `0.0252` | `0.1756` | `0.83` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.037`, `-0.103`, `0.005`) | `0.0911` | `0.0051` | `0.83` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.027`, `0.038`, `0.063`) | `0.0852` | `0.0632` | `0.83` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.051`, `0.006`, `0.059`) | `0.0841` | `0.0590` | `0.79` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.046`, `-0.010`, `0.029`) | `0.0844` | `0.0288` | `0.73` |

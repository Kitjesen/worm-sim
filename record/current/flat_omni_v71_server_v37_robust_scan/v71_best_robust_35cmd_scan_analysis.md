# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Evaluation condition

- condition: `robust_sensor_delay_sat_v1`
- sensor_noise: `{'encoder_pos_noise_std': 0.01, 'encoder_vel_noise_std': 0.02, 'imu_gravity_noise_std': 0.01, 'imu_gyro_noise_std': 0.01}`
- action_delay_steps: `1`
- action_saturation: `0.9`

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0582` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0216` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0234` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0003` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0286` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0003` | `0.0013` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0333` | `0.0584` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0402` | `0.0523` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0289` | `0.0241` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0679` | `0.0486` | 0 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0539` | `0.1133` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5133` | `0.0000` | `0.0000` | `0.0000` | `0.0007` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4419` | `1.1277` | `0.0306` | `1.0996` | `8.5550` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0198` | `0.9716` | `0.0279` | `0.9711` | `8.0958` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8571` | `0.6108` | `0.0095` | `0.6103` | `1.1488` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3648` | `1.0254` | `0.0610` | `1.0229` | `8.7088` | `0.2500` | `0.4498` |
| `mixed_vx_yaw` | `0.4748` | `1.1026` | `0.0560` | `1.0975` | `8.6668` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.035`, `-0.003`, `0.007`) | `0.1155` | `0.0067` | `1.33` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.054`, `-0.018`, `-0.076`) | `0.1035` | `0.0758` | `1.21` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.042`, `-0.081`, `0.019`) | `0.0927` | `0.0188` | `0.87` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.077`, `0.039`, `0.080`) | `0.0797` | `0.0803` | `0.80` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `0.100`) | (`0.043`, `-0.015`, `-0.072`) | `0.0168` | `0.1715` | `0.76` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.036`, `-0.014`, `-0.015`) | `0.0652` | `0.1146` | `0.75` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.065`, `-0.001`, `0.036`) | `0.0818` | `0.0359` | `0.70` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.046`, `-0.005`, `0.022`) | `0.0545` | `0.1220` | `0.67` |

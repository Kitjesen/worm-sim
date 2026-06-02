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
| `planar_rmse_m_s` | `0.0618` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0198` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0255` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0291` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0358` | `0.0576` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0437` | `0.0587` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0293` | `0.0222` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0719` | `0.0389` | 1 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0482` | `0.1094` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5592` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4305` | `1.1134` | `0.0412` | `1.0817` | `8.5608` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0175` | `0.9716` | `0.0379` | `0.9721` | `8.0980` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8887` | `0.6108` | `0.0125` | `0.6127` | `1.0986` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3877` | `1.0205` | `0.0824` | `1.0166` | `8.6953` | `0.2500` | `0.4538` |
| `mixed_vx_yaw` | `0.4608` | `1.1082` | `0.0782` | `1.0975` | `8.5713` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.064`, `-0.052`, `0.033`) | `0.1159` | `0.0333` | `1.37` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.036`, `0.014`, `0.021`) | `0.1050` | `0.0205` | `1.11` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.061`, `-0.010`, `-0.069`) | `0.0940` | `0.0687` | `1.00` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.048`, `0.005`, `0.052`) | `0.0952` | `0.0521` | `0.97` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.059`, `0.000`, `-0.058`) | `0.0857` | `0.0578` | `0.82` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.059`, `-0.002`, `0.012`) | `0.0869` | `0.0116` | `0.76` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.051`, `0.006`, `0.040`) | `0.0498` | `0.1395` | `0.73` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.052`, `0.014`, `-0.025`) | `0.0496` | `0.1254` | `0.64` |

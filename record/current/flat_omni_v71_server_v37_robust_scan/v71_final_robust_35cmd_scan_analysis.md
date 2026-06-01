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
| `planar_rmse_m_s` | `0.0591` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0205` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0215` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0282` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0006` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0317` | `0.0270` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0528` | `0.0519` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0285` | `0.0229` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0671` | `0.0379` | 0 | 0 | 1 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.0519` | `0.1090` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5163` | `0.0000` | `0.0000` | `0.0000` | `0.0001` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4406` | `1.1288` | `0.0311` | `1.0997` | `8.6674` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0181` | `0.9716` | `0.0279` | `0.9700` | `8.1096` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8525` | `0.6108` | `0.0093` | `0.6107` | `1.1402` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3651` | `1.0242` | `0.0602` | `1.0215` | `8.6991` | `0.2500` | `0.4670` |
| `mixed_vx_yaw` | `0.4771` | `1.1025` | `0.0562` | `1.0972` | `8.6593` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.035`, `0.014`, `0.018`) | `0.1043` | `0.0180` | `1.10` |
| `mixed_vx_vy` | (`-0.050`, `0.075`, `0.000`) | (`-0.035`, `-0.018`, `-0.017`) | `0.0940` | `0.0170` | `0.89` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.052`, `-0.001`, `0.038`) | `0.0483` | `0.1384` | `0.71` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.031`, `-0.048`, `-0.100`) | `0.0843` | `0.0004` | `0.71` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.031`, `-0.097`, `-0.013`) | `0.0839` | `0.0125` | `0.71` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.033`, `-0.014`, `0.022`) | `0.0687` | `0.0784` | `0.63` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.050`, `-0.014`, `0.008`) | `0.0789` | `0.0077` | `0.62` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.049`, `-0.042`, `-0.100`) | `0.0609` | `0.0997` | `0.62` |

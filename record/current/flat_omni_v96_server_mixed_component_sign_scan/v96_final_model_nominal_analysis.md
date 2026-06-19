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
| `planar_rmse_m_s` | `0.0597` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0217` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0314` | `0.0800` | pass |
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
| `pure_vx` | 4 | `0.0319` | `0.0296` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0273` | `0.0724` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0243` | 0 | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0716` | `0.0614` | 0 | 2 | 2 | 0 | 1 | 9 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5950` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4269` | `1.1038` | `0.0482` | `1.0965` | `8.4571` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0207` | `0.9716` | `0.0462` | `0.9767` | `8.1217` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9105` | `0.6108` | `0.0157` | `0.6164` | `1.0754` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3588` | `1.0120` | `0.0994` | `1.0120` | `8.6222` | `0.2500` | `0.4571` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.047`, `0.060`, `0.039`) | `0.1455` | `0.0392` | `2.15` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.081`, `0.036`, `0.132`) | `0.1129` | `0.1319` | `1.71` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.043`, `-0.108`, `-0.039`) | `0.0985` | `0.0392` | `1.01` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.025`, `0.060`, `0.110`) | `0.0768` | `0.1097` | `0.89` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.064`, `-0.010`, `0.022`) | `0.0917` | `0.0220` | `0.85` |
| `mixed_vx_vy` | (`-0.050`, `0.075`, `0.000`) | (`-0.014`, `0.064`, `0.134`) | `0.0376` | `0.1340` | `0.59` |
| `mixed_vx_vy` | (`0.100`, `0.037`, `0.000`) | (`0.053`, `-0.021`, `0.027`) | `0.0752` | `0.0271` | `0.58` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.068`, `0.017`, `-0.007`) | `0.0664` | `0.0072` | `0.44` |

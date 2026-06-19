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
| `planar_rmse_m_s` | `0.0632` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0230` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0294` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0330` | `0.0273` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0399` | `0.0671` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0296` | `0.0225` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0747` | `0.0555` | 0 | 2 | 3 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0454` | `0.1006` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5610` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4312` | `1.1153` | `0.0414` | `1.0834` | `8.5778` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0182` | `0.9716` | `0.0383` | `0.9724` | `8.1080` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8898` | `0.6108` | `0.0126` | `0.6127` | `1.0943` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3879` | `1.0202` | `0.0829` | `1.0160` | `8.6957` | `0.2500` | `0.4646` |
| `mixed_vx_yaw` | `0.4614` | `1.1085` | `0.0788` | `1.0979` | `8.5973` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.049`, `0.030`, `0.108`) | `0.1169` | `0.1077` | `1.66` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.056`, `-0.028`, `-0.105`) | `0.1123` | `0.1052` | `1.54` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.041`, `0.005`, `0.016`) | `0.1146` | `0.0163` | `1.32` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.062`, `-0.015`, `-0.005`) | `0.0980` | `0.0049` | `0.96` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.048`, `-0.075`, `0.001`) | `0.0980` | `0.0011` | `0.96` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.055`, `0.001`, `-0.039`) | `0.0886` | `0.0390` | `0.82` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.047`, `-0.009`, `0.007`) | `0.0536` | `0.1072` | `0.58` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.047`, `0.013`, `0.042`) | `0.0130` | `0.1418` | `0.52` |

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
| `planar_rmse_m_s` | `0.0592` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0201` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0221` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0286` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0349` | `0.0345` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0389` | `0.0603` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0288` | `0.0225` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0692` | `0.0429` | 0 | 0 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0481` | `0.1062` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5653` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4308` | `1.1129` | `0.0412` | `1.0814` | `8.5687` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0170` | `0.9716` | `0.0390` | `0.9717` | `8.0459` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8885` | `0.6108` | `0.0127` | `0.6128` | `1.0884` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3603` | `1.0192` | `0.0844` | `1.0153` | `8.6636` | `0.2500` | `0.4658` |
| `mixed_vx_yaw` | `0.4576` | `1.1081` | `0.0785` | `1.0967` | `8.6160` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.048`, `-0.016`, `-0.001`) | `0.1052` | `0.0008` | `1.11` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.052`, `-0.010`, `-0.075`) | `0.0971` | `0.0750` | `1.08` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.034`, `0.020`, `0.045`) | `0.1006` | `0.0446` | `1.06` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.046`, `-0.087`, `-0.020`) | `0.0970` | `0.0201` | `0.95` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.055`, `0.006`, `0.055`) | `0.0925` | `0.0548` | `0.93` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.042`, `-0.004`, `0.032`) | `0.0576` | `0.1319` | `0.77` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.061`, `-0.006`, `-0.067`) | `0.0797` | `0.0669` | `0.75` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.041`, `0.016`, `0.060`) | `0.0182` | `0.1600` | `0.67` |

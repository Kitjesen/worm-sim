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
| `planar_rmse_m_s` | `0.0582` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0202` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0213` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0293` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0256` | `0.0251` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0354` | `0.0695` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0295` | `0.0226` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0693` | `0.0476` | 0 | 0 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0462` | `0.1062` | 0 | 0 | 0 | 0 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5626` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4297` | `1.1134` | `0.0420` | `1.0816` | `8.5481` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0178` | `0.9716` | `0.0388` | `0.9723` | `8.0811` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8909` | `0.6108` | `0.0127` | `0.6128` | `1.0949` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3878` | `1.0202` | `0.0837` | `1.0162` | `8.6699` | `0.2500` | `0.4592` |
| `mixed_vx_yaw` | `0.4595` | `1.1072` | `0.0796` | `1.0963` | `8.5019` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.038`, `0.011`, `0.021`) | `0.1093` | `0.0211` | `1.20` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.052`, `0.008`, `0.073`) | `0.0953` | `0.0726` | `1.04` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.070`, `-0.016`, `-0.064`) | `0.0952` | `0.0642` | `1.01` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.043`, `-0.108`, `-0.019`) | `0.0983` | `0.0185` | `0.97` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.058`, `-0.008`, `-0.029`) | `0.0932` | `0.0291` | `0.89` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.062`, `0.007`, `-0.033`) | `0.0904` | `0.0331` | `0.84` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.055`, `0.002`, `-0.026`) | `0.0447` | `0.1258` | `0.60` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `0.100`) | (`0.042`, `-0.002`, `-0.049`) | `0.0085` | `0.1493` | `0.56` |

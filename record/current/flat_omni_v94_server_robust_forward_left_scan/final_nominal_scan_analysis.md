# Worm V6 command scan strict analysis

Verdict: **PASS**

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
| `planar_rmse_m_s` | `0.0534` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0218` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0313` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0296` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0001` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0245` | `0.0359` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0308` | `0.0770` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0299` | `0.0243` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0638` | `0.0601` | 0 | 0 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0840` | `0.1107` | 0 | 1 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5942` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4256` | `1.1005` | `0.0483` | `1.0935` | `8.4840` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0205` | `0.9716` | `0.0463` | `0.9764` | `8.1373` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.9100` | `0.6108` | `0.0156` | `0.6163` | `1.0763` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3593` | `1.0125` | `0.0994` | `1.0124` | `8.6445` | `0.2500` | `0.4540` |
| `mixed_vx_yaw` | `0.4452` | `1.1034` | `0.0941` | `1.0954` | `8.5410` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.000`, `-0.110`, `-0.168`) | `0.1484` | `0.0683` | `2.32` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.021`, `0.098`, `0.158`) | `0.1262` | `0.0576` | `1.67` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.061`, `0.022`, `-0.005`) | `0.1047` | `0.0049` | `1.10` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.062`, `0.010`, `0.076`) | `0.0397` | `0.1762` | `0.93` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.039`, `-0.108`, `-0.035`) | `0.0947` | `0.0354` | `0.93` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.024`, `0.061`, `0.113`) | `0.0755` | `0.1126` | `0.89` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.059`, `0.002`, `0.055`) | `0.0871` | `0.0551` | `0.83` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.052`, `0.028`, `0.097`) | `0.0676` | `0.0975` | `0.69` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1686` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1714` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0296` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0918` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1231` | `0.0374` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0733` | `0.1058` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0923` | `0.1916` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1982` | `0.0576` | 13 | 7 | 16 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1289` | `0.2149` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5501` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4485` | `1.1393` | `0.0415` | `1.1381` | `7.0767` |
| `pure_vy` | `0.0135` | `1.0459` | `0.0395` | `1.0475` | `6.8727` |
| `pure_yaw` | `0.8192` | `2.7146` | `0.0789` | `2.6965` | `1.6007` |
| `mixed_vx_vy` | `0.3889` | `0.7077` | `0.1261` | `0.7254` | `6.2685` |
| `mixed_vx_yaw` | `0.4885` | `0.9792` | `0.0660` | `0.9876` | `6.8042` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.032`, `0.041`, `0.043`) | `0.2898` | `0.0432` | `8.44` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.054`, `0.015`, `0.054`) | `0.2563` | `0.0544` | `6.64` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.030`, `0.054`, `0.058`) | `0.2553` | `0.0581` | `6.60` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.036`, `0.027`, `0.037`) | `0.2468` | `0.0370` | `6.13` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.052`, `0.021`, `0.069`) | `0.2363` | `0.0686` | `5.70` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.036`, `0.024`, `0.016`) | `0.2198` | `0.0159` | `4.84` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.054`, `0.014`, `0.060`) | `0.2154` | `0.0600` | `4.73` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.083`, `-0.025`, `0.011`) | `0.1687` | `0.2612` | `4.55` |

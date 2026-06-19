# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1509` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1697` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0347` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0888` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1196` | `0.0296` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0619` | `0.1017` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0890` | `0.1897` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1761` | `0.0754` | 11 | 4 | 14 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1272` | `0.2358` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5443` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4475` | `1.1404` | `0.0414` | `1.1390` | `7.0760` |
| `pure_vy` | `0.0133` | `1.0459` | `0.0395` | `1.0474` | `6.9835` |
| `pure_yaw` | `0.8233` | `2.7146` | `0.0753` | `2.7012` | `1.5882` |
| `mixed_vx_vy` | `0.3714` | `1.0902` | `0.0879` | `1.0947` | `7.5054` |
| `mixed_vx_yaw` | `0.4871` | `0.9832` | `0.0645` | `0.9923` | `6.8039` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.063`, `0.027`, `-0.029`) | `0.2576` | `0.0290` | `6.66` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.074`, `-0.015`, `-0.081`) | `0.2418` | `0.0807` | `6.01` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.087`, `0.020`, `0.083`) | `0.2352` | `0.0827` | `5.70` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.071`, `0.002`, `0.075`) | `0.2321` | `0.0750` | `5.53` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.096`, `-0.019`, `-0.092`) | `0.1547` | `0.3417` | `5.31` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.090`, `-0.002`, `0.032`) | `0.1597` | `0.2819` | `4.54` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.066`, `0.021`, `-0.027`) | `0.2077` | `0.0270` | `4.33` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.070`, `0.013`, `-0.036`) | `0.1898` | `0.0360` | `3.63` |

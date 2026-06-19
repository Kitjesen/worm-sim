# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1679` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1697` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0261` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0888` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1196` | `0.0296` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0619` | `0.1017` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0890` | `0.1897` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1987` | `0.0558` | 13 | 9 | 16 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.1272` | `0.2358` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5443` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4475` | `1.1404` | `0.0414` | `1.1390` | `7.0760` |
| `pure_vy` | `0.0133` | `1.0459` | `0.0395` | `1.0474` | `6.9835` |
| `pure_yaw` | `0.8233` | `2.7146` | `0.0753` | `2.7012` | `1.5882` |
| `mixed_vx_vy` | `0.3871` | `0.7081` | `0.1257` | `0.7249` | `6.3319` |
| `mixed_vx_yaw` | `0.4871` | `0.9832` | `0.0645` | `0.9923` | `6.8039` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.037`, `0.027`, `0.027`) | `0.2774` | `0.0266` | `7.72` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.033`, `0.045`, `0.045`) | `0.2410` | `0.0449` | `5.86` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.058`, `0.015`, `0.088`) | `0.2352` | `0.0880` | `5.73` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.038`, `0.035`, `0.017`) | `0.2389` | `0.0171` | `5.72` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.053`, `-0.021`, `-0.029`) | `0.2351` | `0.0285` | `5.55` |
| `mixed_vx_vy` | (`0.125`, `-0.150`, `0.000`) | (`-0.061`, `-0.016`, `0.088`) | `0.2294` | `0.0879` | `5.45` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.096`, `-0.019`, `-0.092`) | `0.1547` | `0.3417` | `5.31` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.051`, `0.013`, `0.050`) | `0.2178` | `0.0503` | `4.81` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.2012` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1697` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0397` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0888` | `0.0800` | fail |
| `wrong_planar_sign_count` | `8` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1196` | `0.0296` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0619` | `0.1017` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0890` | `0.1897` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2423` | `0.1022` | 14 | 10 | 16 | 2 | 2 |
| `mixed_vx_yaw` | 6 | `0.1272` | `0.2358` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5443` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4475` | `1.1404` | `0.0414` | `1.1390` | `7.0760` |
| `pure_vy` | `0.0133` | `1.0459` | `0.0395` | `1.0474` | `6.9835` |
| `pure_yaw` | `0.8233` | `2.7146` | `0.0753` | `2.7012` | `1.5882` |
| `mixed_vx_vy` | `0.3549` | `1.0825` | `0.0887` | `1.0873` | `6.2459` |
| `mixed_vx_yaw` | `0.4871` | `0.9832` | `0.0645` | `0.9923` | `6.8039` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`-0.079`, `0.000`, `0.102`) | `0.3377` | `0.1016` | `11.66` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.032`, `-0.030`, `0.006`) | `0.3345` | `0.0065` | `11.19` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.010`, `-0.035`, `-0.006`) | `0.3031` | `0.0059` | `9.19` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`-0.043`, `-0.138`, `-0.039`) | `0.2928` | `0.0393` | `8.61` |
| `mixed_vx_vy` | (`0.125`, `-0.075`, `0.000`) | (`-0.070`, `0.094`, `0.228`) | `0.2574` | `0.2281` | `7.92` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`-0.003`, `-0.041`, `-0.040`) | `0.2782` | `0.0398` | `7.78` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.001`, `-0.006`, `0.011`) | `0.2613` | `0.0112` | `6.83` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.077`, `0.030`, `0.151`) | `0.2497` | `0.1510` | `6.80` |

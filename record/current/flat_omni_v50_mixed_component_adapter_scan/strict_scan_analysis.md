# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1781` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2003` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0523` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0947` | `0.0800` | fail |
| `wrong_planar_sign_count` | `3` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1252` | `0.0265` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0663` | `0.1023` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0950` | `0.2239` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2110` | `0.2023` | 12 | 6 | 14 | 5 | 5 |
| `mixed_vx_yaw` | 6 | `0.1562` | `0.2158` | 4 | 0 | 4 | 2 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5251` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4482` | `1.1386` | `0.0400` | `1.1388` | `7.0536` |
| `pure_vy` | `0.0101` | `1.0459` | `0.0371` | `1.0447` | `6.9858` |
| `pure_yaw` | `0.8216` | `2.7146` | `0.0560` | `2.7070` | `1.5878` |
| `mixed_vx_vy` | `0.3285` | `1.9858` | `0.0448` | `1.9805` | `8.0743` |
| `mixed_vx_yaw` | `0.4612` | `1.7103` | `0.0401` | `1.7154` | `3.1783` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`0.095`, `-0.099`, `-0.390`) | `0.3483` | `0.3895` | `15.93` |
| `mixed_vx_vy` | (`-0.125`, `-0.150`, `0.000`) | (`0.152`, `-0.152`, `-0.391`) | `0.2765` | `0.3912` | `11.47` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.018`, `0.026`, `0.085`) | `0.2953` | `0.0850` | `8.90` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.085`, `0.022`, `-0.136`) | `0.1669` | `0.3864` | `6.52` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`0.005`, `-0.078`, `-0.001`) | `0.2551` | `0.0012` | `6.51` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.037`, `-0.025`, `-0.134`) | `0.2350` | `0.1337` | `5.97` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.018`, `-0.064`, `0.130`) | `0.2320` | `0.1298` | `5.80` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.060`, `0.019`, `0.048`) | `0.2312` | `0.0478` | `5.40` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1498` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2003` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0362` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0947` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1252` | `0.0265` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0663` | `0.1023` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0950` | `0.2239` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1734` | `0.0657` | 11 | 5 | 15 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1268` | `0.2227` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5251` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4482` | `1.1386` | `0.0400` | `1.1388` | `7.0536` |
| `pure_vy` | `0.0101` | `1.0459` | `0.0371` | `1.0447` | `6.9858` |
| `pure_yaw` | `0.8216` | `2.7146` | `0.0560` | `2.7070` | `1.5878` |
| `mixed_vx_vy` | `0.3693` | `1.0899` | `0.0473` | `1.0918` | `7.5898` |
| `mixed_vx_yaw` | `0.4859` | `0.9834` | `0.0382` | `0.9882` | `6.9002` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.068`, `0.020`, `-0.017`) | `0.2485` | `0.0165` | `6.18` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.079`, `0.003`, `0.050`) | `0.2256` | `0.0496` | `5.15` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.079`, `-0.002`, `0.022`) | `0.2257` | `0.0220` | `5.11` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.073`, `0.024`, `-0.025`) | `0.2176` | `0.0246` | `4.75` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.103`, `-0.041`, `0.041`) | `0.1529` | `0.2915` | `4.46` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.091`, `0.008`, `-0.025`) | `0.1594` | `0.2749` | `4.43` |
| `pure_yaw` | (`0.000`, `0.000`, `-0.250`) | (`0.096`, `-0.048`, `-0.593`) | `0.1072` | `0.3433` | `4.10` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.070`, `-0.003`, `-0.066`) | `0.1935` | `0.0665` | `3.86` |

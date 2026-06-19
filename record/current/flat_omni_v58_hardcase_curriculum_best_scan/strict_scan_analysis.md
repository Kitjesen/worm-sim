# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1566` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2709` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0366` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0951` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1294` | `0.0369` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0762` | `0.1086` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0951` | `0.3029` | 0 | 0 | 0 | 4 | 4 |
| `mixed_vx_vy` | 16 | `0.1808` | `0.0694` | 11 | 5 | 14 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.1458` | `0.2750` | 4 | 0 | 4 | 6 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5469` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4493` | `1.1383` | `0.0426` | `1.1377` | `7.8274` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0130` | `1.0459` | `0.0393` | `1.0472` | `7.9786` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8171` | `2.7146` | `0.0776` | `2.6993` | `1.9416` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3723` | `1.0878` | `0.0894` | `1.0933` | `8.1304` | `0.2500` | `0.4150` |
| `mixed_vx_yaw` | `0.4763` | `1.1412` | `0.0896` | `1.1539` | `8.2114` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.059`, `0.054`, `0.015`) | `0.2794` | `0.0153` | `7.81` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.052`, `0.079`, `0.060`) | `0.2512` | `0.0604` | `6.40` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.079`, `0.002`, `0.094`) | `0.1713` | `0.3443` | `5.90` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `-0.250`) | (`0.046`, `-0.000`, `0.066`) | `0.1713` | `0.3157` | `5.43` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.069`, `0.007`, `0.031`) | `0.2305` | `0.0308` | `5.34` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.075`, `-0.001`, `0.019`) | `0.2302` | `0.0193` | `5.31` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.073`, `0.021`, `-0.021`) | `0.2187` | `0.0212` | `4.80` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `0.250`) | (`0.048`, `0.019`, `0.014`) | `0.1737` | `0.2364` | `4.42` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.078`, `0.033`, `0.049`) | `0.1749` | `0.2006` | `4.07` |
| `pure_vx` | (`-0.250`, `0.000`, `0.000`) | (`-0.065`, `0.060`, `0.002`) | `0.1945` | `0.0024` | `3.78` |

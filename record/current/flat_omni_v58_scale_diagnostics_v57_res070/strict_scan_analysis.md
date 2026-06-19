# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1626` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1939` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0322` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0994` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1291` | `0.0414` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0688` | `0.0937` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.1000` | `0.2168` | 1 | 0 | 2 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1896` | `0.0710` | 11 | 6 | 14 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.1385` | `0.2515` | 4 | 0 | 4 | 4 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5463` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4527` | `1.1362` | `0.0826` | `1.1353` | `6.9674` |
| `pure_vy` | `0.0143` | `1.0459` | `0.0795` | `1.0516` | `6.8932` |
| `pure_yaw` | `0.8225` | `2.7146` | `0.1567` | `2.6831` | `1.6303` |
| `mixed_vx_vy` | `0.3765` | `1.0859` | `0.1723` | `1.1019` | `7.3291` |
| `mixed_vx_yaw` | `0.4937` | `0.9788` | `0.1287` | `0.9990` | `6.6824` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.059`, `-0.025`, `-0.071`) | `0.2590` | `0.0714` | `6.83` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.048`, `-0.001`, `0.040`) | `0.2527` | `0.0404` | `6.43` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.065`, `0.003`, `-0.083`) | `0.1847` | `0.3330` | `6.18` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.066`, `0.010`, `0.058`) | `0.2441` | `0.0583` | `6.04` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.054`, `-0.008`, `-0.040`) | `0.2421` | `0.0396` | `5.90` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.072`, `-0.024`, `0.066`) | `0.1800` | `0.3163` | `5.74` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.060`, `0.007`, `0.070`) | `0.2074` | `0.0698` | `4.42` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.057`, `0.005`, `-0.048`) | `0.2085` | `0.0482` | `4.41` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.067`, `-0.018`, `0.078`) | `0.2052` | `0.0785` | `4.36` |
| `mixed_vx_vy` | (`0.125`, `0.150`, `0.000`) | (`-0.025`, `0.014`, `0.077`) | `0.2026` | `0.0771` | `4.25` |

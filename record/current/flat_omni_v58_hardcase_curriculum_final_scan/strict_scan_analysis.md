# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1588` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2720` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0409` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0959` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1278` | `0.0069` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0647` | `0.1025` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0963` | `0.3041` | 0 | 0 | 1 | 4 | 4 |
| `mixed_vx_vy` | 16 | `0.1851` | `0.0880` | 11 | 7 | 14 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.1468` | `0.2621` | 4 | 0 | 4 | 6 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5511` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4502` | `1.1368` | `0.0423` | `1.1361` | `7.7756` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0124` | `1.0459` | `0.0398` | `1.0478` | `7.9346` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8159` | `2.7146` | `0.0771` | `2.7005` | `1.9587` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3733` | `1.0866` | `0.0898` | `1.0923` | `8.1442` | `0.2500` | `0.4274` |
| `mixed_vx_yaw` | `0.4759` | `1.1408` | `0.0899` | `1.1539` | `8.1910` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.074`, `-0.032`, `-0.101`) | `0.2531` | `0.1011` | `6.66` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.066`, `0.028`, `-0.008`) | `0.2557` | `0.0076` | `6.54` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.052`, `0.080`, `0.064`) | `0.2514` | `0.0639` | `6.42` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.083`, `-0.032`, `-0.016`) | `0.2474` | `0.0156` | `6.13` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.081`, `0.018`, `0.074`) | `0.2384` | `0.0739` | `5.82` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `-0.250`) | (`0.049`, `-0.008`, `0.051`) | `0.1737` | `0.3008` | `5.28` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.078`, `-0.032`, `0.023`) | `0.1751` | `0.2729` | `4.93` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `0.250`) | (`0.049`, `0.019`, `0.022`) | `0.1747` | `0.2280` | `4.35` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.080`, `0.039`, `0.028`) | `0.1741` | `0.2217` | `4.26` |
| `pure_yaw` | (`0.000`, `0.000`, `0.250`) | (`0.098`, `0.046`, `0.593`) | `0.1082` | `0.3433` | `4.12` |

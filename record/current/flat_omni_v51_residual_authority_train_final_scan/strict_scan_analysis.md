# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1553` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1907` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0380` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0919` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1285` | `0.0153` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0638` | `0.1220` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0921` | `0.2132` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1804` | `0.0782` | 11 | 6 | 13 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.1341` | `0.2330` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5315` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4479` | `1.1371` | `0.0408` | `1.1365` | `7.1236` |
| `pure_vy` | `0.0110` | `1.0459` | `0.0385` | `1.0458` | `6.9920` |
| `pure_yaw` | `0.8204` | `2.7146` | `0.0717` | `2.7045` | `1.6016` |
| `mixed_vx_vy` | `0.3702` | `1.0903` | `0.0868` | `1.0945` | `7.5585` |
| `mixed_vx_yaw` | `0.4860` | `0.9845` | `0.0615` | `0.9926` | `6.8926` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.056`, `0.073`, `0.036`) | `0.2956` | `0.0362` | `8.77` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.074`, `-0.022`, `0.061`) | `0.1778` | `0.3110` | `5.58` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.088`, `-0.018`, `-0.001`) | `0.2332` | `0.0007` | `5.44` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.057`, `-0.021`, `0.013`) | `0.2325` | `0.0132` | `5.41` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.073`, `0.005`, `-0.047`) | `0.2291` | `0.0472` | `5.31` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.080`, `-0.008`, `-0.030`) | `0.1699` | `0.2800` | `4.85` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.064`, `0.027`, `-0.032`) | `0.2126` | `0.0319` | `4.54` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.071`, `-0.013`, `-0.068`) | `0.1993` | `0.0675` | `4.09` |

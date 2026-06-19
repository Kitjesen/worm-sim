# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1575` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1794` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0403` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0899` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1273` | `0.0180` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0679` | `0.1023` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0899` | `0.2006` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1832` | `0.0832` | 11 | 6 | 14 | 0 | 3 |
| `mixed_vx_yaw` | 6 | `0.1332` | `0.2158` | 4 | 0 | 4 | 4 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5424` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4476` | `1.1381` | `0.0415` | `1.1373` | `7.0339` |
| `pure_vy` | `0.0132` | `1.0459` | `0.0396` | `1.0474` | `6.9330` |
| `pure_yaw` | `0.8221` | `2.7146` | `0.0751` | `2.7019` | `1.5943` |
| `mixed_vx_vy` | `0.3717` | `1.0889` | `0.0878` | `1.0937` | `7.4520` |
| `mixed_vx_yaw` | `0.4875` | `0.9825` | `0.0643` | `0.9921` | `6.8443` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.067`, `0.030`, `0.003`) | `0.2570` | `0.0033` | `6.61` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.055`, `0.083`, `0.065`) | `0.2513` | `0.0646` | `6.42` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.064`, `-0.018`, `-0.000`) | `0.2504` | `0.0001` | `6.27` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.068`, `-0.010`, `-0.052`) | `0.2417` | `0.0515` | `5.91` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.078`, `-0.006`, `0.049`) | `0.1721` | `0.2994` | `5.20` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.077`, `-0.026`, `0.033`) | `0.2129` | `0.0329` | `4.56` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.085`, `0.027`, `0.023`) | `0.1677` | `0.2270` | `4.10` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.065`, `0.008`, `-0.060`) | `0.1965` | `0.0603` | `3.95` |

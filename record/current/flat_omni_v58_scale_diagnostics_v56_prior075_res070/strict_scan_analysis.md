# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1760` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1900` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0283` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.1164` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1520` | `0.0316` | 3 | 0 | 3 | 0 | 0 |
| `pure_vy` | 4 | `0.1129` | `0.1007` | 0 | 2 | 2 | 0 | 1 |
| `pure_yaw` | 4 | `0.1166` | `0.2124` | 2 | 0 | 4 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1986` | `0.0493` | 13 | 8 | 15 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1453` | `0.2604` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5452` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4634` | `0.8566` | `0.0818` | `0.8620` | `6.3465` |
| `pure_vy` | `0.0181` | `0.7844` | `0.0819` | `0.7960` | `5.9992` |
| `pure_yaw` | `0.8393` | `2.0359` | `0.1472` | `2.0647` | `1.5546` |
| `mixed_vx_vy` | `0.3856` | `0.8143` | `0.1712` | `0.8415` | `6.6585` |
| `mixed_vx_yaw` | `0.5055` | `0.7344` | `0.1215` | `0.7595` | `5.9485` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.039`, `-0.014`, `-0.051`) | `0.2672` | `0.0513` | `7.21` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.034`, `0.007`, `0.005`) | `0.2674` | `0.0054` | `7.15` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.049`, `-0.003`, `0.090`) | `0.2010` | `0.3399` | `6.93` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.043`, `0.005`, `0.067`) | `0.2524` | `0.0668` | `6.48` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.042`, `-0.012`, `0.017`) | `0.2496` | `0.0172` | `6.24` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.036`, `0.038`, `0.015`) | `0.2419` | `0.0155` | `5.86` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.048`, `-0.014`, `-0.014`) | `0.2025` | `0.2638` | `5.84` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.040`, `0.005`, `-0.020`) | `0.2216` | `0.0205` | `4.92` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.051`, `-0.002`, `0.060`) | `0.2131` | `0.0595` | `4.63` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.047`, `-0.012`, `0.014`) | `0.2129` | `0.0142` | `4.54` |

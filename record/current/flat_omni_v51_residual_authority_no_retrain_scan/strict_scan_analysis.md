# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1575` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2102` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0402` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0974` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1252` | `0.0265` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0663` | `0.1023` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0980` | `0.2350` | 1 | 0 | 2 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1837` | `0.0706` | 10 | 7 | 14 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1267` | `0.2267` | 4 | 0 | 4 | 4 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5251` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4482` | `1.1386` | `0.0400` | `1.1388` | `7.0536` |
| `pure_vy` | `0.0101` | `1.0459` | `0.0371` | `1.0447` | `6.9858` |
| `pure_yaw` | `0.8216` | `2.7146` | `0.0699` | `2.7051` | `1.5756` |
| `mixed_vx_vy` | `0.3703` | `1.0896` | `0.0854` | `1.0940` | `7.5978` |
| `mixed_vx_yaw` | `0.4861` | `0.9858` | `0.0614` | `0.9937` | `6.8153` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.064`, `-0.049`, `-0.094`) | `0.2722` | `0.0945` | `7.63` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.066`, `0.026`, `-0.012`) | `0.2547` | `0.0116` | `6.49` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.074`, `-0.018`, `-0.002`) | `0.2435` | `0.0019` | `5.93` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.104`, `-0.036`, `-0.075`) | `0.1506` | `0.3245` | `4.90` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.060`, `0.036`, `-0.002`) | `0.2198` | `0.0018` | `4.83` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.071`, `-0.030`, `-0.033`) | `0.2157` | `0.0334` | `4.68` |
| `pure_yaw` | (`0.000`, `0.000`, `-0.250`) | (`0.101`, `-0.048`, `-0.611`) | `0.1118` | `0.3611` | `4.51` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.080`, `-0.027`, `-0.015`) | `0.1981` | `0.0152` | `3.93` |

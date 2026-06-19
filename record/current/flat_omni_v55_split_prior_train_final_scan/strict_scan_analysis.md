# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.2018` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1761` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0407` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0907` | `0.0800` | fail |
| `wrong_planar_sign_count` | `7` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1273` | `0.0268` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0581` | `0.1063` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0909` | `0.1969` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2423` | `0.0922` | 14 | 9 | 16 | 0 | 3 |
| `mixed_vx_yaw` | 6 | `0.1290` | `0.2334` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5520` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4507` | `1.1401` | `0.0415` | `1.1391` | `7.0285` |
| `pure_vy` | `0.0147` | `1.0459` | `0.0404` | `1.0480` | `6.9380` |
| `pure_yaw` | `0.8242` | `2.7146` | `0.0755` | `2.6977` | `1.5926` |
| `mixed_vx_vy` | `0.3575` | `1.0832` | `0.0897` | `1.0880` | `6.2499` |
| `mixed_vx_yaw` | `0.4905` | `0.9798` | `0.0659` | `0.9891` | `6.7922` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.027`, `-0.033`, `0.003`) | `0.3324` | `0.0033` | `11.05` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`-0.072`, `-0.032`, `0.080`) | `0.3252` | `0.0803` | `10.73` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.011`, `-0.037`, `-0.004`) | `0.3037` | `0.0039` | `9.22` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.068`, `0.065`, `0.175`) | `0.2818` | `0.1750` | `8.70` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`-0.030`, `-0.128`, `-0.036`) | `0.2812` | `0.0364` | `7.94` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`-0.002`, `-0.040`, `-0.037`) | `0.2768` | `0.0373` | `7.70` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.002`, `-0.005`, `0.012`) | `0.2610` | `0.0117` | `6.81` |
| `mixed_vx_vy` | (`0.125`, `-0.075`, `0.000`) | (`-0.082`, `0.027`, `0.152`) | `0.2306` | `0.1525` | `5.90` |

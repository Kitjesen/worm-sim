# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1516` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1183` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0356` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0671` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1254` | `0.0188` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0681` | `0.0991` | 0 | 0 | 0 | 0 | 1 |
| `pure_yaw` | 4 | `0.0672` | `0.1323` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.1755` | `0.0807` | 11 | 5 | 13 | 1 | 1 |
| `mixed_vx_yaw` | 6 | `0.1331` | `0.2826` | 4 | 0 | 4 | 5 | 0 |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.056`, `-0.031`, `-0.008`) | `0.2655` | `0.0078` | `7.05` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.075`, `0.041`, `0.001`) | `0.2591` | `0.0011` | `6.72` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `0.250`) | (`-0.011`, `-0.067`, `-0.115`) | `0.1324` | `0.3653` | `5.09` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `-0.250`) | (`-0.012`, `0.065`, `0.112`) | `0.1302` | `0.3622` | `4.97` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.076`, `-0.012`, `-0.009`) | `0.2218` | `0.0093` | `4.92` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.076`, `0.016`, `-0.010`) | `0.2196` | `0.0104` | `4.83` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.079`, `-0.009`, `-0.037`) | `0.1711` | `0.2130` | `4.06` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.078`, `-0.011`, `0.076`) | `0.1721` | `0.1737` | `3.72` |

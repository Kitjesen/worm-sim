# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1011` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.0188` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0503` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `6` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0269` | `0.0328` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0655` | `0.0913` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0287` | `0.0210` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.1213` | `0.1208` | 5 | 1 | 7 | 2 | 4 |
| `mixed_vx_yaw` | 6 | `0.1135` | `0.2236` | 2 | 0 | 2 | 2 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5656` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4253` | `1.1093` | `0.0461` | `1.1037` | `8.5261` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0190` | `0.8769` | `0.0457` | `0.8842` | `8.5777` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8907` | `0.6108` | `0.0154` | `0.6161` | `1.0910` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3019` | `1.6050` | `0.1087` | `1.5940` | `8.7195` | `0.2500` | `0.5183` |
| `mixed_vx_yaw` | `0.4510` | `1.7624` | `0.1073` | `1.7803` | `6.7985` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`0.102`, `-0.018`, `-0.476`) | `0.2024` | `0.3757` | `7.62` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`0.119`, `-0.148`, `-0.238`) | `0.2305` | `0.2382` | `6.73` |
| `mixed_vx_vy` | (`-0.050`, `-0.075`, `0.000`) | (`0.139`, `-0.136`, `-0.263`) | `0.1985` | `0.2629` | `5.67` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`0.066`, `0.020`, `0.277`) | `0.1675` | `0.1767` | `3.59` |
| `mixed_vx_vy` | (`0.050`, `-0.037`, `0.000`) | (`-0.107`, `0.012`, `0.111`) | `0.1648` | `0.1109` | `3.02` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `0.100`) | (`0.076`, `0.029`, `0.397`) | `0.0391` | `0.2972` | `2.36` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`-0.019`, `-0.021`, `0.003`) | `0.1532` | `0.0032` | `2.35` |
| `mixed_vx_vy` | (`-0.050`, `-0.037`, `0.000`) | (`-0.051`, `0.078`, `0.156`) | `0.1156` | `0.1561` | `1.95` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0598` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0188` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0328` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0269` | `0.0328` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0655` | `0.0913` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0287` | `0.0210` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0659` | `0.0519` | 0 | 2 | 2 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0717` | `0.1095` | 0 | 1 | 1 | 0 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5656` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4253` | `1.1093` | `0.0461` | `1.1037` | `8.5261` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0190` | `0.8769` | `0.0457` | `0.8842` | `8.5777` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8907` | `0.6108` | `0.0154` | `0.6161` | `1.0910` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3562` | `1.0201` | `0.0992` | `1.0224` | `8.6486` | `0.2500` | `0.4389` |
| `mixed_vx_yaw` | `0.4474` | `1.1099` | `0.0902` | `1.1064` | `8.5804` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.013`, `-0.102`, `-0.142`) | `0.1346` | `0.0418` | `1.85` |
| `pure_vy` | (`0.000`, `-0.037`, `0.000`) | (`-0.087`, `0.045`, `0.128`) | `0.1200` | `0.1281` | `1.85` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.060`, `0.037`, `0.004`) | `0.1191` | `0.0044` | `1.42` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.074`, `0.028`, `0.087`) | `0.1067` | `0.0873` | `1.33` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.068`, `0.027`, `0.075`) | `0.0417` | `0.1746` | `0.94` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.035`, `0.069`, `0.135`) | `0.0951` | `0.0351` | `0.93` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.027`, `0.059`, `0.100`) | `0.0790` | `0.0998` | `0.87` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.024`, `-0.126`, `-0.027`) | `0.0896` | `0.0267` | `0.82` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0629` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0186` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0330` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0283` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0266` | `0.0187` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0801` | `0.1083` | 0 | 1 | 1 | 0 | 0 |
| `pure_yaw` | 4 | `0.0285` | `0.0208` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0663` | `0.0504` | 0 | 1 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0837` | `0.1119` | 0 | 1 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5634` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4224` | `1.1082` | `0.0468` | `1.1025` | `8.4543` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0184` | `0.8769` | `0.0463` | `0.8840` | `8.5881` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8888` | `0.6108` | `0.0155` | `0.6161` | `1.0889` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3555` | `1.0207` | `0.0999` | `1.0228` | `8.6587` | `0.2500` | `0.4277` |
| `mixed_vx_yaw` | `0.4473` | `1.1112` | `0.0913` | `1.1070` | `8.5635` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `pure_vy` | (`0.000`, `-0.037`, `0.000`) | (`-0.077`, `0.085`, `0.168`) | `0.1444` | `0.1683` | `2.79` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.010`, `0.107`, `0.179`) | `0.1404` | `0.0790` | `2.13` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.014`, `-0.098`, `-0.138`) | `0.1305` | `0.0382` | `1.74` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.057`, `0.046`, `0.025`) | `0.1283` | `0.0250` | `1.66` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.054`, `0.010`, `0.074`) | `0.0466` | `0.1742` | `0.98` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.029`, `0.049`, `0.096`) | `0.0829` | `0.0962` | `0.92` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.032`, `-0.124`, `-0.006`) | `0.0956` | `0.0056` | `0.91` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.043`, `-0.004`, `0.050`) | `0.0915` | `0.0495` | `0.90` |

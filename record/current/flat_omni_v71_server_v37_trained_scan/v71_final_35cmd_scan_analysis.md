# Worm V6 command scan strict analysis

Verdict: **PASS**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0615` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0186` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0359` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0284` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0369` | `0.0189` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0273` | `0.0796` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0286` | `0.0208` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0734` | `0.0732` | 0 | 1 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0816` | `0.1174` | 0 | 1 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5817` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4241` | `1.1047` | `0.0461` | `1.0988` | `8.5436` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0179` | `0.9716` | `0.0473` | `0.9762` | `8.1191` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8997` | `0.6108` | `0.0155` | `0.6163` | `1.0840` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3556` | `1.0162` | `0.0998` | `1.0180` | `8.6337` | `0.2500` | `0.4335` |
| `mixed_vx_yaw` | `0.4443` | `1.1063` | `0.0919` | `1.1005` | `8.5819` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.053`, `-0.057`, `-0.123`) | `0.1405` | `0.1226` | `2.35` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.007`, `-0.106`, `-0.155`) | `0.1404` | `0.0549` | `2.05` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.018`, `0.099`, `0.162`) | `0.1288` | `0.0619` | `1.76` |
| `mixed_vx_vy` | (`-0.050`, `-0.075`, `0.000`) | (`-0.003`, `-0.161`, `-0.048`) | `0.0986` | `0.0476` | `1.03` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.048`, `0.007`, `0.036`) | `0.0968` | `0.0362` | `0.97` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.056`, `0.011`, `0.072`) | `0.0450` | `0.1722` | `0.94` |
| `mixed_vx_vy` | (`0.100`, `-0.037`, `0.000`) | (`0.066`, `0.023`, `0.123`) | `0.0692` | `0.1233` | `0.86` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.023`, `0.061`, `0.108`) | `0.0745` | `0.1081` | `0.85` |

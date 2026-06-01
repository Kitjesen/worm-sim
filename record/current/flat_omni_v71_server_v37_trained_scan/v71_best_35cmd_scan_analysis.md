# Worm V6 command scan strict analysis

Verdict: **PASS**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0590` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0185` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0341` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0284` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0328` | `0.0540` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0309` | `0.0829` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0285` | `0.0207` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0702` | `0.0509` | 1 | 0 | 2 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.0749` | `0.1106` | 0 | 1 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5805` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4254` | `1.1063` | `0.0462` | `1.1007` | `8.6095` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0176` | `0.9716` | `0.0472` | `0.9763` | `8.1046` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8987` | `0.6108` | `0.0154` | `0.6162` | `1.0819` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3555` | `1.0160` | `0.0998` | `1.0177` | `8.6568` | `0.2500` | `0.4408` |
| `mixed_vx_yaw` | `0.4440` | `1.1058` | `0.0917` | `1.1000` | `8.5965` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.024`, `0.102`, `0.157`) | `0.1272` | `0.0572` | `1.70` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.022`, `-0.090`, `-0.129`) | `0.1191` | `0.0286` | `1.44` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.053`, `-0.061`, `0.008`) | `0.1041` | `0.0077` | `1.09` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.065`, `0.020`, `-0.005`) | `0.1012` | `0.0051` | `1.03` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.031`, `0.054`, `0.102`) | `0.0833` | `0.1024` | `0.96` |
| `mixed_vx_vy` | (`-0.100`, `0.075`, `0.000`) | (`-0.070`, `-0.014`, `-0.047`) | `0.0942` | `0.0468` | `0.94` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `0.100`) | (`0.068`, `-0.018`, `-0.061`) | `0.0363` | `0.1614` | `0.78` |
| `mixed_vx_yaw` | (`0.100`, `0.000`, `-0.100`) | (`0.061`, `0.003`, `0.056`) | `0.0391` | `0.1561` | `0.76` |

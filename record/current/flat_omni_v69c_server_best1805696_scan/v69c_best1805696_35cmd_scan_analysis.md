# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0640` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0189` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0354` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0289` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0266` | `0.0241` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0722` | `0.0905` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0291` | `0.0212` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0702` | `0.0505` | 0 | 1 | 1 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.0819` | `0.1093` | 0 | 1 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5684` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4241` | `1.1084` | `0.0468` | `1.1026` | `8.4942` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0199` | `0.8769` | `0.0463` | `0.8837` | `8.5925` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8928` | `0.6108` | `0.0157` | `0.6161` | `1.0905` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3568` | `1.0193` | `0.0997` | `1.0209` | `8.6440` | `0.2500` | `0.4467` |
| `mixed_vx_yaw` | `0.4471` | `1.1091` | `0.0914` | `1.1047` | `8.5637` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.010`, `0.111`, `0.172`) | `0.1435` | `0.0722` | `2.19` |
| `pure_vy` | (`0.000`, `-0.037`, `0.000`) | (`-0.096`, `0.052`, `0.131`) | `0.1313` | `0.1312` | `2.15` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.017`, `-0.090`, `-0.142`) | `0.1226` | `0.0421` | `1.55` |
| `mixed_vx_vy` | (`-0.100`, `-0.075`, `0.000`) | (`-0.058`, `0.041`, `0.015`) | `0.1234` | `0.0152` | `1.53` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.031`, `0.056`, `0.104`) | `0.0829` | `0.1041` | `0.96` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.053`, `-0.010`, `0.018`) | `0.0971` | `0.0182` | `0.95` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.059`, `0.012`, `0.033`) | `0.0959` | `0.0332` | `0.95` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.036`, `-0.072`, `0.009`) | `0.0865` | `0.0091` | `0.75` |

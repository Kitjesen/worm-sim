# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0631` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0188` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0325` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0285` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0385` | `0.0257` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0813` | `0.1065` | 0 | 1 | 1 | 0 | 0 |
| `pure_yaw` | 4 | `0.0287` | `0.0211` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0648` | `0.0497` | 0 | 0 | 0 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0776` | `0.1161` | 0 | 0 | 2 | 0 | 2 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5787` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4256` | `1.1064` | `0.0460` | `1.1011` | `8.5280` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0214` | `0.8769` | `0.0456` | `0.8842` | `8.5784` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8983` | `0.6108` | `0.0155` | `0.6161` | `1.0837` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3568` | `1.0165` | `0.0990` | `1.0181` | `8.6329` | `0.2500` | `0.4394` |
| `mixed_vx_yaw` | `0.4462` | `1.1067` | `0.0916` | `1.1017` | `8.6341` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `pure_vy` | (`0.000`, `-0.037`, `0.000`) | (`-0.080`, `0.074`, `0.168`) | `0.1374` | `0.1678` | `2.59` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.009`, `-0.099`, `-0.150`) | `0.1345` | `0.0497` | `1.87` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.026`, `0.088`, `0.138`) | `0.1150` | `0.0381` | `1.36` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.030`, `0.049`, `0.100`) | `0.0841` | `0.0997` | `0.96` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.043`, `-0.090`, `-0.003`) | `0.0939` | `0.0032` | `0.88` |
| `mixed_vx_yaw` | (`0.050`, `0.000`, `-0.100`) | (`0.056`, `0.017`, `0.080`) | `0.0177` | `0.1803` | `0.84` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.052`, `-0.002`, `0.027`) | `0.0874` | `0.0267` | `0.78` |
| `mixed_vx_vy` | (`-0.100`, `-0.037`, `0.000`) | (`-0.063`, `0.041`, `0.023`) | `0.0868` | `0.0229` | `0.77` |

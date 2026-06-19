# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1566` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1760` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0379` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0913` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1238` | `0.0280` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0784` | `0.1246` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0915` | `0.1968` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1815` | `0.0718` | 12 | 7 | 15 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1303` | `0.2345` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5519` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4501` | `1.1376` | `0.0428` | `1.1363` | `7.0155` |
| `pure_vy` | `0.0143` | `1.0459` | `0.0401` | `1.0476` | `6.9596` |
| `pure_yaw` | `0.8215` | `2.7146` | `0.0810` | `2.6961` | `1.5880` |
| `mixed_vx_vy` | `0.3738` | `1.0875` | `0.0900` | `1.0918` | `7.4224` |
| `mixed_vx_yaw` | `0.4904` | `0.9772` | `0.0661` | `0.9854` | `6.8657` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.056`, `0.068`, `0.025`) | `0.2921` | `0.0247` | `8.55` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.074`, `-0.011`, `-0.063`) | `0.2381` | `0.0626` | `5.77` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.085`, `0.011`, `0.084`) | `0.1654` | `0.3338` | `5.52` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.077`, `0.003`, `0.011`) | `0.2310` | `0.0106` | `5.34` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.081`, `0.008`, `0.102`) | `0.2208` | `0.1016` | `5.13` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.062`, `0.050`, `-0.001`) | `0.2252` | `0.0006` | `5.07` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.079`, `0.014`, `0.010`) | `0.1715` | `0.2404` | `4.38` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.084`, `-0.032`, `-0.023`) | `0.1970` | `0.0233` | `3.89` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.069`, `0.022`, `-0.019`) | `0.1887` | `0.0187` | `3.57` |
| `pure_vx` | (`-0.250`, `0.000`, `0.000`) | (`-0.067`, `0.021`, `-0.020`) | `0.1843` | `0.0198` | `3.41` |

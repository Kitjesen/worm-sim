# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1632` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2142` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0364` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.1048` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1350` | `0.0256` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0721` | `0.0874` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.1051` | `0.2395` | 1 | 0 | 3 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1891` | `0.0813` | 11 | 7 | 15 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1373` | `0.2555` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5452` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4505` | `1.1389` | `0.0832` | `1.1378` | `7.0524` |
| `pure_vy` | `0.0144` | `1.0459` | `0.0791` | `1.0510` | `6.8529` |
| `pure_yaw` | `0.8242` | `2.7146` | `0.1529` | `2.6860` | `1.5970` |
| `mixed_vx_vy` | `0.3751` | `1.0872` | `0.1709` | `1.1045` | `7.3034` |
| `mixed_vx_yaw` | `0.4912` | `0.9816` | `0.1286` | `1.0016` | `6.6853` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.074`, `0.040`, `0.126`) | `0.2595` | `0.1261` | `7.13` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.037`, `0.000`, `-0.017`) | `0.2605` | `0.0171` | `6.79` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.066`, `0.007`, `0.112`) | `0.1838` | `0.3624` | `6.66` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.070`, `-0.021`, `0.056`) | `0.2480` | `0.0565` | `6.23` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.051`, `-0.052`, `-0.105`) | `0.2363` | `0.1046` | `5.86` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.057`, `0.012`, `-0.024`) | `0.2374` | `0.0235` | `5.65` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.068`, `0.002`, `-0.034`) | `0.1823` | `0.2840` | `5.34` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.065`, `-0.031`, `0.053`) | `0.2135` | `0.0531` | `4.63` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.061`, `0.012`, `-0.064`) | `0.2079` | `0.0644` | `4.42` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.062`, `-0.019`, `0.013`) | `0.1966` | `0.0133` | `3.87` |

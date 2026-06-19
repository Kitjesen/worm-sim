# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1703` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1978` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0269` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0967` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1254` | `0.0356` | 2 | 0 | 3 | 0 | 0 |
| `pure_vy` | 4 | `0.0648` | `0.0899` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0969` | `0.2211` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2008` | `0.0479` | 13 | 9 | 15 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.1301` | `0.2272` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5364` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4481` | `1.1389` | `0.0404` | `1.1380` | `7.0754` |
| `pure_vy` | `0.0121` | `1.0459` | `0.0402` | `1.0465` | `6.9766` |
| `pure_yaw` | `0.8233` | `2.7146` | `0.0734` | `2.7035` | `1.5661` |
| `mixed_vx_vy` | `0.3844` | `0.7074` | `0.1254` | `0.7235` | `6.3558` |
| `mixed_vx_yaw` | `0.4874` | `0.9843` | `0.0629` | `0.9935` | `6.8916` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.032`, `0.029`, `0.013`) | `0.2820` | `0.0126` | `7.96` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.045`, `0.026`, `0.063`) | `0.2703` | `0.0628` | `7.40` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.041`, `0.012`, `-0.003`) | `0.2499` | `0.0030` | `6.25` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.034`, `0.040`, `0.040`) | `0.2450` | `0.0402` | `6.04` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.051`, `0.009`, `0.047`) | `0.2441` | `0.0470` | `6.01` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.081`, `-0.003`, `0.044`) | `0.1688` | `0.2943` | `5.02` |
| `mixed_vx_vy` | (`0.125`, `-0.150`, `0.000`) | (`-0.058`, `-0.038`, `0.061`) | `0.2149` | `0.0610` | `4.71` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.039`, `0.033`, `0.028`) | `0.2153` | `0.0276` | `4.65` |

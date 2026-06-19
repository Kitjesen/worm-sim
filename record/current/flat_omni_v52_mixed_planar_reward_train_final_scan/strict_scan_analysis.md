# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1534` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1978` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0348` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0967` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1254` | `0.0356` | 2 | 0 | 3 | 0 | 0 |
| `pure_vy` | 4 | `0.0648` | `0.0899` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0969` | `0.2211` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1783` | `0.0823` | 11 | 5 | 13 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1301` | `0.2272` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5364` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4481` | `1.1389` | `0.0404` | `1.1380` | `7.0754` |
| `pure_vy` | `0.0121` | `1.0459` | `0.0402` | `1.0465` | `6.9766` |
| `pure_yaw` | `0.8233` | `2.7146` | `0.0734` | `2.7035` | `1.5661` |
| `mixed_vx_vy` | `0.3702` | `1.0893` | `0.0886` | `1.0935` | `7.4897` |
| `mixed_vx_yaw` | `0.4874` | `0.9843` | `0.0629` | `0.9935` | `6.8916` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.068`, `0.025`, `-0.015`) | `0.2524` | `0.0146` | `6.37` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.070`, `-0.003`, `0.026`) | `0.2365` | `0.0265` | `5.61` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.082`, `0.009`, `0.056`) | `0.2318` | `0.0561` | `5.45` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.081`, `-0.003`, `0.044`) | `0.1688` | `0.2943` | `5.02` |
| `mixed_vx_vy` | (`-0.125`, `-0.150`, `0.000`) | (`0.073`, `-0.144`, `-0.192`) | `0.1980` | `0.1919` | `4.84` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.070`, `0.034`, `-0.014`) | `0.2143` | `0.0140` | `4.60` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.068`, `0.017`, `-0.025`) | `0.2040` | `0.0248` | `4.18` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.090`, `0.014`, `0.003`) | `0.1611` | `0.2473` | `4.12` |

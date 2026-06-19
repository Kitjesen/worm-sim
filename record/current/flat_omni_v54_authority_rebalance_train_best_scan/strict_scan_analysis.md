# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1678` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1877` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0304` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0959` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1260` | `0.0448` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0858` | `0.1237` | 1 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0960` | `0.2098` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1954` | `0.0535` | 13 | 7 | 16 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.1264` | `0.2285` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5462` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4482` | `1.1377` | `0.0412` | `1.1366` | `7.0691` |
| `pure_vy` | `0.0139` | `1.0459` | `0.0394` | `1.0474` | `6.9343` |
| `pure_yaw` | `0.8201` | `2.7146` | `0.0764` | `2.6993` | `1.5877` |
| `mixed_vx_vy` | `0.3883` | `0.7064` | `0.1260` | `0.7233` | `6.2659` |
| `mixed_vx_yaw` | `0.4878` | `0.9821` | `0.0644` | `0.9903` | `6.8136` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.033`, `0.034`, `0.030`) | `0.2845` | `0.0298` | `8.12` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.052`, `0.010`, `0.044`) | `0.2549` | `0.0435` | `6.54` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.050`, `0.014`, `0.091`) | `0.2417` | `0.0906` | `6.05` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.036`, `0.043`, `0.049`) | `0.2393` | `0.0491` | `5.79` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.034`, `0.030`, `0.024`) | `0.2401` | `0.0243` | `5.78` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.092`, `-0.031`, `0.062`) | `0.1606` | `0.3117` | `5.01` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.038`, `0.038`, `0.021`) | `0.2154` | `0.0209` | `4.65` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.058`, `0.003`, `0.052`) | `0.2071` | `0.0515` | `4.36` |

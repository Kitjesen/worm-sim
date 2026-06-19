# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1733` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1954` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0245` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.1196` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1530` | `0.0237` | 3 | 0 | 3 | 0 | 0 |
| `pure_vy` | 4 | `0.1021` | `0.0798` | 0 | 1 | 2 | 0 | 1 |
| `pure_yaw` | 4 | `0.1197` | `0.2185` | 3 | 0 | 4 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1961` | `0.0511` | 13 | 7 | 16 | 0 | 0 |
| `mixed_vx_yaw` | 6 | `0.1446` | `0.2503` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5463` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4659` | `0.8558` | `0.0824` | `0.8620` | `6.3791` |
| `pure_vy` | `0.0187` | `0.7844` | `0.0832` | `0.7975` | `5.9574` |
| `pure_yaw` | `0.8370` | `2.0359` | `0.1493` | `2.0631` | `1.5780` |
| `mixed_vx_vy` | `0.3866` | `0.8129` | `0.1772` | `0.8404` | `6.5565` |
| `mixed_vx_yaw` | `0.5063` | `0.7327` | `0.1227` | `0.7577` | `5.9737` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.040`, `0.005`, `0.053`) | `0.2612` | `0.0533` | `6.89` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.041`, `0.009`, `0.013`) | `0.2625` | `0.0134` | `6.89` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.051`, `-0.020`, `0.058`) | `0.1998` | `0.3081` | `6.36` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.054`, `0.002`, `0.075`) | `0.2452` | `0.0751` | `6.15` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.048`, `0.004`, `-0.016`) | `0.2017` | `0.2664` | `5.84` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.039`, `0.038`, `-0.032`) | `0.2390` | `0.0321` | `5.74` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.034`, `-0.009`, `-0.029`) | `0.2258` | `0.0290` | `5.12` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.045`, `-0.007`, `-0.025`) | `0.2210` | `0.0248` | `4.90` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.048`, `-0.015`, `0.016`) | `0.2103` | `0.0157` | `4.43` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.057`, `-0.004`, `0.044`) | `0.2084` | `0.0444` | `4.39` |

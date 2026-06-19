# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1527` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1933` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0379` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0965` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1275` | `0.0188` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0799` | `0.1040` | 0 | 0 | 1 | 0 | 0 |
| `pure_yaw` | 4 | `0.0967` | `0.2161` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1754` | `0.0892` | 10 | 6 | 13 | 1 | 1 |
| `mixed_vx_yaw` | 6 | `0.1284` | `0.2484` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5463` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4502` | `1.1362` | `0.0420` | `1.1351` | `7.0443` |
| `pure_vy` | `0.0144` | `1.0459` | `0.0401` | `1.0482` | `6.9778` |
| `pure_yaw` | `0.8206` | `2.7146` | `0.0785` | `2.6982` | `1.5752` |
| `mixed_vx_vy` | `0.3730` | `1.0864` | `0.0888` | `1.0910` | `7.4193` |
| `mixed_vx_yaw` | `0.4897` | `0.9786` | `0.0645` | `0.9868` | `6.8380` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.062`, `0.036`, `0.001`) | `0.2645` | `0.0009` | `6.99` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.075`, `-0.013`, `-0.001`) | `0.2397` | `0.0006` | `5.74` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.090`, `-0.001`, `0.098`) | `0.1603` | `0.3484` | `5.61` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.084`, `0.002`, `-0.061`) | `0.1665` | `0.3107` | `5.19` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.076`, `0.021`, `-0.017`) | `0.2167` | `0.0168` | `4.70` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.085`, `-0.015`, `0.002`) | `0.2129` | `0.0019` | `4.53` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.073`, `-0.025`, `-0.070`) | `0.2035` | `0.0704` | `4.26` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.067`, `-0.002`, `-0.056`) | `0.1973` | `0.0560` | `3.97` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.101`, `-0.042`, `-0.079`) | `0.1896` | `0.0792` | `3.75` |
| `pure_yaw` | (`0.000`, `0.000`, `0.250`) | (`0.092`, `0.045`, `0.576`) | `0.1025` | `0.3263` | `3.71` |

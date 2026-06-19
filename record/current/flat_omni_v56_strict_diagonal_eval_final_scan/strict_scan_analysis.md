# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1538` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1831` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0376` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0980` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1230` | `0.0270` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0656` | `0.0942` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0984` | `0.2047` | 0 | 0 | 2 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1792` | `0.0617` | 11 | 6 | 14 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1258` | `0.2510` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5517` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4488` | `1.1395` | `0.0424` | `1.1380` | `7.0209` |
| `pure_vy` | `0.0138` | `1.0459` | `0.0397` | `1.0477` | `6.9606` |
| `pure_yaw` | `0.8240` | `2.7146` | `0.0784` | `2.6980` | `1.5762` |
| `mixed_vx_vy` | `0.3725` | `1.0904` | `0.0888` | `1.0947` | `7.3908` |
| `mixed_vx_yaw` | `0.4904` | `0.9801` | `0.0664` | `0.9892` | `6.8334` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.063`, `-0.045`, `-0.068`) | `0.2699` | `0.0683` | `7.40` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.066`, `0.049`, `0.008`) | `0.2713` | `0.0076` | `7.36` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.093`, `0.041`, `0.168`) | `0.1619` | `0.4177` | `6.98` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.082`, `-0.010`, `0.030`) | `0.2185` | `0.0298` | `4.80` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.066`, `0.045`, `0.007`) | `0.2119` | `0.0067` | `4.49` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.070`, `0.009`, `-0.041`) | `0.1984` | `0.0410` | `3.98` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.096`, `0.031`, `0.076`) | `0.1875` | `0.0755` | `3.66` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.064`, `0.038`, `-0.007`) | `0.1893` | `0.0072` | `3.58` |

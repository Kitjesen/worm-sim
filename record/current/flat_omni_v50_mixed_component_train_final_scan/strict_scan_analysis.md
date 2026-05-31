# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1801` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2075` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0514` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0963` | `0.0800` | fail |
| `wrong_planar_sign_count` | `4` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1262` | `0.0378` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0732` | `0.0793` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0966` | `0.2320` | 0 | 0 | 2 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2130` | `0.1428` | 12 | 7 | 16 | 4 | 5 |
| `mixed_vx_yaw` | 6 | `0.1519` | `0.1517` | 4 | 0 | 4 | 2 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5337` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4500` | `1.1390` | `0.0392` | `1.1397` | `7.0895` |
| `pure_vy` | `0.0124` | `1.0459` | `0.0369` | `1.0462` | `6.9908` |
| `pure_yaw` | `0.8286` | `2.7146` | `0.0568` | `2.7063` | `1.5847` |
| `mixed_vx_vy` | `0.3363` | `1.7142` | `0.0444` | `1.7146` | `7.7321` |
| `mixed_vx_yaw` | `0.4761` | `1.3746` | `0.0379` | `1.3798` | `3.9576` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`0.118`, `-0.137`, `-0.307`) | `0.3679` | `0.3070` | `15.89` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.014`, `-0.022`, `0.006`) | `0.3150` | `0.0059` | `9.92` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.040`, `-0.040`, `-0.099`) | `0.2397` | `0.0994` | `5.99` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.091`, `-0.031`, `-0.006`) | `0.2414` | `0.0064` | `5.83` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.011`, `-0.092`, `-0.055`) | `0.2397` | `0.0550` | `5.82` |
| `mixed_vx_vy` | (`-0.125`, `-0.150`, `0.000`) | (`0.082`, `-0.141`, `-0.239`) | `0.2070` | `0.2390` | `5.71` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.033`, `-0.107`, `0.083`) | `0.2192` | `0.0826` | `4.97` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.071`, `0.003`, `0.015`) | `0.1786` | `0.2650` | `4.95` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1515` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1196` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0365` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0631` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1268` | `0.0139` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0723` | `0.1035` | 0 | 0 | 0 | 0 | 1 |
| `pure_yaw` | 4 | `0.0633` | `0.1337` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.1747` | `0.0692` | 12 | 5 | 14 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1281` | `0.2884` | 4 | 0 | 4 | 6 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5251` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4471` | `1.1385` | `0.0402` | `1.1386` | `7.0910` |
| `pure_vy` | `0.0103` | `1.0459` | `0.0373` | `1.0449` | `6.9904` |
| `pure_yaw` | `0.8215` | `2.7146` | `0.0564` | `2.7073` | `1.2665` |
| `mixed_vx_vy` | `0.3687` | `1.0904` | `0.0475` | `1.0923` | `7.5830` |
| `mixed_vx_yaw` | `0.4860` | `0.9862` | `0.0380` | `0.9908` | `6.8554` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.070`, `0.035`, `0.006`) | `0.2585` | `0.0057` | `6.68` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.073`, `-0.005`, `0.019`) | `0.2286` | `0.0187` | `5.23` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `0.250`) | (`-0.012`, `-0.066`, `-0.115`) | `0.1311` | `0.3650` | `5.05` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.081`, `0.010`, `0.061`) | `0.2197` | `0.0611` | `4.92` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `-0.250`) | (`-0.018`, `0.053`, `0.101`) | `0.1191` | `0.3508` | `4.49` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.079`, `0.029`, `-0.012`) | `0.2096` | `0.0116` | `4.40` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.077`, `-0.010`, `-0.022`) | `0.1729` | `0.2277` | `4.28` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.076`, `0.020`, `-0.021`) | `0.1978` | `0.0207` | `3.92` |

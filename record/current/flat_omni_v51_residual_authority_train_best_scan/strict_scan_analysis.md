# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1561` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1846` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0388` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0954` | `0.0800` | fail |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1348` | `0.0246` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0901` | `0.1402` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0955` | `0.2064` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1775` | `0.0813` | 10 | 6 | 14 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1278` | `0.2397` | 4 | 0 | 4 | 3 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5243` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4483` | `1.1365` | `0.0397` | `1.1370` | `7.0832` |
| `pure_vy` | `0.0099` | `1.0459` | `0.0378` | `1.0462` | `7.0140` |
| `pure_yaw` | `0.8213` | `2.7146` | `0.0709` | `2.7050` | `1.5911` |
| `mixed_vx_vy` | `0.3696` | `1.0903` | `0.0857` | `1.0954` | `7.5411` |
| `mixed_vx_yaw` | `0.4864` | `0.9856` | `0.0611` | `0.9938` | `6.8860` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.080`, `0.026`, `0.072`) | `0.2444` | `0.0719` | `6.10` |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.068`, `0.001`, `-0.041`) | `0.2365` | `0.0413` | `5.64` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.068`, `0.011`, `-0.026`) | `0.2295` | `0.0257` | `5.28` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.083`, `-0.005`, `0.045`) | `0.2277` | `0.0453` | `5.23` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.087`, `-0.005`, `0.071`) | `0.1631` | `0.3207` | `5.23` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.084`, `-0.010`, `-0.048`) | `0.1664` | `0.2978` | `4.99` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.059`, `-0.035`, `-0.091`) | `0.1953` | `0.0914` | `4.02` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.062`, `-0.017`, `0.004`) | `0.1966` | `0.0037` | `3.87` |

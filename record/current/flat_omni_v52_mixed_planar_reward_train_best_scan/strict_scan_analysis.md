# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1546` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1945` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0389` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0961` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1277` | `0.0207` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0697` | `0.0964` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0962` | `0.2175` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1790` | `0.0719` | 12 | 7 | 14 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.1297` | `0.2304` | 4 | 0 | 4 | 3 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5355` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4472` | `1.1367` | `0.0402` | `1.1364` | `7.1350` |
| `pure_vy` | `0.0118` | `1.0459` | `0.0397` | `1.0465` | `7.0098` |
| `pure_yaw` | `0.8233` | `2.7146` | `0.0728` | `2.7039` | `1.5968` |
| `mixed_vx_vy` | `0.3702` | `1.0887` | `0.0880` | `1.0938` | `7.4935` |
| `mixed_vx_yaw` | `0.4862` | `0.9853` | `0.0629` | `0.9946` | `6.8406` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.059`, `0.063`, `0.011`) | `0.2863` | `0.0108` | `8.20` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.082`, `-0.013`, `-0.079`) | `0.1683` | `0.3293` | `5.54` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.063`, `0.011`, `-0.039`) | `0.2332` | `0.0395` | `5.48` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.089`, `0.019`, `0.122`) | `0.2079` | `0.1222` | `4.70` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.071`, `-0.029`, `-0.022`) | `0.2160` | `0.0221` | `4.68` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.071`, `0.033`, `-0.006`) | `0.2093` | `0.0058` | `4.38` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.094`, `-0.033`, `0.014`) | `0.1592` | `0.2638` | `4.27` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.070`, `-0.016`, `-0.070`) | `0.2023` | `0.0695` | `4.21` |

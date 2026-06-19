# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1509` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1646` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0372` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0873` | `0.0800` | fail |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1281` | `0.0366` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0638` | `0.0973` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0874` | `0.1840` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.1746` | `0.0697` | 12 | 4 | 15 | 0 | 2 |
| `mixed_vx_yaw` | 6 | `0.1246` | `0.2068` | 4 | 0 | 4 | 3 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5452` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4491` | `1.1390` | `0.0410` | `1.1383` | `7.0543` |
| `pure_vy` | `0.0135` | `1.0459` | `0.0393` | `1.0478` | `6.9380` |
| `pure_yaw` | `0.8220` | `2.7146` | `0.0766` | `2.6998` | `1.6009` |
| `mixed_vx_vy` | `0.3719` | `1.0882` | `0.0883` | `1.0927` | `7.4408` |
| `mixed_vx_yaw` | `0.4878` | `0.9818` | `0.0642` | `0.9903` | `6.7936` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.065`, `0.040`, `0.002`) | `0.2655` | `0.0017` | `7.05` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.083`, `-0.005`, `0.031`) | `0.2277` | `0.0309` | `5.21` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.069`, `0.017`, `-0.035`) | `0.2249` | `0.0351` | `5.09` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.081`, `-0.014`, `-0.025`) | `0.2170` | `0.0252` | `4.73` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.088`, `0.018`, `0.006`) | `0.1635` | `0.2438` | `4.16` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.072`, `0.003`, `-0.044`) | `0.1944` | `0.0435` | `3.83` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.078`, `-0.000`, `0.026`) | `0.1875` | `0.0262` | `3.53` |
| `pure_vx` | (`-0.250`, `0.000`, `0.000`) | (`-0.068`, `-0.031`, `-0.061`) | `0.1841` | `0.0608` | `3.48` |

# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1812` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2137` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0550` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0979` | `0.0800` | fail |
| `wrong_planar_sign_count` | `5` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1211` | `0.0577` | 1 | 0 | 3 | 0 | 0 |
| `pure_vy` | 4 | `0.0768` | `0.1153` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0982` | `0.2390` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2148` | `0.1811` | 12 | 7 | 15 | 5 | 5 |
| `mixed_vx_yaw` | 6 | `0.1478` | `0.1804` | 4 | 0 | 4 | 2 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5270` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4469` | `1.1373` | `0.0398` | `1.1376` | `7.0450` |
| `pure_vy` | `0.0110` | `1.0459` | `0.0371` | `1.0458` | `7.0488` |
| `pure_yaw` | `0.8238` | `2.7146` | `0.0558` | `2.7072` | `1.5793` |
| `mixed_vx_vy` | `0.3364` | `1.7137` | `0.0448` | `1.7138` | `7.7377` |
| `mixed_vx_yaw` | `0.4755` | `1.3737` | `0.0378` | `1.3785` | `3.9226` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`0.124`, `-0.146`, `-0.317`) | `0.3736` | `0.3170` | `16.47` |
| `mixed_vx_vy` | (`-0.125`, `-0.150`, `0.000`) | (`0.166`, `-0.159`, `-0.403`) | `0.2916` | `0.4026` | `12.56` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.012`, `-0.005`, `0.033`) | `0.3042` | `0.0334` | `9.28` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.057`, `-0.067`, `-0.173`) | `0.2397` | `0.1727` | `6.49` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.083`, `-0.037`, `-0.015`) | `0.2502` | `0.0150` | `6.26` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.012`, `-0.107`, `0.066`) | `0.2404` | `0.0659` | `5.89` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.097`, `-0.002`, `-0.075`) | `0.1530` | `0.3251` | `4.98` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.030`, `-0.052`, `0.021`) | `0.2207` | `0.0214` | `4.88` |

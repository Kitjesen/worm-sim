# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1842` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.2003` | `0.2000` | fail |
| `mean_off_axis_speed_m_s` | `0.0511` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0947` | `0.0800` | fail |
| `wrong_planar_sign_count` | `5` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1252` | `0.0265` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0663` | `0.1023` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0950` | `0.2239` | 0 | 0 | 1 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2191` | `0.1741` | 13 | 6 | 15 | 3 | 5 |
| `mixed_vx_yaw` | 6 | `0.1525` | `0.1759` | 4 | 0 | 4 | 2 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5251` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4482` | `1.1386` | `0.0400` | `1.1388` | `7.0536` |
| `pure_vy` | `0.0101` | `1.0459` | `0.0371` | `1.0447` | `6.9858` |
| `pure_yaw` | `0.8216` | `2.7146` | `0.0560` | `2.7070` | `1.5878` |
| `mixed_vx_vy` | `0.3352` | `1.7135` | `0.0452` | `1.7128` | `7.7590` |
| `mixed_vx_yaw` | `0.4758` | `1.3736` | `0.0382` | `1.3782` | `3.8943` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`0.144`, `-0.147`, `-0.352`) | `0.3936` | `0.3520` | `18.59` |
| `mixed_vx_vy` | (`-0.125`, `-0.150`, `0.000`) | (`0.194`, `-0.140`, `-0.455`) | `0.3194` | `0.4547` | `15.37` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.012`, `-0.005`, `0.029`) | `0.3046` | `0.0287` | `9.30` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.089`, `-0.046`, `-0.034`) | `0.2537` | `0.0337` | `6.46` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`0.042`, `-0.044`, `-0.128`) | `0.2395` | `0.1284` | `6.15` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.081`, `-0.004`, `-0.062`) | `0.1688` | `0.3119` | `5.28` |
| `mixed_vx_vy` | (`-0.250`, `-0.075`, `0.000`) | (`-0.029`, `-0.030`, `0.025`) | `0.2255` | `0.0246` | `5.10` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`0.034`, `-0.112`, `0.056`) | `0.2191` | `0.0558` | `4.88` |

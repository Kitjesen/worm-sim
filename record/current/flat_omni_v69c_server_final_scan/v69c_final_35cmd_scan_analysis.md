# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.0601` | `0.1000` | pass |
| `yaw_rmse_rad_s` | `0.0189` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0323` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0289` | `0.0800` | pass |
| `wrong_planar_sign_count` | `1` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.0473` | `0.0426` | 0 | 0 | 0 | 0 | 0 |
| `pure_vy` | 4 | `0.0684` | `0.0916` | 0 | 0 | 1 | 0 | 1 |
| `pure_yaw` | 4 | `0.0290` | `0.0211` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.0626` | `0.0482` | 1 | 0 | 1 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.0697` | `0.0927` | 0 | 0 | 1 | 0 | 1 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5683` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4262` | `1.1082` | `0.0464` | `1.1021` | `8.5725` | `0.0000` | `0.0000` |
| `pure_vy` | `0.0193` | `0.8769` | `0.0458` | `0.8838` | `8.5799` | `0.0000` | `0.0000` |
| `pure_yaw` | `0.8924` | `0.6108` | `0.0155` | `0.6161` | `1.0904` | `0.0000` | `0.0000` |
| `mixed_vx_vy` | `0.3569` | `1.0193` | `0.0987` | `1.0209` | `8.6639` | `0.2500` | `0.4556` |
| `mixed_vx_yaw` | `0.4471` | `1.1091` | `0.0911` | `1.1041` | `8.5767` | `0.0000` | `0.0000` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `pure_vy` | (`0.000`, `-0.037`, `0.000`) | (`-0.095`, `0.039`, `0.130`) | `0.1216` | `0.1300` | `1.90` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `-0.100`) | (`-0.014`, `-0.096`, `-0.148`) | `0.1288` | `0.0480` | `1.72` |
| `mixed_vx_vy` | (`0.050`, `-0.075`, `0.000`) | (`-0.051`, `-0.102`, `-0.007`) | `0.1042` | `0.0072` | `1.09` |
| `mixed_vx_yaw` | (`-0.100`, `0.000`, `0.100`) | (`-0.037`, `0.075`, `0.125`) | `0.0976` | `0.0252` | `0.97` |
| `mixed_vx_vy` | (`0.100`, `0.075`, `0.000`) | (`0.048`, `0.002`, `0.048`) | `0.0895` | `0.0484` | `0.86` |
| `mixed_vx_vy` | (`0.050`, `0.075`, `0.000`) | (`-0.024`, `0.053`, `0.097`) | `0.0772` | `0.0974` | `0.83` |
| `pure_vx` | (`-0.100`, `0.000`, `0.000`) | (`-0.055`, `0.074`, `0.041`) | `0.0864` | `0.0413` | `0.79` |
| `mixed_vx_vy` | (`0.100`, `-0.075`, `0.000`) | (`0.059`, `-0.003`, `0.051`) | `0.0827` | `0.0512` | `0.75` |

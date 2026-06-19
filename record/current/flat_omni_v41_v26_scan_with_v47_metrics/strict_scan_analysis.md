# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.1517` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1105` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0360` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0000` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0659` | `0.0800` | pass |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0000` | `0.0002` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1195` | `0.0337` | 2 | 0 | 3 | 0 | 0 |
| `pure_vy` | 4 | `0.0724` | `0.0984` | 0 | 0 | 0 | 0 | 1 |
| `pure_yaw` | 4 | `0.0661` | `0.1236` | 0 | 0 | 0 | 0 | 0 |
| `mixed_vx_vy` | 16 | `0.1762` | `0.0552` | 12 | 5 | 15 | 0 | 1 |
| `mixed_vx_yaw` | 6 | `0.1334` | `0.2976` | 4 | 0 | 4 | 6 | 0 |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.066`, `0.027`, `0.011`) | `0.2555` | `0.0106` | `6.53` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`0.069`, `-0.025`, `0.006`) | `0.2514` | `0.0057` | `6.32` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `0.250`) | (`-0.007`, `-0.073`, `-0.130`) | `0.1384` | `0.3795` | `5.52` |
| `mixed_vx_yaw` | (`-0.125`, `0.000`, `-0.250`) | (`-0.011`, `0.065`, `0.115`) | `0.1312` | `0.3647` | `5.05` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.069`, `-0.018`, `0.015`) | `0.2245` | `0.0153` | `5.04` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.081`, `0.016`, `-0.013`) | `0.2157` | `0.0134` | `4.66` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `-0.250`) | (`0.080`, `-0.001`, `0.001`) | `0.1700` | `0.2510` | `4.47` |
| `mixed_vx_yaw` | (`0.250`, `0.000`, `0.250`) | (`0.083`, `-0.036`, `0.017`) | `0.1705` | `0.2327` | `4.26` |

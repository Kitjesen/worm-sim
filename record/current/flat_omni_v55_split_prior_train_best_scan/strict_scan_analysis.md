# Worm V6 command scan strict analysis

Verdict: **FAIL**

## Proof rule

Six primitive directions are a necessary condition, not a sufficient condition, for continuous `vx/vy/yaw` tracking. A policy is accepted only if the scan satisfies component RMSE, direction-sign, off-axis, stop, and yaw-only drift gates. One failed command class is a constructive counterexample to the continuous-tracking claim.

## Acceptance gate

| Condition | Measured | Target | Status |
| --- | ---: | ---: | --- |
| `planar_rmse_m_s` | `0.2023` | `0.1000` | fail |
| `yaw_rmse_rad_s` | `0.1635` | `0.2000` | pass |
| `mean_off_axis_speed_m_s` | `0.0423` | `0.0800` | pass |
| `zero_command_mean_speed_m_s` | `0.0001` | `0.0200` | pass |
| `yaw_only_mean_planar_speed_m_s` | `0.0889` | `0.0800` | fail |
| `wrong_planar_sign_count` | `10` | `0` | fail |
| `wrong_yaw_sign_count` | `0` | `0` | pass |

Dominant failure group: `mixed_vx_vy`

## Command-class decomposition

| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | yaw exceed | off-axis exceed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stop` | 1 | `0.0001` | `0.0003` | 0 | 0 | 0 | 0 | 0 |
| `pure_vx` | 4 | `0.1198` | `0.0382` | 2 | 0 | 2 | 0 | 0 |
| `pure_vy` | 4 | `0.0698` | `0.0953` | 0 | 0 | 0 | 0 | 0 |
| `pure_yaw` | 4 | `0.0891` | `0.1827` | 0 | 0 | 0 | 2 | 4 |
| `mixed_vx_vy` | 16 | `0.2432` | `0.1105` | 14 | 7 | 16 | 2 | 3 |
| `mixed_vx_yaw` | 6 | `0.1288` | `0.2463` | 4 | 0 | 4 | 4 | 0 |

## Telemetry by command class

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stop` | `0.5473` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pure_vx` | `0.4489` | `1.1402` | `0.0416` | `1.1396` | `7.0734` |
| `pure_vy` | `0.0137` | `1.0459` | `0.0403` | `1.0475` | `6.9455` |
| `pure_yaw` | `0.8243` | `2.7146` | `0.0740` | `2.7004` | `1.5963` |
| `mixed_vx_vy` | `0.3556` | `1.0824` | `0.0889` | `1.0875` | `6.2554` |
| `mixed_vx_yaw` | `0.4887` | `0.9807` | `0.0644` | `0.9902` | `6.8276` |

## Worst commands

| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | Planar err | Yaw err | Score |
| --- | --- | --- | ---: | ---: | ---: |
| `mixed_vx_vy` | (`-0.250`, `-0.150`, `0.000`) | (`-0.053`, `0.106`, `0.235`) | `0.3235` | `0.2348` | `11.84` |
| `mixed_vx_vy` | (`0.250`, `0.150`, `0.000`) | (`-0.031`, `-0.045`, `-0.012`) | `0.3424` | `0.0117` | `11.73` |
| `mixed_vx_vy` | (`0.250`, `-0.075`, `0.000`) | (`-0.081`, `-0.017`, `0.098`) | `0.3356` | `0.0975` | `11.50` |
| `mixed_vx_vy` | (`-0.250`, `0.150`, `0.000`) | (`-0.010`, `-0.039`, `-0.012`) | `0.3054` | `0.0117` | `9.33` |
| `mixed_vx_vy` | (`0.250`, `0.075`, `0.000`) | (`-0.005`, `-0.039`, `-0.037`) | `0.2790` | `0.0371` | `7.82` |
| `mixed_vx_vy` | (`0.250`, `-0.150`, `0.000`) | (`0.004`, `-0.196`, `-0.176`) | `0.2499` | `0.1757` | `7.02` |
| `mixed_vx_vy` | (`-0.250`, `0.075`, `0.000`) | (`-0.002`, `-0.007`, `0.009`) | `0.2613` | `0.0093` | `6.83` |
| `mixed_vx_vy` | (`0.125`, `-0.075`, `0.000`) | (`-0.090`, `0.016`, `0.135`) | `0.2336` | `0.1348` | `5.91` |

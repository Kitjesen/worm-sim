# Worm V6 Paper Results Summary

Only current-contract V6 artifacts are used for numeric paper claims. Existing stale eval/scan files are listed as `stale` with blank metric cells until they are regenerated under the current observation, control timing, and actuator contract.

## Figures

- [Fixed mode speed](fixed_mode_speed.svg)
- [Blend scan speed](blend_scan_speed.svg)
- [Paper claim analysis](paper_claims.md)

## Fixed Modes

| Terrain | Mode | RL speed mm/s | Success | Slip proxy | RL action/m | CMA-ES speed mm/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| flat | worm | 17.196 | 1.000 | 0.940 | 22422.792 | 31.940 | RL done, CMA-ES done |
| flat | snake | 17.944 | 1.000 | 0.910 | 16909.708 | 86.240 | RL done, CMA-ES done |
| flat | mixed | 10.451 | 0.800 | 0.962 | 1017909077.175 | 200.590 | RL done, CMA-ES done |
| sand | worm |  |  |  |  | 18.160 | RL stale, CMA-ES done |
| sand | snake |  |  |  |  | 5.800 | RL stale, CMA-ES done |
| sand | mixed |  |  |  |  | 9.730 | RL stale, CMA-ES done |
| slope | worm |  |  |  |  | -25.590 | RL stale, CMA-ES done |
| slope | snake |  |  |  |  | -5.810 | RL stale, CMA-ES done |
| slope | mixed |  |  |  |  | 34.860 | RL stale, CMA-ES done |

## Robustness Eval

| Terrain | Mode | Robust speed mm/s | Robust success | Robust slip proxy | Robust termination | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| flat | worm | 13.770 | 1.000 | 0.950 | 0.000 | done |
| flat | snake | -14.457 | 0.000 | 1.000 | 0.000 | done |
| flat | mixed | 31.325 | 1.000 | 0.901 | 0.000 | done |
| sand | worm |  |  |  |  | stale |
| sand | snake |  |  |  |  | stale |
| sand | mixed |  |  |  |  | stale |
| slope | worm |  |  |  |  | stale |
| slope | snake |  |  |  |  | stale |
| slope | mixed |  |  |  |  | stale |

## Best Current-Contract Blend Scan

| Terrain | Policy | Best gait_blend | Speed mm/s | Success | Slip proxy | Action/m | Termination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flat | random | 0.750 | 24.832 | 1.000 | 0.912 | 22572.737 | 0.000 |

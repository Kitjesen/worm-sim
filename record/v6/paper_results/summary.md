# Worm V6 Paper Results Summary

## Figures

- [Fixed mode speed](fixed_mode_speed.svg)
- [Blend scan speed](blend_scan_speed.svg)
- [Paper claim analysis](paper_claims.md)

## Fixed Modes

| Terrain | Mode | RL speed mm/s | Success | Slip proxy | RL action/m | CMA-ES speed mm/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| flat | worm | 20.003 | 1.000 | 0.935 | 21158.266 | 31.940 | RL done, CMA-ES done |
| flat | snake | 16.612 | 1.000 | 0.918 | 23407.068 | 86.240 | RL done, CMA-ES done |
| flat | mixed | 5.445 | 0.800 | 0.983 | 578042.530 | 200.590 | RL done, CMA-ES done |
| sand | worm | 2.376 | 0.600 | 0.974 | 2076934416.559 | 18.160 | RL done, CMA-ES done |
| sand | snake | 4.964 | 0.400 | 0.978 | 179642.377 | 5.800 | RL done, CMA-ES done |
| sand | mixed | 3.749 | 0.600 | 0.979 | 2189056151.072 | 9.730 | RL done, CMA-ES done |
| slope | worm | -13.208 | 0.000 | 1.000 | 6609793015.003 | -25.590 | RL done, CMA-ES done |
| slope | snake | -10.439 | 0.800 | 0.895 | 297102881.760 | -5.810 | RL done, CMA-ES done |
| slope | mixed | -11.201 | 0.000 | 1.000 | 7859092471.170 | 34.860 | RL done, CMA-ES done |

## Robustness Eval

| Terrain | Mode | Robust speed mm/s | Robust success | Robust slip proxy | Robust termination | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| flat | worm | -0.399 | 0.600 | 0.991 | 0.000 | done |
| flat | snake | 8.665 | 0.400 | 0.933 | 0.000 | done |
| flat | mixed | -13.509 | 0.200 | 0.996 | 0.000 | done |
| sand | worm | 3.931 | 0.400 | 0.976 | 0.000 | done |
| sand | snake | 3.438 | 0.600 | 0.982 | 0.000 | done |
| sand | mixed | 4.311 | 0.400 | 0.976 | 0.000 | done |
| slope | worm | -24.403 | 0.000 | 1.000 | 0.600 | done |
| slope | snake | -42.730 | 0.000 | 1.000 | 0.800 | done |
| slope | mixed | -12.709 | 0.200 | 0.991 | 0.200 | done |

## Best Blend Scan

| Terrain | Policy | Best gait_blend | Speed mm/s | Success | Slip proxy | Action/m | Termination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flat | random | 0.500 | 16.657 | 1.000 | 0.939 | 24685.836 | 0.000 |
| sand | random | 1.000 | 4.880 | 0.600 | 0.978 | 974420834.014 | 0.000 |
| slope | random | 0.000 | -2.086 | 0.400 | 0.986 | 4938631714.022 | 0.000 |

# Worm V6 Paper Claim Analysis

## Terrain Mode Selection

| Terrain | Status | Best fixed mode | Fixed speed mm/s | Best gait_blend | Blend label | Blend speed mm/s | Delta vs fixed |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| flat | done | snake | 17.944 | 0.750 | mixed | 24.832 | 6.888 |
| sand | incomplete | None |  |  | None |  |  |
| slope | incomplete | None |  |  | None |  |  |

## Cross-Terrain Averages

| Policy | Speed mm/s | Success | Termination |
| --- | ---: | ---: | ---: |
| fixed worm | 17.196 | 1.000 | 0.000 |
| fixed snake | 17.944 | 1.000 | 0.000 |
| fixed mixed | 10.451 | 0.800 | 0.000 |
| adaptive best blend | 24.832 | 1.000 | 0.000 |

## Claim Assessments

| Claim | Status |
| --- | --- |
| Continuous gait_blend exposes terrain-dependent mode preferences. | incomplete |
| Adaptive best-blend random policy improves cross-terrain average speed over the best single fixed mode. | incomplete |
| Adaptive best-blend random policy improves or preserves cross-terrain success rate versus the best single fixed mode. | incomplete |
| Adaptive best-blend random policy improves or preserves cross-terrain termination rate versus the most stable single fixed mode. | incomplete |

## Cautions

- Current-contract cross-terrain claims are incomplete until these terrains have fresh eval and scan evidence: sand, slope.

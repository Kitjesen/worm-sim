# Worm V6 Paper Claim Analysis

## Terrain Mode Selection

| Terrain | Best fixed mode | Fixed speed mm/s | Best gait_blend | Blend label | Blend speed mm/s | Delta vs fixed |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| flat | worm | 20.003 | 0.500 | mixed | 16.657 | -3.345 |
| sand | snake | 4.964 | 1.000 | snake | 4.880 | -0.084 |
| slope | snake | -10.439 | 0.000 | worm | -2.086 | 8.353 |

## Cross-Terrain Averages

| Policy | Speed mm/s | Success | Termination |
| --- | ---: | ---: | ---: |
| fixed worm | 3.057 | 0.533 | 0.133 |
| fixed snake | 3.712 | 0.733 | 0.067 |
| fixed mixed | -0.669 | 0.467 | 0.000 |
| adaptive best blend | 6.484 | 0.667 | 0.000 |

## Claim Assessments

| Claim | Status |
| --- | --- |
| Continuous gait_blend exposes terrain-dependent mode preferences. | supported |
| Adaptive best-blend random policy improves cross-terrain average speed over the best single fixed mode. | supported |
| Adaptive best-blend random policy improves or preserves cross-terrain success rate versus the best single fixed mode. | not_supported |
| Adaptive best-blend random policy improves or preserves cross-terrain termination rate versus the most stable single fixed mode. | supported |

## Cautions

- flat: best continuous random-policy blend does not beat the best terrain-specific fixed policy speed.
- sand: best continuous random-policy blend does not beat the best terrain-specific fixed policy speed.
- Success rate is not improved; frame Goal 3 around average speed and termination stability, not success.

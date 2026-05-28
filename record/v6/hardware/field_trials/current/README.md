# Worm V6 Hardware Field Trial Package

This package prepares the real-robot validation step for the paper.
It contains templates and commands only; it is not proof of a hardware run.

The current audit is complete only after one validated CSV and one video
reference exist for each terrain: flat, sand, and slope.

## Terrain Plan

| Terrain | Bundle | Recommended gait_blend | Trial README |
| --- | --- | ---: | --- |
| flat | `record/v6/deploy_bundles/flat_random` | 0.500 | `record/v6/hardware/field_trials/current/flat/README.md` |
| sand | `record/v6/deploy_bundles/sand_random` | 1.000 | `record/v6/hardware/field_trials/current/sand/README.md` |
| slope | `record/v6/deploy_bundles/slope_random` | 0.000 | `record/v6/hardware/field_trials/current/slope/README.md` |

## Final Audit Command

```powershell
python src\v6\paper_status_v6.py --refresh-audit
```

## Hardware Status Command

```powershell
python src\v6\hardware_trial_status_v6.py --write-report
```

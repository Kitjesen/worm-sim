# Worm V6 Hardware Deploy Preflight

Complete: `true`
Observation ABI: `a2970eeac045b4838117f76b0381d9a4c0b0bf540a7b3834f64d700aa059fe55`

This is a deploy-chain readiness check. It is not real hardware validation evidence.

## Terrain Bundles

| Terrain | Status | gait_blend | Bundle | Max abs action |
| --- | --- | ---: | --- | ---: |
| flat | `ok` | 0.500 | `record/v6/deploy_bundles/flat_random` | 0.20000000298023224 |
| sand | `ok` | 1.000 | `record/v6/deploy_bundles/sand_random` | 0.20000000298023224 |
| slope | `ok` | 0.000 | `record/v6/deploy_bundles/slope_random` | 0.20000000298023224 |

## Global ABI Checks

| Check | Status | Detail |
| --- | --- | --- |
| obs_dim | `ok` | 80 |
| observation_columns | `ok` | 80 |
| raw_hardware_columns | `ok` | 84 |
| sensor_counts | `ok` | {'actuated_joints': 11, 'slide_joints': 6, 'yaw_joints': 5, 'segment_imus': 7} |

## Reproduce

```powershell
python src\v3\preflight_hardware_deploy_v6.py --strict
```

# Worm V6 Hardware Deploy Preflight

Complete: `false`
Observation ABI: `672933c397be678aeae9067d299f38bfe69c748729beeab948cb339bb727183b`
Actuator contract: `e0a3640f4c778e407838ee3fe70634c2277a5489349c2150f75f6ac9e9b3bdcf`

This is a deploy-chain readiness check. It is not real hardware validation evidence.

## Terrain Bundles

| Terrain | Status | gait_blend | Bundle | Max abs action |
| --- | --- | ---: | --- | ---: |
| flat | `ok` | 0.750 | `record/v6/deploy_bundles/flat_random` | 0.20000000298023224 |
| sand | `failed` | 0.500 | `record/v6/deploy_bundles/sand_random` | 0.20000000298023224 |
| slope | `failed` | 0.500 | `record/v6/deploy_bundles/slope_random` | 0.20000000298023224 |

## Global ABI Checks

| Check | Status | Detail |
| --- | --- | --- |
| obs_dim | `ok` | 80 |
| observation_columns | `ok` | 80 |
| raw_hardware_columns | `ok` | 84 |
| sensor_counts | `ok` | {'actuated_joints': 11, 'slide_joints': 6, 'yaw_joints': 5, 'segment_imus': 7} |
| actuator_contract | `ok` | e0a3640f4c778e407838ee3fe70634c2277a5489349c2150f75f6ac9e9b3bdcf |

## Reproduce

```powershell
python src\v6\preflight_hardware_deploy_v6.py --strict
```

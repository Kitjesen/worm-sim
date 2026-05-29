# Worm V6 Hardware Deploy Preflight

Complete: `false`
Observation ABI: `83bf1ec48b816810335fabc08800e2e11f551c7ce72e96a94431e6afa1840d59`
Actuator contract: `e0a3640f4c778e407838ee3fe70634c2277a5489349c2150f75f6ac9e9b3bdcf`

This is a deploy-chain readiness check. It is not real hardware validation evidence.

## Terrain Bundles

| Terrain | Status | Sample gait metadata | Bundle | Max abs action |
| --- | --- | ---: | --- | ---: |
| flat | `failed` | 0.750 | `record/v6/deploy_bundles/flat_random` |  |
| sand | `failed` | 0.500 | `record/v6/deploy_bundles/sand_random` |  |
| slope | `failed` | 0.500 | `record/v6/deploy_bundles/slope_random` |  |

## Global ABI Checks

| Check | Status | Detail |
| --- | --- | --- |
| obs_dim | `ok` | 80 |
| observation_columns | `ok` | 80 |
| raw_hardware_columns | `ok` | 85 |
| sensor_counts | `ok` | {'actuated_joints': 11, 'slide_joints': 6, 'yaw_joints': 5, 'segment_imus': 7} |
| actuator_contract | `ok` | e0a3640f4c778e407838ee3fe70634c2277a5489349c2150f75f6ac9e9b3bdcf |

## Errors

- flat: runtime_predict: 'cmd_vel_norm'
- sand: runtime_predict: 'cmd_vel_norm'
- slope: runtime_predict: 'cmd_vel_norm'

## Reproduce

```powershell
python src\v6\preflight_hardware_deploy_v6.py --strict
```

# V61 yaw activity scaling continuation

## Goal

Continue from `flat_random_v60_low_yaw_envelope_from_v59best` and make the
flat low-yaw envelope track commanded yaw magnitude instead of only yaw sign.

## Current V60 evidence

Run:
`runs/worm_v6_ppo_flat_random_v60_low_yaw_envelope_from_v59best`

Best checkpoint:
`best_timestep = 1078016`

Full fixed schedule:

- `wrong_planar_sign_count = 0`
- `wrong_yaw_sign_count = 0`
- `planar_success_rate = 0.8095`
- `yaw_success_rate = 0.8095`
- `planar_velocity_rmse_m_s = 0.1331`
- `yaw_rate_rmse_rad_s = 0.1239`
- `straight_violation_count = 3`
- `stationary_violation_count = 4`

Low-yaw envelope scan:
`v60_low_yaw_envelope_scan_6s.json`

- `num_commands = 13`
- `planar_rmse_m_s = 0.0488`
- `yaw_rmse_rad_s = 0.2645`
- `planar_sign_rate = 1.0`
- `yaw_sign_rate = 1.0`
- `zero_command_mean_speed_m_s = 0.000015`
- `yaw_only_mean_planar_speed_m_s = 0.0723`
- `yaw_error_exceed_count = 4`

Interpretation:

- V60 solved sign separation in the low-yaw envelope.
- V60 still overdrives pure yaw magnitude: commands around `0.10-0.125 rad/s`
  produce about `0.40 rad/s`.
- Pure yaw still carries planar drift of about `0.06-0.08 m/s`.

## V61 change

Keep the deployable ABI unchanged:

- observation: 80D
- action: 12D = 11D residual motor action + 1D learned latent gait gate

Change only the deterministic action adapter:

```text
old pure-yaw activity:
  s_c = 1

new pure-yaw activity:
  s_c = clip(0.20 + 0.80 * |cmd_yaw_norm|, 0, 1)
```

This keeps full in-place yaw authority at the maximum yaw command, but prevents
small yaw commands from saturating the yaw prior.

## Test and train sequence

1. Run reward/observation/metric tests.
2. Run no-retrain low-yaw envelope scan using V60 best under the V61 adapter.
3. If yaw RMSE improves without losing yaw sign, start:

```powershell
python src\v6\train_v6.py `
  --terrain flat `
  --gait-mode random `
  --run-label flat_random_v61_yaw_activity_scaled_from_v60best `
  --command-curriculum low_yaw_envelope `
  --timesteps 1400000 `
  --train-chunk-timesteps 200000 `
  --resume runs\worm_v6_ppo_flat_random_v60_low_yaw_envelope_from_v59best\best_model.zip `
  --allow-contract-resume `
  --allow-curriculum-resume `
  --policy-net-arch 512,256,128 `
  --value-net-arch 512,256,128 `
  --learning-rate 3e-5 `
  --directional-eval-freq-steps 20000 `
  --directional-eval-seconds 6.0 `
  --n-envs 4 `
  --device cpu
```

## V61 acceptance target

Low-yaw envelope:

- `wrong_planar_sign_count == 0`
- `wrong_yaw_sign_count == 0`
- `planar_rmse_m_s <= 0.06`
- `yaw_rmse_rad_s <= 0.12`
- `yaw_only_mean_planar_speed_m_s <= 0.05`
- `zero_command_mean_speed_m_s <= 0.02`

Do not claim continuous full omnidirectional tracking until mixed vx/vy and
vx/yaw commands also pass the broader scan.

## V62 update: linear pure-yaw activity

The V61 trained run did not materially improve over the no-retrain adapter
scan:

- V61 best low-yaw scan: `yaw_rmse_rad_s = 0.1721`
- V61 final low-yaw scan: `yaw_rmse_rad_s = 0.1774`
- Pure-yaw planar drift stayed near `0.07 m/s`

A no-retrain ablation on V61 final showed that reducing the pure-yaw activity
floor is more effective than reducing the slide component:

| yaw activity min | pure-yaw yaw RMSE | pure-yaw mean planar speed |
| --- | ---: | ---: |
| 0.20 | 0.194 | 0.072 |
| 0.10 | 0.174 | 0.067 |
| 0.05 | 0.163 | 0.065 |
| 0.00 | 0.144 | 0.060 |

Therefore V62/V33 changes the pure-yaw activity rule to:

```text
s_c = |cmd_yaw_norm|
```

The next train run should resume from V61 final or best with:

```powershell
python src\v6\train_v6.py `
  --terrain flat `
  --gait-mode random `
  --run-label flat_random_v62_yaw_linear_activity_from_v61final `
  --command-curriculum low_yaw_envelope `
  --timesteps 1600000 `
  --train-chunk-timesteps 200000 `
  --resume runs\worm_v6_ppo_flat_random_v61_yaw_activity_scaled_from_v60best\final_model.zip `
  --allow-contract-resume `
  --allow-curriculum-resume `
  --policy-net-arch 512,256,128 `
  --value-net-arch 512,256,128 `
  --learning-rate 3e-5 `
  --directional-eval-freq-steps 20000 `
  --directional-eval-seconds 6.0 `
  --n-envs 4 `
  --device cpu
```

## V63/V34 update: calibrated pure-yaw gain

V62 did not improve the low-yaw scan beyond the V33 no-retrain adapter result:

- V62 best low-yaw scan: `yaw_rmse_rad_s = 0.1276`
- V62 final low-yaw scan: `yaw_rmse_rad_s = 0.1314`
- V62 checkpoint `1451008` low-yaw scan: `yaw_rmse_rad_s = 0.1268`

A follow-up gain ablation on checkpoint `1451008` showed the pure-yaw primitive
is still overpowered even when `s_c = |cmd_yaw_norm|`:

| yaw activity gain | pure-yaw yaw RMSE | pure-yaw mean planar speed |
| --- | ---: | ---: |
| 1.00 | 0.144 | 0.059 |
| 0.85 | 0.111 | 0.053 |
| 0.70 | 0.073 | 0.043 |
| 0.60 | 0.046 | 0.035 |
| 0.50 | 0.023 | 0.028 |

The 13-command low-yaw table with `gain = 0.50` passed the first-stage
low-yaw target:

- `planar_rmse_m_s = 0.0502`
- `yaw_rmse_rad_s = 0.0209`
- `planar_sign_rate = 1.0`
- `yaw_sign_rate = 1.0`
- `zero_command_mean_speed_m_s = 0.000007`
- `yaw_only_mean_planar_speed_m_s = 0.0285`
- `wrong_planar_sign_count = 0`
- `wrong_yaw_sign_count = 0`

V34 therefore fixes the adapter rule as:

```text
s_c = 0.50 * |cmd_yaw_norm|
```

This is a low-yaw feasible-envelope setting. Do not use it to claim high-yaw
turn-rate tracking until a separate high-yaw scan passes.

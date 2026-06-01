# V63 mixed-composition training continuation

## Goal

Continue from the accepted V34 low-yaw feasible-envelope result and train a
flat-terrain policy that keeps low-yaw control stable while improving mixed
`vx/vy/yaw` command composition.

This stage is still a flat simulation stage. It must not be claimed as full
omnidirectional deployment until the broader scan and videos pass.

## Starting point

Selected V34 policy:

```text
runs/worm_v6_ppo_flat_random_v62_yaw_linear_activity_from_v61final/checkpoints/worm_v6_ppo_1451008_steps.zip
```

Paired VecNormalize:

```text
runs/worm_v6_ppo_flat_random_v62_yaw_linear_activity_from_v61final/checkpoints/worm_v6_ppo_vecnormalize_1451008_steps.pkl
```

V34 low-yaw scan:

```text
runs/worm_v6_ppo_flat_random_v62_yaw_linear_activity_from_v61final/v34_gain050_ckpt1451008_low_yaw_scan_6s.json
```

Accepted low-yaw evidence:

- `wrong_planar_sign_count = 0`
- `wrong_yaw_sign_count = 0`
- `planar_rmse_m_s = 0.0505`
- `yaw_rmse_rad_s = 0.0197`
- `zero_command_mean_speed_m_s = 0.00004`
- `yaw_only_mean_planar_speed_m_s = 0.0278`

## V63 training objective

Use the existing `mixed_composition_repair` curriculum to expose the policy to:

- pure stop
- pure axial `vx`
- pure lateral `vy`
- pure yaw
- diagonal planar commands
- forward/reverse plus yaw commands

The main target is reducing cross-axis interference while retaining the V34
low-yaw sign and magnitude behavior.

## Training command

```powershell
python src\v6\train_v6.py `
  --terrain flat `
  --gait-mode random `
  --run-label flat_random_v63_mixed_composition_from_v34ckpt1451008 `
  --command-curriculum mixed_composition_repair `
  --timesteps 1651008 `
  --train-chunk-timesteps 200000 `
  --resume runs\worm_v6_ppo_flat_random_v62_yaw_linear_activity_from_v61final\checkpoints\worm_v6_ppo_1451008_steps.zip `
  --allow-contract-resume `
  --allow-curriculum-resume `
  --policy-net-arch 512,256,128 `
  --value-net-arch 512,256,128 `
  --learning-rate 1e-5 `
  --directional-eval-freq-steps 20000 `
  --directional-eval-seconds 6.0 `
  --n-envs 4 `
  --device cpu
```

## Acceptance checks

V63 must be evaluated with two scans:

1. Low-yaw retention scan using the same 13-command V34 envelope.
2. Mixed-composition scan with diagonal planar and forward-yaw commands.

Stage target:

- Low-yaw scan:
  - `wrong_planar_sign_count == 0`
  - `wrong_yaw_sign_count == 0`
  - `yaw_rmse_rad_s <= 0.05`
  - `yaw_only_mean_planar_speed_m_s <= 0.04`
- Mixed scan:
  - `wrong_planar_sign_count == 0`
  - `wrong_yaw_sign_count == 0`
  - `fixed_lateral_speed_gate_passed == true`
  - `planar_rmse_m_s <= 0.10`
  - `yaw_rmse_rad_s <= 0.12`
  - `off_axis_exceed_count <= 2`

## Boundary for paper language

If V63 passes only the low-yaw retention scan, describe it as:

```text
low-yaw feasible-envelope tracking with mixed-command training in progress
```

If V63 passes both scans, describe it as:

```text
flat-terrain mixed planar/yaw command tracking prototype
```

Do not describe this as full continuous omnidirectional tracking until a dense
continuous command scan and long videos pass.

## V63 result and V64 follow-up

V63 trained for one 200k-step chunk from the V34 checkpoint using
`mixed_composition_repair`.

Run:

```text
runs/worm_v6_ppo_flat_random_v63_mixed_composition_from_v34ckpt1451008
```

Best checkpoint:

```text
runs/worm_v6_ppo_flat_random_v63_mixed_composition_from_v34ckpt1451008/best_model.zip
```

V63 retained the low-yaw envelope:

- `planar_rmse_m_s = 0.0483`
- `yaw_rmse_rad_s = 0.0196`
- `wrong_planar_sign_count = 0`
- `wrong_yaw_sign_count = 0`
- `yaw_only_mean_planar_speed_m_s = 0.0284`

The mixed low-yaw scan did not improve enough:

- `planar_rmse_m_s = 0.1605`
- `vx_error_exceed_count = 17`
- `planar_error_exceed_count = 16`
- `fixed_lateral_strict_gate_passed = true`

The largest errors are concentrated in commands such as
`vx = +/-0.25 m/s, vy = +/-0.15 m/s`, where the policy currently produces only
about `0.05-0.07 m/s` axial speed. This means the next stage should first prove
continuous tracking inside a physically reachable low-speed envelope rather
than continuing to optimize against an unreachable high-speed grid.

V64 therefore introduces the `feasible_mixed_low_speed` curriculum:

- `|vx| in [0.04, 0.10] m/s`
- `|vy| in [0.03, 0.075] m/s`
- `|yaw| in [0.08, 0.125] rad/s`

Training command:

```powershell
python src\v6\train_v6.py `
  --terrain flat `
  --gait-mode random `
  --run-label flat_random_v64_feasible_mixed_low_speed_from_v63best `
  --command-curriculum feasible_mixed_low_speed `
  --timesteps 1731008 `
  --train-chunk-timesteps 200000 `
  --resume runs\worm_v6_ppo_flat_random_v63_mixed_composition_from_v34ckpt1451008\best_model.zip `
  --allow-contract-resume `
  --allow-curriculum-resume `
  --policy-net-arch 512,256,128 `
  --value-net-arch 512,256,128 `
  --learning-rate 1e-5 `
  --directional-eval-freq-steps 20000 `
  --directional-eval-seconds 6.0 `
  --n-envs 4 `
  --device cpu
```

V64 acceptance target:

- `wrong_planar_sign_count == 0`
- `wrong_yaw_sign_count == 0`
- `planar_rmse_m_s <= 0.08`
- `yaw_rmse_rad_s <= 0.05`
- `fixed_lateral_strict_gate_passed == true`
- `off_axis_exceed_count <= 1`

## V64/V65/V66 results

Current selected candidate:

```text
runs/worm_v6_ppo_flat_random_v64_feasible_mixed_low_speed_from_v63best/best_model.zip
```

V64 best feasible mixed scan:

- `planar_rmse_m_s = 0.0836`
- `yaw_rmse_rad_s = 0.0201`
- `wrong_planar_sign_count = 2`
- `wrong_yaw_sign_count = 0`
- `fixed_lateral_strict_gate_passed = true`
- `off_axis_exceed_count = 2`

V64 best low-yaw retention scan:

- `planar_rmse_m_s = 0.0496`
- `yaw_rmse_rad_s = 0.0191`
- `wrong_planar_sign_count = 0`
- `wrong_yaw_sign_count = 0`

Failure diagnosis:

- The remaining errors are not yaw errors.
- The two wrong planar signs are the forward diagonal cases
  `vx=+0.10, vy=+/-0.075`.
- The robot reverses or nearly reverses the axial component while producing the
  requested lateral component.

V65 added `feasible_forward_diagonal_repair` to oversample those forward
diagonal cases. It was rejected because it degraded lateral-right performance:

- `planar_rmse_m_s = 0.0855`
- `wrong_planar_sign_count = 3`
- `fixed_lateral_strict_gate_passed = false`

V66 returned to the balanced `feasible_mixed_low_speed` curriculum with a lower
learning rate from V64 best. It was also rejected because it did not beat V64:

- `planar_rmse_m_s = 0.0846`
- `wrong_planar_sign_count = 3`
- `fixed_lateral_strict_gate_passed = true`

Next target:

V67 should keep V64 best as the resume point and change the selection/eval
schedule to match the feasible envelope. The current training best selector
still evaluates high-speed commands such as `vx=0.25, vy=0.15`, which are not
the V64/V65/V66 target and can select checkpoints that are not optimal for the
low-speed feasible-envelope claim.

## V67 feasible-envelope contract update

The global deployable command envelope is now the first-stage feasible range:

- `vx in [-0.10, 0.10] m/s`
- `vy in [-0.075, 0.075] m/s`
- `yaw_rate in [-0.125, 0.125] rad/s`

This is now used by the environment command normalization, training best-eval
schedule, and the default command scan grid. The 80D observation and 12D action
ABI are unchanged.

Adapter changes:

- V35 reduces pure-yaw activity gain from `0.50` to `0.25` after the yaw range
  was narrowed. This keeps pure-yaw physical activity close to the previously
  validated low-yaw envelope.
- V36 keeps left lateral primitive scale at `1.00` and sets right lateral
  primitive runtime scale to `0.75`. A `0.50` trial under-shot and flipped the
  right-lateral sign, while `1.00` over-produced right lateral speed.

No-retrain scans from V64 best:

| candidate | planar RMSE | yaw RMSE | wrong planar | wrong yaw | planar exceed | off-axis exceed | right lateral vy |
|---|---:|---:|---:|---:|---:|---:|---:|
| V35 yaw gain 0.25 | 0.0725 | 0.0195 | 0 | 0 | 2 | 3 | -0.1619 |
| V36 right scale 0.50 | 0.0755 | 0.0195 | 2 | 0 | 3 | 4 | +0.0236 |
| V36 right scale 0.75 | 0.0609 | 0.0195 | 0 | 0 | 1 | 3 | -0.1143 |

Current selected candidate:

```text
runs/worm_v6_ppo_flat_random_v64_feasible_mixed_low_speed_from_v63best/best_model.zip
```

Use it with the current V36 action adapter. The corresponding scan artifact is:

```text
runs/worm_v6_ppo_flat_random_v64_feasible_mixed_low_speed_from_v63best/v36_range_lateralright075_baseline_scan_6s.json
```

V67 PPO continuation was run from the same V64 best checkpoint with:

- `learning_rate = 5e-6`
- `train_chunk_timesteps = 100000`
- `directional_eval_freq_steps = 0`

Online directional eval was disabled because allocating the 21-case MuJoCo eval
environment hit local memory limits. Resume compatibility now treats
`eval_command` as part of the experimental contract override when
`--allow-contract-resume` is explicitly passed, so old high-speed eval tables do
not block deliberate command-envelope migration.

V67 checkpoint scans:

| candidate | planar RMSE | yaw RMSE | wrong planar | wrong yaw | planar exceed | off-axis exceed | right lateral vy |
|---|---:|---:|---:|---:|---:|---:|---:|
| ckpt 1611008 | 0.0677 | 0.0196 | 0 | 0 | 3 | 3 | -0.0926 |
| ckpt 1631008 | 0.0686 | 0.0196 | 1 | 0 | 1 | 3 | -0.0856 |
| ckpt 1651008 | 0.0615 | 0.0195 | 0 | 0 | 2 | 2 | -0.0857 |
| ckpt 1671008 | 0.0641 | 0.0195 | 0 | 0 | 1 | 1 | -0.1152 |
| ckpt 1691008 | 0.0634 | 0.0194 | 0 | 0 | 2 | 2 | -0.1295 |
| final | 0.0666 | 0.0193 | 0 | 0 | 3 | 2 | -0.0491 |

Conclusion:

The useful improvement came from correcting the feasible speed envelope and
adapter priors, not from the V67 PPO continuation. V67 did not clearly beat the
V36 no-retrain baseline. The next training step should avoid another broad
random continuation and instead use a lower learning rate plus a hard-case
curriculum focused on:

- reverse plus negative lateral commands;
- forward/yaw and reverse/yaw off-axis coupling;
- right-lateral speed overshoot without losing sign;
- keeping pure yaw stationary and non-compact.

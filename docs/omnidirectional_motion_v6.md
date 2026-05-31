# Worm V6 Omnidirectional Motion Plan

## Goal

Make the V6 deployable policy learn body-frame omnidirectional commands:

- `cmd_vx_m_s`: forward and reverse body-frame speed.
- `cmd_vy_m_s`: left and right body-frame lateral speed.
- `cmd_yaw_rad_s`: signed yaw-rate command.

The deployable ABI stays unchanged:

- Observation: 80D real-sensor-compatible state.
- Action: 12D policy output = 11D residual motor action + 1D learned latent gait gate.
- `gait_blend` is learned from the last policy action dimension unless fixed for evaluation.

## Implemented Code Changes

- Added command curricula in `src/v6/worm_env_v6.py`:
  - `straight`: signed `vx`, no `vy/yaw`.
  - `planar`: signed `vx` or signed `vy`, no yaw.
  - `yaw`: signed yaw, no translation.
  - `omni`: primitives covering forward, reverse, lateral left/right, yaw left/right, forward+yaw, diagonal, and random mixed commands.
- Updated reward contract to `omni_curriculum_gate_v5`.
- Added reward telemetry in `info`:
  - `body_vx_m_s`
  - `body_vy_m_s`
  - `body_yaw_rate_rad_s`
  - `planar_velocity_error_m_s`
  - `yaw_rate_error_rad_s`
  - `command_aligned_speed_m_s`
  - `off_axis_speed_m_s`
- Replaced yaw-only best-model selection with an omnidirectional schedule in `src/v6/train_v6.py`.
- Added planar direction gate plus yaw direction gate for persistent best model selection.
- Updated `src/v6/eval_v6.py` so success is measured in the commanded body-frame direction, not only world `-X` distance.

## Training Commands

Start with curriculum stages when debugging a new reward/policy:

```powershell
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum straight --timesteps 300000 --n-envs 4 --device cpu
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum planar --timesteps 600000 --n-envs 4 --device cpu --resume runs\worm_v6_ppo_flat_random\final_model.zip
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum yaw --timesteps 900000 --n-envs 4 --device cpu --resume runs\worm_v6_ppo_flat_random\final_model.zip
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum omni --timesteps 1200000 --n-envs 4 --device cpu --resume runs\worm_v6_ppo_flat_random\final_model.zip
```

Direct all-direction training entry:

```powershell
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum omni --timesteps 1000000 --n-envs 4 --device cpu
```

## Evaluation Commands

Evaluate each primitive direction explicitly:

```powershell
python src\v6\eval_v6.py --terrain flat --gait-mode random --cmd-vx 0.20 --cmd-vy 0.00 --cmd-yaw 0.0
python src\v6\eval_v6.py --terrain flat --gait-mode random --cmd-vx -0.20 --cmd-vy 0.00 --cmd-yaw 0.0
python src\v6\eval_v6.py --terrain flat --gait-mode random --cmd-vx 0.00 --cmd-vy 0.10 --cmd-yaw 0.0
python src\v6\eval_v6.py --terrain flat --gait-mode random --cmd-vx 0.00 --cmd-vy -0.10 --cmd-yaw 0.0
python src\v6\eval_v6.py --terrain flat --gait-mode random --cmd-vx 0.00 --cmd-vy 0.00 --cmd-yaw 0.5
python src\v6\eval_v6.py --terrain flat --gait-mode random --cmd-vx 0.00 --cmd-vy 0.00 --cmd-yaw -0.5
```

Key JSON metrics to inspect:

- `mean_body_vx_m_s`
- `mean_body_vy_m_s`
- `mean_yaw_rate_rad_s`
- `mean_commanded_planar_distance_mm`
- `mean_planar_tracking_error_m_s`
- `mean_yaw_tracking_error_rad_s`
- `planar_success_rate`
- `yaw_success_rate`
- `success_rate`

## Current Status

The all-direction training and evaluation pipeline is implemented and now uses
the strict 35-command scan as the acceptance gate. The latest V41-V59 flat line
keeps the 80D observation ABI and 12D residual-plus-gait-gate action ABI fixed,
with actor and critic networks set to `512-256-128`. The current first-stage
command box is `vx=+/-0.25 m/s`, `vy=+/-0.15 m/s`, and
`yaw=+/-0.25 rad/s`; the yaw range is narrowed while yaw-only drift is still
being repaired.

Current verified status:

- six primitive directions and fixed left/right lateral gates are usable;
- zero command is clean;
- yaw signs are generally separated;
- continuous `vx/vy/yaw` tracking is **not** accepted yet.

Latest rejected experiment:

```text
runs/worm_v6_ppo_flat_random_v59_yaw_preserve_hardcase_from_v56best/
record/v6/omni_v59_yaw_preserve_hardcase/strict_scan_analysis.md
```

V59 uses `omni_directional_offaxis_yaw_v28` and adds
`mixed_planar_yaw_preserve_repair` sampling from V56 best. It keeps the hard
mixed-planar cases from V58 but preserves yaw-only and mixed-yaw samples so yaw
does not collapse. The best strict scan reports `planar_rmse_m_s=0.1561`,
`yaw_rmse_rad_s=0.1887`, and zero wrong signs. The final scan passes the fixed
lateral strict gate but still has `planar_rmse_m_s=0.1591` and one wrong
planar sign. Until a scan passes the strict gate, the correct paper wording is
"weak omnidirectional prototype" or "six-direction primitive controller", not
continuous body-frame velocity tracking.

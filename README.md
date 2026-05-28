# Worm Sim

Deployable MuJoCo simulation and RL pipeline for a snake/worm multimodal robot.

The current main line is **V6 deployable bimodal locomotion**: one robot body can use worm-like peristaltic extension/contraction, snake-like lateral undulation, and continuous blends between them. The paper goal is to study which mode works best on flat ground, sand, and slopes, using only sensors that can exist on the real robot.

Repository: https://github.com/Kitjesen/worm-sim

## Current Main Claim

This is now a **snake + worm dual-mode robot project**, not only an open-loop worm gait demo.

The mode interface is:

| `gait_blend` | Mode | Meaning |
| ---: | --- | --- |
| `0.0` | `worm` | Peristaltic / extension-contraction dominant |
| `1.0` | `snake` | Serpentine / yaw undulation dominant |
| `0.0 < gait_blend < 1.0` | `mixed` | Continuous hybrid gait |
| sampled during training | `random` | Continuous mode-conditioned policy for blend scans and deployment |

The latest paper-facing implementation is under `src/v6/`.  Old
`src/v3/*_v6.py` paths are compatibility wrappers only; they are not the
current project identity.

## Why This Version Matters

The deployable RL policy is constrained to realistic inputs:

- velocity command, yaw-rate command, and `gait_blend`
- actuated joint encoder positions
- actuated joint encoder velocities
- previous action
- per-segment IMU projected gravity
- per-segment IMU angular velocity
- phase clock

The policy observation does **not** use:

- base linear velocity
- global position
- global yaw
- MuJoCo freejoint state
- external localization as a policy input

Reward calculation may still use simulator truth for training, but policy inference cannot.

## Robot And Actuator Contract

The V6 model has:

- 7 segment IMUs
- 11 actuated joints
- 6 slide joints for extension/contraction
- 5 yaw joints for snake-like bending
- 80-dimensional deployable observation vector

The current actuator contract is centralized in `src/v6/motor_contract_v6.py`:

| Group | Target range | MuJoCo actuator abstraction |
| --- | --- | --- |
| slide | `[-0.05, 0.0] m` | position servo, `kp=800 N/m`, force limit `50 N` |
| yaw | `[-1.57, 1.57] rad` | position servo, `kp=200 Nm/rad`, torque limit `20 Nm` |

The peristaltic actuation period is fixed at **1.0 s**. Control runs at 50 Hz.

These are deployability guards, not a final vendor motor model. Real hardware identification still needs speed limits, current/voltage limits, backlash, deadband, thermal derating, and controller PID details.

## Current Progress

Last reproducibility baseline before the V6 directory migration:
`07cdfeb Make deployable multimodal progress reproducible`.

Current status files:

- [current progress](record/v6/paper_results/current_progress.md)
- [results index](record/v6/paper_results/results_index.md)
- [completion audit](record/v6/paper_results/completion_audit.md)
- [observation contract](record/v6/paper_results/observation_contract.md)
- [observation source audit](record/v6/paper_results/observation_source_audit.md)
- [hardware validation summary](record/v6/paper_results/hardware_validation_summary.md)

Current status:

| Terrain | Mode | Current-contract PPO status |
| --- | --- | --- |
| flat | worm | trained to 1,015,808 steps |
| flat | snake | trained to 1,015,808 steps |
| flat | mixed | trained to 1,015,808 steps |
| flat | random | trained to 1,015,808 steps; deploy bundle, evals, and `gait_blend` scan generated |
| sand | worm | current-contract retraining in progress; 524,288 / 1,000,000 steps |
| sand | snake/mixed/random | old artifacts exist, retraining under current contract still needed |
| slope | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |

Important caution: cross-terrain summaries now mark stale sand/slope artifacts as `stale` and leave their metric cells blank. Final cross-terrain paper claims should wait until sand/slope retraining plus `deploy eval scan summary audit` are rerun under the current actuator contract.

## Current Effect Summary

The strongest current evidence is structural and flat-ground:

- Deployable observation contract passes audit.
- Flat `worm`, `snake`, and `mixed` fixed-mode PPO models have been retrained with the current 1 s peristaltic timing and actuator limits.
- Flat `random` has reached the same 1,015,808-step threshold and now has a deployable TorchScript bundle.
- On flat ground, the current random-policy blend scan is best at `gait_blend=0.75`: 24.832 mm/s, success 1.0. `gait_blend=1.0` is close at 24.116 mm/s, and pure worm `0.0` is weak at 1.876 mm/s.
- Flat robust eval is mixed: `mixed` is strongest under the current noise/saturation test at 31.325 mm/s, while `snake` fails robustly with negative speed.
- `sand/worm` current-contract retraining has reached 524,288 steps after the latest resume chunk; it is improved training evidence, but still below the 1M formal threshold.
- New training launches accept `--device auto/cpu/cuda`; the latest completed sand chunk was started by the older CPU-only command, while subsequent chunks can use CUDA for PPO network updates.
- Hardware pipeline templates and preflight checks exist; flat deploy preflight passes, but sand/slope fail until their current-contract bundles are regenerated. Real flat/sand/slope hardware logs are still pending.

Interim result tables and figures:

- [fixed-mode summary](record/v6/paper_results/fixed_mode_summary.csv)
- [fixed-mode speed figure](record/v6/paper_results/fixed_mode_speed.svg)
- [gait_blend scan summary](record/v6/paper_results/blend_scan_summary.csv)
- [gait_blend speed figure](record/v6/paper_results/blend_scan_speed.svg)
- [paper claim analysis](record/v6/paper_results/paper_claims.md)

## Viewable Videos

Representative committed videos:

- [1 s gait mode comparison](record/v6/videos/gait_modes_comparison_1s_720p.mp4)
- [worm mode](record/v6/videos/worm_v6_worm.mp4)
- [snake mode](record/v6/videos/worm_v6_snake.mp4)
- [combined/mixed mode](record/v6/videos/worm_v6_combined.mp4)
- [gait comparison 720p](record/v6/videos/gait_comparison_1280x720.mp4)
- [training arena 720p](record/v6/videos/training_arena_1280x720.mp4)
- [flat random preview](record/v6/videos/eval_flat_random.mp4)
- [sand random preview](record/v6/videos/eval_sand_random.mp4)
- [slope random preview](record/v6/videos/eval_slope_random.mp4)

Large 4K videos are intentionally not committed because ordinary GitHub repositories reject files over 100 MB without Git LFS.

## Trajectory Plots

Segment-level trajectory and time-history plots are under:

- [trajectory folder](record/v6/trajectory/)
- [flat trajectory report](record/v6/trajectory/flat_trajectory_report.md)
- [sand trajectory report](record/v6/trajectory/sand_trajectory_report.md)
- [slope trajectory report](record/v6/trajectory/slope_trajectory_report.md)

These plots include absolute positions, relative motion, and per-segment trajectories. Earlier plotting bugs that made all segments appear to start from zero should be treated as invalid; current trajectory outputs separate absolute and relative views.

## Quick Start

Install the usual Python/MuJoCo stack first:

```powershell
pip install mujoco stable-baselines3 gymnasium torch numpy matplotlib pandas
```

Check current project status:

```powershell
python src\v6\paper_status_v6.py --refresh-audit
```

Run the observation-source audit:

```powershell
python src\v6\audit_observation_sources_v6.py --strict
```

Train one formal chunk:

```powershell
python src\v6\run_paper_pipeline_v6.py --preset formal --stage train --terrain flat --train-modes random --timesteps 1000000 --train-chunk-timesteps 600000 --n-envs 4 --device auto --resume --resume-partial --max-records 1
```

Run post-training exports, evaluations, scans, summaries, and audit:

```powershell
python src\v6\run_paper_pipeline_v6.py --preset formal --stage deploy eval scan summary audit --timesteps 1000000 --n-envs 4 --resume
```

Run hardware preflight:

```powershell
python src\v6\preflight_hardware_deploy_v6.py --strict
```

Training device selection:

- `--device auto` uses CUDA for the PPO network when PyTorch sees a CUDA GPU, otherwise CPU.
- `--device cuda` fails fast if CUDA is unavailable.
- MuJoCo environment stepping is still CPU-bound, so a 3090 helps most when paired with enough CPU cores and a higher `--n-envs` setting.

## Main Scripts

| File | Purpose |
| --- | --- |
| `src/v6/worm_v6.py` | V6 robot/MJCF model and gait utilities |
| `src/v6/worm_env_v6.py` | Gymnasium environment with deployable observation design |
| `src/v6/motor_contract_v6.py` | actuator range and mapping contract |
| `src/v6/observation_contract_v6.py` | observation ABI and source contract |
| `src/v6/train_v6.py` | PPO training entry point |
| `src/v6/eval_v6.py` | fixed-mode and robust evaluation |
| `src/v6/run_terrain_experiments.py` | terrain/mode experiment runner |
| `src/v6/run_paper_pipeline_v6.py` | paper pipeline orchestrator |
| `src/v6/scan_gait_blend_v6.py` | continuous `gait_blend` scan |
| `src/v6/deploy_policy_v6.py` | export/replay deployable policy bundles |
| `src/v6/hardware_policy_runtime_v6.py` | runtime wrapper for hardware policy inference |
| `src/v6/check_controller_stream_v6.py` | controller stream validation |
| `src/v6/validate_hardware_log_v6.py` | hardware log schema and evidence validation |

## Hardware Validation Path

The intended hardware evidence path is:

1. Export deployable random-policy bundle for each terrain.
2. Run controller stream self-check.
3. Capture raw encoder/IMU/action stream on flat, sand, and slope.
4. Attach video evidence for each run.
5. Validate each log with `validate_hardware_log_v6.py`.
6. Summarize hardware status with `hardware_trial_status_v6.py` and `summarize_hardware_validation_v6.py`.

Templates and examples are committed under:

- [hardware folder](record/v6/hardware/)
- [current field trial manifest](record/v6/hardware/field_trials/current/hardware_trial_manifest.json)

## Verification Commands Used Recently

```powershell
python -m compileall -q src\v6
python src\v6\test_motor_contract_v6.py
python src\v6\test_deployable_obs_v6.py
python src\v6\test_observation_contract_v6.py
python src\v6\test_deploy_policy_v6.py
python src\v6\test_hardware_obs_builder_v6.py
python src\v6\test_hardware_policy_runtime_v6.py
python src\v6\test_hardware_preflight_v6.py
python src\v6\test_controller_stream_check_v6.py
python src\v6\audit_observation_sources_v6.py --strict
python src\v6\test_goal_audit_v6.py
python src\v6\test_resume_partial_v6.py
```

## Repository Layout

```text
docs/
  deployable_multimodal_v6_paper_plan.md
  hardware_validation_checklist_v6.md
record/v6/
  hardware/          hardware templates, preflight, trial status
  paper_results/     summaries, audits, figures, paper evidence
  trajectory/        segment trajectory plots and CSVs
  videos/            committed viewable videos
runs/
  worm_v6_ppo_flat_worm/
  worm_v6_ppo_flat_snake/
  worm_v6_ppo_flat_mixed/
src/v6/
  *_v6.py            current deployable simulation, RL, eval, hardware tools
```

## Legacy Work

V4 open-loop worm and pipe-crawling demos are still useful historical prototypes, but they are no longer the paper main line. The current paper target is the V6 deployable snake/worm multimodal RL pipeline.

## Remaining Work

- Retrain sand and slope policies under the current actuator contract.
- Regenerate all eval, robust eval, `gait_blend` scan, summary, and audit artifacts.
- Collect real hardware logs and videos on flat, sand, and slope.
- Rebuild final paper figures after the above are complete.

## Author

Hongsen Pang ([@Kitjesen](https://github.com/Kitjesen)), BSRL Lab.

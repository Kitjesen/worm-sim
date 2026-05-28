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

The current RL action path is a deployable residual policy:

- a phase/gait-blend gait prior generates the nominal worm/snake/mixed action
- PPO outputs a bounded normalized residual
- the deployed command is `clip(gait_prior + residual, -1, 1)`

This keeps the paper focus on a learnable, mode-conditioned policy while
avoiding a cold-start controller that has to rediscover the entire gait from
random saturated actions.

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

Last pre-residual baseline before the V6 directory migration:
`07cdfeb Make deployable multimodal progress reproducible`.

The current training contract is stricter than that baseline:

- reward contract: `forward_progress_v3`
- action adapter: `gait_prior_residual_v1`
- residual exploration: `low_noise_residual_v1`
- fixed eval command: `cmd_vel=0.025 m/s`, `cmd_yaw=0`, no command resampling

Older 1M-step artifacts that do not match these contracts are treated as stale
by the audit and should not be used for paper claims.

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
| flat | worm | current low-noise residual contract, 278,528 / 1,000,000 steps |
| flat | snake/mixed/random | old artifacts exist, retraining under current contract still needed |
| sand | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |
| slope | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |

Important caution: the audit now marks old flat/sand/slope 1M-step artifacts as
`stale` when their reward, action-adapter, actuator, timing, or residual
exploration contract does not match the current deployable line. Final
cross-terrain paper claims should wait until all modes are retrained plus
`deploy eval scan summary audit` are rerun under the current contracts.

## Current Effect Summary

The strongest current evidence is structural plus a fresh flat/worm smoke result:

- Deployable observation contract passes audit.
- The current policy observation is 80-D and remains limited to command,
  encoders, per-segment IMUs, previous action, and phase.
- The residual exploration scale was reduced after diagnosing random saturated
  residual actions as the cause of large negative training episode rewards.
- Fresh `flat/worm` low-noise residual training has reached 278,528 steps.
- Fixed-command robust 10 s eval of the early best `flat/worm` checkpoint:
  153.4 mm forward, 15.34 mm/s, 3.1 mm lateral drift, success 1.0.
- Fixed-command robust 10 s eval of the latest final checkpoint:
  186.2 mm forward, 18.62 mm/s, 17.2 mm lateral drift, success 1.0.
- Low-resolution video rollout of the early best checkpoint: 5.0 s, 110.1 mm
  forward, 22.0 mm/s, no termination.
- Low-resolution video rollout of the latest final checkpoint: 5.0 s, 121.7
  mm forward, 24.34 mm/s, no termination.
- Hardware pipeline templates and preflight checks exist; real flat/sand/slope
  hardware logs are still pending.

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
- [current flat/worm low-noise residual rollout](record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_65k_best_20260529.mp4)
- [current flat/worm low-noise final rollout](record/v6/videos/eval_flat_worm_reward_v3_prior_lownoise_278k_final_20260529.mp4)

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
python src\v6\run_paper_pipeline_v6.py --preset formal --stage train --terrain flat --train-modes worm --timesteps 1000000 --train-chunk-timesteps 100000 --n-envs 4 --device cpu --resume --resume-partial --max-records 1
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

- `--device cpu` is the current recommended default for SB3 MLP-PPO.
- `--device cuda` fails fast if CUDA is unavailable; `--device auto` selects CUDA when PyTorch detects it.
- MuJoCo environment stepping is still CPU-bound, and local CUDA testing showed poor MLP-PPO utilization, so a 3090 will not help linearly unless paired with enough CPU workers or a GPU-parallel simulator backend.

## Main Scripts

| File | Purpose |
| --- | --- |
| `src/v6/worm_v6.py` | V6 robot/MJCF model and gait utilities |
| `src/v6/worm_env_v6.py` | Gymnasium environment with deployable observation design |
| `src/v6/motor_contract_v6.py` | actuator range and mapping contract |
| `src/v6/observation_contract_v6.py` | observation ABI and source contract |
| `src/v6/train_v6.py` | PPO training entry point |
| `src/v6/eval_v6.py` | fixed-mode and robust evaluation |
| `src/v6/record_eval_video_v6.py` | low-overhead deterministic MP4 recorder |
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
python src\v6\test_reward_contract_v6.py
python src\v6\test_paper_pipeline_plan_v6.py
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

- Continue `flat/worm` from 278,528 to 1,000,000 current-contract steps.
- Retrain flat snake/mixed/random plus all sand and slope policies under the current reward/action/exploration contracts.
- Regenerate all eval, robust eval, `gait_blend` scan, summary, and audit artifacts.
- Collect real hardware logs and videos on flat, sand, and slope.
- Rebuild final paper figures after the above are complete.

## Author

Hongsen Pang ([@Kitjesen](https://github.com/Kitjesen)), BSRL Lab.

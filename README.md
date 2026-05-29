# Worm Sim

Deployable MuJoCo simulation and RL pipeline for a snake/worm multimodal robot.

The current main line is **V6 deployable bimodal locomotion**: one robot body can use worm-like peristaltic extension/contraction, snake-like lateral undulation, and continuous blends between them. The paper goal is to study which mode works best on flat ground, sand, and slopes, using only sensors that can exist on the real robot.

Repository: https://github.com/Kitjesen/worm-sim

## Current Main Claim

This is now a **snake + worm dual-mode robot project**, not only an open-loop worm gait demo.

The deployable command interface is body-frame velocity control:

| Command | Meaning |
| --- | --- |
| `cmd_vx_m_s` | forward/reverse body-frame velocity target |
| `cmd_vy_m_s` | lateral body-frame velocity target |
| `cmd_yaw_rad_s` | yaw-rate target |

`gait_blend` is no longer an externally commanded policy observation for the
main learned controller. PPO outputs an extra action gate that maps to the
continuous gait blend:

| `gait_blend` | Mode | Meaning |
| ---: | --- | --- |
| `0.0` | `worm` | Peristaltic / extension-contraction dominant |
| `1.0` | `snake` | Serpentine / yaw undulation dominant |
| `0.0 < gait_blend < 1.0` | `mixed` | Continuous hybrid gait |
| learned by policy | `random` | Autonomous mode selection from the 12th policy action |

The latest paper-facing implementation is under `src/v6/`.  Old
`src/v3/*_v6.py` paths are compatibility wrappers only; they are not the
current project identity.

## Why This Version Matters

The deployable RL policy is constrained to realistic inputs:

- body-frame `vx`, `vy`, and yaw-rate command
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

- a CMA-ES gait-anchor prior generates the nominal worm/snake/mixed action
- PPO outputs a bounded normalized residual plus a learned gait gate
- the deployed command is `clip(gait_prior + residual, -1, 1)`

This keeps the paper focus on a learnable, self-selected multimodal policy while
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

## Spring-Steel Strip Calibration

The full V6 training model still uses a fast equivalent slide-joint spring.
Real spring-steel strip flexibility is now modeled in a separate single-segment
calibration tool so it can be inspected and fitted without slowing PPO training.

Run a quick smoke calibration:

```powershell
python src\v6\spring_steel_calibration_v6.py --quick
```

Run the fuller displacement sweep:

```powershell
python src\v6\spring_steel_calibration_v6.py --out-dir record\v6\spring_steel_calibration_full --compressions-mm 0,5,10,15,20,25,30,35,40,45,50
```

The tool writes:

- `single_segment_steel_strip.xml`: generated MuJoCo cable-strip model
- `force_displacement.csv`: target compression, actual compression, and force
- `calibration_summary.json`: config and fitted stiffness values
- `calibration_report.md`: readable run report

Latest full-sweep smoke result:

- fit source: actual MuJoCo slide position, not the requested servo target
- linear stiffness with intercept: `160.433 N/m`
- zero-intercept equivalent stiffness for the V6 slide spring: `335.566 N/m`
- secant stiffness at max compression: `266.633 N/m`
- max target/actual compression mismatch: `3.125 mm`

This supports the current `300 N/m` V6 slide stiffness as a reasonable
placeholder, but it is not a final material-identification result. The cable
bend/twist parameters still need to be fitted against measured steel-strip
force-displacement data.

## Real Robot Parameter Identification

Generate bench-test templates for the physical robot:

```powershell
python src\v6\hardware_parameter_id_v6.py --write-templates
```

Templates are written under:

```text
record/v6/hardware/parameter_id
```

The first real file to collect should be the spring-steel force-displacement
CSV. After filling it with measured compression and load-cell force, fit the
equivalent V6 slide spring:

```powershell
python src\v6\hardware_parameter_id_v6.py --spring-csv record\v6\hardware\parameter_id\spring_steel_force_displacement_YYYYMMDD.csv
```

Collection details are in
[real robot parameter collection](docs/real_robot_parameter_collection_v6.md).
The detailed GitHub TODO plan is
[real robot experiment TODO](docs/real_robot_experiment_todo_v6.md).

## Current Progress

The latest contract correction was made after comparing PPO rollouts with the
stronger 4K CMA-ES gait-comparison video. That video is an open-loop CMA-ES
baseline, not a learned PPO policy. Its strongest flat full-combined gait is
`247.97 mm/s` from `runs/cmaes_speed_full/best_gait.json`.

The old PPO reward target was too conservative:

- old reward contract: `forward_progress_v3`
- old action adapter: `gait_prior_residual_v1`
- old fixed eval command: `cmd_vel=0.025 m/s`, `cmd_yaw=0`

That trained the policy to track roughly `25 mm/s`, so it was never a fair
attempt to beat the `247.97 mm/s` CMA-ES full-combined baseline.

The corrected current contract is:

- reward contract: `omni_auto_gate_v4`
- action adapter: `cmaes_tri_anchor_auto_gate_v2`
- residual policy scale: `0.35`
- command range: `cmd_vx in [-0.25, 0.25] m/s`,
  `cmd_vy in [-0.15, 0.15] m/s`, `cmd_yaw in [-0.5, 0.5] rad/s`
- gait anchors: `gait_blend=0.0` peristaltic, `0.5` full combined, `1.0`
  serpentine, with piecewise-linear blends between anchors
- best-model selection evaluates auto-blend left/straight/right yaw cases
  instead of only straight-line motion
- best-model selection now uses `directional_sign_gate_v1`: positive yaw
  commands must produce positive yaw delta, negative yaw commands must produce
  negative yaw delta, and straight commands must stay within a small yaw drift
  tolerance before a checkpoint can become `best_model`

Older PPO artifacts that do not match these contracts are treated as stale by
the audit and should not be used for paper claims.

Current status files:

- [current progress](record/v6/paper_results/current_progress.md)
- [flat directional reward v3 progress](record/v6/paper_results/flat_directional_reward_v3_progress_20260529.md)
- [results index](record/v6/paper_results/results_index.md)
- [completion audit](record/v6/paper_results/completion_audit.md)
- [observation contract](record/v6/paper_results/observation_contract.md)
- [observation source audit](record/v6/paper_results/observation_source_audit.md)
- [hardware validation summary](record/v6/paper_results/hardware_validation_summary.md)

Current status:

| Terrain | Mode | Current-contract PPO status |
| --- | --- | --- |
| flat | worm | old 1M low-speed run exists but is stale under the auto-gated contract |
| flat | random | previous high-speed random residual PPO reached ~311k / 1M under `high_speed_directional_v3`; it is now stale under `omni_auto_gate_v4` |
| flat | snake/mixed | previous videos are diagnostic only; fixed-mode ablation retraining still needed |
| sand | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |
| slope | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |

Important caution: the audit now marks old flat/sand/slope 1M-step artifacts as
`stale` when their reward, action-adapter, actuator, timing, or residual
exploration contract does not match the current deployable line. Final
cross-terrain paper claims should wait until all modes are retrained plus
`deploy eval scan summary audit` are rerun under the current contracts.

## Current Effect Summary

The strongest current evidence is structural plus a fresh corrected-prior smoke
result:

- Deployable observation contract passes audit.
- The current policy observation is 80-D and remains limited to command,
  encoders, per-segment IMUs, previous action, and phase.
- The current command slot is `vx/vy/yaw`; `gait_blend` is selected by the
  policy's 12th action and is exported only as an action-derived diagnostic.
- The old negative PPO results are explained: saturated residuals plus a
  low-speed `25 mm/s` reward target were the wrong training setup.
- The action prior now reuses the three CMA-ES anchors that generated the
  stronger comparison video.
- Fresh zero-residual `flat/worm` CMA-ES-anchor rollout: 5.0 s, `166.8 mm`
  forward, `33.35 mm/s`, `1.9 mm` lateral drift, no termination.
- Fresh zero-residual `flat/mixed` CMA-ES-anchor rollout: 5.0 s, `695.8 mm`
  forward, `139.16 mm/s`, `56.0 mm` lateral drift, no termination.
- Fresh zero-residual `flat/snake` CMA-ES-anchor rollout: 5.0 s, `340.4 mm`
  forward, `68.09 mm/s`, `80.3 mm` lateral drift, no termination.
- The corrected deployable prior is still slower than the raw 4K full-combined
  CMA-ES baseline (`247.97 mm/s`) because V6 now evaluates the anchors through
  the deployable 1 s phase clock instead of hidden simulator time.
- The previous `flat/random` residual PPO reached roughly `311k / 1M` steps
  under `high_speed_directional_v3`. Best balanced eval reward was `1946.40`
  at `194,688` steps. This is now stale for paper claims because the ABI moved
  to `vx/vy/yaw` commands and learned gait gating.
- Current fixed-blend PPO videos are not yet stronger than the 4K CMA-ES
  baseline. The 300k v3 straight-command speeds are worm `38.32 mm/s`, mixed
  `133.48 mm/s`, and snake `62.89 mm/s`; the local 4K CMA-ES comparison still
  shows the stronger raw full-combined gait at `247.97 mm/s`.
- Directional control remains the main failure. With mixed mode and
  `cmd_yaw=+0.5`, the 300k v3 policy still turns negative
  (`yaw_delta=-0.516 rad`); with `cmd_yaw=-0.5`, it turns negative
  (`yaw_delta=-0.383 rad`). This means right-turn behavior is present, but
  left/right command separation has not been learned.
- The next training run must start/continue under `omni_auto_gate_v4` and must
  pass the direction gate before saving a new best model.
- Hardware pipeline templates and preflight checks exist; real flat/sand/slope
  hardware logs are still pending.

Interim result tables and figures:

- [fixed-mode summary](record/v6/paper_results/fixed_mode_summary.csv)
- [fixed-mode speed figure](record/v6/paper_results/fixed_mode_speed.svg)
- [fixed-gate gait_blend ablation summary](record/v6/paper_results/blend_scan_summary.csv)
- [fixed-gate gait_blend ablation figure](record/v6/paper_results/blend_scan_speed.svg)
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
- [corrected flat/worm CMA-ES-anchor prior rollout](record/v6/videos/eval_flat_worm_cmaes_prior_20260529.mp4)
- [corrected flat/mixed CMA-ES-anchor prior rollout](record/v6/videos/eval_flat_mixed_cmaes_prior_20260529.mp4)
- [corrected flat/snake CMA-ES-anchor prior rollout](record/v6/videos/eval_flat_snake_cmaes_prior_20260529.mp4)
- [flat/worm high-speed PPO random-policy rollout](record/v6/videos/eval_flat_worm_ppo_highspeed_random_213k_best_20260529.mp4)
- [flat/mixed high-speed PPO random-policy rollout](record/v6/videos/eval_flat_mixed_ppo_highspeed_random_213k_best_20260529.mp4)
- [flat/snake high-speed PPO random-policy rollout](record/v6/videos/eval_flat_snake_ppo_highspeed_random_213k_best_20260529.mp4)
- [flat/mixed high-speed PPO left-yaw rollout](record/v6/videos/eval_flat_mixed_left_ppo_highspeed_random_213k_best_20260529.mp4)
- [flat/mixed high-speed PPO right-yaw rollout](record/v6/videos/eval_flat_mixed_right_ppo_highspeed_random_213k_best_20260529.mp4)
- [flat/worm v3 balanced PPO 300k rollout](record/v6/videos/eval_flat_worm_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4)
- [flat/mixed v3 balanced PPO 300k rollout](record/v6/videos/eval_flat_mixed_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4)
- [flat/snake v3 balanced PPO 300k rollout](record/v6/videos/eval_flat_snake_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4)
- [flat/mixed v3 balanced PPO 300k left-yaw rollout](record/v6/videos/eval_flat_mixed_left_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4)
- [flat/mixed v3 balanced PPO 300k right-yaw rollout](record/v6/videos/eval_flat_mixed_right_ppo_highspeed_random_v3_300k_balanced_best_20260529.mp4)
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
python src\v6\run_paper_pipeline_v6.py --preset formal --stage train --terrain flat --train-modes random --timesteps 1000000 --train-chunk-timesteps 100000 --n-envs 4 --device cpu --resume --resume-partial --max-records 1
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
- Local CUDA testing on this MLP-PPO task produced the expected SB3 warning about poor GPU utilization. The first CUDA chunk ran at about `333 it/s`; the CPU continuation of the same run reached about `362 it/s`.
- MuJoCo environment stepping is still CPU-bound, so a 3090 will not help linearly unless paired with enough CPU workers or a GPU-parallel simulator backend.

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
| `src/v6/scan_gait_blend_v6.py` | fixed-gate `gait_blend` ablation scan |
| `src/v6/deploy_policy_v6.py` | export/replay deployable policy bundles |
| `src/v6/hardware_policy_runtime_v6.py` | runtime wrapper for hardware policy inference |
| `src/v6/check_controller_stream_v6.py` | controller stream validation |
| `src/v6/validate_hardware_log_v6.py` | hardware log schema and evidence validation |
| `src/v6/hardware_parameter_id_v6.py` | real-robot parameter templates and spring-steel stiffness fitting |

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
python src\v6\test_hardware_parameter_id_v6.py
python src\v6\audit_observation_sources_v6.py --strict
python src\v6\test_goal_audit_v6.py
python src\v6\test_resume_partial_v6.py
python src\v6\test_cmaes_prior_v6.py
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
  cmaes_speed_peristaltic/
  cmaes_speed_full/
  cmaes_speed_serpentine/
  worm_v6_ppo_flat_random/
src/v6/
  *_v6.py            current deployable simulation, RL, eval, hardware tools
```

## Legacy Work

V4 open-loop worm and pipe-crawling demos are still useful historical prototypes, but they are no longer the paper main line. The current paper target is the V6 deployable snake/worm multimodal RL pipeline.

## Remaining Work

- Restart flat random residual PPO under `omni_auto_gate_v4` toward the 1M formal target.
- Improve yaw/directional command learning; current mixed-mode left command still turns with the wrong sign.
- Retrain flat worm/snake plus all sand and slope policies under the current reward/action contracts.
- Regenerate all eval, robust eval, fixed-gate ablation scan, summary, and audit artifacts.
- Collect real hardware logs and videos on flat, sand, and slope.
- Rebuild final paper figures after the above are complete.

## Author

Hongsen Pang ([@Kitjesen](https://github.com/Kitjesen)), BSRL Lab.

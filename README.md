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

The real slide drive is asymmetric: the servo pulls a rope to contract, while
the spring-steel strips passively return the segment during release. A reduced
unilateral force model is now captured in
`src/v6/unilateral_slide_actuator_v6.py` and tested by:

```powershell
python src\v6\test_unilateral_slide_actuator_v6.py
```

This model is the intended bridge from the current MuJoCo position-servo
abstraction to the Isaac Lab explicit actuator implementation.

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

## Isaac Lab Migration

Full migration to Isaac Lab is feasible, but it should be staged. MuJoCo stays
as the calibration/debug baseline, and Isaac Lab becomes the GPU RL target after
we prove articulation, observation, action, reward, and render parity.

The migration plan is tracked in
[Isaac Lab migration plan](docs/isaaclab_migration_plan_v6.md). It includes:

- the unilateral servo-rope plus passive spring-steel slide model;
- the same 80-D deployable observation contract;
- the same 12-D policy action with learned gait gate;
- a visual-only steel-strip rendering path for videos;
- a later optional deformable-strip experiment for single-segment calibration.

## Current Progress

### Latest Flat V64-V68 Status

The active flat baseline is now the V64 best checkpoint evaluated with the
current V36 feasible-speed adapter. The command envelope is intentionally
narrowed to a physical first-stage target: `vx in [-0.10, 0.10] m/s`,
`vy in [-0.075, 0.075] m/s`, and `yaw in [-0.125, 0.125] rad/s`.

The scan tool's default grid now matches the documented 35-command acceptance
surface when `--include-forward-yaw` is used. The latest default-guard scan is:

```text
record/current/flat_omni_v68_v50_default_guard_35cmd_scan/scan_35_commands_default_guard_6s.json
```

Key metrics: `num_commands=35`, `planar_rmse_m_s=0.05985`,
`yaw_rmse_rad_s=0.01880`, `wrong_planar_sign_count=1`,
`wrong_yaw_sign_count=0`, `zero_command_mean_speed_m_s=0.000048`, and
fixed left/right lateral strict gates pass. Strict analysis still rejects this
as a completed continuous tracker because the remaining wrong planar sign is in
the `mixed_vx_vy` group.

V50 componentwise mixed-command composition remains default-off. Re-enabling it
with `WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1` on the same checkpoint gives
`planar_rmse_m_s=0.10105` and `wrong_planar_sign_count=6`, so it is still an
ablation result rather than the accepted controller.

Training is being moved off the local Windows machine. The 3090 server has a
working `thunder2` environment with CUDA-visible PyTorch, MuJoCo, Gymnasium,
and Stable-Baselines3. New long runs should use a clean Git checkout under
`/home/bsrl/hongsenpang/codex_runs`; the older
`/home/bsrl/hongsenpang/worm_project` directory is a manual copy and should not
be treated as the authoritative training checkout.

### Historical Flat V41-V59 Status

The newest flat line is V41-V59. It keeps the deployable ABI fixed:

- observation remains 80D;
- policy action remains `11D residual + 1D learned latent gait gate`;
- actor and critic are both `512-256-128`;
- action adapter is `cmaes_tri_anchor_auto_gate_directional_v31`;
- latest reward contract is `omni_directional_offaxis_yaw_v28`;
- latest diagnostic curriculum is `mixed_planar_yaw_preserve_repair`.
- first-stage yaw command range is currently `+/-0.25 rad/s`, narrowed from
  the earlier `+/-0.5 rad/s` target while yaw-only stationarity is repaired.

V26 added searched left/right lateral primitives on the same 1 s deployable
phase clock. V27 keeps the same hardware-facing interface and fixes the
mixed-yaw prior sign so mixed `vx+yaw` commands follow the commanded yaw sign.

Best conservative flat baseline remains:

```text
runs/worm_v6_ppo_flat_random_lateral_primitives_v41_from_v40final/best_model.zip
record/current/flat_omni_v41_v26_scan_after40k/scan_35_commands_best.json
```

| Metric | Value |
| --- | ---: |
| `planar_rmse_m_s` | `0.1517` |
| `yaw_rmse_rad_s` | `0.1105` |
| `planar_sign_rate` | `1.00` |
| `yaw_sign_rate` | `1.00` |
| `fixed_lateral_strict_gate_passed` | `true` |

V47-V56 added stricter command-class proof, componentwise tracking cost,
mixed-composition repair sampling, scan telemetry, and a mixed-yaw sign adapter
fix. V50 also tested componentwise mixed-prior composition, but that path is
default-off because the smoke result regressed mixed-planar sign reliability.
V51/V52 then tested residual-authority and mixed-planar reward repairs. They
did not yet pass continuous tracking. V53 continued with a mixed-planar
curriculum, and V54 tested a prior/residual authority rebalance that is also
default-off because it regressed planar RMSE. V55 tested a split-channel
mixed-planar prior and rejected it because it reintroduced wrong planar signs.
V56 added full-speed diagonal commands to best-model selection; it preserves
signs but still fails planar RMSE and yaw-only drift. V57 added a
full-diagonal mixed-planar deficit reward and continued from V56 best, but the
strict scan regressed sign reliability. V58 then tested whether the issue was
only residual/prior scale and whether oversampling hard mixed diagonals would
repair the target class. The scale sweep worsened the scan, and the hardcase
curriculum restored zero sign errors but worsened yaw RMSE. V59 then added a
yaw-preserving hardcase curriculum and a stronger yaw-only stationarity cap.
It recovered yaw RMSE relative to V58 but still failed continuous planar
tracking. The latest strict result is:

```text
record/v6/omni_v59_yaw_preserve_hardcase/strict_scan_analysis.md
```

| Candidate | Planar RMSE | Yaw RMSE | Wrong planar signs | Wrong yaw signs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| V49 adapter scan | `0.1515` | `0.1196` | `0` | `0` | partial improvement, still rejected |
| V50 componentwise-prior final smoke | `0.1801` | `0.2075` | `4` | `0` | rejected |
| V50 default guard scan | `0.1498` | `0.2003` | `0` | `0` | baseline protected, still rejected |
| V51 residual-authority final | `0.1553` | `0.1907` | `0` | `0` | rejected |
| V52 mixed-planar reward final | `0.1534` | `0.1978` | `0` | `0` | rejected |
| V53 mixed-planar curriculum final | `0.1509` | `0.1697` | `0` | `0` | rejected |
| V54 authority-rebalance final | `0.1686` | `0.1714` | `0` | `0` | rejected, default-off |
| V54 default guard on V53 final | `0.1509` | `0.1697` | `0` | `0` | default protected, still rejected |
| V55 split-prior no-retrain | `0.2012` | `0.1697` | `8` | `0` | rejected, default-off |
| V55 split-prior train final | `0.2018` | `0.1761` | `7` | `0` | rejected, default-off |
| V56 strict-diagonal eval best | `0.1509` | `0.1646` | `0` | `0` | rejected |
| V57 full-diagonal reward best | `0.1527` | `0.1933` | `1` | `0` | rejected |
| V57 full-diagonal reward final | `0.1566` | `0.1760` | `1` | `0` | rejected |
| V58 hardcase curriculum best | `0.1566` | `0.2709` | `0` | `0` | rejected |
| V58 hardcase curriculum final | `0.1588` | `0.2720` | `0` | `0` | rejected |
| V59 yaw-preserve hardcase best | `0.1561` | `0.1887` | `0` | `0` | rejected |
| V59 yaw-preserve hardcase final | `0.1591` | `0.1895` | `1` | `0` | rejected |

Detailed proof note:

```text
docs/omni_v47_strict_command_tracking_proof.md
```

Current interpretation: six primitive signs and fixed lateral gates are good,
but continuous `vx/vy/yaw` tracking is **not solved yet**. The strict proof
shows the dominant failure is still mixed command composition:

- `mixed_vx_vy` cannot independently satisfy both planar components;
- `mixed_vx_yaw` improved after the V49 sign fix but still lacks enough yaw
  authority;
- V50's stronger componentwise prior is an ablation result, not the accepted
  default; enable it only with `WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1`;
- V51/V52 keep the safer prior path and improve yaw RMSE/sign preservation,
  but mixed `vx/vy` planar RMSE remains too high;
- V53 improves yaw RMSE with a mixed-planar curriculum but still fails planar
  RMSE and yaw-only drift;
- V54's mixed-planar authority rebalance is an ablation result, not the
  accepted default; enable it only with
  `WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE=1`;
- V55's split-channel mixed planar prior is also an ablation result; enable it
  only with `WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR=1`;
- V57 confirms that reward pressure alone did not increase mixed-planar
  residual authority enough; `mixed_vx_vy` remains the dominant failure group;
- V58 confirms that no-retrain residual/prior scaling is not enough; the
  hardcase curriculum restores sign reliability but increases yaw RMSE and
  still leaves full-speed reverse diagonals weak;
- V59 confirms that preserving yaw samples during hardcase training avoids the
  severe V58 yaw regression, but full-speed mixed `vx/vy` composition is still
  not solved;
- V56/V59 remain diagnostic candidates rather than accepted trackers until a
  candidate passes the strict scan gate.

The detailed movement table is in
[V41-V59 movement summary](docs/omni_v41_v46_motion_summary.md).

### Latest Flat V29 Status

V29 is kept below as historical context. It keeps the same
deployable ABI but updates the learning setup:

- observation remains 80D;
- policy action remains `11D residual + 1D learned latent gait gate`;
- actor and critic are both `512-256-128`;
- action adapter is `cmaes_tri_anchor_auto_gate_directional_v23`;
- the accepted V29 checkpoint was trained under
  `omni_directional_offaxis_yaw_v16`.

The V23 gait gate is speed-adaptive for axial commands: slow forward/reverse
commands stay closer to the worm/peristaltic center, while full-speed
forward/reverse commands return to the faster mixed center. The current valid
flat candidate is:

```text
runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip
```

Fixed-command results:

| Command | Measured response | Threshold | Status |
| --- | --- | --- | --- |
| forward | `body_vx = 0.1384 m/s` | `> 0.12` | pass |
| reverse | `body_vx = -0.0810 m/s` | `< -0.03` | pass |
| lateral_left | `body_vy = 0.0328 m/s` | `> 0.03` | pass, but high off-axis |
| lateral_right | `body_vy = -0.0296 m/s` | `< -0.03` | borderline fail |
| yaw_left | `yaw_rate = 0.3768 rad/s` | `> 0.03` | pass, but translates |
| yaw_right | `yaw_rate = -0.4206 rad/s` | `< -0.03` | pass, but translates |

V29 is not a completed continuous omnidirectional tracker. The 35-command scan
still reports `planar_rmse_m_s=0.1665`, `yaw_rmse_rad_s=0.1179`,
`planar_sign_rate=0.84`, and yaw-only planar drift of about `0.0966 m/s`.
Until lateral off-axis motion and yaw-only translation are solved, this should
be described as a **six-direction primitive controller / weak omnidirectional
prototype**, not final arbitrary velocity tracking.

Detailed notes and exact artifact paths are in
[V29 training log](docs/omni_v29_training_log.md).
The latest HD visual pass is tracked in
[V29 HD recording log](docs/omni_v29_hd_recording_log.md).
The longer 15 s visual pass is tracked in
[V29 15 s recording log](docs/omni_v29_long15_recording_log.md).
The organized video entry point is
[video index](record/VIDEO_INDEX.md).

After the HD pass, V30-V33 were run as short flat repair experiments. They did
not beat V29:

| Version | Main change | Result |
| --- | --- | --- |
| V30 | stronger lateral/yaw stationary penalties plus pure lateral/yaw oversampling | worsened `forward_yaw_left` to wrong-sign |
| V31 | experimental mixed planar+yaw action prior | worsened both forward-yaw signs; reverted |
| V32 | targeted mixed-yaw curriculum sampling | restored zero wrong-yaw at best, but tracking did not improve |
| V33 | V32 continuation with `--learning-rate 1e-4` | did not solve drift and regressed to one wrong-yaw case |
| V34 | low-LR continuation from V29 best | did not beat V29; yaw success improved but one wrong-yaw case returned |
| V35 | axis-separation repair from V29 best | fixed yaw-only prior scaling path; yaw-only drift improved, planar tracking still weak |
| V36 | continuous rejoin from V35 final | yaw RMSE improved, planar RMSE worsened; diagnostic only |
| V37 | worm-centered lateral prior from V36 final | planar sign improved to `0.96`, but lateral speed remained too weak |
| V38 | lateral speed-deficit reward from V37 final | yaw-only planar drift improved to `0.060 m/s`, but planar RMSE worsened |

The retained post-V29 code changes are the targeted mixed-yaw sampling contract
(`omni_directional_offaxis_yaw_v18`) and configurable PPO learning rate. V29 is
now superseded as the active development line by V41-V59 diagnostics, although
V41 is still only a weak omnidirectional prototype rather than accepted
continuous velocity tracking. See
[V30-V33 repair log](docs/omni_v30_v33_repair_log.md).
The latest V34 continuation is recorded in
[V34 training log](docs/omni_v34_training_log.md).
The V35/V36 axis-separation diagnostic is recorded in
[V35/V36 training log](docs/omni_v35_v36_axis_separation_log.md).
The V37/V38 lateral repair diagnostic is recorded in
[V37/V38 training log](docs/omni_v37_v38_lateral_repair_log.md).

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

The corrected current contract line is now:

- reward contract: `omni_directional_offaxis_yaw_v10`
- action adapter: `cmaes_tri_anchor_auto_gate_directional_v12`
- residual policy scale: `0.35`
- command range: `cmd_vx in [-0.25, 0.25] m/s`,
  `cmd_vy in [-0.15, 0.15] m/s`, `cmd_yaw in [-0.5, 0.5] rad/s`
- gait anchors: `gait_blend=0.0` peristaltic, `0.5` full combined, `1.0`
  serpentine, with piecewise-linear blends between anchors
- best-model selection evaluates auto-blend left/straight/right yaw cases
  instead of only straight-line motion
- best-model selection now uses `omni_tracking_gate_v2`: planar and yaw command
  signs must be correct, `planar_success_rate >= 0.85`,
  `yaw_success_rate >= 0.70`, and zero-yaw commands may have at most one yaw
  drift violation before a checkpoint is formally accepted.
- v9/v10 lateral prior correction: both lateral commands use the same `+pi/2`
  slide phase offset, and right-lateral mirrors only the yaw-anchor sign. This
  keeps the action adapter deployable because it depends only on `obs[0:3]`
  command and phase clock.
- v10 also raises the lateral prior floor to `0.80` and adds extra pure
  right-yaw authority. This improves the fixed 6 s demonstration without
  changing the deployable ABI.
- v12 adds zero-yaw heading trims to the command-conditioned prior: axial
  forward/reverse commands get a phase offset, and lateral commands get a
  small yaw trim. This is still deployable because it depends only on
  `obs[0:3]` command and phase clock, and it keeps the 80D observation plus
  12D policy-action ABI unchanged.

Older PPO artifacts that do not match these contracts are treated as stale by
the audit and should not be used for paper claims.

### Historical Flat V12 Omni Training Status

V12 is kept as a historical flat six-direction gate result. It is no longer the
active current line because the adapter/reward contracts have moved on to
V26/V23 and the current target is continuous `vx/vy/yaw` tracking.

The historical flat candidate used:

```text
runs/worm_v6_ppo_flat_random_heading_omni_v11_from_hhbest/best_model.zip
```

Formal directional gate summary:

```text
runs/worm_v6_ppo_flat_random_heading_omni_v11_from_hhbest/v12_directional_eval_summary.json
```

| Metric | Value | Required | Status |
| --- | ---: | ---: | --- |
| `wrong_planar_sign_count` | `0` | `0` | pass |
| `wrong_yaw_sign_count` | `0` | `0` | pass |
| `planar_success_rate` | `1.00` | `>= 0.85` | pass |
| `yaw_success_rate` | `0.875` | `>= 0.70` | pass |
| `straight_violation_count` | `1` | `<= 1` | pass |

Fixed 6 s command-threshold evidence:

| Command | Measured response | Required | Status |
| --- | --- | --- | --- |
| forward | `body_vx = 0.1886 m/s` | `> 0.12` | pass |
| reverse | `body_vx = -0.0767 m/s` | `< -0.03` | pass |
| lateral_left | `body_vy = 0.0376 m/s` | `> 0.03` | pass |
| lateral_right | `body_vy = -0.0393 m/s` | `< -0.03` | pass |
| yaw_left | `yaw_rate = 0.0324 rad/s` | `> 0.03` | pass |
| yaw_right | `yaw_rate = -0.0379 rad/s` | `< -0.03` | pass |

Current artifacts:

- `record/v6/omni_v12_heading_prior_artifacts/*.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/*_trajectory.png`
- `record/v6/omni_v12_heading_prior_artifacts/*_trajectory.csv`
- `record/v6/omni_v12_heading_prior_artifacts/*_eval.json`
- `record/v6/omni_v12_heading_prior_artifacts/gait_comparison_v12_flat_6cmd.mp4`
- `record/v6/omni_v12_heading_prior_artifacts/v12_flat_6cmd_summary.json`

Remaining flat limitations: lateral commands still have large off-axis forward
motion, yaw-only commands drift translationally, and the current improvement
comes from a command-conditioned V12 prior plus learned residual policy rather
than a fully retrained V12 long run. Do not make sand/slope claims until these
artifacts are repeated with robust tests and terrain transfer.

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
| flat | random | current V41 best passes correct signs and fixed lateral strict gates, but continuous `vx/vy/yaw` tracking is not accepted |
| flat | snake/mixed | previous videos are diagnostic only; fixed-mode ablation retraining still needed |
| sand | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |
| slope | worm/snake/mixed/random | old artifacts exist, retraining under current contract still needed |

Important caution: the audit now marks old flat/sand/slope 1M-step artifacts as
`stale` when their reward, action-adapter, actuator, timing, or residual
exploration contract does not match the current deployable line. Final
cross-terrain paper claims should wait until all modes are retrained plus
`deploy eval scan summary audit` are rerun under the current contracts.

## Current Effect Summary

The strongest current evidence is structural plus the latest flat V41-V59
repair line:

- Deployable observation contract passes audit.
- The current policy observation is 80-D and remains limited to command,
  encoders, per-segment IMUs, previous action, and phase.
- The current command slot is `vx/vy/yaw`; `gait_blend` is selected by the
  policy's 12th action and is exported only as an action-derived diagnostic.
- V26 lateral primitives pass adapter prior-only lateral checks and make the
  fixed left/right lateral gates pass in the 35-command scan.
- The current best flat candidate is still a weak omnidirectional prototype:
  it has correct signs for planar and yaw commands, but not accepted continuous
  velocity tracking because `planar_rmse_m_s` remains about `0.15`.
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
- Historical directional control has a flat V12 six-direction gate pass, but
  the current V41-V59 line should still be described as weak omnidirectional
  control until the 35-command continuous tracking gate passes.
- The next training/evaluation work should not jump straight to paper claims:
  rerun robust flat tests, compare learned gate against fixed worm/mixed/snake
  gates, and then transfer the same ABI to sand and slope.
- Hardware pipeline templates and preflight checks exist; real flat/sand/slope
  hardware logs are still pending.

Interim result tables and figures:

- [fixed-mode summary](record/v6/paper_results/fixed_mode_summary.csv)
- [fixed-mode speed figure](record/v6/paper_results/fixed_mode_speed.svg)
- [fixed-gate gait_blend ablation summary](record/v6/paper_results/blend_scan_summary.csv)
- [fixed-gate gait_blend ablation figure](record/v6/paper_results/blend_scan_speed.svg)
- [paper claim analysis](record/v6/paper_results/paper_claims.md)

## Viewable Videos

Current flat V29 videos with visual spring-steel strips and head speed overlay:

- [flat V29 long 15 s 3x2 UHD comparison, 3840x2160](record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_4k_uhd.mp4)
- [flat V29 long 15 s 3x2 comparison, 3840x1440](record/current/flat_omni_v29_long15/gait_comparison_3x2_15s_3840x1440.mp4)
- [flat V29 long forward](record/current/flat_omni_v29_long15/forward_15s_1080p.mp4)
- [flat V29 long reverse](record/current/flat_omni_v29_long15/reverse_15s_1080p.mp4)
- [flat V29 long lateral left](record/current/flat_omni_v29_long15/lateral_left_15s_1080p.mp4)
- [flat V29 long lateral right](record/current/flat_omni_v29_long15/lateral_right_15s_1080p.mp4)
- [flat V29 long yaw left](record/current/flat_omni_v29_long15/yaw_left_15s_1080p.mp4)
- [flat V29 long yaw right](record/current/flat_omni_v29_long15/yaw_right_15s_1080p.mp4)
- [flat V29 short HD 3x2 comparison, 3840x1440](record/current/flat_omni_v29_hd/gait_comparison_3x2_3840x1440.mp4)
- [flat V29 short HD 3x2 UHD comparison, 3840x2160](record/current/flat_omni_v29_hd/gait_comparison_3x2_4k_uhd.mp4)
- [flat V29 six-command comparison](record/v6/omni_v29_speed_gate_videos/gait_comparison_3x2.mp4)
- [flat V29 forward](record/v6/omni_v29_speed_gate_videos/forward.mp4)
- [flat V29 reverse](record/v6/omni_v29_speed_gate_videos/reverse.mp4)
- [flat V29 lateral left](record/v6/omni_v29_speed_gate_videos/lateral_left.mp4)
- [flat V29 lateral right](record/v6/omni_v29_speed_gate_videos/lateral_right.mp4)
- [flat V29 yaw left](record/v6/omni_v29_speed_gate_videos/yaw_left.mp4)
- [flat V29 yaw right](record/v6/omni_v29_speed_gate_videos/yaw_right.mp4)

Latest diagnostic V35 videos with visual spring-steel strips and head speed
overlay:

- [flat V35 long 20 s 3x2 UHD comparison, 3840x2160](record/current/flat_omni_v35_axis_sep_long20/gait_comparison_3x2_20s_4k_uhd.mp4)
- [flat V35 long 20 s 3x2 comparison, 3840x1440](record/current/flat_omni_v35_axis_sep_long20/gait_comparison_3x2_20s_3840x1440.mp4)
- [flat V35 long forward](record/current/flat_omni_v35_axis_sep_long20/forward_20s_1080p.mp4)
- [flat V35 long reverse](record/current/flat_omni_v35_axis_sep_long20/reverse_20s_1080p.mp4)
- [flat V35 long lateral left](record/current/flat_omni_v35_axis_sep_long20/lateral_left_20s_1080p.mp4)
- [flat V35 long lateral right](record/current/flat_omni_v35_axis_sep_long20/lateral_right_20s_1080p.mp4)
- [flat V35 long yaw left](record/current/flat_omni_v35_axis_sep_long20/yaw_left_20s_1080p.mp4)
- [flat V35 long yaw right](record/current/flat_omni_v35_axis_sep_long20/yaw_right_20s_1080p.mp4)

Latest diagnostic V37 videos with visual spring-steel strips and head speed
overlay:

- [flat V37 long 30 s 3x2 UHD comparison, 3840x2160](record/current/flat_omni_v37_lateral_wormcenter_long30/gait_comparison_3x2_30s_4k_uhd.mp4)
- [flat V37 long 30 s 3x2 comparison, 3840x1440](record/current/flat_omni_v37_lateral_wormcenter_long30/gait_comparison_3x2_30s_3840x1440.mp4)
- [flat V37 long forward](record/current/flat_omni_v37_lateral_wormcenter_long30/forward_30s_1080p.mp4)
- [flat V37 long reverse](record/current/flat_omni_v37_lateral_wormcenter_long30/reverse_30s_1080p.mp4)
- [flat V37 long lateral left](record/current/flat_omni_v37_lateral_wormcenter_long30/lateral_left_30s_1080p.mp4)
- [flat V37 long lateral right](record/current/flat_omni_v37_lateral_wormcenter_long30/lateral_right_30s_1080p.mp4)
- [flat V37 long yaw left](record/current/flat_omni_v37_lateral_wormcenter_long30/yaw_left_30s_1080p.mp4)
- [flat V37 long yaw right](record/current/flat_omni_v37_lateral_wormcenter_long30/yaw_right_30s_1080p.mp4)

Current flat V12 command videos:

- [flat V12 six-command comparison](record/v6/omni_v12_heading_prior_artifacts/gait_comparison_v12_flat_6cmd.mp4)
- [flat V12 forward](record/v6/omni_v12_heading_prior_artifacts/forward.mp4)
- [flat V12 reverse](record/v6/omni_v12_heading_prior_artifacts/reverse.mp4)
- [flat V12 lateral left](record/v6/omni_v12_heading_prior_artifacts/lateral_left.mp4)
- [flat V12 lateral right](record/v6/omni_v12_heading_prior_artifacts/lateral_right.mp4)
- [flat V12 yaw left](record/v6/omni_v12_heading_prior_artifacts/yaw_left.mp4)
- [flat V12 yaw right](record/v6/omni_v12_heading_prior_artifacts/yaw_right.mp4)

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

- Keep V41 best as the current flat planar baseline while improving continuous
  `vx/vy/yaw` tracking under `omni_directional_offaxis_yaw_v23` +
  `cmaes_tri_anchor_auto_gate_directional_v26`.
- Improve trajectory quality: mixed `vx/vy` commands under-track lateral
  components, reverse is weak, and yaw-only commands still translate while
  turning.
- Retrain flat worm/snake plus all sand and slope policies under the current reward/action contracts.
- Regenerate all eval, robust eval, fixed-gate ablation scan, summary, and audit artifacts.
- Collect real hardware logs and videos on flat, sand, and slope.
- Rebuild final paper figures after the above are complete.

## Author

Hongsen Pang ([@Kitjesen](https://github.com/Kitjesen)), BSRL Lab.

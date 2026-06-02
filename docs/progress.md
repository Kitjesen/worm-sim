# Progress

## Current Target (2026-06-01)

V50 mixed-command composition repair has been implemented and rejected as the
default control path. V51/V52 then tested the safer follow-up: keep the
V49/V41 prior path, increase mixed-command residual authority, and add a
mixed-planar component/sign reward. V53 then continued with a mixed-planar
curriculum, V54 tested a more aggressive prior/residual authority rebalance,
V55 tested a split-channel mixed-planar prior, V56 added full-speed diagonal
commands to the best-model selection schedule, and V57 tested an explicit
full-diagonal speed-deficit reward. V58 then tested no-retrain residual/prior
scale changes, searched deployable open-loop mixed-planar primitives, and ran a
hardcase mixed-diagonal curriculum from V56 best. V59 then added a
yaw-preserving hardcase curriculum and a stronger yaw-only stationarity cap.
These produced useful ablations but still do **not** solve continuous tracking
on flat terrain.

The continuing goal is to keep the deployable interface fixed while repairing
the failure modes exposed by the strict 35-command scan:

- keep the 80D observation ABI unchanged: body-frame `vx`, `vy`, `yaw`
  command, encoders, previous action, segment IMUs, and 1 s phase clock;
- keep the 12D action ABI unchanged: 11D residual motor action plus 1D learned
  latent gait gate;
- keep actor and critic networks at `512-256-128`;
- current first-stage feasible command limits are `vx=+/-0.10 m/s`,
  `vy=+/-0.075 m/s`, and `yaw=+/-0.125 rad/s`; the command envelope is kept
  intentionally narrow until mixed planar tracking and yaw-only stationarity
  are reliable;
- use `src/v6/analyze_command_scan_v6.py` and the strict scan report as the
  acceptance certificate, not video appearance alone.

The flat V37 adapter plus V70-best policy first passed the strict scan in a
no-retrain evaluation. Server-trained V71, V72, V73, and V74 nominal checkpoints now
also pass the strict 35-command scan under the same deployable ABI. The V71 HD
video set is recorded. Robust scans with sensor noise, one-step action delay,
and `0.90` action saturation are archived through V74, and they show the policy
is still a nominal flat strict-scan checkpoint rather than a robust continuous
tracker. The remaining robust failure is a mixed `vx/vy` counterexample.

The flat acceptance targets are:

- `wrong_planar_sign_count == 0`;
- `wrong_yaw_sign_count == 0`;
- `planar_rmse_m_s <= 0.10`;
- `yaw_rmse_rad_s <= 0.20`;
- `mixed_vx_vy` planar exceed below `8/16`;
- `mixed_vx_yaw` yaw exceed below `3/6`.

Current V74 evidence:

- The server is the only active training machine. The V74 train and scan jobs
  completed on the server; no local Windows training was used for this run.
- Server checkout:
  `/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-v71-clean`.
- Server environment: `wormv6_np2`, cloned from `thunder2`, with NumPy `2.2.6`,
  PyTorch `2.7.0+cu128`, CUDA available, SB3 `2.7.0`, MuJoCo `3.3.3`, and
  Gymnasium `1.2.0`. `thunder2` remains unchanged for Isaac Lab.
- V70 failed because low-speed right lateral used the full-speed right-lateral
  prior at reduced command activity. A zero-residual server probe showed
  `cmd_vy=-0.0375 m/s` produced the wrong sign with the old reduced-amplitude
  prior.
- V71/V37 adds a separately searched slow-right lateral primitive for dominant
  negative-`vy` commands with normalized magnitude up to `0.65`. The 80D
  observation ABI and 12D residual-plus-gate action ABI are unchanged.
- The V37 no-retrain scan with the V70 best checkpoint is accepted:
  `record/current/flat_omni_v71_v37_noretrain_scan/`,
  `planar_rmse_m_s=0.05689`, `yaw_rmse_rad_s=0.01885`,
  `wrong_planar_sign_count=0`, `wrong_yaw_sign_count=0`,
  `planar_error_exceed_count=0`.
- V71 trained best and final checkpoints are archived in
  `record/current/flat_omni_v71_server_v37_trained_scan/`.
  V71 best: `planar_rmse_m_s=0.05902`, `yaw_rmse_rad_s=0.01852`,
  `wrong_planar_sign_count=0`, `wrong_yaw_sign_count=0`,
  `fixed_lateral_strict_gate_passed=true`, analyzer `accepted=true`.
  V71 final: `planar_rmse_m_s=0.06150`, `yaw_rmse_rad_s=0.01860`,
  `wrong_planar_sign_count=0`, `wrong_yaw_sign_count=0`,
  `fixed_lateral_strict_gate_passed=true`, analyzer `accepted=true`.
- V71 robust scans are archived in
  `record/current/flat_omni_v71_server_v37_robust_scan/`. The evaluation
  condition is `robust_sensor_delay_sat_v1`: encoder-position noise `0.01`,
  encoder-velocity noise `0.02`, IMU gravity noise `0.01`, IMU gyro noise
  `0.01`, one action-delay step, and action saturation `0.90`.
  V71 best robust: analyzer `accepted=false`,
  `failed_conditions=["wrong_planar_sign_count"]`,
  `dominant_failure_group=mixed_vx_vy`,
  `planar_rmse_m_s=0.05816`, `yaw_rmse_rad_s=0.02155`,
  `wrong_planar_sign_count=1`, `wrong_yaw_sign_count=0`,
  `fixed_lateral_strict_gate_passed=true`. The concrete wrong-sign command is
  `cmd=(+0.05,+0.075,0)`, measured as
  `body_vx=-0.03496 m/s`, `body_vy=-0.00321 m/s`.
  V71 final robust is also rejected with the same wrong-sign command and
  additionally loses the fixed lateral-left strict gate.
- V71 best was recorded on the server under
  `record/current/flat_omni_v71_server_v37_hd20/`. The folder contains six
  20 s fixed-command HD videos, a 24 s continuous sweep video, a `3840x1440`
  3x2 comparison video, trajectory CSV/PNG files, telemetry CSV/PNG files, and
  `video_manifest.md`. Videos use the visual spring-steel strip renderer and
  head-speed overlay. Manifest command-direction summary: forward
  `66.06 mm/s`, reverse `-53.53 mm/s`, lateral-left `31.95 mm/s`,
  lateral-right `37.34 mm/s`, yaw-left `0.047 rad/s`, yaw-right
  `-0.047 rad/s`, and mean gait gate ranges from about `0.005` in right
  lateral to about `0.894` in yaw-only commands. Telemetry mean body-frame
  lateral velocity is positive for lateral-left (`+0.053 m/s`) and negative for
  lateral-right (`-0.118 m/s`).
- V72 server repair training has been started from the V71 best checkpoint
  with `command_curriculum=feasible_forward_diagonal_repair`, actor/critic
  `512-256-128`, and unchanged 80D/12D ABI. This is targeting the robust
  forward-left diagonal failure before any sand/slope expansion. Robust sensor
  perturbations are currently used for scan evaluation, not silently mixed into
  a V71-compatible resume contract.
- V72 server repair training completed at `2,211,768` total steps. The scan
  artifacts are archived in
  `record/current/flat_omni_v72_server_forward_left_repair_scan/`. V72 best
  nominal and V72 final nominal are both accepted by the strict 35-command
  analyzer:
  - V72 best nominal: `planar_rmse_m_s=0.05555`,
    `yaw_rmse_rad_s=0.01943`, `wrong_planar_sign_count=0`,
    `wrong_yaw_sign_count=0`, `fixed_lateral_strict_gate_passed=true`;
  - V72 final nominal: `planar_rmse_m_s=0.06127`,
    `yaw_rmse_rad_s=0.01922`, `wrong_planar_sign_count=0`,
    `wrong_yaw_sign_count=0`, `fixed_lateral_strict_gate_passed=true`.
  The robust scan is still rejected for `wrong_planar_sign_count` in the
  `mixed_vx_vy` group. V72 final robust keeps the fixed lateral strict gate but
  still fails `cmd=(+0.05,+0.075,0)`, measured as
  `body_vx=-0.03233 m/s`, `body_vy=+0.00594 m/s`.
- V73 server robust fine-tune continued from V72 final with robust sensor
  noise, one action-delay step, action saturation `0.90`, and the explicit
  robust-resume override. The scan artifacts are archived in
  `record/current/flat_omni_v73_server_robust_forward_left_scan/`.
  V73 best and final nominal scans are accepted:
  - V73 best nominal: `planar_rmse_m_s=0.06052`,
    `yaw_rmse_rad_s=0.02059`, `wrong_planar_sign_count=0`,
    `wrong_yaw_sign_count=0`, `fixed_lateral_strict_gate_passed=true`;
  - V73 final nominal: `planar_rmse_m_s=0.05662`,
    `yaw_rmse_rad_s=0.02059`, `wrong_planar_sign_count=0`,
    `wrong_yaw_sign_count=0`, `fixed_lateral_strict_gate_passed=true`.
  Robust scans still fail `wrong_planar_sign_count=1` in `mixed_vx_vy`:
  - V73 best robust: `planar_rmse_m_s=0.06293`,
    `yaw_rmse_rad_s=0.02084`, `wrong_yaw_sign_count=0`,
    `fixed_lateral_strict_gate_passed=true`;
  - V73 final robust: `planar_rmse_m_s=0.05837`,
    `yaw_rmse_rad_s=0.02132`, `wrong_yaw_sign_count=0`,
    `fixed_lateral_strict_gate_passed=true`.
  The remaining robust wrong-sign command is still `cmd=(+0.05,+0.075,0)`,
  measured in V73 final robust as `body_vx=-0.03165 m/s`,
  `body_vy=+0.01286 m/s`.
- V74 server targeted repair continued from V73 final with
  `command_curriculum=robust_forward_left_diagonal_repair`, robust sensor
  noise, one action-delay step, action saturation `0.90`, actor/critic
  `512-256-128`, and unchanged 80D/12D ABI. The scan artifacts are archived in
  `record/current/flat_omni_v74_server_robust_forward_left_targeted_scan/`.
  V74 best and final nominal scans are accepted:
  - V74 best nominal: `planar_rmse_m_s=0.05529`,
    `yaw_rmse_rad_s=0.01920`, `wrong_planar_sign_count=0`,
    `wrong_yaw_sign_count=0`, `fixed_lateral_strict_gate_passed=true`;
  - V74 final nominal: `planar_rmse_m_s=0.05925`,
    `yaw_rmse_rad_s=0.01919`, `wrong_planar_sign_count=0`,
    `wrong_yaw_sign_count=0`, `fixed_lateral_strict_gate_passed=true`.
  Robust scans still fail `wrong_planar_sign_count=1` in `mixed_vx_vy`:
  - V74 best robust: `planar_rmse_m_s=0.05602`,
    `yaw_rmse_rad_s=0.02028`, `wrong_yaw_sign_count=0`,
    `fixed_lateral_strict_gate_passed=true`;
  - V74 final robust: `planar_rmse_m_s=0.05915`,
    `yaw_rmse_rad_s=0.02019`, `wrong_yaw_sign_count=0`,
    `fixed_lateral_strict_gate_passed=false`.
  The repeated robust wrong-sign command remains `cmd=(+0.05,+0.075,0)`.
  V74 best robust measures it as `body_vx=-0.03253 m/s`,
  `body_vy=+0.01780 m/s`; V74 final robust measures it as
  `body_vx=-0.03531 m/s`, `body_vy=+0.01460 m/s`. This targeted curriculum did
  not solve the robust diagonal failure and slightly regressed robust lateral
  left in the final checkpoint.

V49 fixed part of the mixed-yaw sign issue, reducing `mixed_vx_yaw` yaw RMSE
from `0.2884` to `0.2145`, but overall planar RMSE stayed at `0.1515` and
`mixed_vx_vy` remained the dominant failure class. V50 confirmed that simply
injecting stronger composed priors is not sufficient. V51/V52 show that
residual authority plus mixed-planar reward can preserve zero wrong signs and
bring yaw RMSE under the gate, but planar RMSE is still too high. V53 improves
yaw RMSE further, while V54 shows that simply reducing mixed-planar prior
authority and increasing residual authority regresses planar RMSE. V55 shows
that a naive slide/yaw split prior reintroduces wrong planar signs, V56 shows
that full-speed diagonal selection coverage preserves signs but does not reduce
planar RMSE enough, V57 shows that a stronger full-diagonal reward alone does
not make the policy use enough residual authority, and V58 shows that hardcase
oversampling can restore sign reliability but still does not fix mixed-planar
magnitude or yaw-only drift. V59 shows that preserving yaw samples during
hardcase training recovers yaw RMSE from the V58 regression, but full-speed
mixed `vx/vy` composition still fails the strict planar RMSE gate.

## Done

- V6 MuJoCo model, terrain presets, and actuator contract exist.
- Deployable 80-D observation structure exists.
- CMA-ES anchor priors exist for worm, full-combined, and snake modes.
- Paper audit, observation audit, hardware-log validation, deploy export, and
  plotting scaffolds exist.
- Single-segment spring-steel strip calibration now exists under
  `src/v6/spring_steel_calibration_v6.py`, with XML/CSV/JSON/report outputs
  and a test script.
- Real-robot parameter-identification templates now exist under
  `record/v6/hardware/parameter_id`, with a spring-steel force-displacement
  fitter in `src/v6/hardware_parameter_id_v6.py`.
- A detailed real-robot experiment TODO is now tracked in
  `docs/real_robot_experiment_todo_v6.md`.
- A simulator-independent unilateral slide actuator formula now exists in
  `src/v6/unilateral_slide_actuator_v6.py`, with a smoke test. It captures
  servo-rope pulling for contraction plus passive spring-steel return for
  release.
- An Isaac Lab migration plan now exists in
  `docs/isaaclab_migration_plan_v6.md`, including the reduced actuation model
  and visual-only steel-strip rendering path.
- Server training is now active on the 3090 machine, not the local Windows
  workstation. The clean Git checkout is
  `/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-omni`, branch
  `codex/real-robot-experiment-todo`. The checkout may show local modified
  files because server artifacts are preserved, but the active V71 source files
  are synchronized from this branch.
- A Worm-only server Conda environment, `wormv6_np2`, was created by cloning
  `thunder2` and upgrading NumPy to `2.2.6`, because tracked V64/V67
  VecNormalize pickle files require NumPy 2 module names. The original
  `thunder2` environment remains unchanged for Isaac Lab.
- V69b was stopped immediately after startup because it accidentally used the
  training script default learning rate (`3e-4`) instead of the V67 fine-tuning
  rate (`5e-6`).
- V71 server fine-tuning completed in tmux session
  `worm_v71_v37_np2_lowlr` with run label
  `flat_random_v71_server_v37_slow_right_prior_lowlr_from_v70best_np2`.
  It resumed from V70 best at `1,966,768` steps and saved final artifacts at
  `2,096,768` steps after the SB3 chunk overshot the nominal `100,000` target
  to `131,072` rollout steps. It used `continuous_omni`,
  `learning_rate=2e-6`, `n_envs=8`, CUDA device mapped from physical GPU 2,
  and actor/critic `512-256-128`.
- First V69c directional eval is only a startup sanity point, not a result:
  `planar_sign_rate=1.00`, `yaw_sign_rate=0.33`,
  `planar_rmse_m_s=0.054`, `yaw_rmse_rad_s=0.063`, with
  `wrong_planar=0` and `wrong_yaw=1`.
- A preliminary server-side V69c early-best 35-command scan was run while the
  long run continued. Artifacts are in
  `record/current/flat_omni_v69c_server_early_scan/`. It is not accepted:
  `num_commands=35`, `planar_rmse_m_s=0.06289`,
  `yaw_rmse_rad_s=0.01856`, `wrong_planar_sign_count=1`,
  `wrong_yaw_sign_count=0`, `planar_error_exceed_count=2`, and the dominant
  failure group is still `mixed_vx_vy`. This confirms the low-learning-rate
  server resume is stable but has not solved the remaining mixed planar sign
  failure yet.
- The V69c chunk completed at about `1,835,696` steps. Its later best
  checkpoint at `1,805,696` and its final checkpoint both still fail the strict
  scan on exactly the same acceptance condition:
  `wrong_planar_sign_count=1`.
  - Best artifact:
    `record/current/flat_omni_v69c_server_best1805696_scan/`,
    `planar_rmse_m_s=0.06403`, `yaw_rmse_rad_s=0.01892`.
  - Final artifact:
    `record/current/flat_omni_v69c_server_final_scan/`,
    `planar_rmse_m_s=0.06010`, `yaw_rmse_rad_s=0.01891`.
  The concrete counterexample in the final scan is slow right lateral:
  `cmd=(0, -0.0375, 0)` produces `body_vy=+0.03865 m/s` and
  `body_vx=-0.09485 m/s`.
- V70 preparation adds `slow_lateral_left` and `slow_lateral_right` to the
  best-model selection schedule and adds a `slow_lateral_right_repair`
  curriculum. This directly covers the strict-scan counterexample while
  retaining left-lateral, full-right-lateral, zero-command, axial, mixed
  right-lateral, and yaw samples.
- Previous flat/random PPO diagnostics reached about 311k steps, but are stale
  under the new auto-gated contract.
- Flat V6 omnidirectional training has been pushed through yaw, planar repair,
  and off-axis/yaw A/B runs while keeping the 80D observation ABI and 12D action
  ABI fixed.
- The v9 action adapter keeps the deployable ABI fixed but changes the lateral
  direction prior: both lateral commands use the same `+pi/2` slide phase
  offset, while right-lateral mirrors only the yaw-anchor sign. This was based
  on prior-only diagnosis showing that the previous `-pi/2` right-lateral phase
  cancelled rightward thrust.
- Focused flat curricula now exist for `lateral_right`, `yaw_right`, and
  `right_recovery`.
- The V10 action adapter keeps the deployable ABI fixed while increasing
  lateral prior authority and pure right-yaw authority. It can now generate
  six fixed-command videos that pass the simple 6 s velocity/yaw-rate
  thresholds.
- Current V10 policy/video artifacts were generated under
  `record/v6/omni_v10_prior_ablation_artifacts`, including six command videos,
  metrics JSON files, per-segment trajectory CSVs, per-segment trajectory
  plots, and a fixed 6 s summary JSON.
- The V12 action adapter keeps the same 80D observation and 12D policy-action
  ABI while adding command-conditioned zero-yaw heading trims: axial
  forward/reverse commands receive a phase offset and lateral commands receive
  a small yaw trim.
- The current flat/random V12 candidate passes `omni_tracking_gate_v2` with
  `planar_success_rate=1.00`, `yaw_success_rate=0.875`, zero wrong signs, and
  `straight_violation_count=1`.
- Current V12 artifacts were generated under
  `record/v6/omni_v12_heading_prior_artifacts`, including six command videos,
  a 3x2 comparison video, fixed 6 s eval JSON files, per-segment trajectory
  CSVs, and per-segment trajectory plots.
- V13 keeps the same 80D observation and 12D policy-action ABI while adding
  command-magnitude activity scaling. This fixes the zero-speed command failure:
  the restored 35-command scan reports `zero_command_mean_speed_m_s=0.000105`.
- A `continuous_omni` curriculum now samples stop, low-speed, axis, planar,
  yaw, and mixed `vx/vy/yaw` commands for arbitrary-command training.
- A 35-command tracking scan tool now exists at
  `src/v6/scan_command_tracking_v6.py`. The latest V13 scan is under
  `record/v6/omni_v13_command_scan`.
- V13 six-command videos and trajectory plots were generated under
  `record/v6/omni_v13_continuous_artifacts`.
- A V14 continuous-vector prior blend was tested and intentionally left
  disabled because short training worsened continuous planar tracking
  (`planar_rmse_m_s` about `0.189`, `planar_sign_rate=0.76`).
- The low-overhead recorder now renders visual-only spring-steel strips and a
  head-attached average-velocity overlay. The MuJoCo strip injection now reuses
  the shared steel-strip geometry model with about 50 mm pre-bend compression
  plus up to another 50 mm active compression.
- Updated six-command flat videos with steel strips and head speed overlay are
  under `record/v6/omni_v13_steel_speed_6cmd`, including
  `gait_comparison_v13_flat_6cmd_steel_speed.mp4`.
- Multi-agent review agreed that the current controller should be described as
  a weak omnidirectional prototype or six-direction primitive controller, not
  as arbitrary continuous velocity tracking.
- The training best-model selection contract was upgraded to
  `omni_tracking_scan_v3`. The held-out selection schedule now includes stop,
  slow commands, mixed planar commands, and half-rate yaw commands, and the
  selection score penalizes planar velocity RMSE, yaw-rate RMSE, off-axis
  speed, and zero-command drift.
- A V15 flat/random continuation was run from the V13 best model under the new
  `omni_tracking_scan_v3` selection contract:
  `runs/worm_v6_ppo_flat_random_continuous_tracking_v15_from_v13best`. The run
  reached `622368` total timesteps and selected a best checkpoint at `537680`
  timesteps.
- V15 generated a full viewable artifact set under
  `record/v6/omni_v15_tracking_artifacts`: six fixed-command MP4 videos, a 3x2
  comparison MP4, metrics JSON files, per-segment trajectory CSV files, and
  per-segment trajectory plots. The videos use visual spring-steel strips and
  the head-attached speed overlay by default.
- The V15 comparison video was checked as a non-empty `1920x720`, 10 fps,
  60-frame MP4. The trajectory CSVs contain `head` plus `seg1` through `seg6`
  with monotonic timestamps.
- The V16 action adapter keeps the same 80D observation and 12D policy-action
  ABI but raises yaw-only prior authority symmetrically for left/right yaw:
  `YAW_ONLY_PRIOR_SCALE_FLOOR=0.30`, `YAW_ONLY_SLIDE_PRIOR_SCALE=0.80`,
  `YAW_ONLY_YAW_PRIOR_SCALE=5.00`, and
  `YAW_RIGHT_ONLY_YAW_PRIOR_SCALE=5.00`.
- The `continuous_omni` curriculum now includes extra yaw-only repair samples
  and reverse/mixed zero-yaw repair samples while preserving the same command
  interface.
- A V16 flat/random continuation was run from the V15 best model:
  `runs/worm_v6_ppo_flat_random_continuous_tracking_v16_yawprior_from_v15best`.
  The run reached `652368` total timesteps and selected a best checkpoint at
  `597680` timesteps.
- V16 viewable artifacts were generated under
  `record/v6/omni_v16_tracking_artifacts`, including six fixed-command videos,
  a 3x2 comparison video, metrics JSON files, per-segment trajectory CSVs, and
  per-segment trajectory plots.
- V23/V16 revised the latent gait-gate contract without changing the
  deployable ABI. The policy still outputs `11D residual + 1D gait gate`, but
  the effective axial command center is now speed-adaptive: slow axial commands
  use a worm-like center (`0.35`), while full-speed axial commands return to the
  faster mixed center (`0.50`). This addresses the V25/V26 failure mode where a
  fixed axial gate near `0.40` restored visible worm motion but capped forward
  speed.
- Actor and critic defaults are now both `512-256-128`. The actor-transfer tool
  now writes `training_config.json`, so transferred actors can be resumed by
  `train_v6.py` instead of being rejected as missing metadata.
- V29 was trained as the then-current flat six-direction candidate:
  `runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate`.
  It resumes the faster V22 actor into a fresh `512-256-128` critic under the
  V23/V16 contract.
- V29 restores the fixed forward/reverse thresholds: forward reaches
  `0.1384 m/s`, reverse reaches `-0.0810 m/s`, and yaw signs remain separated
  (`yaw_left=0.3768 rad/s`, `yaw_right=-0.4206 rad/s`). Low-speed axial gate is
  about `0.36`, while full-speed axial gate is about `0.48`.
- V29 artifacts were generated under
  `record/v6/omni_v29_speed_gate_videos`: six fixed-command MP4 videos, a 3x2
  comparison MP4, per-command trajectory CSV files, trajectory plots, and a
  35-command scan JSON/CSV. The MP4s include visual spring-steel strips and the
  head speed overlay; the fixed-command videos verify as `1280x720`, `25 fps`,
  `150 frames`, `6.0 s`.
- Detailed V29 notes are in `docs/omni_v29_training_log.md`.
- HD V29 direction videos were regenerated under
  `record/current/flat_omni_v29_hd`: six H.264 `1920x1080`,
  25 fps, 6 s single-command videos with visual spring-steel strips and head
  speed overlay, plus `3840x1440` and `3840x2160` 3x2 comparison videos.
  The HD rollouts confirm forward/reverse and yaw signs, but also make the
  remaining lateral off-axis and yaw-only translation problems obvious. Exact
  metrics are in `docs/omni_v29_hd_recording_log.md`.
- Longer V29 direction videos were recorded under
  `record/current/flat_omni_v29_long15`: six H.264 `1920x1080`,
  25 fps, 15 s single-command videos and 15 s 3x2 comparison videos. The
  recorder now accumulates yaw over time, so long yaw videos are not corrupted
  by final-angle wraparound. Exact metrics are in
  `docs/omni_v29_long15_recording_log.md`.
- V30-V33 were run as flat repair attempts after the HD recording pass. None
  beat V29. V30 made `forward_yaw_left` wrong-sign, V31 mixed planar+yaw prior
  worsened both forward-yaw signs and was reverted, V32 restored zero wrong-yaw
  at its best checkpoint but did not improve tracking, and V33 low-LR
  fine-tuning still regressed to one wrong-yaw case. The retained code changes
  are targeted mixed-yaw curriculum sampling and configurable PPO learning
  rate. Details are in `docs/omni_v30_v33_repair_log.md`.
- V34 was run as a low-learning-rate continuation from V29 best. It completed
  to `325,536` total timesteps and improved held-out yaw success from `0.7647`
  to `0.8235`, but the selected best checkpoint still failed tracking and
  direction gates, reintroduced one wrong-yaw case, and did not beat V29 by
  selection score. The 35-command scan reports `planar_rmse_m_s=0.1635`,
  `yaw_rmse_rad_s=0.1128`, `planar_sign_rate=0.84`, `yaw_sign_rate=1.00`, and
  yaw-only planar drift about `0.1046 m/s`. Details are in
  `docs/omni_v34_training_log.md`.
- V35/V36 were run as the next flat diagnostic after finding that the v23
  yaw-only prior advertised slide/yaw scaling but returned the raw in-place yaw
  prior before applying those scales. V24 fixes that path, v19 raises yaw-only
  stationary and lateral-only forward-drift penalties, and a new
  `axis_separation` curriculum samples one command axis at a time. V35 reduced
  yaw-only planar drift to `0.0631 m/s`; V36 reduced yaw RMSE to
  `0.0994 rad/s` after switching back to `continuous_omni`. Neither run
  improved planar tracking enough to replace V29. Details are in
  `docs/omni_v35_v36_axis_separation_log.md`.
- V35 20 s direction videos were generated under
  `record/current/flat_omni_v35_axis_sep_long20`: six H.264 `1920x1080`,
  25 fps, 20 s single-command videos plus `3840x1440` and `3840x2160` 3x2
  comparison videos. The long-video recorder now avoids VecNormalize
  auto-reset at the 20 s episode limit, so final pose metrics are not reset
  contaminated.
- V37/V38 were run as a lateral repair diagnostic after prior-only analysis
  showed the mixed-centered lateral prior was dominated by forward off-axis
  speed. V25 moves dominant lateral commands to a worm-centered `-pi/2`
  lateral prior, improving the V36 35-command planar sign rate from `0.84` to
  `0.96`, but lateral speed remains below threshold. V38 adds a pure-lateral
  speed-deficit reward, reducing yaw-only planar drift to `0.0600 m/s` at the
  scanned `422880` checkpoint, but planar RMSE worsens to `0.1677 m/s`.
  Details are in `docs/omni_v37_v38_lateral_repair_log.md`.
- V37 30 s direction videos were generated under
  `record/current/flat_omni_v37_lateral_wormcenter_long30`: six H.264
  `1920x1080`, 25 fps, 30 s single-command videos plus `3840x1440` and
  `3840x2160` 3x2 comparison videos. They include visual spring-steel strips
  and the head speed overlay.
- V39 fixed a pure-lateral reward bug: the lateral-only speed deficit now uses
  signed body-frame lateral velocity instead of off-axis speed. The V39
  35-command scan still fails the new explicit lateral gate
  (`left vy=+0.0231 m/s`, `right vy=-0.0158 m/s`, threshold `0.03 m/s`), so
  V39 is not accepted. Residual-scale probes up to `0.70` also fail, showing
  that evaluation-time residual authority is not enough.
- V40 was run from the V37 final model with the V21 reward,
  `continuous_omni` curriculum, actor/critic `512-256-128`, and 6 s directional
  eval every 20k steps. It is still not accepted. The final held-out eval has
  `tracking_gate_passed=false`, `direction_gate_passed=false`,
  `planar_rmse_m_s=0.1056`, `yaw_rmse_rad_s=0.1952`; the final independent
  35-command scan has `planar_rmse_m_s=0.1630`, `yaw_rmse_rad_s=0.1127`, and
  fails fixed lateral speed (`left vy=+0.0260 m/s`, `right vy=-0.0097 m/s`).
  Details are in
  `docs/omni_v39_v40_lateral_reward_and_tracking_log.md`.
- The recorder now supports optional gate/action telemetry CSV and plot output
  for each video. It records raw gate action, learned/deployed gait blend,
  desired gait blend, prior component, residual component, and applied action
  without changing the deployable 80D observation or 12D action contract.
- V40 final was recorded for 30 s per command under
  `record/current/flat_omni_v40_v21_long30`: six H.264 `1920x1080`, 25 fps
  single-command videos, trajectory plots, telemetry plots, and `3840x2160`
  3x2 comparison video. The comparison verifies as 30.0 s and 750 frames.
  Details are in `docs/omni_v40_long30_recording_log.md`.
- V41 adds a dedicated lateral primitive search path and V26 action adapter.
  The adapter keeps the same 80D observation and 12D action ABI, but dominant
  lateral commands now use independently searched left/right 1 s phase-clock
  primitives instead of the older transformed forward anchor. Prior-only
  integrated smoke passes both lateral sides:
  left `body_vy=0.0814 m/s`, right `body_vy=-0.1784 m/s`, both within the
  forward-drift and yaw-drift thresholds.
- Re-scanning the V40 final PPO policy with the V26 adapter immediately fixes
  the previous fixed-lateral gate failure: 35-command scan reports
  `planar_rmse_m_s=0.1535`, `yaw_rmse_rad_s=0.1127`,
  `planar_sign_rate=1.00`, `yaw_sign_rate=1.00`, and
  `fixed_lateral_strict_gate_passed=true`.
- V41/V42/V43 continuation training was run from the V40/V41 line with
  actor/critic `512-256-128`. The best current planar scan is V41 best:
  `planar_rmse_m_s=0.1517`, `yaw_rmse_rad_s=0.1105`,
  `planar_sign_rate=1.00`, `yaw_sign_rate=1.00`, and lateral strict gate
  passed. V42 final improves yaw RMSE to `0.1009` but has slightly worse
  planar RMSE. V43's first V22-reward continuation did not improve the
  35-command scan.
- V44 added a `mixed_planar_repair` curriculum under reward contract
  `omni_directional_offaxis_yaw_v23` and ran one 40k continuation from V41
  best. It preserved correct signs and fixed-lateral strict gates, but did not
  beat V41 best on the 35-command scan: V44 final has
  `planar_rmse_m_s=0.1540`, `yaw_rmse_rad_s=0.1193`, and yaw-only planar drift
  `0.0602 m/s`. A continuous-vector-prior probe was worse
  (`planar_rmse_m_s=0.1872`, `planar_sign_rate=0.76`), and residual-scale
  sweeps did not beat V41 best.
- V45/V46 were run as the next optimization attempts. V45 axis-separation
  repair from V41 best did not beat V41: V45 best has
  `planar_rmse_m_s=0.1552`, `yaw_rmse_rad_s=0.1165`, and
  `planar_sign_rate=0.96`; V45 final recovered signs but lost fixed-lateral
  strict acceptance. The V45 reverse axial primitive search also failed
  acceptance (`signed_axial_m_s=0.0621`, target `>=0.10`), and the V45 yaw
  primitive search failed turn-rate acceptance (`mean_signed_yaw=0.2186 rad/s`,
  target `>=0.30`).
- V46 added reward contract `omni_directional_offaxis_yaw_v24`, which penalizes
  residual cancellation of the pure-axial peristaltic slide prior. It is a
  useful diagnostic for the observed loss of visible worm-like actuation, but
  it did not replace V41. V46 best reports `planar_rmse_m_s=0.1547`,
  `yaw_rmse_rad_s=0.1141`, and `planar_sign_rate=0.96`; V46 final reports
  `planar_rmse_m_s=0.1569`, `yaw_rmse_rad_s=0.1152`, and
  `planar_sign_rate=1.00`.
- V47 added a stricter command-tracking proof path and componentwise tracking
  cost. It preserved correct signs but did not beat V41: V47 final smoke
  reports `planar_rmse_m_s=0.1516`, `yaw_rmse_rad_s=0.1183`, and still fails
  mixed-command composition.
- V48 added `mixed_composition_repair` sampling from V47 final, oversampling
  `mixed_vx_vy` and `mixed_vx_yaw`. The local smoke continued from `652,256`
  to `701,408` policy steps but did not produce a real improvement:
  `planar_rmse_m_s=0.1515`, `yaw_rmse_rad_s=0.1196`,
  `mixed_vx_vy` planar exceed `14/16`, and `mixed_vx_yaw` yaw exceed `6/6`.
- V49 changed the mixed `vx+yaw` yaw-anchor sign in the action adapter without
  retraining. It improved `mixed_vx_yaw` yaw RMSE from `0.2884` to `0.2145`
  and yaw exceed from `6/6` to `4/6`, but overall planar RMSE stayed at
  `0.1515`; the candidate is still rejected.
- V50 added an experimental componentwise mixed-command prior plus a deployment
  parity fix for lateral/yaw prior transforms. No-retrain scans and a short
  continuation from V48 final were rejected: the final V50 smoke reports
  `planar_rmse_m_s=0.1801`, `yaw_rmse_rad_s=0.2075`,
  `wrong_planar_sign_count=4`, and `wrong_yaw_sign_count=0`. The experimental
  mixed-prior path is now default-off behind
  `WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1`.
- A default-guard scan with the V50 code path and the experimental switch off
  preserved the safe sign behavior: `planar_rmse_m_s=0.1498`,
  `yaw_rmse_rad_s=0.2003`, `wrong_planar_sign_count=0`, and
  `wrong_yaw_sign_count=0`. It still fails continuous tracking, with
  `mixed_vx_vy` remaining the dominant failure group.
- V51 added command-conditioned residual authority without changing the 80D/12D
  deployable ABI: mixed `vx/vy` residual multiplier `1.80`, mixed yaw
  multiplier `1.60`, and pure-yaw multiplier `1.25`. A 50k continuation from
  V48 final preserved yaw signs and improved yaw RMSE in the final scan to
  `0.1907`, but planar RMSE remained `0.1553`; the candidate is still rejected.
- V52 added reward contract `omni_directional_offaxis_yaw_v26` with mixed
  planar component tracking and sign penalties, then continued 50k from V51
  final. The final scan reports `planar_rmse_m_s=0.1534`,
  `yaw_rmse_rad_s=0.1978`, `wrong_planar_sign_count=0`, and
  `wrong_yaw_sign_count=0`. This is a small improvement over V51 final but
  still fails the flat continuous-tracking gate; `mixed_vx_vy` remains the
  dominant failure group.
- V53 continued from V52 final with `mixed_planar_repair` curriculum after
  explicitly allowing a curriculum-resume. The final strict scan reports
  `planar_rmse_m_s=0.1509`, `yaw_rmse_rad_s=0.1697`,
  `wrong_planar_sign_count=0`, and `wrong_yaw_sign_count=0`. This is the
  current safest local candidate on yaw/sign gates, but it still fails planar
  RMSE and yaw-only drift.
- V54 tested a mixed-planar authority rebalance:
  `WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE=1` lowers the dominant
  mixed-planar prior to `0.65` and raises the mixed-planar residual multiplier
  to `2.50`. No-retrain and 50k continuation scans regressed planar RMSE
  (`0.1686` final after training), so the ablation is default-off. The V54
  default guard on V53 final reproduces `planar_rmse_m_s=0.1509`,
  `yaw_rmse_rad_s=0.1697`, and zero sign errors.
- V55 added `WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR=1`, a default-off
  split-channel prior where slide actuators follow the signed axial primitive
  and yaw actuators follow the signed lateral primitive for mixed `vx/vy`.
  No-retrain and short continuation scans regressed to
  `planar_rmse_m_s≈0.202` with `7-10` wrong planar signs, so this hypothesis is
  rejected as a default control path.
- V56 added full-speed diagonal commands to `best_eval_schedule`:
  `(±0.25, ±0.15, 0)`. The best strict scan reports
  `planar_rmse_m_s=0.1509`, `yaw_rmse_rad_s=0.1646`,
  `wrong_planar_sign_count=0`, `wrong_yaw_sign_count=0`, and
  `yaw_only_mean_planar_speed_m_s=0.0873`. This is the safest local candidate
  on selection coverage and yaw RMSE, but it is still rejected.
- V57 added reward contract `omni_directional_offaxis_yaw_v27` with a
  full-diagonal mixed-planar deficit term. It continued from V56 best at
  `918,016` steps with the same `512-256-128` actor/critic and fixed 80D/12D
  ABI. The best strict scan reports `planar_rmse_m_s=0.1527`,
  `yaw_rmse_rad_s=0.1933`, `wrong_planar_sign_count=1`, and
  `yaw_only_mean_planar_speed_m_s=0.0965`; the final scan reports
  `planar_rmse_m_s=0.1566`, `yaw_rmse_rad_s=0.1760`,
  `wrong_planar_sign_count=1`, and `yaw_only_mean_planar_speed_m_s=0.0913`.
  V57 is rejected. The telemetry still shows `mixed_vx_vy` residual L2 around
  `0.09` against prior L2 around `1.09`, so reward pressure alone did not give
  the learned residual enough authority to compose both planar components.
- V58 first ran no-retrain scale diagnostics on V56/V57 checkpoints. Reducing
  the residual scale to `0.70`, or combining `gait_prior_scale=0.75` with
  `policy_residual_scale=0.70`, worsened planar/yaw tracking rather than
  revealing a hidden correct policy. V58 also added
  `src/v6/search_mixed_planar_prior_v6.py`; a small CMA search found one
  deployable 1 s phase-clock `forward_right` diagonal primitive that validates
  at `(0.2186, -0.0359) m/s`, but `forward_left` and both reverse diagonals
  failed validation. The `mixed_planar_hardcase_repair` curriculum then
  continued from V56 best. Its best strict scan reports
  `planar_rmse_m_s=0.1566`, `yaw_rmse_rad_s=0.2709`,
  `wrong_planar_sign_count=0`, `wrong_yaw_sign_count=0`, and
  `yaw_only_mean_planar_speed_m_s=0.0951`; the final scan reports
  `planar_rmse_m_s=0.1588`, `yaw_rmse_rad_s=0.2720`, zero sign errors, and
  `yaw_only_mean_planar_speed_m_s=0.0959`. V58 is rejected: hardcase sampling
  restores signs but worsens yaw RMSE and still leaves `mixed_vx_vy` as the
  dominant failure group.
- V59 added reward contract `omni_directional_offaxis_yaw_v28`, a
  `mixed_planar_yaw_preserve_repair` curriculum, and a higher yaw-only
  stationarity penalty cap. It continued from V56 best with the same 80D/12D
  deployable ABI and `512-256-128` actor/critic. The best strict scan reports
  `planar_rmse_m_s=0.1561`, `yaw_rmse_rad_s=0.1887`,
  `wrong_planar_sign_count=0`, `wrong_yaw_sign_count=0`, and
  `yaw_only_mean_planar_speed_m_s=0.0677`; the final scan reports
  `planar_rmse_m_s=0.1591`, `yaw_rmse_rad_s=0.1895`,
  `wrong_planar_sign_count=1`, `wrong_yaw_sign_count=0`, and
  `yaw_only_mean_planar_speed_m_s=0.0678`. V59 is rejected: yaw recovers
  compared with V58 and fixed lateral strict gate passes in the final scan, but
  full-speed mixed `vx/vy` commands still fail continuous tracking.
- Latest detailed movement table and artifact list:
  `docs/omni_v41_v46_motion_summary.md`.

## 2026-06-01 V68 35-Command Scan and Server Migration

- The scan tool default now matches the documented 35-command acceptance
  surface. With `--include-forward-yaw`, the default grid is 25 planar
  `vx/vy` commands, 4 nonzero pure-yaw commands plus the zero yaw row already
  present in the planar grid, and 6 forward-yaw commands. A regression check in
  `src/v6/test_omni_eval_metrics_v6.py` asserts that the default scan command
  count is `35`.
- Re-scanning the V64 best checkpoint under the current V36 adapter and the
  feasible command range gives:
  `record/current/flat_omni_v68_v50_default_guard_35cmd_scan/scan_35_commands_default_guard_6s.json`.
  Summary: `num_commands=35`, `planar_rmse_m_s=0.05985`,
  `yaw_rmse_rad_s=0.01880`, `wrong_planar_sign_count=1`,
  `wrong_yaw_sign_count=0`, `zero_command_mean_speed_m_s=0.000048`,
  `yaw_only_mean_planar_speed_m_s=0.02850`, and fixed left/right lateral strict
  gates pass. Strict analysis rejects it only on `wrong_planar_sign_count`;
  the dominant failure group remains `mixed_vx_vy`.
- Enabling the V50 componentwise mixed-command prior on the same checkpoint
  still regresses the full scan:
  `record/current/flat_omni_v68_v50_mixed_composition_35cmd_scan/scan_35_commands_mixed_composition_6s.json`.
  Summary: `planar_rmse_m_s=0.10105`, `yaw_rmse_rad_s=0.01880`,
  `wrong_planar_sign_count=6`, `wrong_yaw_sign_count=0`, and
  `planar_error_exceed_count=8`. Strict analysis rejects it on
  `planar_rmse_m_s` and `wrong_planar_sign_count`. Therefore V50 mixed-command
  componentwise composition remains an ablation behind
  `WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1`, not the default controller.
- Tests rerun after the scan fix:
  `test_reward_contract_v6.py`, `test_deployable_obs_v6.py`,
  `test_omni_eval_metrics_v6.py`, and `test_command_scan_analysis_v6.py`.
- Training should now move off the local Windows machine. The 3090 server is
  reachable over SSH, sees 8 CUDA devices in the `thunder2` Conda environment,
  and has `torch 2.7.0+cu128`, `mujoco 3.3.3`, `gymnasium 1.2.0`, and
  `stable_baselines3 2.7.0`. A clean Git-based training checkout should be used
  under `/home/bsrl/hongsenpang/codex_runs` instead of the older manually
  copied `/home/bsrl/hongsenpang/worm_project` directory.

## 2026-06-02 V76 Server Hardcase Selection

- Training and post-evaluation were run on the server clean checkout, not on
  the local Windows machine:
  `/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training`.
- MuJoCo PPO used the server `wormv6_np2` environment: Python `3.11`, NumPy
  `2.2.6`, PyTorch `2.7.0+cu128`, SB3 `2.7.0`, MuJoCo `3.3.3`, and Gymnasium
  `1.2.0`. The 8x RTX 3090 host is available, but this SB3/MuJoCo setup remains
  largely CPU-step limited.
- Run label:
  `flat_random_v76_server_hardcase_selection_from_v74final_np2`. It continues
  from V74 final, keeps the 80D observation ABI, keeps the 12D residual plus
  learned-gate action ABI, and uses actor/critic `512-256-128`.
- Local pulled artifacts:
  `record/current/flat_omni_v76_server_hardcase_selection_scan/` and
  `record/current/flat_omni_v76_server_hardcase_selection_videos/`.
- V76 best nominal scan: `planar_rmse_m_s=0.05713`,
  `yaw_rmse_rad_s=0.01889`, `wrong_planar_sign_count=0`,
  `wrong_yaw_sign_count=0`, and `fixed_lateral_strict_gate_passed=true`.
- V76 best robust scan: `planar_rmse_m_s=0.05680`,
  `yaw_rmse_rad_s=0.02052`, `wrong_planar_sign_count=1`,
  `wrong_yaw_sign_count=0`, and `fixed_lateral_strict_gate_passed=true`.
- The remaining robust counterexample is still the slow forward-left diagonal
  `cmd=(+0.05,+0.075,0)`. Under V76 best robust it measures
  `body_vx=-0.03818 m/s`, `body_vy=-0.00377 m/s`, so the projected planar sign
  is still wrong.
- Eight server-recorded HD videos were pulled locally: six 12 s fixed-command
  videos, one 12 s hard forward-left diagnostic, and one 24 s continuous command
  sweep. They include visual spring-steel strips and head-speed overlay.
- A narrow, default-off hardcase gate flag,
  `WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1`, was added for the next server
  retrain. It is not an accepted default result until trained and rescanned.

## 2026-06-02 V77 Server Hardcase Gate Retrain

- V77 was trained on the server only, from the V76 best checkpoint, with
  `WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1`.
- Server-only training is now the active rule for long runs. The verified
  server checkout is
  `/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training`, using
  `/home/bsrl/miniconda3/envs/wormv6_np2/bin/python` with Python 3.11.14,
  NumPy 2.2.6, PyTorch 2.7.0+cu128, CUDA available on 8 GPUs, MuJoCo 3.3.3,
  Gymnasium 1.2.0, and SB3 2.7.0. Local Windows is reserved for pulling
  artifacts, documentation, and lightweight validation, not training.
- Run label:
  `flat_random_v77_server_hardcase_gate_from_v76best_np2`. It keeps the 80D
  observation ABI, keeps the 12D residual plus learned-gate action ABI, and
  uses actor/critic `512-256-128`.
- Local pulled artifacts:
  `record/current/flat_omni_v77_server_hardcase_gate_scan/` and
  `record/current/flat_omni_v77_server_hardcase_gate_videos/`.
- V77 best nominal scan: `planar_rmse_m_s=0.05513`,
  `yaw_rmse_rad_s=0.01856`, `wrong_planar_sign_count=0`,
  `wrong_yaw_sign_count=0`, and `fixed_lateral_strict_gate_passed=true`.
- V77 best robust scan: `planar_rmse_m_s=0.05967`,
  `yaw_rmse_rad_s=0.02028`, `wrong_planar_sign_count=0`,
  `wrong_yaw_sign_count=0`, `planar_error_exceed_count=0`, and
  `fixed_lateral_strict_gate_passed=true`.
- The repeated robust hardcase is now fixed for the V77 best checkpoint:
  `cmd=(+0.05,+0.075,0)` measures `body_vx=-0.03548 m/s`,
  `body_vy=+0.02522 m/s`, `yaw_rate=+0.03460 rad/s`, and
  `planar_sign_ok=true`. The deployed gate is no longer near the pure lateral
  gate: mean deployed gate is `0.255`, desired gate is `0.28`, and learned gate
  is `0.465`.
- V77 final is not the accepted candidate. Its robust scan regresses to
  `wrong_planar_sign_count=1` and `fixed_lateral_strict_gate_passed=false`.
- Eight server-recorded HD videos were pulled locally: six 12 s fixed-command
  videos, one 12 s hard forward-left diagnostic, and one 24 s continuous command
  sweep. They include visual spring-steel strips and head-speed overlay.

## 2026-06-02 V85b Server Mixed-Planar Hardcase Diagnostic

- V85b was trained on the server only, from the V77 best checkpoint, with
  `WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1` and the
  `mixed_planar_hardcase_repair` curriculum. The Windows machine was used only
  for pulling artifacts and documentation.
- Run label:
  `flat_random_v85b_server_mixed_planar_hardcase_from_v77best_np2`. It keeps
  the 80D observation ABI, keeps the 12D residual plus learned-gate action ABI,
  and uses actor/critic `512-256-128`.
- Local pulled artifacts:
  `record/current/flat_omni_v85b_server_mixed_planar_hardcase_scan/` and
  `record/current/flat_omni_v85b_server_mixed_planar_hardcase_videos/`.
- V85b best nominal scan: `planar_rmse_m_s=0.05620`,
  `yaw_rmse_rad_s=0.01861`, `wrong_planar_sign_count=0`,
  `wrong_yaw_sign_count=0`, `off_axis_exceed_count=2`,
  `planar_error_exceed_count=1`, and
  `fixed_lateral_strict_gate_passed=true`.
- V85b best robust scan: `planar_rmse_m_s=0.06144`,
  `yaw_rmse_rad_s=0.02042`, `wrong_planar_sign_count=2`,
  `wrong_yaw_sign_count=0`, `planar_error_exceed_count=2`, and
  `fixed_lateral_strict_gate_passed=true`.
- V85b final robust scan improves the robust sign count but still fails:
  `planar_rmse_m_s=0.06254`, `yaw_rmse_rad_s=0.02027`,
  `wrong_planar_sign_count=1`, `wrong_yaw_sign_count=0`,
  `off_axis_exceed_count=1`, and `planar_error_exceed_count=5`.
- The remaining V85b final robust sign counterexample is
  `cmd=(+0.05,+0.075,0)`, which measures
  `body_vx=-0.03670 m/s`, `body_vy=+0.01979 m/s`,
  `yaw_rate=+0.01997 rad/s`, and mean deployed gate `0.256`.
- Nine server-recorded HD videos were pulled locally: six 12 s fixed-command
  videos, two 12 s hard diagonal diagnostics, and one 24 s continuous command
  sweep. They include visual spring-steel strips and head-speed overlay.
- V85b is not the accepted flat checkpoint. It is useful evidence that the
  mixed-planar hardcase curriculum preserves nominal signs but still does not
  make robust slow forward-left diagonal tracking reliable.

## 2026-06-02 V86b Server Robust Forward-Left Repair

- V86b was trained and post-evaluated on the server only, from the V85b final
  checkpoint, with the `robust_forward_left_diagonal_repair` curriculum. The
  Windows machine was used only to pull artifacts and update documentation.
- The verified server checkout remains
  `/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training`, currently at
  Git commit `49ab85b`. The training/evaluation Python is
  `/home/bsrl/miniconda3/envs/wormv6_np2/bin/python`, with Python `3.11.14`,
  NumPy `2.2.6`, PyTorch `2.7.0+cu128`, CUDA available on 8 RTX 3090 GPUs,
  MuJoCo `3.3.3`, Gymnasium `1.2.0`, and SB3 `2.7.0`.
- Run label:
  `flat_random_v86b_server_robust_forward_left_from_v85bfinal_np2`. It keeps
  the 80D observation ABI, keeps the 12D residual plus learned-gate action ABI,
  and uses actor/critic `512-256-128`.
- Local pulled artifacts:
  `record/current/flat_omni_v86b_server_robust_forward_left_scan/` and
  `record/current/flat_omni_v86b_server_robust_forward_left_videos/`.
- Server environment verification passed:
  `test_reward_contract_v6.py`, `test_deployable_obs_v6.py`,
  `test_omni_eval_metrics_v6.py`, and
  `test_visual_steel_strip_geometry_v6.py`.
- V86b best nominal scan: `planar_rmse_m_s=0.05702`,
  `yaw_rmse_rad_s=0.01897`, `wrong_planar_sign_count=0`,
  `wrong_yaw_sign_count=0`, `off_axis_exceed_count=3`, and
  `planar_error_exceed_count=2`.
- V86b best robust scan: `planar_rmse_m_s=0.05855`,
  `yaw_rmse_rad_s=0.02027`, `wrong_planar_sign_count=0`,
  `wrong_yaw_sign_count=0`, `off_axis_exceed_count=1`,
  `planar_error_exceed_count=2`, and
  `fixed_lateral_strict_gate_passed=true`.
- V86b final robust scan is not the candidate because it regresses to
  `wrong_planar_sign_count=1`, even though it reduces
  `planar_error_exceed_count` to `1`.
- The V86b best robust remaining hard failures are no longer the V85b
  `(+0.05,+0.075,0)` sign failure. They are magnitude/crosstalk cases:
  `cmd=(-0.10,+0.075,0)` has `planar_error_m_s=0.10286`, and
  `cmd=(+0.05,-0.075,0)` has `planar_error_m_s=0.10707` with
  `off_axis_speed_m_s=0.09834`.
- Nine server-recorded HD videos were pulled locally: six 12 s fixed-command
  videos, two 12 s hard diagonal diagnostics, and one 24 s continuous command
  sweep. They include visual spring-steel strips and head-speed overlay.
- V86b is a useful robust sign repair result, but it is not yet a complete
  continuous velocity tracker because magnitude fidelity and off-axis crosstalk
  still exceed the current scan thresholds.

## Not Done

- The current V77 best checkpoint fixes the repeated robust hardcase sign gate,
  but it is still not a complete arbitrary continuous velocity tracker.
- V85b does not supersede V77 as the accepted robust sign-gate candidate:
  V85b final robust still has one planar sign failure and five planar magnitude
  errors over threshold.
- V86b best robust removes those planar sign failures, but it still does not
  supersede V77 as the paper-facing checkpoint because it has
  `planar_error_exceed_count=2` and `off_axis_exceed_count=1`.
- Remaining flat limitations are magnitude accuracy and crosstalk, not the
  repeated robust sign failure: V77 best nominal has `off_axis_exceed_count=3`
  and `planar_error_exceed_count=1`.
- Fixed-gate ablations, deploy bundles, final paper summaries, and sand/slope
  transfer must still be regenerated after the robust flat gate is solved.
- Mixed `vx/vy` commands remain the limiting case for strong continuous
  tracking claims; yaw-only and fixed lateral signs are much more stable than in
  the early V13-V29 line.
- Arbitrary continuous velocity tracking is not solved yet. V13 stops cleanly
  at zero command, but the 35-command scan still has
  `planar_rmse_m_s=0.1628` and `yaw_rmse_rad_s=0.3189`.
- The first V15 run under `omni_tracking_scan_v3` did not pass the flat
  continuous-tracking goal. Its best held-out selection summary has
  `tracking_gate_passed=false` and `direction_gate_passed=false`.
- V15 best passes the target for planar RMSE on the 17-command selection set
  (`planar_velocity_rmse_m_s=0.0940`), off-axis speed
  (`mean_off_axis_speed_m_s=0.0381`), and zero-command drift
  (`zero_command_mean_speed_m_s=0.000019`). It still fails yaw-rate RMSE
  (`yaw_rate_rmse_rad_s=0.2453`, target `<=0.20`) and straight yaw drift
  (`straight_violation_count=3`, target `<=1`).
- The independent 35-command scan for V15 best remains weak:
  `planar_rmse_m_s=0.1604`, `yaw_rmse_rad_s=0.3205`,
  `planar_sign_rate=0.80`, and `yaw_sign_rate=1.00`. The V15 final checkpoint
  was slightly worse than the V15 best checkpoint on this scan.
- V16 improves yaw magnitude but still fails the flat continuous-tracking goal.
  Its best held-out selection summary reports `tracking_gate_passed=false`,
  `direction_gate_passed=false`, `planar_velocity_rmse_m_s=0.0976`,
  `yaw_rate_rmse_rad_s=0.2285`, `straight_violation_count=2`, and two
  yaw-only stationary-drift violations.
- The independent 35-command scan for V16 best reports
  `planar_rmse_m_s=0.1621`, `yaw_rmse_rad_s=0.2868`,
  `planar_sign_rate=0.80`, and `yaw_sign_rate=1.00`. This is a yaw-RMSE
  improvement over V15, but planar tracking did not improve and yaw-only
  commands now translate more.
- V29 is not a complete continuous omnidirectional tracker. The held-out
  selection still reports `tracking_gate_passed=false`,
  `direction_gate_passed=false`, `planar_velocity_rmse_m_s=0.1070`,
  `yaw_rate_rmse_rad_s=0.1729`, `straight_violation_count=3`, and
  `stationary_violation_count=4`. The 35-command scan reports
  `planar_rmse_m_s=0.1665`, `yaw_rmse_rad_s=0.1179`,
  `planar_sign_rate=0.84`, `yaw_sign_rate=1.00`, and yaw-only planar drift of
  about `0.0966 m/s`.
- V30-V33 are not accepted policies. They are historical repair diagnostics
  superseded by the V41-V44 line.
- V35/V36 are also not accepted policies. They show that the yaw-only prior
  scaling fix helps reduce pure-yaw planar drift, but lateral-only and mixed
  planar commands still dominate the remaining failure mode.
- V39/V40 are not accepted. V39 fixes the lateral reward contract but does not
  produce enough lateral speed; V40's final and best scans still fail fixed
  lateral speed, especially right lateral motion.
- V41-V77 improve or preserve lateral gates, mixed-yaw sign, feasible command
  ranges, and server-side nominal/robust sign-gate acceptance. The limiting
  metric is no longer the repeated robust hardcase sign; it is continuous
  magnitude fidelity and off-axis crosstalk.
- No formal 1M-step PPO artifacts exist under the current V26/V23 omni
  contract.
- Fixed-mode eval, robust eval, auto-gated random-policy deploy bundles, and
  final paper summaries must be regenerated.
- Real flat/sand/slope hardware logs and videos are still missing.
- The spring-steel cable parameters are not yet identified from real
  force-displacement measurements.
- Isaac Lab is not installed in the current local Windows environment, so no
  Isaac render smoke test has been run locally yet.
- Motor step-response, IMU static-calibration, contact-drag, and mass/geometry
  CSVs are still empty templates until real bench data is collected.

## Current Verdict

The project framework is substantial, but the paper is not complete. The latest
flat V77 best line keeps the deployable ABI fixed, trains only on the server,
uses actor/critic `512-256-128`, fixes the repeated robust mixed-planar sign
failure, and provides viewable HD videos with visual spring-steel strips and
head-speed overlay. V85b adds a complete diagnostic video/scan set but does not
supersede V77 because robust slow forward-left still fails. V86b fixes the V85b
robust planar sign failures in the best robust checkpoint, but still leaves
planar magnitude and off-axis exceedances. The current best result is still not
"any velocity command can be tracked" with high fidelity: V77 best robust
passes the cleanest sign-gate line, while V86b identifies the next failure mode
as robust diagonal magnitude/crosstalk rather than pure sign confusion.

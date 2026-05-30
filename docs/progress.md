# Progress

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
- V29 was trained as the current valid flat candidate:
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
  `record/v6/omni_v29_speed_gate_videos_hd1080`: six H.264 `1920x1080`,
  25 fps, 6 s single-command videos with visual spring-steel strips and head
  speed overlay, plus `3840x1440` and `3840x2160` 3x2 comparison videos.
  The HD rollouts confirm forward/reverse and yaw signs, but also make the
  remaining lateral off-axis and yaw-only translation problems obvious. Exact
  metrics are in `docs/omni_v29_hd_recording_log.md`.

## Not Done

- Flat direction/yaw command control has a current accepted candidate, but it
  is not yet a complete paper result.
- Robust flat eval, fixed-gate ablations, deploy bundles, and final paper
  summaries must still be regenerated under the V12 adapter.
- Lateral commands still show large off-axis forward motion, and yaw-only
  commands still translate while turning; these are limitations to address
  before making strong omnidirectional-tracking claims.
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
- No formal 1M-step PPO artifacts exist under the current V12 omni contract.
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
flat V12/V13 line passes the formal flat six-direction gate while keeping the
deployable ABI fixed, and V15 now provides a documented continuous-tracking
attempt under the stronger `omni_tracking_scan_v3` selection contract. The
project is not yet at "any velocity command can be tracked": V15 stops cleanly
and preserves signed yaw separation, while V16 increases yaw authority and
reduces yaw RMSE. The main blockers remain yaw-only translation, straight-line
yaw drift, mixed planar commands, and off-axis coupling.

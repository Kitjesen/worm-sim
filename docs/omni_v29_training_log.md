# Worm V6 Omni Tracking Log - V29

Date: 2026-05-31

## Goal

Push flat-terrain Worm V6 from a six-direction prototype toward continuous
body-frame `vx/vy/yaw` tracking while keeping the deployable interface fixed:

- Observation ABI: 80D (`vx, vy, yaw` command, encoders, previous action,
  seven segment IMUs, 1 s phase clock).
- Action ABI: 12D (`11D residual motor action + 1D learned latent gait gate`).
- Actor: `512-256-128`.
- Critic: `512-256-128`.

## Implementation Changes

- `action_adapter_v6.py`: action adapter bumped to
  `cmaes_tri_anchor_auto_gate_directional_v23`.
- `worm_env_v6.py`: reward contract bumped to
  `omni_directional_offaxis_yaw_v16`.
- `train_v6.py`: default actor and critic networks changed to
  `512,256,128`.
- `transfer_actor_to_new_critic_v6.py`: actor-transfer checkpoints now write
  a matching `training_config.json`, so `train_v6.py --resume` can verify and
  resume them instead of silently starting from scratch.
- `record_eval_video_v6.py`: video writer now uses the effective frame rate
  implied by MuJoCo control-step subsampling. A 6 s run with 150 recorded
  frames is written as 25 fps, so playback duration remains 6 s.

## Speed-Adaptive Gait Gate

The learned gate is still produced by the policy, but the deployed effective
gate is command-centered:

```text
z_learned = clip(0.5 + 0.5 * 3.0 * a_gate, 0, 1)
z_g = clip(c(cmd) + 0.70 * (z_learned - 0.5), 0, 1)
```

For pure axial translation, the command center is speed-adaptive:

```text
c_axial(|vx|) = 0.35 + alpha * (0.50 - 0.35)
alpha = clip((|vx_norm| - 0.20) / (0.80 - 0.20), 0, 1)
```

This keeps slow axial commands visibly worm-like while letting full-speed
forward/reverse commands use the faster mixed anchor.

## Runs

### V27

Run:
`runs/worm_v6_ppo_flat_random_continuous_tracking_v27_speed_adaptive_gate_from_v26best`

Source:
`runs/worm_v6_ppo_flat_random_continuous_tracking_v26_actor512critic512_actortransfer/best_model.zip`

Result:

- Best selection score: `-1005953.10`.
- `tracking_gate_passed=false`, `direction_gate_passed=false`.
- Forward improved from the V26 level but remained below the fixed-command
  target: `body_vx=0.1097 m/s`.
- Conclusion: V27 proved the speed-adaptive gate helps axial speed, but the
  V26 actor had already adapted to the slower axial center.

### V28

Attempted to transfer the V22 actor to a fresh `512-256-128` critic, but the
transfer script did not write `training_config.json`. `train_v6.py` correctly
rejected the resume with `missing training_config.json` and trained from
scratch. This run is not counted as a valid continuation result.

Fix applied:
`transfer_actor_to_new_critic_v6.py` now writes the target training config.

### V29

Run:
`runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate`

Source:
`runs/worm_v6_ppo_flat_random_continuous_tracking_v22_actor512_gate_gain_from_v21best/best_model.zip`

Transfer:

- Actor copied: true.
- Critic copied: false.
- Source timestep: `230000`.
- Target actor/critic arch: `pi=[512,256,128]`, `vf=[512,256,128]`.

Best held-out selection:

- `tracking_gate_passed=false`.
- `direction_gate_passed=false`.
- `planar_velocity_rmse_m_s=0.1070`.
- `yaw_rate_rmse_rad_s=0.1729`.
- `mean_off_axis_speed_m_s=0.0318`.
- `zero_command_mean_speed_m_s=0.0000177`.
- `planar_success_rate=0.7647`.
- `yaw_success_rate=0.7647`.
- `wrong_planar_sign_count=0`.
- `wrong_yaw_sign_count=0`.
- `straight_violation_count=3`.
- `stationary_violation_count=4`.

Six fixed-command summary:

| Case | Effective gate | Learned gate | vx m/s | vy m/s | yaw rad/s | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| forward | 0.4777 | 0.4682 | 0.1384 | 0.0018 | -0.0103 | correct:straight_ok |
| reverse | 0.4881 | 0.4831 | -0.0810 | 0.0252 | -0.0166 | correct:straight_ok |
| lateral_left | 0.4763 | 0.4661 | 0.1085 | 0.0328 | 0.0777 | correct:straight_drift |
| lateral_right | 0.4809 | 0.4727 | 0.1060 | -0.0296 | -0.0804 | correct:straight_drift |
| yaw_left | 0.8206 | 0.4580 | 0.0875 | 0.0423 | 0.3768 | stationary_drift:correct |
| yaw_right | 0.8396 | 0.4852 | 0.1026 | -0.0347 | -0.4206 | stationary_drift:correct |

35-command scan:

- Artifact: `record/v6/omni_v29_speed_gate_videos/scan_35_commands.json`.
- `num_commands=35`.
- `planar_rmse_m_s=0.1665`.
- `yaw_rmse_rad_s=0.1179`.
- `planar_sign_rate=0.84`.
- `yaw_sign_rate=1.00`.
- `zero_command_mean_speed_m_s=0.000048`.
- `yaw_only_mean_planar_speed_m_s=0.0966`.

## Artifacts

- Six videos:
  `record/v6/omni_v29_speed_gate_videos/{forward,reverse,lateral_left,lateral_right,yaw_left,yaw_right}.mp4`.
- 3x2 comparison:
  `record/v6/omni_v29_speed_gate_videos/gait_comparison_3x2.mp4`.
- Per-command trajectory CSV and trajectory plot PNG files in:
  `record/v6/omni_v29_speed_gate_videos/`.
- 35-command scan JSON/CSV:
  `record/v6/omni_v29_speed_gate_videos/scan_35_commands.json`,
  `record/v6/omni_v29_speed_gate_videos/scan_35_commands.csv`.

Video verification:

- Each fixed-command MP4: `1280x720`, `25 fps`, `150 frames`, `6.0 s`.
- 3x2 comparison MP4: `1920x720`, `25 fps`, `150 frames`, `6.0 s`.
- Steel-strip visual overlay and head speed overlay are enabled.

## Current Verdict

V29 is the best current flat candidate for the revised objective. It restores
full-speed forward/reverse motion while keeping low-speed axial commands
worm-like, and the learned/effective gait gate now separates yaw from axial
translation.

It is still not a completed continuous omnidirectional tracker. The remaining
blockers are:

- lateral commands still have large forward off-axis motion,
- yaw-only commands still translate while turning,
- the independent 35-command scan has high planar RMSE,
- fixed-command `direction_gate_passed` and `tracking_gate_passed` remain false.

Paper wording should remain:
`six-direction primitive controller / weak omnidirectional prototype` until
the lateral and yaw-only drift issues are solved.

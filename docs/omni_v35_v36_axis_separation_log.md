# Worm V6 V35/V36 Axis-Separation Training Log

Date: 2026-05-31

## Purpose

Continue the flat omnidirectional-control line after V34 failed to beat V29.
The immediate target was not a paper claim; it was to reduce two blocking
failure modes:

- pure yaw commands translated forward while turning;
- pure lateral commands leaked into large axial/off-axis motion.

The deployable ABI remained fixed:

- observation: 80D `vx, vy, yaw + encoders + previous_action + 7 IMUs + 1 s phase clock`;
- policy action: 12D `11D residual motor action + 1D learned latent gait gate`;
- actor/critic: `512-256-128`.

## Code Changes

Action adapter `cmaes_tri_anchor_auto_gate_directional_v24` fixes a real bug in
the yaw-only prior path. The previous contract advertised
`YAW_ONLY_SLIDE_PRIOR_SCALE` and `YAW_ONLY_YAW_PRIOR_SCALE`, but the
`uses_inplace_yaw_prior` branch returned the raw in-place yaw prior before those
scales were applied. V24 applies the slide/yaw scales and clips the resulting
prior.

Reward contract `omni_directional_offaxis_yaw_v19` increases the penalties for
axis contamination:

| Term | Old | New | Reason |
| --- | ---: | ---: | --- |
| `W_YAW_STATIONARY` | `3.0` | `8.0` | penalize planar speed during yaw-only commands |
| `W_LATERAL_ONLY_FORWARD_DRIFT` | `2.0` | `5.0` | penalize forward drift during lateral-only commands |

A new `axis_separation` curriculum samples one active command axis at a time:
stop, forward/reverse, left/right lateral, and left/right yaw. This is meant as
a repair curriculum before returning to continuous `vx/vy/yaw` commands.

The recorder was also fixed for long clips. Recording exactly the 20 s episode
limit through `VecNormalize.step()` caused the vector environment to auto-reset
before the final pose was measured. `record_eval_video_v6.py` now steps the raw
simulation environment directly and normalizes observations manually, so 20 s
metrics and trajectories use the terminal state rather than the reset state.

## Runs

### V35: axis-separation repair

```text
runs/worm_v6_ppo_flat_random_axis_separation_v35_noeval_from_v29best
```

- resume: `runs/worm_v6_ppo_flat_random_continuous_tracking_v29_v22actor_critic512_speed_gate/best_model.zip`
- curriculum: `axis_separation`
- learning rate: `1e-4`
- envs: `1`
- directional eval during training: disabled to avoid local memory exhaustion
- completed timesteps: `300960`

### V36: continuous rejoin

```text
runs/worm_v6_ppo_flat_random_continuous_rejoin_v36_from_v35final
```

- resume: `runs/worm_v6_ppo_flat_random_axis_separation_v35_noeval_from_v29best/final_model.zip`
- curriculum: `continuous_omni`
- learning rate: `7.5e-5`
- envs: `1`
- directional eval during training: disabled to avoid local memory exhaustion
- completed timesteps: `341920`

## 35-Command Scan Results

| Version | Planar RMSE m/s | Yaw RMSE rad/s | Planar sign | Yaw sign | Zero speed m/s | Yaw-only planar speed m/s | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V29 | `0.1665` | `0.1179` | `0.84` | `1.00` | near zero | `0.0966` | current accepted viewable baseline |
| V34 | `0.1635` | `0.1128` | `0.84` | `1.00` | `0.00005` | `0.1046` | rejected |
| V35 | `0.1658` | `0.1090` | `0.84` | `1.00` | `0.00005` | `0.0631` | diagnostic improvement in yaw-only drift |
| V36 | `0.1673` | `0.0994` | `0.84` | `1.00` | `0.00005` | `0.0671` | yaw RMSE improved, planar tracking worsened |

V35/V36 therefore improved yaw-only planar drift relative to V29/V34, but did
not solve continuous planar tracking. The current accepted policy remains V29
until a run reduces lateral/off-axis error without losing the fixed axial and
yaw primitives.

## Long V35 Videos

The V35 final model was recorded for 20 s per command with visual spring-steel
strips and the head speed overlay:

```text
record/current/flat_omni_v35_axis_sep_long20
```

| Case | Forward speed mm/s | Lateral drift mm | Yaw rate rad/s | Notes |
| --- | ---: | ---: | ---: | --- |
| forward | `127.83` | `203.4` | `0.024` | usable forward motion |
| reverse | `-69.01` | `1123.9` | `0.016` | reverse sign works, large lateral drift |
| lateral_left | `23.42` | `999.2` | `-0.086` | lateral command still heavily contaminated |
| lateral_right | `21.76` | `863.9` | `0.089` | right lateral remains contaminated |
| yaw_left | `17.46` | `164.5` | `0.125` | yaw sign works, but turn rate is weak |
| yaw_right | `18.12` | `168.8` | `-0.122` | yaw sign works, but turn rate is weak |

Viewable files:

- `record/current/flat_omni_v35_axis_sep_long20/gait_comparison_3x2_20s_4k_uhd.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/gait_comparison_3x2_20s_3840x1440.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/forward_20s_1080p.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/reverse_20s_1080p.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/lateral_left_20s_1080p.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/lateral_right_20s_1080p.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/yaw_left_20s_1080p.mp4`
- `record/current/flat_omni_v35_axis_sep_long20/yaw_right_20s_1080p.mp4`

The 4K comparison video was verified as `3840x2160`, `25 fps`, `20.0 s`, and
`500` frames. Each single-command video was verified as `1920x1080`, `25 fps`,
`20.0 s`, and `500` frames.

## Verification

Passed after the code changes:

```powershell
python src\v6\test_reward_contract_v6.py
python src\v6\test_deployable_obs_v6.py
python src\v6\test_omni_eval_metrics_v6.py
python src\v6\test_visual_steel_strip_geometry_v6.py
```

Generated scan artifacts:

- `record/current/flat_omni_v35_axis_sep_scan/scan_35_commands_final.json`
- `record/current/flat_omni_v35_axis_sep_scan/scan_35_commands_final.csv`
- `record/current/flat_omni_v36_continuous_rejoin_scan/scan_35_commands_final.json`
- `record/current/flat_omni_v36_continuous_rejoin_scan/scan_35_commands_final.csv`

## Conclusion

This update produced a useful partial improvement, not a solved controller.
The yaw-only in-place prior scaling bug was fixed, and yaw-only planar drift
fell from about `0.0966-0.1046 m/s` to about `0.063-0.067 m/s`. However,
lateral-only and mixed planar commands still leak into axial/off-axis motion,
and V36 did not improve planar RMSE.

The next technical step is a lateral-specific repair stage that changes the
model-selection target, not just the reward. The next accepted run must lower
planar RMSE toward `<=0.10 m/s` and eliminate lateral sign failures while
keeping yaw-only planar speed below the current V35/V36 level.

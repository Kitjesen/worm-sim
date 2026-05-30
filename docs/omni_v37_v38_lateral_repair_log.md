# Worm V6 V37/V38 Lateral Repair Log

Date: 2026-05-31

## Purpose

Continue the flat omnidirectional line after V35/V36. The target was to reduce
pure-lateral forward leakage while preserving the deployable contract:

- observation: 80D `vx, vy, yaw + encoders + previous_action + 7 IMUs + 1 s phase clock`;
- policy action: 12D `11D residual motor action + 1D learned latent gait gate`;
- actor/critic: `512-256-128`.

This is a diagnostic update, not a completed continuous `vx/vy/yaw` tracking
result.

## Code Changes

Action adapter `cmaes_tri_anchor_auto_gate_directional_v25` changes dominant
lateral commands after prior-only diagnosis showed the previous mixed-centered
`+pi/2` lateral prior was dominated by forward off-axis speed:

- lateral gait gate center: `0.50 -> 0.00`;
- lateral slide phase offset: `+pi/2 -> -pi/2`;
- right lateral still mirrors only the yaw-anchor sign.

The V36 model was scanned under the v25 adapter before retraining. That probe
showed the adapter change reduces the sign/off-axis failure, but does not
restore enough lateral speed.

Reward contract `omni_directional_offaxis_yaw_v20` adds a narrow pure-lateral
speed-deficit penalty:

- `W_LATERAL_ONLY_SPEED_DEFICIT = 4.0`;
- `LATERAL_ONLY_PROGRESS_TARGET_M_S = 0.06`.

The intent is to prevent the policy from satisfying the pure-lateral repair
objective by barely moving laterally.

The long-video recorder was also fixed for clips beyond the 20 s environment
time limit. It now steps the raw MuJoCo environment directly, records the first
time-limit step in metrics, and continues rendering until the requested video
duration unless the robot actually terminates.

## Runs

### V37: lateral worm-centered repair

```text
runs/worm_v6_ppo_flat_random_lateral_wormcenter_v37_from_v36final
```

- resume: `runs/worm_v6_ppo_flat_random_continuous_rejoin_v36_from_v35final/final_model.zip`
- curriculum: `axis_separation`
- learning rate: `1e-4`
- envs: `1`
- completed timesteps: `382880`

### V38: lateral speed-deficit repair

```text
runs/worm_v6_ppo_flat_random_lateral_speeddeficit_v38_from_v37final
```

- resume: `runs/worm_v6_ppo_flat_random_lateral_wormcenter_v37_from_v36final/final_model.zip`
- curriculum: `axis_separation`
- learning rate: `1e-4`
- envs: `1`
- scanned checkpoint: `checkpoints/worm_v6_ppo_422880_steps.zip`
- note: the local training command was interrupted after checkpoint creation,
  before `final_model.zip` was written.

## 35-Command Scan Results

| Version | Planar RMSE m/s | Yaw RMSE rad/s | Planar sign | Yaw sign | Zero speed m/s | Yaw-only planar speed m/s | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V36 original/v24 | `0.1673` | `0.0994` | `0.84` | `1.00` | `0.00005` | `0.0671` | baseline before v25 probe |
| V36 model + v25 probe | `0.1608` | `0.0994` | `0.96` | `1.00` | `0.00005` | `0.0671` | adapter improves planar sign/off-axis, not enough lateral speed |
| V37/v25 | `0.1627` | `0.1214` | `0.96` | `1.00` | `0.00005` | `0.0634` | yaw-only planar drift slightly lower, yaw RMSE worse |
| V38/v25+v20 | `0.1677` | `0.1181` | `0.96` | `1.00` | `0.00005` | `0.0600` | yaw-only planar drift lower, planar tracking worse |

Fixed 6 s command responses:

| Version | forward vx | reverse vx | lateral left vy | lateral right vy | yaw left rate | yaw right rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V37 | `0.1426` | `-0.0709` | `0.0219` | `-0.0124` | `0.3794` | `-0.3436` |
| V38 checkpoint | `0.1314` | `-0.0710` | `0.0257` | `-0.0141` | `0.3641` | `-0.3461` |

Both V37 and V38 still fail the fixed lateral thresholds:

- lateral_left requires `body_vy > 0.03 m/s`;
- lateral_right requires `body_vy < -0.03 m/s`.

## Adapter Sweep

A focused V37 adapter probe swept lateral gate center and lateral yaw trim:

```text
record/current/flat_omni_v37_lateral_adapter_probe_sweep
```

The sweep confirmed that v25's root-cause claim is only half solved:

- worm-centered lateral commands reduce forward leakage;
- simply moving `COMMAND_GATE_LATERAL_CENTER` upward does not restore robust
  lateral speed and can flip left/right lateral signs.

The next run should therefore not repeat the same axis-separation setup. It
needs either a stronger lateral primitive search or model selection that
explicitly scores pure-lateral velocity, zero-yaw yaw drift, and mixed-yaw sign.

## 30 s Viewable Videos

V37 was recorded for 30 s per command with visual spring-steel strips and head
speed overlay:

```text
record/current/flat_omni_v37_lateral_wormcenter_long30
```

| Case | Forward speed mm/s | Lateral drift mm | Yaw rate rad/s | Notes |
| --- | ---: | ---: | ---: | --- |
| forward | `89.75` | `1935.6` | `0.066` | forward still works but drifts over 30 s |
| reverse | `-36.48` | `2133.3` | `0.035` | reverse sign works but drifts |
| lateral_left | `16.40` | `295.9` | `0.013` | lateral displacement is visible but below the speed target |
| lateral_right | `8.61` | `44.2` | `-0.001` | right lateral remains too weak |
| yaw_left | `14.27` | `181.8` | `0.098` | yaw sign works, turn rate weak |
| yaw_right | `12.49` | `154.4` | `-0.088` | yaw sign works, turn rate weak |

Viewable files:

- `record/current/flat_omni_v37_lateral_wormcenter_long30/gait_comparison_3x2_30s_4k_uhd.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/gait_comparison_3x2_30s_3840x1440.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/forward_30s_1080p.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/reverse_30s_1080p.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/lateral_left_30s_1080p.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/lateral_right_30s_1080p.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/yaw_left_30s_1080p.mp4`
- `record/current/flat_omni_v37_lateral_wormcenter_long30/yaw_right_30s_1080p.mp4`

The 4K comparison video verifies as `3840x2160`, `25 fps`, `30.0 s`, and
`750` frames. Each single-command video verifies as `1920x1080`, `25 fps`,
`30.0 s`, and `750` frames.

## Verification

Passed after the code changes:

```powershell
python src\v6\test_reward_contract_v6.py
python src\v6\test_deployable_obs_v6.py
python src\v6\test_omni_eval_metrics_v6.py
python src\v6\test_visual_steel_strip_geometry_v6.py
python -m py_compile src\v6\record_eval_video_v6.py
```

Generated scan artifacts:

- `record/current/flat_omni_v36_v25_adapter_probe_scan/scan_35_commands_final.json`
- `record/current/flat_omni_v36_v25_adapter_probe_scan/scan_35_commands_final.csv`
- `record/current/flat_omni_v37_lateral_wormcenter_scan/scan_35_commands_final.json`
- `record/current/flat_omni_v37_lateral_wormcenter_scan/scan_35_commands_final.csv`
- `record/current/flat_omni_v38_lateral_speeddeficit_scan/scan_35_commands_422880.json`
- `record/current/flat_omni_v38_lateral_speeddeficit_scan/scan_35_commands_422880.csv`

## Conclusion

This update produced useful evidence but not a solved controller. V25 reduces
pure-lateral forward leakage and improves planar sign rate from `0.84` to
`0.96`, but it also weakens lateral speed. V38's new lateral speed-deficit
reward slightly improves pure-lateral `vy` and yaw-only planar drift, but it
does not meet the fixed lateral thresholds and worsens planar RMSE.

The next technical step should be a stronger lateral primitive or model
selection loop, not another identical axis-separation continuation.

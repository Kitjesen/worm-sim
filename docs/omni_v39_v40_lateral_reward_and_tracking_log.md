# V39-V40 Lateral Reward Fix and Continuous Tracking Log

Date: 2026-05-31

This log records the current flat-terrain attempt to move Worm V6 from a
six-direction prototype toward continuous `vx/vy/yaw` tracking while preserving
the deployable ABI:

- observation: 80D `vx, vy, yaw + encoders + previous action + 7 IMUs + phase`;
- action: 12D `11D residual motor action + 1D learned latent gait gate`;
- actor and critic: `512-256-128`;
- simulator: MuJoCo.

## Code Changes

V39 fixed a reward bug in the pure-lateral speed-deficit term.

Previous behavior:

- the variable used for signed lateral progress could be overwritten by
  `off_axis_speed / speed_scale`;
- a pure-lateral command could therefore be partially rewarded for any
  perpendicular motion, even if the signed body-frame lateral velocity was too
  small.

Current behavior:

- `lateral_progress = body_vy * sign(cmd_vy)`;
- the lateral-only speed deficit is computed against signed body-frame lateral
  velocity;
- the general off-axis penalty still uses true off-axis speed.

The reward contract is now:

```text
omni_directional_offaxis_yaw_v21
```

The scan tool now reports explicit pure-lateral gates instead of hiding lateral
failure inside an average RMSE:

- `fixed_lateral_left_body_vy_m_s`
- `fixed_lateral_right_body_vy_m_s`
- `fixed_lateral_speed_gate_passed`
- `fixed_lateral_strict_gate_passed`

The fixed speed gate is intentionally aligned with the paper acceptance
threshold:

```text
lateral_left:  body_vy >= +0.03 m/s
lateral_right: body_vy <= -0.03 m/s
```

The recorder also now exports optional gate/action telemetry:

- raw gate action;
- learned and deployed gait blend;
- desired gait blend and gate error;
- prior component, residual component, and final applied action per actuator.

This does not change the 80D observation or 12D action ABI.

## V39 Short Continuation

Run:

```powershell
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum axis_separation --run-label flat_random_lateral_signedvy_v39_from_v38ckpt --timesteps 462880 --train-chunk-timesteps 40000 --n-envs 1 --device cpu --resume runs\worm_v6_ppo_flat_random_lateral_speeddeficit_v38_from_v37final\checkpoints\worm_v6_ppo_422880_steps.zip --allow-curriculum-resume --allow-contract-resume --learning-rate 1e-4 --directional-eval-freq-steps 0
```

Scan artifact:

```text
record/current/flat_omni_v39_lateral_signedvy_scan/scan_35_commands_final.json
```

Summary:

| Metric | Value |
| --- | ---: |
| planar RMSE | `0.1688 m/s` |
| yaw RMSE | `0.1183 rad/s` |
| planar sign rate | `0.96` |
| yaw sign rate | `1.00` |
| yaw-only mean planar drift | `0.0632 m/s` |
| lateral-left body vy | `+0.0231 m/s` |
| lateral-right body vy | `-0.0158 m/s` |
| fixed lateral speed gate | `false` |
| fixed lateral strict gate | `false` |

Interpretation:

- V21 is a necessary fix, but a 40k axis-separation continuation did not solve
  lateral speed.
- Right lateral remains the clearest bottleneck.
- V39 is not an accepted policy.

## Residual Authority Probe

The V39 final model was rescanned with only evaluation-time residual-scale
changes:

| Residual scale | left vy | right vy | gate |
| ---: | ---: | ---: | --- |
| `0.45` | `+0.0199` | `-0.0148` | fail |
| `0.55` | `+0.0220` | `-0.0114` | fail |
| `0.70` | `+0.0253` | `-0.0104` | fail |

Conclusion: simply increasing policy residual authority at evaluation time does
not recover lateral motion. The policy and/or lateral primitive has to be
trained or redesigned.

## V40 Continuous-Omni Restart

V40 starts again from the V37 final model rather than the V38/V39 lateral-only
basin:

```powershell
python src\v6\train_v6.py --terrain flat --gait-mode random --command-curriculum continuous_omni --run-label flat_random_continuous_omni_v40_v21_from_v37final --timesteps 900000 --train-chunk-timesteps 80000 --n-envs 1 --device cpu --resume runs\worm_v6_ppo_flat_random_lateral_wormcenter_v37_from_v36final\final_model.zip --allow-curriculum-resume --allow-contract-resume --learning-rate 7.5e-5 --directional-eval-freq-steps 20000 --directional-eval-seconds 6.0
```

Interim `442880`-step held-out directional eval:

| Case | body vx | body vy | yaw rate | status |
| --- | ---: | ---: | ---: | --- |
| forward | `+0.1321` | `+0.0255` | `+0.0794` | `correct:straight_drift` |
| reverse | `-0.0727` | `+0.0034` | `-0.0391` | `correct:straight_drift` |
| lateral_left | `+0.0282` | `+0.0225` | `+0.1106` | `correct:straight_drift` |
| lateral_right | `+0.0146` | `-0.0127` | `-0.0866` | `correct:straight_drift` |
| yaw_left | `+0.0572` | `+0.0287` | `+0.3736` | `stationary_drift:correct` |
| yaw_right | `+0.0565` | `-0.0266` | `-0.3639` | `stationary_drift:correct` |

Aggregate at this checkpoint:

| Metric | Value |
| --- | ---: |
| direction gate passed | `false` |
| tracking gate passed | `false` |
| planar RMSE | `0.0997 m/s` |
| yaw RMSE | `0.1914 rad/s` |
| mean off-axis speed | `0.0223 m/s` |
| yaw-only mean planar drift | `0.0625 m/s` |
| planar success rate | `0.7647` |
| yaw success rate | `0.4706` |
| straight violation count | `7` |

Current interpretation:

- the continuous tracking metrics are close on planar/yaw RMSE;
- the fixed lateral speed gate still fails;
- straight-line yaw drift and yaw-only translation are still too high;
- V40 should continue to its chunk end and then be scanned with the 35-command
  tool before any video is promoted as a result.

Final `462880`-step held-out directional eval:

| Case | body vx | body vy | yaw rate | status |
| --- | ---: | ---: | ---: | --- |
| forward | `+0.1209` | `-0.0064` | `+0.0284` | `correct:straight_ok` |
| reverse | `-0.0557` | `+0.0751` | `+0.0447` | `correct:straight_drift` |
| lateral_left | `+0.0251` | `+0.0273` | `+0.1183` | `correct:straight_drift` |
| lateral_right | `+0.0144` | `-0.0065` | `-0.0784` | `correct:straight_drift` |
| yaw_left | `+0.0662` | `+0.0331` | `+0.4122` | `stationary_drift:correct` |
| yaw_right | `+0.0669` | `-0.0310` | `-0.4063` | `stationary_drift:correct` |

Final held-out aggregate:

| Metric | Value |
| --- | ---: |
| direction gate passed | `false` |
| tracking gate passed | `false` |
| planar RMSE | `0.1056 m/s` |
| yaw RMSE | `0.1952 rad/s` |
| mean off-axis speed | `0.0276 m/s` |
| planar success rate | `0.7059` |
| yaw success rate | `0.5882` |
| wrong yaw sign count | `1` |
| straight violation count | `5` |

The independent 35-command scan was also run for both V40 final and V40 best:

| Model | planar RMSE | yaw RMSE | planar sign | yaw sign | left vy | right vy | lateral gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V40 final | `0.1630` | `0.1127` | `0.96` | `1.00` | `+0.0260` | `-0.0097` | fail |
| V40 best | `0.1645` | `0.1182` | `0.96` | `1.00` | `+0.0238` | `-0.0127` | fail |

Artifacts:

```text
record/current/flat_omni_v40_v21_scan/scan_35_commands_final.json
record/current/flat_omni_v40_v21_scan/scan_35_commands_final.csv
record/current/flat_omni_v40_v21_scan/scan_35_commands_best.json
record/current/flat_omni_v40_v21_scan/scan_35_commands_best.csv
```

Final V40 verdict: not accepted. The run preserves planar/yaw sign separation
in the scan, but it still lacks enough signed lateral speed, especially to the
right.

## Verification

Passed after the V21 scan/telemetry changes:

```powershell
python src\v6\test_reward_contract_v6.py
python src\v6\test_deployable_obs_v6.py
python src\v6\test_omni_eval_metrics_v6.py
python -m py_compile src\v6\record_eval_video_v6.py src\v6\worm_env_v6.py src\v6\scan_command_tracking_v6.py
```

Telemetry smoke output:

```text
record/current/telemetry_smoke/smoke_prior_lateral.mp4
record/current/telemetry_smoke/smoke_prior_lateral_telemetry.csv
record/current/telemetry_smoke/smoke_prior_lateral_telemetry.png
```

## Next Decision

If V40 still fails the lateral gate after the full chunk, the next step should
not be another identical continuation. The likely next branch is:

1. run a lateral primitive search that directly optimizes signed body-frame
   `vy` with yaw-drift and `vx` leakage penalties;
2. add the best lateral-left and lateral-right primitives as separate command
   priors;
3. restart continuous-omni training from the best V37/V40 checkpoint with the
   stronger lateral priors and the V21 reward.

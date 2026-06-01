# V70-V71 Server Training Log

Date: 2026-06-01

## Server Environment

- Host: `fe91fae6a6756695.natapp.cc:12346`
- User: `bsrl`
- Training checkout: `/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-omni`
- Branch: `codex/real-robot-experiment-todo`
- MuJoCo/RL conda env: `wormv6_np2`
- Isaac Lab env kept unchanged: `thunder2`
- `wormv6_np2` notes: cloned from `thunder2`, NumPy `2.2.6`, PyTorch
  `2.7.0+cu128`, `cma` installed for open-loop prior search.

No RL training should be run on the local Windows workstation.

## V70 Result

Run label:
`flat_random_v70_server_slow_lateral_right_repair_lowlr_from_v69cfinal_np2`

V70 oversampled the strict-scan counterexample
`cmd=(0, -0.0375, 0)` through the `slow_lateral_right_repair` curriculum and
added slow lateral commands to best-model selection.

Strict 35-command scan artifact:

```text
record/current/flat_omni_v70_server_best_final_scan/
```

Result: not accepted. The main remaining failure was still slow right lateral:
the policy moved left for `cmd_vy=-0.0375 m/s`.

Server zero-residual diagnosis showed this was not only a policy-learning
problem. The full-speed right-lateral primitive moves right at full authority,
but when the command activity scale reduces it for low-speed commands, it can
enter a wrong-sign contact basin and move left.

## V71/V37 Adapter Fix

Action adapter version:
`cmaes_tri_anchor_auto_gate_directional_v37`

Change:

- keep the 80D observation ABI unchanged;
- keep the 12D action ABI unchanged;
- keep the learned latent gait gate unchanged;
- add a separately searched `right_slow` lateral primitive;
- use `right_slow` only for dominant negative-`vy` commands with normalized
  magnitude `<= 0.65`.

Slow-right prior search artifacts:

```text
record/current/flat_omni_v71_slow_right_prior_search/
record/current/flat_omni_v71_slow_right_prior_search_multiseed/
```

The multi-seed selected primitive targets `vy=-0.0375 m/s` under deployment
activity scale `0.60`. Validation aggregate:

- mean signed lateral speed: `0.03710 m/s`
- mean absolute forward speed: `0.00484 m/s`
- mean absolute yaw rate: `0.10735 rad/s`
- wrong sign count: `0/10`
- accepted seed count by the simple lateral gate: `8/10`

## V37 No-Retrain Scan

Model:
`runs/worm_v6_ppo_flat_random_v70_server_slow_lateral_right_repair_lowlr_from_v69cfinal_np2/best_model.zip`

Artifact:

```text
record/current/flat_omni_v71_v37_noretrain_scan/
```

Strict analyzer verdict: accepted.

Key metrics:

- `num_commands=35`
- `planar_rmse_m_s=0.05689`
- `yaw_rmse_rad_s=0.01885`
- `wrong_planar_sign_count=0`
- `wrong_yaw_sign_count=0`
- `planar_error_exceed_count=0`
- fixed lateral strict gate: passed

This is the first flat random controller combination in this line that passes
the strict 35-command scan. It is still a no-retrain adapter-policy
combination. V71 fine-tuning then produced trained best and final checkpoints
under the V37 action-adapter contract.

## V71 Training

tmux session:
`worm_v71_v37_np2_lowlr`

Run label:
`flat_random_v71_server_v37_slow_right_prior_lowlr_from_v70best_np2`

Resume:
`runs/worm_v6_ppo_flat_random_v70_server_slow_lateral_right_repair_lowlr_from_v69cfinal_np2/best_model.zip`

Training settings:

- terrain: `flat`
- gait mode: `random`
- command curriculum: `continuous_omni`
- learning rate: `2e-6`
- n envs: `8`
- device: CUDA, mapped from physical GPU 2
- actor network: `512,256,128`
- critic network: `512,256,128`
- nominal target timesteps for this chunk: `2,066,768`
- saved final checkpoint step: `2,096,768`

Log:

```text
server_logs/flat_random_v71_server_v37_slow_right_prior_lowlr_from_v70best_np2.log
```

The SB3 rollout chunk finished at `131,072` fresh rollout steps, so the saved
final checkpoint is `2,096,768` rather than exactly `2,066,768`.

## V71 Trained Strict Scan

Artifact:

```text
record/current/flat_omni_v71_server_v37_trained_scan/
```

V71 best:

- analyzer verdict: accepted
- `planar_rmse_m_s=0.05902`
- `yaw_rmse_rad_s=0.01852`
- `wrong_planar_sign_count=0`
- `wrong_yaw_sign_count=0`
- `fixed_lateral_strict_gate_passed=true`

V71 final:

- analyzer verdict: accepted
- `planar_rmse_m_s=0.06150`
- `yaw_rmse_rad_s=0.01860`
- `wrong_planar_sign_count=0`
- `wrong_yaw_sign_count=0`
- `fixed_lateral_strict_gate_passed=true`

The directional best-model summary still marks `direction_gate_passed=false`
for some yaw/straight-yaw cases, so the result should be stated narrowly:
V71 passes the current nominal strict 35-command scan on flat terrain. It does
not yet pass the robust scan described below.

## V71 Robust Scan

Artifact:

```text
record/current/flat_omni_v71_server_v37_robust_scan/
```

Evaluation condition:

- `eval_condition=robust_sensor_delay_sat_v1`
- encoder position noise `0.01`
- encoder velocity noise `0.02`
- IMU gravity noise `0.01`
- IMU gyro noise `0.01`
- action delay `1` control step
- action saturation `0.90`

V71 best robust:

- analyzer verdict: rejected
- failed condition: `wrong_planar_sign_count`
- dominant failure group: `mixed_vx_vy`
- `planar_rmse_m_s=0.05816`
- `yaw_rmse_rad_s=0.02155`
- `wrong_planar_sign_count=1`
- `wrong_yaw_sign_count=0`
- `fixed_lateral_strict_gate_passed=true`
- concrete counterexample: `cmd=(+0.05,+0.075,0)` measured
  `body_vx=-0.03496 m/s`, `body_vy=-0.00321 m/s`

V71 final robust:

- analyzer verdict: rejected
- failed condition: `wrong_planar_sign_count`
- dominant failure group: `mixed_vx_vy`
- `planar_rmse_m_s=0.05908`
- `yaw_rmse_rad_s=0.02047`
- `wrong_planar_sign_count=1`
- `wrong_yaw_sign_count=0`
- `fixed_lateral_strict_gate_passed=false`
- concrete counterexample: `cmd=(+0.05,+0.075,0)` measured
  `body_vx=-0.03489 m/s`, `body_vy=+0.01432 m/s`

Conclusion: V71 should be cited as the current nominal flat strict-scan
checkpoint, not as a robust continuous tracker. The next repair target is the
low-speed forward-left diagonal under the feasible command envelope.

## V72 Forward-Left Repair Scan

Training:

```text
runs/worm_v6_ppo_flat_random_v72_server_forward_left_repair_from_v71best_np2/
```

Artifacts:

```text
record/current/flat_omni_v72_server_forward_left_repair_scan/
```

V72 continued from the V71 best checkpoint with
`command_curriculum=feasible_forward_diagonal_repair`, actor/critic
`512-256-128`, unchanged 80D observation ABI, and unchanged 12D
residual-plus-gate action ABI. The chunk saved the final checkpoint at
`2,211,768` total steps.

Nominal scans:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V72 best nominal | true | `0.05555` | `0.01943` | `0` | `0` | true |
| V72 final nominal | true | `0.06127` | `0.01922` | `0` | `0` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V72 best robust | false | `0.06319` | `0.01990` | `1` | `0` | false |
| V72 final robust | false | `0.06425` | `0.01985` | `1` | `0` | true |

The robust counterexample remains `cmd=(+0.05,+0.075,0)`. V72 final robust
measures `body_vx=-0.03233 m/s`, `body_vy=+0.00594 m/s`, which is a slight
improvement over V71 but still has the wrong projected planar sign. The next
iteration should repair this specific diagonal without sacrificing the nominal
strict-scan acceptance or fixed lateral gates.

## V73 Robust Resume Scan

Artifact:

```text
record/current/flat_omni_v73_server_robust_forward_left_scan/
```

V73 continued from the V72 final checkpoint in the clean server checkout:

```text
/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-v71-clean
```

The run used `wormv6_np2` with Python `3.11`, NumPy `2.2.6`, PyTorch
`2.7.0+cu128`, SB3 `2.7.0`, MuJoCo `3.3.3`, and Gymnasium `1.2.0`.
Training used the robust sensor/delay/saturation condition directly:
encoder-position noise `0.01`, encoder-velocity noise `0.02`, IMU gravity
noise `0.01`, IMU gyro noise `0.01`, one action-delay step, and action
saturation `0.90`. The actor and critic remained `512-256-128`, and the 80D
observation plus 12D residual-plus-gate action ABI did not change.

Nominal scans:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V73 best nominal | true | `0.06052` | `0.02059` | `0` | `0` | true |
| V73 final nominal | true | `0.05662` | `0.02059` | `0` | `0` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V73 best robust | false | `0.06293` | `0.02084` | `1` | `0` | true |
| V73 final robust | false | `0.05837` | `0.02132` | `1` | `0` | true |

V73 improves robust planar RMSE relative to V72 final and preserves the fixed
lateral strict gate, but it still does not pass the robust acceptance gate. The
remaining robust counterexample is still the slow forward-left diagonal
`cmd=(+0.05,+0.075,0)`. V73 final robust measures
`body_vx=-0.03165 m/s`, `body_vy=+0.01286 m/s`, so the policy still reverses
the forward component under the robust perturbation.

## V74 Targeted Robust Forward-Left Scan

Artifact:

```text
record/current/flat_omni_v74_server_robust_forward_left_targeted_scan/
```

V74 continued from the V73 final checkpoint in the clean server checkout:

```text
/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-v71-clean
```

The run used the same server-only `wormv6_np2` environment and kept the
deployable ABI unchanged: 80D observation, 12D residual-plus-gate action, and
actor/critic `512-256-128`. Training used the targeted
`robust_forward_left_diagonal_repair` curriculum with robust sensor noise,
one action-delay step, and action saturation `0.90`.

Nominal scans:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V74 best nominal | true | `0.05529` | `0.01920` | `0` | `0` | true |
| V74 final nominal | true | `0.05925` | `0.01919` | `0` | `0` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V74 best robust | false | `0.05602` | `0.02028` | `1` | `0` | true |
| V74 final robust | false | `0.05915` | `0.02019` | `1` | `0` | false |

V74 preserves nominal strict-scan acceptance but does not solve robust
continuous tracking. The repeated robust counterexample is still the slow
forward-left diagonal `cmd=(+0.05,+0.075,0)`. V74 best robust measures
`body_vx=-0.03253 m/s`, `body_vy=+0.01780 m/s`; V74 final robust measures
`body_vx=-0.03531 m/s`, `body_vy=+0.01460 m/s`. The final checkpoint also
regresses the fixed lateral-left strict speed gate under robust perturbation,
so V74 best is the better robust candidate but is still rejected.

## V71 HD Video Evidence

Artifact:

```text
record/current/flat_omni_v71_server_v37_hd20/
```

The V71 best checkpoint was recorded on the server with visual spring-steel
strips and head-speed overlay. The directory contains:

- six 20 s fixed-command `1920x1080` videos;
- one 24 s continuous command sweep `1920x1080` video;
- one `3840x1440` 3x2 comparison video;
- metrics JSON, trajectory CSV/PNG, telemetry CSV/PNG, and stdout logs for each
  command;
- `video_manifest.json` and `video_manifest.md`.

Manifest command-direction summary:

| Case | Duration | Frames | Speed mm/s | Yaw rad/s | Mean gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward | 20.0 | 334 | 66.06 | 0.065 | 0.403 |
| reverse | 20.0 | 334 | -53.53 | 0.034 | 0.482 |
| lateral_left | 20.0 | 334 | 31.95 | 0.130 | 0.028 |
| lateral_right | 20.0 | 334 | 37.34 | -0.065 | 0.005 |
| yaw_left | 20.0 | 334 | 4.69 | 0.047 | 0.894 |
| yaw_right | 20.0 | 334 | 4.42 | -0.047 | 0.893 |
| continuous_sweep | 24.0 | 400 | -23.09 | 0.003 | 0.293 |

The comparison video probes as `3840x1440`, `334` frames, `16.67 fps`, with a
non-black sample-frame mean of `160.38`.

Telemetry mean body-frame lateral velocity has the intended fixed-command sign:
lateral-left is `+0.053 m/s`, and lateral-right is `-0.118 m/s`.

Next required output:

- preserve the nominal strict-scan acceptance while repairing the robust
  `cmd=(+0.05,+0.075,0)` mixed-planar counterexample;
- rescan the next checkpoint under both nominal and robust conditions.

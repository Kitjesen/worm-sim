# V70-V89a Server Training Log

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

## V75 Mixed-Planar Continuous Gate Rejection

V75 tested a stronger mixed-planar gate idea after the repeated robust
`cmd=(+0.05,+0.075,0)` failure, but it is not accepted as a default controller.

The no-retrain V75 gate on the V74 final checkpoint regressed scan quality:

| Scan | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | ---: | ---: | ---: | ---: | --- |
| V75 no-retrain nominal | `0.06939` | `0.01955` | `1` | `0` | true |
| V75 no-retrain robust | `0.07895` | `0.01935` | `4` | `0` | true |

An early retrain around `2.57M` total steps also did not recover the gate:
the best directional summary reported `planar_success=0.913`,
`yaw_success=0.696`, `wrong_planar=1`, `wrong_yaw=1`, and
`straight_violation=6`; the last summary regressed to `planar_success=0.826`,
`yaw_success=0.609`, `wrong_planar=3`, `wrong_yaw=2`, and
`straight_violation=7`.

Conclusion: the broad V75 gate is rejected. Future hardcase gate experiments
must stay narrow and default-off until trained and rescanned.

## V76 Hardcase Selection Scan And HD Videos

Artifacts:

```text
record/current/flat_omni_v76_server_hardcase_selection_scan/
record/current/flat_omni_v76_server_hardcase_selection_videos/
```

V76 continued from the V74 final checkpoint in the clean server checkout:

```text
/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training
```

The run used the server `wormv6_np2` environment and kept the deployable ABI
unchanged: 80D observation, 12D residual-plus-gate action, and actor/critic
`512-256-128`. Training used the `robust_forward_left_diagonal_repair`
curriculum with robust sensor noise, one action-delay step, and action
saturation `0.90`. The best-eval schedule also includes the repeated
mixed-planar hardcases.

Nominal scans:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V76 best nominal | true | `0.05713` | `0.01889` | `0` | `0` | true |
| V76 final nominal | false | `0.05699` | `0.01858` | `1` | `0` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V76 best robust | false | `0.05680` | `0.02052` | `1` | `0` | true |
| V76 final robust | false | `0.05900` | `0.02040` | `1` | `0` | true |

Repeated hardcase measurements:

| Checkpoint | body vx | body vy | body yaw | Planar sign ok | Mean deployed gate | Mean learned gate |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| V76 best nominal | `-0.02105` | `+0.05937` | `+0.11899` | true | `0.032` | `0.454` |
| V76 best robust | `-0.03818` | `-0.00377` | `+0.00683` | false | `0.025` | `0.469` |
| V76 final nominal | `-0.03298` | `+0.04879` | `+0.08741` | true | `0.031` | `0.453` |
| V76 final robust | `-0.03647` | `+0.01880` | `+0.02311` | false | `0.025` | `0.467` |

The learned latent gate is around `0.45-0.47`, but the deployed gate stays near
the lateral primitive (`0.025-0.032`) for the hardcase. This explains why the
policy still loses the forward component under robust perturbation. A narrow,
default-off hardcase gate flag was added for the next server retrain:

```text
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
```

The V76 video directory contains six 12 s fixed-command `1920x1080` videos, one
12 s hard forward-left diagnostic video, one 24 s continuous sweep, metrics
JSON, trajectory CSV/PNG, telemetry CSV/PNG, and `video_manifest.md/json`.

Manifest command-direction summary:

| Case | Speed mm/s | Yaw rad/s | Mean gate |
| --- | ---: | ---: | ---: |
| forward | `75.77` | `0.083` | `0.400` |
| reverse | `-66.91` | `0.037` | `0.490` |
| lateral_left | `12.30` | `0.134` | `0.031` |
| lateral_right | `-22.74` | `-0.014` | `0.005` |
| yaw_left | `8.32` | `0.076` | `0.900` |
| yaw_right | `7.31` | `-0.075` | `0.904` |
| hard_forward_left | `-5.80` | `0.115` | `0.032` |
| continuous_sweep | `-19.41` | `0.031` | `0.296` |

Conclusion: V76 best nominal is the current flat strict-scan candidate with
viewable videos. It is not a robust continuous tracker yet. The next accepted
output is V77: train with the narrow hardcase gate enabled on the server, then
rescan both nominal and robust conditions.

## V77 Narrow Hardcase Gate Scan And HD Videos

Artifacts:

```text
record/current/flat_omni_v77_server_hardcase_gate_scan/
record/current/flat_omni_v77_server_hardcase_gate_videos/
```

V77 continued from the V76 best checkpoint in the same clean server checkout:

```text
/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training
```

The run kept the deployable ABI unchanged: 80D observation, 12D
residual-plus-gate action, and actor/critic `512-256-128`. The only intended
adapter change was enabling the narrow gate:

```text
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
```

Nominal scans:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V77 best nominal | true | `0.05513` | `0.01856` | `0` | `0` | true |
| V77 final nominal | true | `0.06047` | `0.01863` | `0` | `0` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V77 best robust | true | `0.05967` | `0.02028` | `0` | `0` | true |
| V77 final robust | false | `0.06419` | `0.02037` | `1` | `0` | false |

Repeated hardcase measurements:

| Checkpoint | body vx | body vy | body yaw | Planar sign ok | Mean deployed gate | Mean learned gate |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| V77 best nominal | `-0.02468` | `+0.06090` | `+0.10799` | true | `0.253` | `0.454` |
| V77 best robust | `-0.03548` | `+0.02522` | `+0.03460` | true | `0.255` | `0.465` |
| V77 final nominal | `-0.03001` | `+0.05138` | `+0.09517` | true | `0.253` | `0.450` |
| V77 final robust | `-0.03600` | `+0.01510` | `+0.01205` | false | `0.255` | `0.465` |

V77 best robust fixes the repeated V71-V76 robust sign failure. The mechanism is
also visible in telemetry: the deployed gate for the hardcase moves from the
near-lateral V76 value (`~0.025`) to the intended narrow-gate value (`~0.255`,
target `0.28`), while the learned gate remains around `0.46`.

This is still not a complete arbitrary continuous tracking solution. V77 best
nominal has `off_axis_exceed_count=3` and `planar_error_exceed_count=1`; V77
final robust regresses. The paper-facing checkpoint for this iteration is V77
best, not V77 final.

The V77 video directory contains six 12 s fixed-command `1920x1080` videos, one
12 s hard forward-left diagnostic video, one 24 s continuous sweep, metrics
JSON, trajectory CSV/PNG, telemetry CSV/PNG, and `video_manifest.md/json`.

Manifest command-direction summary:

| Case | Speed mm/s | Yaw rad/s | Mean gate |
| --- | ---: | ---: | ---: |
| forward | `73.10` | `0.063` | `0.397` |
| reverse | `-58.33` | `0.039` | `0.489` |
| lateral_left | `13.88` | `0.136` | `0.032` |
| lateral_right | `2.72` | `-0.040` | `0.006` |
| yaw_left | `8.68` | `0.077` | `0.900` |
| yaw_right | `6.84` | `-0.073` | `0.905` |
| hard_forward_left | `-7.01` | `0.111` | `0.247` |
| continuous_sweep | `-21.22` | `-0.001` | `0.296` |

Conclusion: V77 best is the current flat robust sign-gate candidate with
viewable videos. The next work is to verify whether additional mixed-planar
hardcase repair can improve magnitude and diagonal robustness without losing
V77's robust sign gate.

## V85b Mixed-Planar Hardcase Diagnostic And HD Videos

Artifacts:

```text
record/current/flat_omni_v85b_server_mixed_planar_hardcase_scan/
record/current/flat_omni_v85b_server_mixed_planar_hardcase_videos/
```

V85b continued from the V77 best checkpoint on the clean server checkout with
the narrow hardcase gate enabled:

```text
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
```

The run used `mixed_planar_hardcase_repair`, preserved the 80D observation ABI,
preserved the 12D residual-plus-gate action ABI, and used actor/critic
`512-256-128`.

Nominal scans:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V85b best nominal | true | `0.05620` | `0.01861` | `0` | `0` | `2` | `1` | true |
| V85b final nominal | true | `0.05675` | `0.01881` | `0` | `0` | `2` | `1` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V85b best robust | false | `0.06144` | `0.02042` | `2` | `0` | `0` | `2` | true |
| V85b final robust | false | `0.06254` | `0.02027` | `1` | `0` | `1` | `5` | true |

Remaining robust diagonal failures:

| Checkpoint | Command | body vx | body vy | body yaw | Planar sign ok | Mean deployed gate |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| V85b best robust | `(+0.05,-0.075,0)` | `-0.06787` | `-0.03134` | `+0.06050` | false | `0.223` |
| V85b best robust | `(+0.05,+0.075,0)` | `-0.03492` | `+0.01553` | `+0.02562` | false | `0.256` |
| V85b final robust | `(+0.05,+0.075,0)` | `-0.03670` | `+0.01979` | `+0.01997` | false | `0.256` |

The final checkpoint fixes the right-diagonal robust sign failure that appears
in the best checkpoint, but it still fails the slow forward-left diagonal and
has more planar magnitude errors. V85b is therefore a diagnostic result, not a
new accepted flat checkpoint.

The V85b video directory contains six 12 s fixed-command `1920x1080` videos,
two 12 s hard diagonal diagnostic videos, one 24 s continuous sweep, metrics
JSON, trajectory CSV/PNG, telemetry CSV/PNG, and `video_manifest.md/json`.

Manifest command-direction summary:

| Case | Speed mm/s | Yaw rad/s | Mean gate |
| --- | ---: | ---: | ---: |
| forward | `64.38` | `0.083` | `0.399` |
| reverse | `-55.19` | `0.050` | `0.490` |
| lateral_left | `1.92` | `0.126` | `0.032` |
| lateral_right | `-10.34` | `-0.043` | `0.005` |
| yaw_left | `7.62` | `0.077` | `0.901` |
| yaw_right | `7.46` | `-0.075` | `0.905` |
| hard_forward_left | `-0.37` | `0.125` | `0.248` |
| hard_forward_right | `21.62` | `-0.079` | `0.212` |
| continuous_sweep | `-21.86` | `0.007` | `0.295` |

Conclusion: V85b shows that broad mixed-planar hardcase training preserves the
nominal sign gate, but the robust diagonal controller is still not reliable.
The next server run is V86b: resume from V85b final with
`robust_forward_left_diagonal_repair` and accept it only if both nominal and
robust 35-command scans pass.

## V86b Robust Forward-Left Repair And HD Videos

Artifacts:

```text
record/current/flat_omni_v86b_server_robust_forward_left_scan/
record/current/flat_omni_v86b_server_robust_forward_left_videos/
```

V86b continued from the V85b final checkpoint on the clean server checkout with
the same deployable interfaces: 80D observation ABI, 12D residual-plus-gate
action ABI, actor/critic `512-256-128`, and
`WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1`.

The server environment was verified after the run with the `wormv6_np2`
interpreter:

```text
/home/bsrl/miniconda3/envs/wormv6_np2/bin/python
Python 3.11.14, NumPy 2.2.6, PyTorch 2.7.0+cu128, MuJoCo 3.3.3,
Gymnasium 1.2.0, SB3 2.7.0, CUDA available on 8 RTX 3090 GPUs
```

The server test set passed:

```text
test_reward_contract_v6.py
test_deployable_obs_v6.py
test_omni_eval_metrics_v6.py
test_visual_steel_strip_geometry_v6.py
```

Nominal scans:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V86b best nominal | true | `0.05702` | `0.01897` | `0` | `0` | `3` | `2` | true |
| V86b final nominal | true | `0.05730` | `0.01909` | `0` | `0` | `3` | `2` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V86b best robust | true | `0.05855` | `0.02027` | `0` | `0` | `1` | `2` | true |
| V86b final robust | false | `0.05765` | `0.02012` | `1` | `0` | `1` | `1` | true |

Remaining V86b best-robust hard failures:

| Command | body vx | body vy | body yaw | Planar error | Off-axis | Mean deployed gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(-0.10,+0.075,0)` | `-0.05537` | `-0.01768` | `-0.07050` | `0.10286` | `0.04736` | `0.495` |
| `(+0.05,-0.075,0)` | `-0.05531` | `-0.09431` | `-0.00548` | `0.10707` | `0.09834` | `0.222` |

Manifest command-direction summary:

| Case | Speed mm/s | Yaw rad/s | Mean gate |
| --- | ---: | ---: | ---: |
| forward | `69.55` | `0.038` | `0.396` |
| reverse | `-24.98` | `0.066` | `0.491` |
| lateral_left | `5.28` | `0.127` | `0.032` |
| lateral_right | `-37.63` | `-0.025` | `0.005` |
| yaw_left | `8.65` | `0.077` | `0.901` |
| yaw_right | `6.25` | `-0.075` | `0.906` |
| hard_forward_left | `-1.47` | `0.116` | `0.247` |
| hard_forward_right | `-55.70` | `-0.026` | `0.212` |
| continuous_sweep | `-18.51` | `0.011` | `0.295` |

Conclusion: V86b fixes the V85b robust sign failures in the best checkpoint but
does not yet meet the continuous-tracking target because planar magnitude and
off-axis errors remain above threshold. It is a server-verified diagnostic
candidate, not the final accepted flat policy.

## V87 V50 Mixed-Command Composition Server Scan

Artifacts:

```text
record/current/flat_omni_v87_server_v50_mixed_composition_scan/
```

V87 continued from the V86b best checkpoint on the server checkout and directly
tested the V50 mixed-command composition branch:

```text
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1
```

The run label was
`flat_random_v87_server_v50_mixed_composition_from_v86bbest_np2`. It preserved
the deployable interfaces: 80D observation ABI, 12D residual-plus-gate action
ABI, and actor/critic `512-256-128`.

Nominal scans:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V87 best nominal | false | `0.09427` | `0.01897` | `3` | `0` | `4` | `7` | true |
| V87 final nominal | false | `0.10410` | `0.01918` | `4` | `0` | `5` | `8` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V87 best robust | false | `0.09867` | `0.02018` | `3` | `0` | `6` | `8` | true |
| V87 final robust | false | `0.09861` | `0.02019` | `4` | `0` | `5` | `6` | true |

Representative robust regressions:

| Checkpoint | Command | body vx | body vy | body yaw | Planar error | Off-axis |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V87 best robust | `(-0.10,-0.075,0)` | `+0.099` | `-0.139` | `-0.236` | `0.209` | `0.171` |
| V87 best robust | `(-0.05,-0.075,0)` | `+0.164` | `-0.117` | `-0.292` | `0.218` | `0.201` |
| V87 best robust | `(+0.10,+0.075,0)` | `-0.012` | `-0.032` | `-0.049` | `0.155` | `0.019` |

Conclusion: V87 is a negative V50 ablation. Enabling componentwise
mixed-command composition preserves the fixed lateral strict gates, but it
regresses planar sign reliability and mixed-planar magnitude/crosstalk. It
should not be promoted as the default control path.

## V88 Default Adapter Mixed-Composition Repair Server Scan

Artifacts:

```text
record/current/flat_omni_v88_server_mixed_composition_default_scan/
```

V88 used the same `mixed_composition_repair` curriculum as V87 but returned to
the default hardcase-gated adapter. It kept:

```text
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
```

and did not enable `WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION`. The run label
was `flat_random_v88_server_mixed_composition_default_from_v86bbest_np2`. The
80D observation ABI, 12D residual-plus-gate action ABI, and actor/critic
`512-256-128` were unchanged.

The server test set passed after the run with
`/home/bsrl/miniconda3/envs/wormv6_np2/bin/python`:

```text
test_reward_contract_v6.py
test_deployable_obs_v6.py
test_omni_eval_metrics_v6.py
test_visual_steel_strip_geometry_v6.py
```

Nominal scans:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V88 best nominal | true | `0.05777` | `0.01888` | `0` | `0` | `2` | `2` | true |
| V88 final nominal | true | `0.05331` | `0.01880` | `0` | `0` | `3` | `2` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted sign gate | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V88 best robust | false | `0.05743` | `0.02015` | `1` | `0` | `1` | `1` | true |
| V88 final robust | false | `0.05911` | `0.01977` | `1` | `0` | `0` | `2` | true |

Remaining V88 final-robust hard failures:

| Command | body vx | body vy | body yaw | Planar error | Planar sign ok |
| --- | ---: | ---: | ---: | ---: | --- |
| `(+0.05,+0.075,0)` | `-0.032` | `+0.017` | `+0.032` | `0.100` | false |
| `(+0.10,-0.075,0)` | `+0.042` | `+0.007` | `+0.041` | `0.100` | true |

Conclusion: V88 confirms that the default hardcase-gated adapter is the better
path than the V50 componentwise composition flag. V88 final nominal is the
strongest clean scan in this branch, but robust evaluation still has one
mixed-planar sign failure and two planar-error exceedances. The current flat
controller is therefore still a strong diagnostic prototype, not a final
continuous `vx/vy/yaw` tracker.

## V89a Robust Forward-Diagonal Repair Server Scan

Artifacts:

```text
record/current/flat_omni_v89a_server_robust_forward_diag_scan/
```

V89a continued from the V88 final checkpoint:

```text
runs/worm_v6_ppo_flat_random_v88_server_mixed_composition_default_from_v86bbest_np2/final_model.zip
```

The first V89 launch used a formal target below the resumed checkpoint's
`3,007,128` accumulated steps, so it did not train and is excluded from the
experimental result. V89a corrected the target to `3,107,128` and ran an
actual continuation under `robust_forward_left_diagonal_repair` with robust
sensor noise, one action-delay step, and `0.90` action saturation. It kept:

```text
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=0
```

The 80D observation ABI, 12D residual-plus-gate action ABI, and actor/critic
`512-256-128` were unchanged.

Nominal scans:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V89a best nominal | true | `0.06092` | `0.01886` | `0` | `0` | `4` | `2` | true |
| V89a final nominal | true | `0.05324` | `0.01922` | `0` | `0` | `2` | `0` | true |

Robust scans under `robust_sensor_delay_sat_v1`:

| Checkpoint | Accepted | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Off-axis exceed | Planar-error exceed | Lateral strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V89a best robust | false | `0.06180` | `0.01985` | `1` | `0` | `1` | `2` | true |
| V89a final robust | false | `0.06322` | `0.02015` | `1` | `0` | `1` | `3` | true |

Remaining V89a final-robust hard failures:

| Command | body vx | body vy | body yaw | Planar error | Planar sign ok |
| --- | ---: | ---: | ---: | ---: | --- |
| `(+0.05,+0.075,0)` | `-0.041` | `+0.005` | `+0.016` | `0.115` | false |
| `(+0.10,-0.075,0)` | `+0.049` | `+0.030` | `+0.108` | `0.117` | true |
| `(-0.10,+0.075,0)` | `-0.056` | `-0.028` | `-0.105` | `0.112` | true |

Conclusion: V89a final nominal is now the cleanest nominal flat scan because it
removes all nominal planar-error exceedances while preserving zero planar/yaw
sign failures. Robust evaluation still fails on mixed `vx/vy` commands. The
next run should keep the V50 componentwise composition flag disabled and use a
robust mixed-planar repair that also preserves yaw-only and forward-yaw cases.

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

Next required output after V89a:

- keep the V50 componentwise mixed-command composition flag default-off;
- continue from the V89a/V88 default-adapter line with a robust mixed-planar
  magnitude/sign repair target that preserves yaw-only and forward-yaw cases;
- accept the next flat checkpoint only if nominal and robust 35-command scans
  have zero planar/yaw sign failures and reduce mixed-planar magnitude/off-axis
  exceedances;
- regenerate videos, deploy bundle, learned-gate ablations, and sand/slope
  transfer only after the robust flat gate is solved.

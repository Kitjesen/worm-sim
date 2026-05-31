# Worm V6 strict command-tracking proof note

Date: 2026-05-31

## Question

The current research target is not only to show six visible movement
directions. The paper-level claim is stronger:

```text
continuous body-frame vx/vy/yaw command tracking on flat terrain
```

The current evidence must therefore separate two statements:

1. Six primitive commands produce the correct direction signs.
2. Arbitrary mixed `vx/vy/yaw` commands are tracked with bounded component
   error.

The first statement is necessary but not sufficient for the second.

## Formal acceptance rule

Let a fixed-command scan contain rows

```text
r_i = (c_i, y_i)
c_i = [vx_cmd, vy_cmd, yaw_rate_cmd]
y_i = [vx_meas, vy_meas, yaw_rate_meas]
e_i = y_i - c_i
```

A policy may be called continuous body-frame command tracking only if the scan
satisfies all gates:

```text
planar_rmse = sqrt(mean(||[e_vx, e_vy]||_2^2)) <= 0.10 m/s
yaw_rmse    = sqrt(mean(e_yaw^2))              <= 0.20 rad/s
wrong_planar_sign_count = 0
wrong_yaw_sign_count    = 0
mean_off_axis_speed     <= 0.08 m/s
zero_command_speed      <= 0.02 m/s
yaw_only_planar_speed   <= 0.08 m/s
```

In addition, the scan is decomposed into command classes:

```text
stop, pure_vx, pure_vy, pure_yaw,
mixed_vx_vy, mixed_vx_yaw, mixed_vy_yaw, full_mixed
```

This gives a constructive counterexample test: if any command class has a high
exceedance count or a worst command with error above the bound, then the
continuous-tracking claim is rejected even when primitive directions look good
in video.

The implementation is:

```text
src/v6/analyze_command_scan_v6.py
src/v6/test_command_scan_analysis_v6.py
```

## Evidence

Sources:

```text
record/current/flat_omni_v41_v26_scan_with_v47_metrics/strict_scan_analysis.md
record/current/flat_omni_v47_tracking_cost_scan/strict_scan_analysis.md
record/current/flat_omni_v48_mixed_composition_repair_scan/strict_scan_analysis.md
record/current/flat_omni_v49_mixed_yaw_sign_adapter_scan/strict_scan_analysis.md
```

V41, V47, V48, and the V49 adapter scan all fail the same formal acceptance
gate:

| Candidate | Planar RMSE | Yaw RMSE | Wrong planar sign | Wrong yaw sign | Dominant failure |
| --- | ---: | ---: | ---: | ---: | --- |
| V41 best rescanned | `0.1517` | `0.1105` | `0` | `0` | `mixed_vx_vy` |
| V47 final smoke | `0.1516` | `0.1183` | `0` | `0` | `mixed_vx_vy` |
| V48 final smoke | `0.1515` | `0.1196` | `0` | `0` | `mixed_vx_vy` |
| V49 adapter scan | `0.1515` | `0.1196` | `0` | `0` | `mixed_vx_vy` |

Command-class decomposition:

| Candidate | Class | Planar RMSE | Yaw RMSE | Planar exceed | Yaw exceed |
| --- | --- | ---: | ---: | ---: | ---: |
| V41 | `pure_vx` | `0.1195` | `0.0337` | `3/4` | `0/4` |
| V41 | `pure_vy` | `0.0724` | `0.0984` | `0/4` | `0/4` |
| V41 | `pure_yaw` | `0.0661` | `0.1236` | `0/4` | `0/4` |
| V41 | `mixed_vx_vy` | `0.1762` | `0.0552` | `15/16` | `0/16` |
| V41 | `mixed_vx_yaw` | `0.1334` | `0.2976` | `4/6` | `6/6` |
| V47 | `pure_vx` | `0.1254` | `0.0188` | `2/4` | `0/4` |
| V47 | `pure_vy` | `0.0681` | `0.0991` | `0/4` | `0/4` |
| V47 | `pure_yaw` | `0.0672` | `0.1323` | `0/4` | `0/4` |
| V47 | `mixed_vx_vy` | `0.1755` | `0.0807` | `13/16` | `1/16` |
| V47 | `mixed_vx_yaw` | `0.1331` | `0.2826` | `4/6` | `5/6` |
| V48 | `pure_vx` | `0.1268` | `0.0139` | `2/4` | `0/4` |
| V48 | `pure_vy` | `0.0723` | `0.1035` | `0/4` | `0/4` |
| V48 | `pure_yaw` | `0.0633` | `0.1337` | `0/4` | `0/4` |
| V48 | `mixed_vx_vy` | `0.1747` | `0.0692` | `14/16` | `0/16` |
| V48 | `mixed_vx_yaw` | `0.1281` | `0.2884` | `4/6` | `6/6` |
| V49 | `pure_vx` | `0.1268` | `0.0139` | `2/4` | `0/4` |
| V49 | `pure_vy` | `0.0723` | `0.1035` | `0/4` | `0/4` |
| V49 | `pure_yaw` | `0.0633` | `0.1337` | `0/4` | `0/4` |
| V49 | `mixed_vx_vy` | `0.1747` | `0.0692` | `14/16` | `0/16` |
| V49 | `mixed_vx_yaw` | `0.1305` | `0.2145` | `4/6` | `4/6` |

## V48 telemetry diagnosis

V48 added `mixed_composition_repair`, a curriculum that oversamples
`mixed_vx_vy` and `mixed_vx_yaw` while keeping the deployable 80D observation
and 12D residual-plus-gate action ABI unchanged. The short local smoke trained
from V47 final:

```text
runs/worm_v6_ppo_flat_random_v48_mixed_composition_repair_from_v47final
```

It did not pass the gate. The telemetry added to
`src/v6/scan_command_tracking_v6.py` shows why this was not merely a lack of
training time:

| Class | Mean gait blend | Prior L2 | Residual L2 | Action L2 | Tracking cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pure_yaw` | `0.8215` | `2.7146` | `0.0564` | `2.7073` | `1.2665` |
| `mixed_vx_yaw` | `0.4860` | `0.9862` | `0.0380` | `0.9908` | `6.8554` |
| `mixed_vx_vy` | `0.3687` | `1.0904` | `0.0475` | `1.0923` | `7.5830` |

This means the residual policy is not overpowering the prior. Instead, the
mixed commands are still dominated by a weak prior composition: pure yaw gets a
large yaw-oriented prior, while `mixed_vx_yaw` receives much lower prior/action
authority and the residual stays too small to create the missing yaw response.
For planar mixing, `mixed_vx_vy` keeps correct signs but does not generate
enough independent `vx` and `vy` components.

## V49 adapter sign check

V49 makes the smallest adapter-side correction after the V48 diagnosis:

```text
ACTION_ADAPTER_VERSION = cmaes_tri_anchor_auto_gate_directional_v27
```

The deployable ABI is unchanged. The only intended behavior change is that
mixed `vx+yaw` commands now set the yaw-anchor sign from `cmd_yaw` instead of
the legacy inverted transform. A no-retrain scan using the V48 final model
shows this is partially effective:

- `mixed_vx_yaw` yaw RMSE improved from `0.2884` to `0.2145`;
- `mixed_vx_yaw` yaw exceed improved from `6/6` to `4/6`;
- overall acceptance still fails because planar RMSE remains `0.1515 m/s` and
  `mixed_vx_vy` remains the dominant failure group.

This proves the sign bug was real but not the whole problem. The next adapter
or training change must increase controllable mixed-yaw authority and solve
planar `vx/vy` composition.

## What this proves

The current policy has learned usable primitives:

- stop is clean;
- left/right planar signs are correct;
- yaw signs are correct;
- pure lateral commands are now the most reliable primitive;
- zero and yaw-only drift are within the current gates.

The current policy has not learned continuous command composition:

- mixed `vx/vy` commands do not superpose the two planar components;
- mixed `vx/yaw` commands preserve planar direction but under-track yaw or even
  collapse the turn response;
- the failure is not explained by a missing velocity command in observation,
  because `vx/vy/yaw` are already in `obs[0:3]`;
- the failure is not solved by actor/critic size alone, because V47 uses the
  `512-256-128` actor and critic yet remains above the planar RMSE bound.
- the failure is not solved by short mixed-command oversampling alone, because
  V48 keeps the same dominant failure group and the telemetry shows insufficient
  mixed-command prior authority.
- the mixed-yaw sign correction is useful but insufficient, because V49 reduces
  mixed-yaw error without moving the overall planar RMSE gate.

Therefore, the correct paper wording remains:

```text
weak omnidirectional primitive controller with correct direction signs
```

The wording below is not yet justified:

```text
continuous body-frame vx/vy/yaw velocity tracking
```

## Next technical move

The next training branch should target mixed-command composition directly,
rather than only continuing the same reward:

1. Use `src/v6/analyze_command_scan_v6.py` as the hard selection certificate.
2. Keep V41 as the conservative baseline until a candidate reduces:
   - `mixed_vx_vy` planar exceed below `8/16`;
   - `mixed_vx_yaw` yaw exceed below `3/6`;
   - overall planar RMSE below `0.10 m/s`.
3. Add a new mixed-prior composition mechanism, not only a curriculum change.
   The immediate V50 hypothesis should be: mixed yaw commands need a
   controllable yaw-prior component closer to the pure-yaw authority while
   preserving axial translation.
4. Continue to log learned gait gate, prior action, residual action, final motor
   action, and component tracking cost per command.

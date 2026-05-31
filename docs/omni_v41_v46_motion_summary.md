# V41-V46 flat omni motion summary

Date: 2026-05-31

## Current baseline

Current best flat baseline remains:

```text
runs/worm_v6_ppo_flat_random_lateral_primitives_v41_from_v40final/best_model.zip
record/current/flat_omni_v41_v26_scan_after40k/scan_35_commands_best.json
```

Safe claim: weak omnidirectional prototype / six-direction primitive
controller with snake-worm learned gait gate.

Not yet safe claim: continuous body-frame `vx/vy/yaw` velocity tracking.

The deployable ABI is unchanged:

- observation: 80D, `vx/vy/yaw` command + 11 encoder positions + 11 encoder
  velocities + previous 11D action + 7 segment IMUs + 1 s phase clock;
- action: 12D, `11D residual motor action + 1D learned latent gait gate`;
- actor/critic: `512-256-128`;
- action adapter: `cmaes_tri_anchor_auto_gate_directional_v26`.

The latest reward code is `omni_directional_offaxis_yaw_v24`. V24 adds a
pure-axial prior-preservation term so residual actions are penalized when they
erase the peristaltic slide wave. V46 tested that idea but did not replace V41.

## Version scan comparison

All scans are flat terrain, random gait mode, 6 s per command, and 35 commands
with forward+yaw rows included.

| Candidate | Planar RMSE | Yaw RMSE | Planar sign | Yaw sign | Lateral strict | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| V41 best | `0.1517` | `0.1105` | `1.00` | `1.00` | yes | current baseline |
| V44 final | `0.1540` | `0.1193` | `1.00` | `1.00` | yes | rejected, worse than V41 |
| V45 best | `0.1552` | `0.1165` | `0.96` | `1.00` | yes | rejected, sign regression |
| V45 final | `0.1590` | `0.1182` | `1.00` | `1.00` | no | rejected, lateral gate lost |
| V46 best | `0.1547` | `0.1141` | `0.96` | `1.00` | yes | rejected, worse than V41 |
| V46 final | `0.1569` | `0.1152` | `1.00` | `1.00` | yes | rejected, worse than V41 |

## Current V41 motion table

Source:

```text
record/current/flat_omni_v41_v26_scan_after40k/scan_35_commands_best.csv
```

| Motion | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw_rate)` | Status |
| --- | --- | --- | --- |
| stop | `(0.000, 0.000, 0.000)` | `(-0.0000, -0.0000, -0.0002)` | clean stop |
| forward | `(0.250, 0.000, 0.000)` | `(0.1446, -0.0105, -0.0266)` | direction OK, speed low |
| reverse | `(-0.250, 0.000, 0.000)` | `(-0.0736, 0.0011, -0.0539)` | direction OK, speed low |
| lateral_left | `(0.000, 0.150, 0.000)` | `(-0.0217, 0.0729, 0.1098)` | strict lateral gate passes |
| lateral_right | `(0.000, -0.150, 0.000)` | `(0.0518, -0.1480, -0.1502)` | strict lateral gate passes |
| yaw_left | `(0.000, 0.000, 0.500)` | `(0.0626, 0.0294, 0.3966)` | yaw sign OK, planar drift remains |
| yaw_right | `(0.000, 0.000, -0.500)` | `(0.0574, -0.0274, -0.3742)` | yaw sign OK, planar drift remains |
| mixed_forward_left | `(0.125, 0.075, 0.000)` | `(0.0537, -0.0009, 0.0136)` | lateral component under-tracked |
| mixed_forward_right | `(0.125, -0.075, 0.000)` | `(0.0561, 0.0098, 0.0257)` | lateral component wrong/weak |
| mixed_reverse_left | `(-0.125, 0.075, 0.000)` | `(-0.0375, 0.0346, 0.0279)` | weak but mostly correct |
| mixed_reverse_right | `(-0.125, -0.075, 0.000)` | `(-0.0299, 0.0490, 0.0438)` | lateral component wrong/weak |
| forward_yaw_left | `(0.125, 0.000, 0.250)` | `(0.0447, -0.0112, 0.0245)` | yaw too weak |
| forward_yaw_right | `(0.125, 0.000, -0.250)` | `(0.0557, 0.0331, 0.0437)` | yaw sign wrong |

## V45/V46 experiments

### Axial primitive search

New tool:

```text
src/v6/search_axial_prior_v6.py
record/current/flat_omni_v45_axial_prior_search_reverse6s/axial_prior_search_summary.json
```

The best 6 s reverse primitive validation reached:

- `body_vx_m_s = -0.0621`;
- `signed_axial_m_s = 0.0621`;
- acceptance target: `signed_axial_m_s >= 0.10`;
- accepted: `false`.

This did not beat the V41 learned reverse row (`-0.0736 m/s`), so it was not
integrated.

### Yaw primitive search

New tool:

```text
src/v6/search_yaw_prior_v6.py
record/current/flat_omni_v45_yaw_prior_search_6s/yaw_prior_search_summary.json
```

The best 6 s yaw validation reduced planar drift but did not provide enough
turn rate:

- mean signed yaw rate: `0.2186 rad/s`;
- max planar speed: `0.0334 m/s`;
- acceptance target: min signed yaw rate `>= 0.30 rad/s`;
- accepted: `false`.

This is useful diagnostically, but not a replacement for the current V41 yaw
prior, which turns faster.

### V45 axis repair

Run:

```text
runs/worm_v6_ppo_flat_random_v45_axis_repair_from_v41best
record/current/flat_omni_v45_axis_repair_scan
```

V45 did not beat V41. Best scan regressed planar sign to `0.96`; final scan
recovered signs but lost fixed-lateral strict acceptance.

### V46 axial prior preservation

Run:

```text
runs/worm_v6_ppo_flat_random_v46_axial_prior_preserve_from_v41best
record/current/flat_omni_v46_axial_prior_preserve_scan
```

V46 used reward contract `omni_directional_offaxis_yaw_v24` and penalized
residual cancellation of the pure-axial slide prior. It did not beat V41:

- V46 best: `planar_rmse_m_s = 0.1547`, `planar_sign_rate = 0.96`;
- V46 final: `planar_rmse_m_s = 0.1569`, `planar_sign_rate = 1.00`;
- V46 final reverse improved slightly to `-0.0773 m/s`, but forward dropped to
  `0.1187 m/s`, below the fixed forward threshold of `0.12 m/s`.

## Diagnosis

The robot has useful primitive behaviors but not arbitrary velocity tracking:

- forward/reverse directions exist, but speed is below commanded magnitude;
- lateral left/right is now the strongest solved part because V26 uses searched
  lateral primitives;
- pure yaw turns both ways, but still translates about `0.06 m/s`;
- mixed `vx/vy` does not superpose both planar components reliably;
- forward+yaw is still weak and can produce the wrong yaw sign;
- simple vector blending, global residual-scale sweeps, V45 axis repair, and
  V46 axial prior preservation did not replace V41.

## Next optimization target

Keep V41 as the active baseline. The next change should not be another blind
fine-tune. Highest-value next steps:

1. Use the 35-command scan as the hard model-selection gate after each training
   branch.
2. Add scan/record telemetry for prior component, residual component, learned
   gait gate, and axial cancellation so failures are visible per command.
3. Search or learn dedicated mixed-quadrant planar priors instead of simple
   linear vector blending.
4. Redesign the reverse primitive parameterization before trying another
   reverse-prior integration.
5. Keep sand/slope paused until flat continuous tracking improves.

## Paper-guided V47 direction

The actor-critic snake tracking paper reviewed on 2026-05-31 frames locomotion
as an optimal tracking problem: a guidance/gait generator provides reference
motion, the actor outputs corrective control, and the critic/value objective
penalizes tracking error plus control input. The useful lesson for Worm V6 is
not to add global position to the policy observation. The useful lesson is to
turn the flat omni problem into a stricter command-tracking cost:

```text
e_track = [body_vx - vx_cmd,
           body_vy - vy_cmd,
           yaw_rate - yaw_rate_cmd]
```

The next branch should therefore keep the 80D observation and 12D action ABI
unchanged, but use V41 as the resume point and optimize/evaluate a V47
tracking-cost curriculum:

```text
flat_random_v47_tracking_cost_from_v41best
```

The target is to reduce component tracking errors and threshold-exceedance
counts, while preserving the learned snake-worm gait gate and preventing the
residual action from erasing necessary worm-style slide activity.

Implementation note: V47 uses reward contract
`omni_directional_offaxis_yaw_v25`, which adds a componentwise tracking cost
over `vx`, `vy`, and `yaw_rate` without changing the policy observation or
action ABI.

Detailed paper notes are in:

```text
docs/paper_actor_critic_tracking_snake_robot_notes.md
```

## V47 smoke result

Run:

```text
runs/worm_v6_ppo_flat_random_v47_tracking_cost_from_v41best_noeval
record/current/flat_omni_v47_tracking_cost_scan
```

The first V47 smoke line resumed V41 best and trained under
`omni_directional_offaxis_yaw_v25` with local no-eval chunks. Training moved
from `504,800` policy steps to `652,256` policy steps. The 17-case in-training
directional evaluator could not be used on this machine because allocating all
MuJoCo eval environments at once ran out of memory, so the run was evaluated
after training with the single-environment 35-command scan.

Same-script V41 baseline scan with the new V47 metrics:

```text
record/current/flat_omni_v41_v26_scan_with_v47_metrics/scan_35_commands_best.json
```

| Candidate | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | planar exceed | off-axis exceed | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V41 best rescanned | `0.1517` | `0.1105` | `18` | `5` | `18` | `2` | current baseline |
| V47 final smoke | `0.1516` | `0.1183` | `17` | `5` | `15` | `2` | diagnostic only, not accepted |

V47 preserved correct signs and fixed lateral strict gates:

```text
wrong_planar_sign_count = 0
wrong_yaw_sign_count = 0
fixed_lateral_strict_gate_passed = true
```

It made the exceedance counts slightly better, but did not solve continuous
tracking. The main remaining failure is still mixed-command composition:
reverse plus lateral, forward plus lateral, and forward plus yaw do not
superpose reliably. The componentwise tracking cost is therefore useful as a
diagnostic contract, but this first smoke line is not a replacement for V41.

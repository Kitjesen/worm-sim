# V41-V44 lateral primitive repair and current motion summary

Date: 2026-05-31

## Summary

This round moved the flat omni controller forward, but it is still not a
complete continuous `vx/vy/yaw` tracker.

Main progress:

- added `src/v6/search_lateral_prior_v6.py` for deployable 1 s phase-clock
  lateral primitive search;
- added V26 action adapter
  `cmaes_tri_anchor_auto_gate_directional_v26`;
- replaced dominant lateral transformed-anchor priors with independent left
  and right lateral primitives;
- kept the deployable ABI unchanged: 80-D observation and 12-D policy action;
- added V22/V23 reward contract
  `omni_directional_offaxis_yaw_v23` with signed planar component-deficit
  penalty for under-tracked `vx/vy` components;
- ran V41, V42, V43, and V44 flat continuation/probe experiments from the
  V40/V41 line.

Current claim boundary:

- Safe claim: weak omnidirectional prototype / six-direction primitive
  controller with improved lateral primitive support.
- Not yet safe claim: continuous body-frame `vx/vy/yaw` velocity tracking.

## Lateral primitive search

Search tool:

```powershell
python src\v6\search_lateral_prior_v6.py --method cma --generations 10 --popsize 10 --seconds 6.0 --validation-seconds 6.0 --directions left --out-dir record\current\flat_omni_v41_lateral_prior_search_left6s
python src\v6\search_lateral_prior_v6.py --method cma --generations 10 --popsize 10 --seconds 6.0 --validation-seconds 6.0 --directions right --out-dir record\current\flat_omni_v41_lateral_prior_search_right6s
```

Primitive validation:

| Primitive | signed lateral | body vx | yaw rate | Accepted |
| --- | ---: | ---: | ---: | --- |
| left | `0.0779 m/s` | `-0.0136 m/s` | `0.1232 rad/s` | yes |
| right | `0.1263 m/s` | `0.0102 m/s` | `-0.1130 rad/s` | yes |

Adapter prior-only integrated smoke:

```text
record/current/flat_omni_v41_lateral_adapter_smoke/eval_prior_lateral_v26.json
```

| Command | body vx | body vy | yaw rate | Accepted |
| --- | ---: | ---: | ---: | --- |
| lateral_left | `-0.0189 m/s` | `0.0814 m/s` | `0.1170 rad/s` | yes |
| lateral_right | `0.0443 m/s` | `-0.1784 m/s` | `-0.1123 rad/s` | yes |

This proves the lateral improvement survives the full `WormEnvV6.step()` action
adapter path, not only the standalone search formula.

## Scan comparison

All scans below use flat terrain, 6 s per command, and the 35-command scan with
forward+yaw commands included.

| Candidate | Planar RMSE | Yaw RMSE | Planar sign | Yaw sign | Lateral strict | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| V40 final, old V25 adapter | `0.1630` | `0.1127` | `0.96` | `1.00` | no | right lateral too weak |
| V40 final + V26 adapter | `0.1535` | `0.1127` | `1.00` | `1.00` | yes | no new PPO training |
| V41 best | `0.1517` | `0.1105` | `1.00` | `1.00` | yes | best current planar scan |
| V42 final | `0.1527` | `0.1009` | `1.00` | `1.00` | yes | best yaw RMSE |
| V43 final | `0.1553` | `0.1092` | `1.00` | `1.00` | yes | V22 reward did not improve scan yet |
| V44 vector-blend probe | `0.1872` | `0.1105` | `0.76` | `1.00` | yes | linear primitive blend is worse |
| V44 mixed-planar best | `0.1588` | `0.1067` | `1.00` | `1.00` | yes | not a replacement |
| V44 mixed-planar final | `0.1540` | `0.1193` | `1.00` | `1.00` | yes | slightly lower yaw-only drift, worse planar RMSE than V41 |

Current recommended checkpoint for the next training branch:

```text
runs/worm_v6_ppo_flat_random_lateral_primitives_v41_from_v40final/best_model.zip
```

V42 final is useful if prioritizing yaw-rate RMSE, and V44 final is useful as
a mixed-planar curriculum diagnostic. V41 best is still the better starting
point for planar tracking because planar RMSE is the limiting acceptance
metric.

## Current motion summary

Source:

```text
record/current/flat_omni_v41_v26_scan_after40k/scan_35_commands_best.csv
```

| Motion | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw_rate)` | Main status |
| --- | --- | --- | --- |
| stop | `(0.000, 0.000, 0.000)` | `(-0.0000, -0.0000, -0.0002)` | very clean |
| forward | `(0.250, 0.000, 0.000)` | `(0.1446, -0.0105, -0.0266)` | direction OK, speed low |
| reverse | `(-0.250, 0.000, 0.000)` | `(-0.0736, 0.0011, -0.0539)` | direction OK, speed low |
| lateral_left | `(0.000, 0.150, 0.000)` | `(-0.0217, 0.0729, 0.1098)` | strict lateral gate passes |
| lateral_right | `(0.000, -0.150, 0.000)` | `(0.0518, -0.1480, -0.1502)` | strict lateral gate passes |
| yaw_left | `(0.000, 0.000, 0.500)` | `(0.0626, 0.0294, 0.3966)` | yaw sign OK, planar drift remains |
| yaw_right | `(0.000, 0.000, -0.500)` | `(0.0574, -0.0274, -0.3742)` | yaw sign OK, planar drift remains |
| mixed_forward_left | `(0.125, 0.075, 0.000)` | `(0.0537, -0.0009, 0.0136)` | under-tracks lateral component |
| mixed_forward_right | `(0.125, -0.075, 0.000)` | `(0.0561, 0.0098, 0.0257)` | under-tracks lateral component |
| mixed_reverse_left | `(-0.125, 0.075, 0.000)` | `(-0.0375, 0.0346, 0.0279)` | weak but sign OK |
| mixed_reverse_right | `(-0.125, -0.075, 0.000)` | `(-0.0299, 0.0490, 0.0438)` | lateral component wrong/weak |
| forward_yaw_left | `(0.125, 0.000, 0.250)` | `(0.0447, -0.0112, 0.0245)` | turn rate too weak |
| forward_yaw_right | `(0.125, 0.000, -0.250)` | `(0.0557, 0.0331, 0.0437)` | yaw sign wrong in this row |

## Diagnosis

The new lateral primitives solved the previous fixed-lateral speed bottleneck.
The remaining failures are different:

- axial speed is still too low, especially reverse;
- yaw-only commands still translate forward while turning;
- mixed `vx/vy` commands mostly follow the axial part and under-track lateral;
- mixed forward+yaw commands do not reliably produce the requested yaw sign and
  magnitude.

Prior-only diagnosis under V26:

```text
record/current/flat_omni_v42_prior_policy_diagnosis/prior_only_6s_commands_v26.json
```

Key observation: forward prior-only reaches `0.1831 m/s`, but the learned
policy scan gives about `0.14 m/s`. The residual policy is therefore damping
some useful axial prior motion.

V44 optimization probes:

- Enabling the existing continuous vector blend between six primitive priors
  made the 35-command scan worse (`planar_rmse_m_s=0.1872`,
  `planar_sign_rate=0.76`), so simple linear primitive mixing is not the next
  safe path.
- Sweeping deployed residual scale (`0.00`, `0.10`, `0.20`, `0.30`, `0.45`)
  did not beat V41 best, so lowering residual authority alone is not enough.
- A 40k `mixed_planar_repair` curriculum run preserved the lateral gate but
  did not improve the full 35-command planar RMSE. It slightly reduced
  yaw-only planar drift in the final checkpoint (`0.0602 m/s`) but remains
  above the `0.05 m/s` target.

Reverse and mixed vector commands therefore need stronger primitive/curriculum
support, not just longer continuation or a global residual-scale change.

## Next steps

1. Keep V26 lateral primitives; they are a real improvement.
2. Do not continue V43/V44 blindly; their first 40k repair runs did not beat
   the V41 best 35-command scan.
3. Add a targeted reverse primitive or reverse-specific prior search, because
   reverse remains far below the requested `-0.25 m/s`.
4. Rework mixed-vector control beyond simple linear prior blending; the V44
   vector-blend probe degraded sign rate.
5. Revisit the yaw-only in-place primitive to reduce forward drift below the
   `0.05 m/s` target.
6. Only after flat continuous tracking passes should sand/slope be restarted.

## Verification run

Passed after the V26/V23 edits:

```powershell
python src\v6\test_reward_contract_v6.py
python src\v6\test_deployable_obs_v6.py
python src\v6\test_omni_eval_metrics_v6.py
python src\v6\test_deploy_policy_v6.py
python src\v6\test_cmaes_prior_v6.py
python src\v6\test_visual_steel_strip_geometry_v6.py
```

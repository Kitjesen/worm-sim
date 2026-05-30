# V6 Spring-Steel Strip Modeling Log

## 2026-05-29 Scope

Goal: add a usable spring-steel strip simulation path without making the V6 PPO
training environment too slow.

Current V6 locomotion uses an equivalent slide-joint spring:

- slide stiffness: `300 N/m`;
- slide damping: `10`;
- visual strips are rendered from segment spacing and do not generate force.

This change adds a separate single-segment calibration tool. The tool builds a
MuJoCo model with two plates and 8 elastic cable strips, sweeps compression,
records the holding force, and fits an equivalent spring constant. The fitted
value can later be copied into `SLIDE_JOINT_STIFFNESS` after validation.

## Design Decision

Do not replace the full V6 robot with flexible strips yet. A full 7-segment
model with 8 flexible strips per slide joint adds hundreds of bodies/DOFs and
will make PPO training much slower and less stable.

Use a two-level model:

1. High-fidelity single-segment strip calibration with `mujoco.elasticity.cable`.
2. Fast V6 whole-body training with an equivalent reduced spring model.

## Outputs

The calibration script writes:

- `single_segment_steel_strip.xml`: generated MuJoCo model;
- `force_displacement.csv`: compression and measured force;
- `calibration_summary.json`: fitted equivalent stiffness and config;
- `calibration_report.md`: human-readable run record;
- optional `calibration_video.mp4`: visual sweep video.

Default output directory:

```powershell
record\v6\spring_steel_calibration
```

## Run Command

```powershell
python src\v6\spring_steel_calibration_v6.py --quick
```

For a fuller sweep:

```powershell
python src\v6\spring_steel_calibration_v6.py --compressions-mm 0,5,10,15,20,25,30,35,40,45,50 --record-video
```

The recommended non-video full sweep used for the current smoke result is:

```powershell
python src\v6\spring_steel_calibration_v6.py --out-dir record\v6\spring_steel_calibration_full --compressions-mm 0,5,10,15,20,25,30,35,40,45,50
```

## Modeling Notes

The high-fidelity model uses 8 cable strips around the segment circumference.
Each strip is initialized with a slight radial bow and welded to both plates.
The moving plate is driven by a position actuator along the segment axis; the
steady actuator force is used as the approximate force required to hold the
compression.

The fitted equivalent model is intended for whole-body RL training:

```text
F ~= k_eq * compression + b_eq * compression_rate
```

The first implementation fits only quasi-static `k_eq` from the compression
sweep. Damping identification requires a dynamic release/oscillation experiment
and is intentionally left out of this first calibration tool.

## 2026-05-29 Implementation Log

1. Added `src/v6/spring_steel_calibration_v6.py`.
   - Generates a single-segment MuJoCo XML with two plates and 8
     `mujoco.elasticity.cable` spring-steel strips.
   - Sweeps axial compression and records the actuator holding force.
   - Writes XML, CSV, JSON, and Markdown report outputs.

2. Added `src/v6/test_spring_steel_calibration_v6.py`.
   - First test failure confirmed the new module was missing.
   - Second test failure confirmed the original fit used servo target
     compression instead of actual simulated joint displacement.
   - Third test failure confirmed the report lacked zero-intercept equivalent
     stiffness values needed by the V6 slide spring model.

3. Fixed the displacement contract.
   - CSV now separates `target_compression_*` from
     `actual_compression_*`.
   - Stiffness fitting now prefers `actual_compression_m` / `mean_qpos_m`.
   - The report records the `compression_source` used for the fit.

4. Added equivalent-stiffness diagnostics.
   - `stiffness_n_per_m`: ordinary linear fit with intercept.
   - `zero_intercept_stiffness_n_per_m`: candidate for V6's no-preload slide
     spring.
   - `secant_stiffness_at_max_n_per_m`: force divided by displacement at the
     largest measured compression.

5. Verification commands run:

```powershell
python src\v6\test_spring_steel_calibration_v6.py
python -m compileall -q src\v6\spring_steel_calibration_v6.py src\v6\test_spring_steel_calibration_v6.py
python src\v6\spring_steel_calibration_v6.py --quick
python src\v6\spring_steel_calibration_v6.py --out-dir record\v6\spring_steel_calibration_full --compressions-mm 0,5,10,15,20,25,30,35,40,45,50
```

6. Current generated results:

Quick sweep in `record/v6/spring_steel_calibration`:

- compression source: `actual_compression_m`
- linear stiffness with intercept: `680.514 N/m`
- zero-intercept equivalent stiffness: `1205.220 N/m`
- secant stiffness at max compression: `960.085 N/m`
- fit R2: `0.45419`
- max target/actual compression mismatch: `9.928 mm`

Full sweep in `record/v6/spring_steel_calibration_full`:

- compression source: `actual_compression_m`
- linear stiffness with intercept: `160.433 N/m`
- zero-intercept equivalent stiffness: `335.566 N/m`
- secant stiffness at max compression: `266.633 N/m`
- fit R2: `0.57506`
- max target/actual compression mismatch: `3.125 mm`

## Current Interpretation

The full sweep is the better smoke result because it uses more compression
points and longer settling. Its zero-intercept stiffness of about `336 N/m`
is close to the current V6 reduced model value of `300 N/m`, so the current
whole-body spring constant is a defensible placeholder.

The result is not yet a final steel-material claim. The low R2 shows that the
elastic-strip response is nonlinear and/or path dependent under the current
cable and weld settings. Before a paper claim, the cable bend/twist parameters
should be calibrated against measured force-displacement data from the real
spring-steel strip assembly.

## Actuation Asymmetry

The real slide mechanism is not a bidirectional linear actuator:

- active contraction comes from a servo pulling a rope/cable;
- release is passive and comes from spring-steel elastic return;
- when the command is relaxed, the motor should not actively push extension.

The current V6 command range already respects the one-sided target range
(`[-50 mm, 0 mm]`), but the MuJoCo slide actuator is still a position-servo
abstraction with an idealized force limit. That is acceptable for early policy
development, but the next hardware-calibrated model should replace it with a
unilateral cable/tendon pull model plus passive spring-steel return.

The reduced force-law implementation is now written in
`src/v6/unilateral_slide_actuator_v6.py`. It uses the same slide coordinate as
the current model, where `q=0` is relaxed and `q<0` is contracted:

```text
c = -q
c* = clip(-a, 0, 1) L
T = clip(kp (c* - c - deadband) - kd c_dot, 0, Tmax)
Qq = ks c + ds c_dot - T
```

`T` is rope tension and is never negative, so the motor cannot push extension.
When the command relaxes, `T` goes to zero and the spring-steel term returns
the segment.

This means two real datasets are required:

1. passive spring-steel force-displacement data;
2. active servo-rope step-response data under load.

## Real Data Path

Generate real-robot parameter templates with:

```powershell
python src\v6\hardware_parameter_id_v6.py --write-templates
```

After filling a spring-steel force-displacement CSV from the real bench setup,
fit it with:

```powershell
python src\v6\hardware_parameter_id_v6.py --spring-csv record\v6\hardware\parameter_id\spring_steel_force_displacement_YYYYMMDD.csv
```

See `docs/real_robot_parameter_collection_v6.md` for the exact data collection
protocol.

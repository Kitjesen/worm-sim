# Isaac Lab Migration Plan for Worm V6

## Direct Answer

The real slide mechanism can be represented in Isaac Lab. It should not be
modeled as a bidirectional position servo. The deployable model should split
the slide into two effects:

1. active contraction from servo-rope pulling;
2. passive release from spring-steel elastic return.

The same split should also be used for the next MuJoCo hardware-calibrated
model, because it is the mechanism present on the real robot.

Steel-strip rendering can also be tried in Isaac Sim/Isaac Lab. For RL training,
the recommended first version is visual-only curved strips attached between
segment frames. A physically deformable steel strip is possible as a later
experiment, but it should not be the first training model because it is slower
and harder to calibrate.

## Current Local Constraint

The current local Windows environment does not have `isaaclab` or `omni`
installed, so an Isaac Lab render preview cannot be launched on this machine
yet. The implementation target is an Isaac Sim/Isaac Lab workstation, ideally
the 3090 server.

Checked on 2026-05-29:

```powershell
python -c "import importlib.util; print(importlib.util.find_spec('isaaclab'))"
```

Result: `None`.

## Official Capability Notes

- Isaac Lab articulations support actuator groups over named joints, and the
  actuator API exposes joint position, joint velocity, computed effort, and
  applied effort fields.
- Isaac Lab supports implicit actuators and explicit/custom actuators. The
  custom explicit actuator path is the right place for servo-rope physics.
- Isaac Lab also has a `DeformableObject` API for PhysX soft bodies. This path
  is experimental and requires GPU simulation and deformable mesh assets.

References:

- https://isaac-sim.github.io/IsaacLab/v2.1.1/source/api/lab/isaaclab.actuators.html
- https://isaac-sim.github.io/IsaacLab/v2.1.1/source/how-to/write_articulation_cfg.html
- https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_deformable_object.html

## Slide Actuation Model

Use the slide coordinate already used by V6:

```text
q_i = slide joint position, q_i = 0 relaxed, q_i < 0 contracted
c_i = -q_i = compression
```

The policy still outputs the same normalized slide command. Under the current
V6 convention, negative action means contraction:

```text
c_i^* = clip(-a_i, 0, 1) L
```

where `L = 0.05 m` is the maximum slide stroke.

The passive spring-steel return is:

```text
Q_i^{spring} = k_s c_i + d_s \dot{c}_i
```

where positive generalized force increases `q_i`, so it extends the segment.

The servo-rope can only pull, not push:

```text
T_i = clip(k_p(c_i^* - c_i - \delta) - k_d \dot{c}_i, 0, T_max)
Q_i^{rope} = -T_i
```

The final generalized slide force is:

```text
Q_i = Q_i^{spring} + Q_i^{rope}
    = k_s c_i + d_s \dot{c}_i - T_i
```

This captures the important hardware behavior:

- if the command asks for more contraction than the current compression, the
  rope becomes taut and pulls;
- if the command relaxes, rope tension becomes zero;
- extension comes only from spring-steel return and damping;
- the actuator never actively pushes extension.

The shared simulator-independent implementation is:

```powershell
python src\v6\test_unilateral_slide_actuator_v6.py
```

Source:

- `src/v6/unilateral_slide_actuator_v6.py`

## Isaac Lab Implementation Path

### Phase 1: Rigid Articulation With Reduced Slide Physics

Create a Worm V6 articulation with:

- 7 segment bodies;
- 6 prismatic slide joints;
- 5 yaw revolute joints;
- passive wheel/contact geometry;
- same 80-D deployable observation ABI;
- same 12-D policy action: 11 residual motors plus learned gait gate.

Slide joints:

- do not use a bidirectional position drive for the 6 slide joints;
- compute slide efforts from the unilateral force law above;
- write the 6 slide efforts into the articulation each control step;
- keep yaw joints as ordinary position/torque-limited servo drives initially.

### Phase 2: Isaac Lab RL Task

Build a DirectRLEnv task:

- command: `vx`, `vy`, `yaw`;
- observation: same V6 80-D source contract;
- action adapter: same CMA-ES prior plus residual plus learned gait gate;
- reward: same direction/speed/yaw tracking terms as MuJoCo;
- terrain: flat, sand-like high-friction, slope.

The first parity gate is not high speed. It is:

```text
For identical open-loop commands, Isaac and MuJoCo produce the same sign of
slide compression/release and roughly similar segment-level motion.
```

### Phase 3: Steel-Strip Rendering

For videos and visual inspection, add visual-only curved steel strips:

- 8 strips per slide joint;
- endpoints attached to adjacent segment frames;
- curve shape follows the same parabolic bow as the current MuJoCo renderer;
- bow increases with compression;
- material: dark metallic strip with fixed width/thickness;
- disabled or simplified during large parallel RL training.

This is the practical first Isaac render target because it is visually faithful
but does not add flexible-body DOFs.

### Phase 4: Optional Deformable Steel Strip Experiment

Only after the rigid reduced model trains:

- create a separate single-segment Isaac scene with deformable strip meshes;
- pin/kinematically drive strip endpoints to the segment frames;
- compare force-displacement against the MuJoCo cable calibration and real
  load-cell data;
- do not use this in full PPO training until it is stable and fast enough.

## What This Means for the Paper

The paper claim should be careful:

- training model: hardware-aware reduced actuation model;
- real mechanism: unilateral rope-pull contraction and passive spring-steel
  return;
- visual model: curved steel-strip rendering for interpretability;
- optional high-fidelity model: single-segment flexible-strip calibration, not
  the primary full-body RL environment.

This makes the deployment story stronger than the current position-servo
abstraction because the action no longer assumes the motor can push extension.

## Immediate TODO

1. Install and verify Isaac Sim/Isaac Lab on the 3090 machine.
2. Convert or rebuild the V6 rigid articulation as USD.
3. Port `unilateral_slide_actuator_v6.py` to a torch/Isaac explicit actuator.
4. Add visual-only steel strips using the current MuJoCo strip geometry logic.
5. Run reset/step/render smoke tests before training.
6. Train a short flat/worm policy and compare against MuJoCo open-loop videos.

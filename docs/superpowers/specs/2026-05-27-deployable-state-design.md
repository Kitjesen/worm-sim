# Deployable State Design for Worm/Snake Robot

## Goal

Design the RL observation and action interface so a policy trained in MuJoCo can run on the physical robot. The policy must not depend on MuJoCo-only global truth such as freejoint linear velocity, world pose, COM velocity, or contact truth.

The physical sensor assumption is:

- Each body segment has one IMU.
- Each actuated joint has an encoder.
- The robot uses the existing 11 actuated joints: 6 slide joints and 5 yaw joints.
- The current V6 model represents 7 body segments, so the deployable observation assumes 7 segment IMUs unless hardware revisions change that count.

## Current Dual-Mode Status

The project already has explicit worm/snake dual-mode support in the open-loop simulation scripts:

- `src/v3/worm_v5.py` defines `worm`, `snake`, and `combined` modes.
- `src/v3/worm_v5_1.py` preserves the same three modes on the rigid-body/passive-wheel mesh model.
- `src/v6/worm_v6.py` also exposes `--mode snake|worm|combined` and controls slide joints for worm/peristaltic motion and yaw joints for snake/serpentine motion.

The current RL environment is different: `src/v6/worm_env_v6.py` exposes all 11 actuators directly and does not condition the policy on a discrete locomotion mode. That means the policy can learn mixed slide+yaw behavior, but it is not currently a clean "select worm vs snake vs combined" controller.

For deployment, the recommended interface is a single full-body policy conditioned by a gait blend command:

- `gait_blend = 0.0`: bias toward worm/peristaltic slide motion.
- `gait_blend = 1.0`: bias toward snake/serpentine yaw motion.
- intermediate values: allow combined behavior.

This keeps the robot dual-mode while avoiding three separate policies.

## Observation Contract

The deployable observation should contain only values available on the real robot or generated inside the controller.

Recommended observation:

```text
command:
  v_cmd_norm                 1
  yaw_cmd_norm               1
  gait_blend                 1

actuated joints:
  joint_pos_norm[11]         11
  joint_vel_norm[11]         11
  previous_action[11]        11

per-segment IMU:
  segment_gravity[7, 3]      21
  segment_gyro_norm[7, 3]    21

phase:
  sin_phase                  1
  cos_phase                  1
```

Total recommended dimension: 80.

`segment_gravity` is the gravity direction expressed in each segment's local IMU/body frame, derived from the IMU orientation filter. `segment_gyro_norm` is angular velocity from each IMU, also expressed in the segment frame and normalized by a fixed deployable scale.

Do not include accelerometer linear acceleration in the first deployed policy. It is useful for logging and later contact/slip estimation, but it is noisy under vibration and can easily become a shortcut that fails on hardware.

## State To Remove From Policy Inputs

The following values may remain available in simulation for reward, diagnostics, or termination, but must not be present in the policy observation:

- MuJoCo freejoint linear velocity, including current `base_linvel`.
- World position, world orientation, global heading, and COM state.
- Ground-truth contact flags, contact forces, slip state, or terrain labels.
- Any future privileged state estimator output unless the same estimator is implemented and validated on hardware.

The current `base_angvel` should also be replaced. A single root-body angular velocity from MuJoCo is not the deployable sensor interface. The policy should consume the seven measured segment gyros instead.

## Action Contract

Keep the action space as 11 normalized joint targets:

```text
action[0:6]   slide target commands
action[6:11]  yaw target commands
```

Before sending to hardware, the runtime controller must apply:

- joint limit clipping
- velocity/rate limiting
- low-pass filtering
- command timeout behavior
- actuator saturation handling

The simulator must model the same command path. If hardware runs position servos, the MuJoCo actuator model should approximate position tracking with latency, bandwidth, saturation, and realistic joint velocity limits.

## Normalization

Use fixed physical scales rather than statistics that are hard to reproduce on hardware:

- Slide position: divide by slide range.
- Yaw position: divide by yaw range.
- Joint velocity: divide by hardware-rated max joint velocity.
- Gyro: divide by a fixed angular velocity scale chosen from hardware and expected gait limits.
- Command velocity/yaw rate: divide by the command range.

`VecNormalize` can still be used during early experiments, but a deployable export should freeze all statistics and include a compatibility check against raw hardware observations. A safer final target is explicit fixed normalization.

## Simulation Randomization

Training must inject the hardware imperfections that the policy will see:

- encoder noise, quantization, velocity filtering delay
- IMU gyro noise, bias drift, orientation filter delay
- small IMU mounting rotation errors per segment
- sensor timestamp jitter and occasional stale packets
- action latency and servo command filtering
- motor gain, damping, friction, and mass variation
- terrain and contact friction variation

The training observation path should call the same sensor adapter used by evaluation and hardware deployment, so privileged MuJoCo state cannot accidentally leak into the policy.

## Validation

A policy should not be considered deployable until these checks pass:

- Observation self-test confirms exact dimension, finite values, and no world-truth fields.
- A simulated sensor adapter produces all policy inputs from encoder and IMU-equivalent signals.
- Randomized evaluation succeeds with sensor noise, latency, and action filtering enabled.
- Open-loop `worm`, `snake`, and `combined` references remain available as fallback safety modes.
- Hardware logs can be replayed through the policy observation builder without shape, unit, or timestamp errors.

## Implementation Boundary

This document is a design spec only. The next implementation step should create a deployable observation adapter and a new training environment variant without changing the existing V6 environment in place until tests prove parity.


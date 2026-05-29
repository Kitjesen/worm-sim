# System Patterns

## Contracts First

Training, evaluation, deployment, and audit artifacts must carry explicit
contract versions. Old artifacts are stale when any of these change:

- observation ABI fingerprint;
- reward contract;
- action adapter contract;
- actuator contract;
- best-selection contract;
- training/eval command defaults.

## Deployable Observation Boundary

Policy observations are limited to realistic hardware sources:

- command metadata: body-frame `vx`, `vy`, yaw rate;
- joint encoder position and velocity;
- previous normalized action;
- one IMU per segment for gravity direction and gyro;
- controller phase clock.

Root pose, root velocity, world yaw, contact labels, and slip labels are reward
or analysis signals only.

## Multimodal Control Pattern

Use CMA-ES anchors for worm, full-combined, and snake priors. PPO learns:

- residual corrections over the 11 actuators;
- the gait gate that chooses/blends the anchor family.

Fixed `gait_blend` remains useful for ablation/evaluation, but the random
deployable policy must learn it internally.

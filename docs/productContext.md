# Product Context

## Goal

Build a deployable MuJoCo/PPO research pipeline for a modular snake/worm robot
that can combine peristaltic and serpentine locomotion on flat ground, sand,
and slopes.

## Paper Claim Target

The paper should only claim results proven by current-contract artifacts:

- body-frame `vx/vy/yaw` command following;
- automatic snake/worm gait blending from the policy action gate;
- deployable observations from joint encoders, per-segment IMUs, previous
  action, command metadata, and phase clock;
- fixed-mode and random-policy comparisons on flat, sand, and slope;
- CMA-ES open-loop baselines as a strong non-learning reference.

## Non-Goals

- Do not claim old `forward_progress_v3` or fixed `gait_blend` PPO artifacts as
  paper results.
- Do not use privileged simulator root state as a policy observation.
- Do not treat the 4K CMA-ES comparison video as a learned policy result.

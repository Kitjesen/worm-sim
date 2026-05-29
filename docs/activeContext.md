# Active Context

## Current Focus

The current main line is `src/v6`. The contract is being moved from a
fixed/external `gait_blend` command to:

- observation command: `cmd_vx_m_s`, `cmd_vy_m_s`, `cmd_yaw_rad_s`;
- policy action: 11 residual motor commands plus one learned gait gate;
- deployed action: 11 normalized actuator targets after adding the CMA-ES gait
  prior.

## Main Technical Blocker

Yaw/direction control is not solved. The previous 300k v3 flat/random policy
could move forward but did not reliably separate positive and negative yaw
commands. It is diagnostic only and stale under `omni_auto_gate_v4`.

## Immediate Priorities

1. Retrain flat/random under `omni_auto_gate_v4`.
2. Track direction success as a hard checkpoint gate.
3. Expand to flat fixed-mode ablations, then sand/slope.
4. Regenerate eval, robust eval, scans, deploy bundles, and audit reports.

## Spring-Steel Modeling Status

A separate single-segment MuJoCo cable-strip calibration path now exists. It is
for fitting an equivalent V6 slide spring, not yet for replacing the full
whole-body training model with flexible strips. The latest full sweep gives a
zero-intercept equivalent stiffness of about `336 N/m`, close to the current
`300 N/m` V6 placeholder, but real force-displacement data is still needed.

Real-robot parameter templates now exist in
`record/v6/hardware/parameter_id`. The first useful real dataset should be a
spring-steel force-displacement CSV with measured compression and load-cell
force; it can be fitted with `src/v6/hardware_parameter_id_v6.py`.

Important correction: the real slide drive is servo-rope unilateral pulling.
Contraction is active, release is passive spring-steel return. The current V6
command range is one-sided, but the MuJoCo actuator is still an idealized
position servo; a hardware-calibrated model should add unilateral cable pull
and rope slack/tension behavior.

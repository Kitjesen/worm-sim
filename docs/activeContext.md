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

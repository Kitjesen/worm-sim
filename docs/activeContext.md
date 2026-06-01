# Active Context

## Current Focus

The current main line is `src/v6`. The contract has moved from a fixed/external
`gait_blend` command to:

- observation command: `cmd_vx_m_s`, `cmd_vy_m_s`, `cmd_yaw_rad_s`;
- policy action: 11 residual motor commands plus one learned gait gate;
- deployed action: 11 normalized actuator targets after adding the CMA-ES gait
  prior.

## Main Technical Blocker

Flat primitive signs and fixed left/right lateral gates are now mostly solved
inside the narrowed feasible command envelope, but continuous mixed-command
tracking is still not accepted. The latest V64 best checkpoint with the V36
adapter reaches `planar_rmse_m_s=0.05985` and `yaw_rmse_rad_s=0.01880` on the
corrected 35-command scan, but still has `wrong_planar_sign_count=1` in the
`mixed_vx_vy` group.

The V50 componentwise mixed-command prior remains default-off. Re-enabling it
on the same checkpoint worsens the 35-command scan to
`planar_rmse_m_s=0.10105` and `wrong_planar_sign_count=6`, so it is an ablation
rather than the accepted path. The next technical blocker is learning a robust
mixed `vx/vy` composition without losing the visible worm-like axial actuation
and without changing the 80D observation / 12D action ABI.

## Immediate Priorities

1. Use the 3090 server for further training; do not run long training on the
   local Windows machine.
2. Keep V50 mixed-command composition behind
   `WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1` unless a future strict scan
   proves it beats the default guard.
3. Continue from the V64/V67 feasible-envelope line with selection based on the
   corrected 35-command scan, especially mixed `vx/vy` cases.
4. Compare learned latent gait gate against fixed worm/mixed/snake gates.
5. Regenerate deploy bundles and audit reports before sand/slope transfer.

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

An implementation contract for this reduced physical slide model now exists in
`src/v6/unilateral_slide_actuator_v6.py`. It maps the current negative slide
action convention to desired compression, computes non-negative rope tension,
and adds passive spring-steel return force. This is the formula to port into
Isaac Lab as a custom explicit actuator.

Isaac Lab migration is feasible but must stay dual-track: MuJoCo remains the
calibration/debug baseline, while Isaac Lab becomes the GPU RL target after
articulation, observation, action, reward, and render parity checks. See
`docs/isaaclab_migration_plan_v6.md`.

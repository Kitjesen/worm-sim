# Paper notes: actor-critic optimal tracking for snake robots

Date: 2026-05-31

Local source:

```text
D:\software\webDownload\Actor-Critic_Framework-Based_on_Optimal_Tracking_Strategy_for_Snake_Robots_with_Reinforcement_Learning_Method.pdf
```

Paper:

```text
Dongfang Li et al.,
"Actor-Critic Framework-Based on Optimal Tracking Strategy for Snake Robots
with Reinforcement Learning Method",
IEEE Transactions on Industrial Informatics, accepted 2026.
```

## What the paper is doing

The paper is not a pure black-box gait-discovery method. Its controller is a
tracking controller that combines:

1. line-of-sight guidance for the path/heading reference;
2. a serpentine gait generator for desired joint trajectories;
3. an actor-critic controller that learns the residual optimal control action;
4. a critic/value formulation based on Hamilton-Jacobi-Bellman tracking cost;
5. explicit penalties on tracking error and control input.

The key tracking error state is defined over desired and actual joint position
and velocity:

```text
e = [e_phi, e_vphi]
  = [actual_joint_angle - reference_joint_angle,
     actual_joint_velocity - reference_joint_velocity]
```

The value/cost structure is essentially:

```text
C(e, u) = k_e * e^T M_V e + k_u * u^T u
V(e) = integral_t^infinity C(e(tau), u(tau)) d tau
```

So the policy is optimized to reduce tracking error while keeping the control
input bounded. The actor generates the control action, and the critic estimates
the value/cost-to-go. The paper also evaluates trajectory tracking error, joint
angle error, joint angular velocity error, steering error, torque input, and
control cost, rather than only visual locomotion quality.

## Hardware and task differences from Worm V6

The paper's prototype is a yaw-pitch serial snake robot with passive wheels,
joint encoders, and external link-position locators. Its task is path tracking
on circular and sinusoidal reference paths.

Worm V6 is different:

- it has 6 slide joints plus 5 yaw joints;
- the slide mechanism is a rope-pulled contraction plus passive spring-steel
  return, not a symmetric torque/position joint;
- the deployable policy observation intentionally excludes global position,
  global yaw, and base velocity;
- the paper's external locator state cannot be used as a Worm V6 policy input.

Therefore, the paper should guide the training objective and evaluation
structure, but it should not change the 80D deployable observation ABI.

## What this means for Worm V6

The current V41-V46 issue is not that the robot cannot move. It can produce
six useful motion primitives. The issue is that the policy has not become a
continuous tracking controller:

- single-axis signs are mostly correct;
- lateral primitives are strong;
- forward/reverse speeds are still low;
- yaw-only commands still translate;
- mixed vx/vy commands do not superpose reliably;
- forward+yaw commands remain weak or wrong in yaw sign.

The paper suggests that the next Worm V6 training line should be framed as an
optimal tracking problem, not just a gait-selection problem.

For our deployable command interface:

```text
r_cmd = [vx_cmd, vy_cmd, yaw_rate_cmd]
r_hat = [body_vx, body_vy, body_yaw_rate]
e_track = r_hat - r_cmd
```

The training/evaluation cost should be treated as:

```text
J_track =
    w_vx * (body_vx - vx_cmd)^2
  + w_vy * (body_vy - vy_cmd)^2
  + w_yaw * (yaw_rate - yaw_rate_cmd)^2
  + w_off * off_axis_speed^2
  + w_drift * yaw_drift^2
  + w_u * ||applied_action||^2
  + w_du * ||delta_residual_action||^2
```

For gait preservation, the paper's desired joint trajectory idea maps to our
prior-plus-residual structure:

```text
a_prior = gait_prior(cmd, phase, z_g)
a_res = policy_residual(obs)
a_deploy = clip(a_prior + a_res, -1, 1)
```

The next reward should not force the policy to copy the prior everywhere. It
should only penalize destructive residual behavior when the residual erases the
needed locomotion manifold:

```text
J_prior_cancel =
    I_axial * max(0, -dot(a_prior_slide, a_res_slide)) / ||a_prior_slide||^2
  + I_yaw   * max(0, -dot(a_prior_yaw,   a_res_yaw))   / ||a_prior_yaw||^2
```

This keeps the learned residual useful while making visible worm-style axial
actuation harder to accidentally cancel.

## Next training implication

The next branch should be V47 and should keep the hardware-facing ABI unchanged:

```text
run label: flat_random_v47_tracking_cost_from_v41best
resume: runs/worm_v6_ppo_flat_random_lateral_primitives_v41_from_v40final/best_model.zip
obs: 80D deployable contract
action: 12D = 11D residual + 1D learned latent gait gate
actor/critic: 512-256-128
action adapter: cmaes_tri_anchor_auto_gate_directional_v26
terrain: flat only
```

V47 should not be judged by PPO mean reward alone. It should be judged by the
35-command scan and fixed-command gates:

```text
planar_velocity_rmse_m_s <= 0.10
yaw_rate_rmse_rad_s <= 0.20
wrong planar signs = 0
wrong yaw signs = 0
yaw-only planar drift <= 0.08 m/s
zero-command speed <= 0.02 m/s
```

The paper also supports adding threshold-exceedance counts, because it reports
how often tracking errors exceed task thresholds. Worm V6 should add equivalent
scan metrics:

```text
vx_error_exceed_count
vy_error_exceed_count
yaw_error_exceed_count
yaw_drift_exceed_count
off_axis_exceed_count
prior_cancel_exceed_count
```

## Claim boundary

Until V47 or a later branch passes the continuous scan, the safe paper claim is:

```text
Worm V6 demonstrates a deployable snake-worm dual-mode primitive controller
with learned latent gait gating.
```

The not-yet-safe claim remains:

```text
Worm V6 achieves continuous body-frame vx/vy/yaw velocity tracking.
```

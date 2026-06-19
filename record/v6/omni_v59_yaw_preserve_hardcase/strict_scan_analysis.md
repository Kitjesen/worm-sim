# V59 yaw-preserve hardcase scan

Date: 2026-06-01

## Setup

- Start checkpoint: `runs/worm_v6_ppo_flat_random_v56_strict_diagonal_eval_from_v53final/best_model.zip`
- New run: `runs/worm_v6_ppo_flat_random_v59_yaw_preserve_hardcase_from_v56best`
- Reward contract: `omni_directional_offaxis_yaw_v28`
- Curriculum: `mixed_planar_yaw_preserve_repair`
- Actor network: `512-256-128`
- Critic network: `512-256-128`
- Observation ABI: 80D
- Policy action ABI: 12D, `11D residual + 1D learned latent gait gate`
- Scan: 35 fixed 6 s commands over the first-stage command box:
  - `vx = {-0.25, -0.125, 0, 0.125, 0.25} m/s`
  - `vy = {-0.15, -0.075, 0, 0.075, 0.15} m/s`
  - `yaw = {-0.25, -0.125, 0, 0.125, 0.25} rad/s`
  - plus forward/reverse/yaw mixed commands.

## Results

| Candidate | Planar RMSE | Yaw RMSE | Wrong planar signs | Wrong yaw signs | Fixed lateral strict | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| V59 best | `0.1561 m/s` | `0.1887 rad/s` | `0` | `0` | right side failed | rejected |
| V59 final | `0.1591 m/s` | `0.1895 rad/s` | `1` | `0` | passed | rejected |

V59 improves over V58 on yaw RMSE and yaw-only planar drift:

- V58 best: `yaw_rmse_rad_s=0.2709`, `yaw_only_mean_planar_speed_m_s=0.0951`
- V59 best: `yaw_rmse_rad_s=0.1887`, `yaw_only_mean_planar_speed_m_s=0.0677`
- V59 final: `yaw_rmse_rad_s=0.1895`, `yaw_only_mean_planar_speed_m_s=0.0678`

It does not solve continuous planar tracking:

- target gate: `planar_rmse_m_s <= 0.10`
- V59 best: `planar_rmse_m_s=0.1561`
- V59 final: `planar_rmse_m_s=0.1591`

## Dominant Failures

The worst errors are still full-speed mixed planar commands. The policy tends
to produce a weak axial response and insufficient lateral component instead of
tracking both `vx` and `vy` simultaneously.

Worst V59 final examples:

| Command `(vx, vy, yaw)` | Response `(body_vx, body_vy, yaw_rate)` | Main issue |
| --- | --- | --- |
| `(-0.25, -0.15, 0)` | `(-0.0527, +0.0888, +0.0628)` | lateral sign wrong and reverse too weak |
| `(+0.25, +0.15, 0)` | `(+0.0705, -0.0076, +0.0333)` | lateral component missing |
| `(+0.25, -0.15, 0)` | `(+0.0784, +0.0072, +0.0331)` | lateral component missing/wrong |
| `(-0.25, +0.15, 0)` | `(-0.0780, +0.0132, -0.0277)` | lateral component weak |

Mixed `vx+yaw` right-turn commands also remain weak. For example,
`(+0.25, 0, -0.25)` produced yaw rate `+0.0522 rad/s`, which is the wrong
sign for that command in the V59 final scan.

## Verdict

V59 is a useful diagnostic but not an accepted policy. It shows that preserving
yaw samples during hardcase training prevents the severe yaw regression seen in
V58, but the robot still cannot compose strong axial and lateral components for
continuous `vx/vy` commands.

Safe paper wording remains:

> weak omnidirectional prototype / six-direction primitive controller

Do not claim continuous body-frame velocity tracking until the strict scan
passes planar RMSE, yaw RMSE, zero drift, and sign gates together.

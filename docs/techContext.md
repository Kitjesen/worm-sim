# Tech Context

## Runtime

- Windows/PowerShell workspace.
- MuJoCo + Gymnasium environment in `src/v6/worm_env_v6.py`.
- Stable-Baselines3 PPO training in `src/v6/train_v6.py`.
- 50 Hz control loop, 500 Hz physics stepping.
- Peristaltic actuation period: 1.0 s.

## Current Core Constants

- observation dimension: 80;
- actuator dimension: 11;
- policy action dimension: 12;
- command ranges: `vx [-0.25, 0.25] m/s`, `vy [-0.15, 0.15] m/s`,
  `yaw [-0.5, 0.5] rad/s`;
- reward contract: `omni_auto_gate_v4`;
- action adapter: `cmaes_tri_anchor_auto_gate_v2`.

## Hardware ABI

Raw hardware logs are converted by `src/v6/build_hardware_obs_v6.py` and
validated by `src/v6/validate_hardware_log_v6.py`. The policy CSV must contain
`cmd_vx_norm`, `cmd_vy_norm`, and `cmd_yaw_norm`; it must not contain external
`gait_blend` as an observation column.

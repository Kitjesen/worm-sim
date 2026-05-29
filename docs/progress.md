# Progress

## Done

- V6 MuJoCo model, terrain presets, and actuator contract exist.
- Deployable 80-D observation structure exists.
- CMA-ES anchor priors exist for worm, full-combined, and snake modes.
- Paper audit, observation audit, hardware-log validation, deploy export, and
  plotting scaffolds exist.
- Previous flat/random PPO diagnostics reached about 311k steps, but are stale
  under the new auto-gated contract.

## Not Done

- Direction/yaw command control is not solved.
- No formal 1M-step PPO artifacts exist under `omni_auto_gate_v4`.
- Fixed-mode eval, robust eval, auto-gated random-policy deploy bundles, and
  final paper summaries must be regenerated.
- Real flat/sand/slope hardware logs and videos are still missing.

## Current Verdict

The project framework is substantial, but the paper is not complete. The next
valid result must come from retraining under the `vx/vy/yaw` command ABI with a
learned gait gate and a hard direction-success checkpoint criterion.

# Progress

## Done

- V6 MuJoCo model, terrain presets, and actuator contract exist.
- Deployable 80-D observation structure exists.
- CMA-ES anchor priors exist for worm, full-combined, and snake modes.
- Paper audit, observation audit, hardware-log validation, deploy export, and
  plotting scaffolds exist.
- Single-segment spring-steel strip calibration now exists under
  `src/v6/spring_steel_calibration_v6.py`, with XML/CSV/JSON/report outputs
  and a test script.
- Real-robot parameter-identification templates now exist under
  `record/v6/hardware/parameter_id`, with a spring-steel force-displacement
  fitter in `src/v6/hardware_parameter_id_v6.py`.
- A detailed real-robot experiment TODO is now tracked in
  `docs/real_robot_experiment_todo_v6.md`.
- Previous flat/random PPO diagnostics reached about 311k steps, but are stale
  under the new auto-gated contract.

## Not Done

- Direction/yaw command control is not solved.
- No formal 1M-step PPO artifacts exist under `omni_auto_gate_v4`.
- Fixed-mode eval, robust eval, auto-gated random-policy deploy bundles, and
  final paper summaries must be regenerated.
- Real flat/sand/slope hardware logs and videos are still missing.
- The spring-steel cable parameters are not yet identified from real
  force-displacement measurements.
- Motor step-response, IMU static-calibration, contact-drag, and mass/geometry
  CSVs are still empty templates until real bench data is collected.

## Current Verdict

The project framework is substantial, but the paper is not complete. The next
valid result must come from retraining under the `vx/vy/yaw` command ABI with a
learned gait gate and a hard direction-success checkpoint criterion.

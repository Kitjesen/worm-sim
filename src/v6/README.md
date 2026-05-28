# V6 Paper Pipeline

This directory is the current paper-facing implementation for the deployable
snake/worm multimodal robot.

Use these entry points for new work:

- `run_paper_pipeline_v6.py` for training, export, eval, scans, summaries, and audit
- `worm_env_v6.py` for the deployable 80-dimensional policy observation
- `worm_v6.py` for the MuJoCo robot model and gait utilities
- `motor_contract_v6.py` for actuator limits and normalized action mapping
- `hardware_policy_runtime_v6.py` for hardware-side policy inference

Training accepts `--device auto`, `--device cpu`, or `--device cuda`.
This selects the PPO network device; MuJoCo stepping remains CPU-bound.
Use `--device cpu` by default for SB3 MLP-PPO unless a benchmark shows CUDA is faster.

The old `src/v3/*_v6.py` paths are compatibility wrappers only.

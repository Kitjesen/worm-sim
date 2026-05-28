"""
Smoke tests for resumable formal V6 training helpers.
"""

import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_terrain_experiments import (  # noqa: E402
    control_timing_kwargs,
    latest_resume_model,
)
from motor_contract_v6 import motor_contract  # noqa: E402
from train_v6 import infer_vecnormalize_path  # noqa: E402


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "worm_v6_ppo_flat_random")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        final_model = os.path.join(run_dir, "final_model.zip")
        final_norm = os.path.join(run_dir, "final_model_vecnormalize.pkl")
        ckpt_100 = os.path.join(ckpt_dir, "worm_v6_ppo_100_steps.zip")
        ckpt_200 = os.path.join(ckpt_dir, "worm_v6_ppo_200_steps.zip")
        ckpt_200_norm = os.path.join(
            ckpt_dir, "worm_v6_ppo_vecnormalize_200_steps.pkl")

        touch(final_model)
        touch(final_norm)
        touch(ckpt_100)
        touch(ckpt_200)
        touch(ckpt_200_norm)
        write_json(os.path.join(run_dir, "training_result.json"),
                   {"completed_timesteps": 50})

        assert infer_vecnormalize_path(final_model) == final_norm
        assert infer_vecnormalize_path(ckpt_200) == ckpt_200_norm
        assert latest_resume_model(run_dir, robust=False) is None

        write_json(os.path.join(run_dir, "training_config.json"), {
            "control_timing": control_timing_kwargs(),
            "actuator_contract_fingerprint": (
                motor_contract()["contract_fingerprint"]),
            "sensor_robustness": {
                "encoder_pos_noise_std": 0.0,
                "encoder_vel_noise_std": 0.0,
                "imu_gravity_noise_std": 0.0,
                "imu_gyro_noise_std": 0.0,
                "action_delay_steps": 0,
            },
        })
        assert latest_resume_model(run_dir, robust=False) == ckpt_200

        stale = dict(control_timing_kwargs())
        stale["peristaltic_actuation_period_s"] = 2.0
        write_json(os.path.join(run_dir, "training_config.json"), {
            "control_timing": stale,
            "actuator_contract_fingerprint": (
                motor_contract()["contract_fingerprint"]),
            "sensor_robustness": {
                "encoder_pos_noise_std": 0.0,
                "encoder_vel_noise_std": 0.0,
                "imu_gravity_noise_std": 0.0,
                "imu_gyro_noise_std": 0.0,
                "action_delay_steps": 0,
            },
        })
        assert latest_resume_model(run_dir, robust=False) is None

        write_json(os.path.join(run_dir, "training_config.json"), {
            "control_timing": control_timing_kwargs(),
            "actuator_contract_fingerprint": (
                motor_contract()["contract_fingerprint"]),
            "sensor_robustness": {
                "encoder_pos_noise_std": 0.0,
                "encoder_vel_noise_std": 0.0,
                "imu_gravity_noise_std": 0.0,
                "imu_gyro_noise_std": 0.0,
                "action_delay_steps": 0,
            },
        })
        assert latest_resume_model(run_dir, robust=True) is None

    print("resume-partial helpers passed")


if __name__ == "__main__":
    main()

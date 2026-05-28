"""
Smoke test for the Worm V6 deployable policy export path.

The test builds an untrained PPO model only to verify the deployment contract:
SB3 PPO + VecNormalize -> TorchScript actor + JSON config -> hardware CSV
observation replay -> 11 finite normalized actions plus physical targets.
"""

import csv
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from deploy_policy_v6 import export_policy, replay_csv  # noqa: E402
from train_v6 import make_env  # noqa: E402
from validate_hardware_log_v6 import (  # noqa: E402
    neutral_example_row,
    write_csv,
)
from worm_env_v6 import NUM_ACTUATORS, NUM_SLIDES  # noqa: E402


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        bundle_dir = os.path.join(tmp, "bundle")
        os.makedirs(run_dir, exist_ok=True)

        vec_env = DummyVecEnv([make_env(
            terrain="flat", gait_mode="random", seed=123)])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        try:
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])),
                verbose=0,
                device="cpu",
                seed=123,
            )
            model_path = os.path.join(run_dir, "best_model")
            model.save(model_path)
            vec_path = os.path.join(run_dir, "best_model_vecnormalize.pkl")
            vec_env.save(vec_path)
        finally:
            vec_env.close()

        export_policy(SimpleNamespace(
            model=f"{model_path}.zip",
            vecnormalize=vec_path,
            run_dir=None,
            terrain="flat",
            gait_mode="random",
            gait_blend=0.5,
            out_dir=bundle_dir,
            max_export_diff=1e-5,
        ))
        with open(os.path.join(bundle_dir, "deploy_config.json"),
                  "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["action_mapping"]["slide_indices"] == [0, NUM_SLIDES]
        assert config["action_mapping"]["yaw_indices"] == [
            NUM_SLIDES, NUM_ACTUATORS]
        assert config["action_mapping"]["slide_range_m"] > 0.0
        assert config["action_mapping"]["slide_min_m"] == -0.05
        assert config["action_mapping"]["slide_max_m"] == 0.0
        assert config["action_mapping"]["yaw_range_rad"] > 0.0
        assert config["actuator_contract_fingerprint"] == (
            config["actuator_contract"]["contract_fingerprint"])
        assert config["control_timing"]["peristaltic_actuation_period_s"] == 1.0
        assert config["control_timing"]["phase_freq_hz"] == 1.0

        input_csv = os.path.join(tmp, "hardware_example.csv")
        output_csv = os.path.join(tmp, "hardware_actions.csv")
        write_csv(input_csv, rows=[neutral_example_row()])
        replay_csv(SimpleNamespace(
            bundle_dir=bundle_dir,
            config=None,
            actor=None,
            input_csv=input_csv,
            output_csv=output_csv,
            batch_size=32,
        ))

        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        actions = np.array([
            float(rows[0][f"action_{i:02d}"]) for i in range(NUM_ACTUATORS)
        ], dtype=np.float32)
        assert actions.shape == (NUM_ACTUATORS,)
        assert np.all(np.isfinite(actions))
        assert np.all(actions >= -1.0) and np.all(actions <= 1.0)

        for i in range(NUM_SLIDES):
            target = float(rows[0][f"slide_target_m_{i:02d}"])
            expected = np.clip(
                actions[i] * config["action_mapping"]["slide_range_m"],
                config["action_mapping"]["slide_min_m"],
                config["action_mapping"]["slide_max_m"])
            assert np.isclose(
                target,
                expected,
                atol=1e-6)
        for i in range(NUM_ACTUATORS - NUM_SLIDES):
            src = NUM_SLIDES + i
            target = float(rows[0][f"yaw_target_rad_{i:02d}"])
            expected = np.clip(
                actions[src] * config["action_mapping"]["yaw_range_rad"],
                config["action_mapping"]["yaw_min_rad"],
                config["action_mapping"]["yaw_max_rad"])
            assert np.isclose(
                target,
                expected,
                atol=1e-6)
        print("deploy policy export/replay contract passed")


if __name__ == "__main__":
    main()

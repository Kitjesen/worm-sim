"""
Smoke test for online raw-sensor -> policy action deployment runtime.
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

from build_hardware_obs_v6 import neutral_raw_row, raw_columns  # noqa: E402
from deploy_policy_v6 import export_policy  # noqa: E402
from hardware_policy_runtime_v6 import (  # noqa: E402
    HardwarePolicyRuntime,
    run_jsonl_stream,
    run_raw_csv,
)
from train_v6 import make_env  # noqa: E402
from worm_env_v6 import NUM_ACTUATORS, NUM_SLIDES  # noqa: E402


def write_raw(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def build_bundle(tmp):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run_dir = os.path.join(tmp, "run")
    bundle_dir = os.path.join(tmp, "bundle")
    os.makedirs(run_dir, exist_ok=True)
    vec_env = DummyVecEnv([make_env(
        terrain="flat", gait_mode="random", seed=321)])
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
            seed=321,
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
    return bundle_dir


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = build_bundle(tmp)
        with open(os.path.join(bundle_dir, "deploy_config.json"),
                  "r", encoding="utf-8") as f:
            config = json.load(f)

        runtime = HardwarePolicyRuntime(bundle_dir=bundle_dir)
        row0 = neutral_raw_row()
        row0.update({"terrain": "flat", "mode": "mixed", "gait_blend": 0.5})
        pred0 = runtime.predict(row0)
        assert pred0["action"].shape == (NUM_ACTUATORS,)
        assert pred0["slide_targets_m"].shape == (NUM_SLIDES,)
        assert pred0["yaw_targets_rad"].shape == (
            NUM_ACTUATORS - NUM_SLIDES,)
        assert np.all(np.isfinite(pred0["action"]))

        row1 = neutral_raw_row()
        row1.update({
            "time_s": 0.02,
            "terrain": "flat",
            "mode": "mixed",
            "gait_blend": 0.5,
        })
        _, policy_row1 = runtime.build_observation(row1)
        for i in range(NUM_ACTUATORS):
            assert np.isclose(
                float(policy_row1[f"previous_action_{i:02d}"]),
                pred0["action"][i],
                atol=1e-6)

        raw_path = os.path.join(tmp, "raw.csv")
        out_path = os.path.join(tmp, "runtime_actions.csv")
        policy_log_path = os.path.join(tmp, "runtime_policy_log.csv")
        video_path = os.path.join(tmp, "demo.mp4")
        open(video_path, "wb").close()
        row1["time_s"] = 0.12
        write_raw(raw_path, [row0, row1])
        runtime.reset()
        result = run_raw_csv(
            runtime,
            raw_path,
            out_path,
            policy_log_csv=policy_log_path,
            metadata_overrides={"video_file": video_path},
            validate_policy_log=True,
            expected_terrain="flat",
            require_video=True,
            min_rows=2,
            min_duration_s=0.1,
        )
        assert result["rows"] == 2
        assert result["policy_log_csv"] == policy_log_path
        assert result["policy_log_metrics"]["obs_dim"] == 80
        with open(out_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        actions = np.array([
            float(rows[0][f"action_{i:02d}"]) for i in range(NUM_ACTUATORS)
        ], dtype=np.float32)
        assert np.all(np.isfinite(actions))
        assert np.max(np.abs(actions)) <= 1.0
        assert np.isclose(
            float(rows[0]["slide_target_m_00"]),
            np.clip(
                actions[0] * config["action_mapping"]["slide_range_m"],
                config["action_mapping"]["slide_min_m"],
                config["action_mapping"]["slide_max_m"]),
            atol=1e-6)
        with open(policy_log_path, "r", encoding="utf-8", newline="") as f:
            policy_rows = list(csv.DictReader(f))
        assert np.isclose(float(policy_rows[0]["previous_action_00"]), 0.0)
        assert np.isclose(
            float(policy_rows[1]["previous_action_00"]),
            float(policy_rows[0]["action_00"]),
            atol=1e-6)

        jsonl_in = os.path.join(tmp, "raw.jsonl")
        jsonl_out = os.path.join(tmp, "actions.jsonl")
        jsonl_policy_log = os.path.join(tmp, "jsonl_policy_log.csv")
        write_jsonl(jsonl_in, [row0, row1])
        runtime.reset()
        stream_result = run_jsonl_stream(
            runtime,
            jsonl_in,
            output_jsonl=jsonl_out,
            policy_log_csv=jsonl_policy_log,
            metadata_overrides={"video_file": video_path},
            validate_policy_log=True,
            expected_terrain="flat",
            require_video=True,
            min_rows=2,
            min_duration_s=0.1,
            max_action_delta=0.0,
        )
        assert stream_result["rows"] == 2
        with open(jsonl_out, "r", encoding="utf-8") as f:
            streamed = [json.loads(line) for line in f if line.strip()]
        assert len(streamed) == 2
        assert np.allclose(streamed[0]["action"], 0.0)
        assert len(streamed[0]["raw_action"]) == NUM_ACTUATORS
        with open(jsonl_policy_log, "r", encoding="utf-8", newline="") as f:
            jsonl_policy_rows = list(csv.DictReader(f))
        assert np.isclose(float(jsonl_policy_rows[1]["previous_action_00"]), 0.0)
        assert np.isclose(float(jsonl_policy_rows[1]["action_00"]), 0.0)
        print("hardware policy runtime contract passed")


if __name__ == "__main__":
    main()

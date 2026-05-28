"""
Smoke test for controller JSONL stream health checks.
"""

import json
import os
import sys
import tempfile

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from capture_hardware_stream_v6 import example_stream_row  # noqa: E402
from check_controller_stream_v6 import check_stream, write_markdown  # noqa: E402
from motor_contract_v6 import action_mapping_config, motor_contract  # noqa: E402
from observation_contract_v6 import attach_contract_to_config  # noqa: E402
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    NUM_ACTUATORS,
    OBS_DIM,
)


class ZeroActor(torch.nn.Module):
    def forward(self, raw_obs):
        if raw_obs.dim() == 1:
            raw_obs = raw_obs.unsqueeze(0)
        return raw_obs.new_zeros((raw_obs.shape[0], NUM_ACTUATORS))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def write_bundle(project_root):
    bundle = os.path.join(
        project_root, "record", "v6", "deploy_bundles",
        "flat_random")
    os.makedirs(bundle, exist_ok=True)
    actor_path = os.path.join(bundle, "policy_actor.pt")
    traced = torch.jit.trace(
        ZeroActor(),
        torch.zeros((1, OBS_DIM), dtype=torch.float32),
        check_trace=True,
    )
    traced.save(actor_path)
    config = {
        "terrain": "flat",
        "gait_mode": "random",
        "obs_dim": OBS_DIM,
        "observation_columns": observation_columns(),
        "action_dim": NUM_ACTUATORS,
        "action_columns": [f"action_{i:02d}" for i in range(NUM_ACTUATORS)],
        "action_mapping": action_mapping_config(),
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
    }
    config = attach_contract_to_config(config)
    with open(os.path.join(bundle, "deploy_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return bundle


def stream_rows():
    return [
        example_stream_row(
            time_s=i * 0.05,
            terrain="flat",
            mode="random",
            gait_blend=0.5,
            include_metadata=False,
        )
        for i in range(5)
    ]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = os.path.join(tmp, "controller_stream.jsonl")
        write_jsonl(jsonl_path, stream_rows())

        payload = check_stream(
            jsonl_path,
            terrain="flat",
            mode="random",
            video_file="record/v6/videos/flat_random_hardware_demo.mp4",
            gait_blend=0.5,
            cmd_vel=0.025,
            cmd_yaw=0.0,
            bundle_dir=None,
            project_root=tmp,
        )
        assert payload["complete"]
        assert payload["valid_rows"] == 5
        assert payload["duration_s"] >= 0.1
        assert payload["gravity_norm"]["min"] > 0.9
        assert not payload["policy_runtime"]["enabled"]

        bundle = write_bundle(tmp)
        payload = check_stream(
            jsonl_path,
            terrain="flat",
            mode="random",
            video_file="record/v6/videos/flat_random_hardware_demo.mp4",
            gait_blend=0.5,
            cmd_vel=0.025,
            cmd_yaw=0.0,
            bundle_dir=bundle,
            project_root=tmp,
        )
        assert payload["complete"]
        assert payload["policy_runtime"]["enabled"]
        assert payload["policy_runtime"]["obs_dim"] == OBS_DIM
        assert payload["policy_runtime"]["action_dim"] == NUM_ACTUATORS
        assert payload["policy_runtime"]["max_abs_action"] == 0.0

        bad = stream_rows()
        bad[0].pop("slide_pos_m_00")
        bad_path = os.path.join(tmp, "bad_stream.jsonl")
        write_jsonl(bad_path, bad)
        bad_payload = check_stream(
            bad_path,
            terrain="flat",
            mode="random",
            video_file="record/v6/videos/flat_random_hardware_demo.mp4",
            gait_blend=0.5,
            cmd_vel=0.025,
            cmd_yaw=0.0,
            bundle_dir=None,
            project_root=tmp,
        )
        assert not bad_payload["complete"]
        assert any("slide_pos_m_00" in error
                   for error in bad_payload["errors"])

        md_path = os.path.join(tmp, "stream_check.md")
        write_markdown(md_path, payload)
        assert os.path.exists(md_path)
        assert "Controller Stream Check" in open(
            md_path, "r", encoding="utf-8").read()

    print("controller stream check contract passed")


if __name__ == "__main__":
    main()

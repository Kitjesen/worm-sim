"""
Smoke test for importing a captured hardware trial into audit artifacts.
"""

import csv
import json
import os
import sys
import tempfile

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import neutral_raw_row, raw_columns  # noqa: E402
from import_hardware_trial_v6 import import_trial  # noqa: E402
from motor_contract_v6 import action_mapping_config, motor_contract  # noqa: E402
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_YAW_RANGE,
    NUM_ACTUATORS,
    NUM_POLICY_ACTIONS,
    OBS_DIM,
    OBS_LAYOUT,
)


class ConstantActor(torch.nn.Module):
    def forward(self, obs):
        return torch.zeros(
            (obs.size(0), NUM_ACTUATORS),
            dtype=obs.dtype,
            device=obs.device,
        ) + 0.1


def write_raw(path, terrain="flat", rows=5):
    out_rows = []
    for i in range(rows):
        row = neutral_raw_row()
        row["time_s"] = i * 0.05
        row["terrain"] = terrain
        row["mode"] = "random"
        row["gait_blend"] = 0.0
        out_rows.append(row)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerows(out_rows)


def build_fake_bundle(path):
    os.makedirs(path, exist_ok=True)
    actor_path = os.path.join(path, "policy_actor.pt")
    dummy = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    torch.jit.trace(ConstantActor(), dummy, check_trace=True).save(actor_path)
    config = {
        "format_version": 1,
        "model_type": "test_constant_actor",
        "terrain": "flat",
        "gait_mode": "random",
        "gait_blend": 0.5,
        "obs_dim": OBS_DIM,
        "obs_layout": {
            key: [value.start, value.stop]
            for key, value in OBS_LAYOUT.items()
        },
        "observation_columns": observation_columns(),
        "action_dim": NUM_ACTUATORS,
        "policy_action_dim": NUM_POLICY_ACTIONS,
        "action_columns": [
            f"action_{i:02d}" for i in range(NUM_ACTUATORS)
        ],
        "action_range": [-1.0, 1.0],
        "action_mapping": action_mapping_config(),
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "command_ranges": {
            "cmd_vx_m_s": list(CMD_VX_RANGE),
            "cmd_vy_m_s": list(CMD_VY_RANGE),
            "cmd_yaw_rad_s": list(CMD_YAW_RANGE),
        },
        "torchscript_actor": actor_path,
    }
    with open(os.path.join(path, "deploy_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = os.path.join(tmp, "flat_raw.csv")
        video_source = os.path.join(tmp, "captured_flat.mp4")
        bundle_dir = os.path.join(tmp, "bundle")
        hardware_dir = os.path.join(tmp, "record", "v6", "hardware")
        field_dir = os.path.join(
            tmp, "record", "v6", "hardware", "field_trials", "current")
        videos_dir = os.path.join(tmp, "record", "v6", "videos")
        best_blend_csv = os.path.join(tmp, "best_blend_by_terrain.csv")

        write_raw(raw_csv)
        open(video_source, "wb").close()
        build_fake_bundle(bundle_dir)
        with open(best_blend_csv, "w", encoding="utf-8", newline="") as f:
            f.write("terrain,policy_mode,gait_blend\n")
            f.write("flat,random,0.25\n")

        entry = import_trial(
            terrain="flat",
            raw_csv=raw_csv,
            video_file=video_source,
            mode="random",
            date_stamp="20260528",
            bundle_dir=bundle_dir,
            hardware_dir=hardware_dir,
            field_dir=field_dir,
            videos_dir=videos_dir,
            best_blend_csv=best_blend_csv,
            copy_video=True,
            project_root=tmp,
        )

        policy_log = os.path.join(
            tmp, entry["policy_log_csv"].replace("/", os.sep))
        action_log = os.path.join(tmp, entry["action_csv"].replace("/", os.sep))
        video_copy = os.path.join(tmp, entry["video_file"].replace("/", os.sep))
        manifest = os.path.join(tmp, entry["manifest"].replace("/", os.sep))

        assert entry["gait_blend"] == 0.25
        assert entry["metrics"]["rows"] == 5
        assert entry["metrics"]["duration_s"] >= 0.1
        assert os.path.exists(policy_log)
        assert os.path.exists(action_log)
        assert os.path.exists(video_copy)
        assert os.path.exists(manifest)

        with open(policy_log, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert rows[0]["terrain"] == "flat"
        assert rows[0]["mode"] == "random"
        assert rows[0]["video_file"] == entry["video_file"]
        assert "gait_blend" not in rows[0]
        assert float(rows[0]["action_00"]) == 0.1

        with open(manifest, "r", encoding="utf-8") as f:
            imported = json.load(f)
        assert "flat" in imported["trials"]
        assert imported["trials"]["flat"]["policy_log_csv"] == (
            entry["policy_log_csv"])

    print("hardware trial import contract passed")


if __name__ == "__main__":
    main()

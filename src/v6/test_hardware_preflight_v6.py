"""
Smoke test for hardware deploy preflight checks.
"""

import json
import os
import sys
import tempfile

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from observation_contract_v6 import attach_contract_to_config  # noqa: E402
from motor_contract_v6 import action_mapping_config, motor_contract  # noqa: E402
from preflight_hardware_deploy_v6 import (  # noqa: E402
    preflight_payload,
    write_reports,
)
from prepare_hardware_trials_v6 import TERRAINS  # noqa: E402
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    NUM_ACTUATORS,
    NUM_SLIDES,
    OBS_DIM,
)


class ZeroActor(torch.nn.Module):
    def forward(self, raw_obs):
        if raw_obs.dim() == 1:
            raw_obs = raw_obs.unsqueeze(0)
        return raw_obs.new_zeros((raw_obs.shape[0], NUM_ACTUATORS))


def write_bundle(project_root, terrain, mode="random"):
    bundle_dir = os.path.join(
        project_root, "record", "v6", "deploy_bundles",
        f"{terrain}_{mode}")
    os.makedirs(bundle_dir, exist_ok=True)
    actor_path = os.path.join(bundle_dir, "policy_actor.pt")
    traced = torch.jit.trace(
        ZeroActor(),
        torch.zeros((1, OBS_DIM), dtype=torch.float32),
        check_trace=True,
    )
    traced.save(actor_path)

    config = {
        "format_version": 1,
        "terrain": terrain,
        "gait_mode": mode,
        "gait_blend": 0.5,
        "obs_dim": OBS_DIM,
        "observation_columns": observation_columns(),
        "action_dim": NUM_ACTUATORS,
        "action_columns": [f"action_{i:02d}" for i in range(NUM_ACTUATORS)],
        "action_range": [-1.0, 1.0],
        "action_mapping": action_mapping_config(),
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
    }
    config = attach_contract_to_config(config)
    with open(os.path.join(bundle_dir, "deploy_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return bundle_dir


def main():
    with tempfile.TemporaryDirectory() as tmp:
        best_csv = os.path.join(tmp, "best_blend_by_terrain.csv")
        with open(best_csv, "w", encoding="utf-8", newline="") as f:
            f.write("terrain,policy_mode,gait_blend\n")
            f.write("flat,random,0.5\n")
            f.write("sand,random,1.0\n")
            f.write("slope,random,0.0\n")

        for terrain in TERRAINS:
            write_bundle(tmp, terrain)

        payload = preflight_payload(
            project_root=tmp,
            best_blend_csv=best_csv,
        )
        assert payload["complete"]
        assert len(payload["terrains"]) == len(TERRAINS)
        for row in payload["terrains"]:
            assert row["status"] == "ok"
            assert row["prediction"]["obs_dim"] == OBS_DIM
            assert row["prediction"]["action_dim"] == NUM_ACTUATORS
            assert row["prediction"]["max_abs_action"] == 0.0

        json_path = os.path.join(tmp, "preflight.json")
        md_path = os.path.join(tmp, "preflight.md")
        written = write_reports(
            json_path=json_path,
            md_path=md_path,
            project_root=tmp,
            best_blend_csv=best_csv,
        )
        assert written["complete"]
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

    print("hardware deploy preflight contract passed")


if __name__ == "__main__":
    main()

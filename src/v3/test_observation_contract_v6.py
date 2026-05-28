"""
Smoke test for the deployable Worm V6 observation ABI contract.
"""

import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from observation_contract_v6 import (  # noqa: E402
    attach_contract_to_config,
    observation_contract,
    refresh_bundle_contract,
    stable_hash,
    write_contract,
)
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    OBS_DIM,
    OBS_LAYOUT,
    PERISTALTIC_ACTUATION_PERIOD_S,
    PHASE_FREQ,
)


def main():
    payload = observation_contract()
    assert payload["obs_dim"] == OBS_DIM
    assert payload["observation_columns"] == observation_columns()
    assert payload["obs_layout"]["phase_clock"] == [
        OBS_LAYOUT["phase_clock"].start,
        OBS_LAYOUT["phase_clock"].stop,
    ]
    assert payload["control_timing"]["peristaltic_actuation_period_s"] == (
        PERISTALTIC_ACTUATION_PERIOD_S)
    assert payload["control_timing"]["phase_freq_hz"] == PHASE_FREQ
    assert payload["control_timing"]["peristaltic_actuation_period_s"] == 1.0
    assert payload["abi_fingerprint"] == stable_hash({
        key: value for key, value in payload.items()
        if key != "abi_fingerprint"
    })

    group_names = [group["name"] for group in payload["groups"]]
    assert group_names == [
        "command",
        "joint_pos",
        "joint_vel",
        "previous_action",
        "segment_gravity",
        "segment_gyro",
        "phase_clock",
    ]
    forbidden = " ".join(payload["forbidden_policy_inputs"]).lower()
    assert "base linear velocity" in forbidden
    assert "global pose" in forbidden

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "observation_contract.json")
        md_path = os.path.join(tmp, "observation_contract.md")
        written = write_contract(json_path, md_path)
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["abi_fingerprint"] == written["abi_fingerprint"]

        bundle = os.path.join(tmp, "bundle")
        os.makedirs(bundle, exist_ok=True)
        config_path = os.path.join(bundle, "deploy_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"obs_dim": OBS_DIM}, f)
        refresh_bundle_contract(bundle)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["observation_contract"]["obs_dim"] == OBS_DIM
        assert config["observation_contract_fingerprint"] == (
            payload["abi_fingerprint"])

        attached = attach_contract_to_config({"obs_dim": OBS_DIM})
        assert attached["observation_contract_fingerprint"] == (
            payload["abi_fingerprint"])

    print("observation contract passed")


if __name__ == "__main__":
    main()

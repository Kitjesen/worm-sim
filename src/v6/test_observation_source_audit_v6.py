"""
Smoke test for the Worm V6 deployable observation source audit.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from audit_observation_sources_v6 import build_audit, write_audit  # noqa: E402


EXPECTED_GROUPS = [
    "command",
    "joint_pos",
    "joint_vel",
    "previous_action",
    "segment_gravity",
    "segment_gyro",
    "phase_clock",
]


def main():
    payload = build_audit()
    assert payload["complete"], payload
    assert payload["obs_dim"] == 80
    assert payload["forbidden_policy_observation_hits"] == {}
    assert payload["missing_group_source_tokens"] == {}
    assert [group["group"] for group in payload["groups"]] == EXPECTED_GROUPS
    assert payload["layout_matches_columns"]

    reward_only = payload["reward_only_privileged_hits"]
    assert "forward_speed" in reward_only
    assert "lateral_speed" in reward_only
    assert "yaw_rate" in reward_only
    assert "termination_height" in reward_only
    assert "termination_nan_qpos" in reward_only

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "observation_source_audit.json")
        md_path = os.path.join(tmp, "observation_source_audit.md")
        written = write_audit(json_path, md_path)
        assert written["complete"]
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        text = open(md_path, "r", encoding="utf-8").read()
        assert "# Worm V6 Observation Source Audit" in text
        assert "Forbidden Policy Observation Hits" in text
        assert "Reward-Only Privileged Hits" in text

    print("observation source audit contract passed")


if __name__ == "__main__":
    main()

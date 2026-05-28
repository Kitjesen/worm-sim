"""
Smoke test for sim sensor collection -> raw hardware CSV -> 80-D policy CSV.
"""

import csv
import os
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from collect_sim_hardware_log_v6 import collect_sim_hardware_log  # noqa: E402
from validate_hardware_log_v6 import validate_csv  # noqa: E402
from worm_env_v6 import NUM_ACTUATORS, NUM_IMUS, OBS_DIM  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "sim_flat_mixed_raw.csv")
        policy_path = os.path.join(tmp, "sim_flat_mixed_policy.csv")
        result = collect_sim_hardware_log(
            output_raw=raw_path,
            output_policy=policy_path,
            terrain="flat",
            gait_mode="mixed",
            gait_blend=0.5,
            duration_s=0.06,
            action_source="sine",
            seed=123,
        )
        assert result["rows"] >= 1
        metrics = validate_csv(policy_path)
        assert metrics["obs_dim"] == OBS_DIM

        with open(raw_path, "r", encoding="utf-8", newline="") as f:
            raw_rows = list(csv.DictReader(f))
        with open(policy_path, "r", encoding="utf-8", newline="") as f:
            policy_rows = list(csv.DictReader(f))
        assert len(raw_rows) == len(policy_rows) == result["rows"]

        gravity = np.array([
            [
                float(policy_rows[0][f"segment_gravity_{seg:02d}_{axis}"])
                for axis in ("x", "y", "z")
            ]
            for seg in range(NUM_IMUS)
        ])
        assert np.all(np.isfinite(gravity))
        assert np.allclose(np.linalg.norm(gravity, axis=1), 1.0, atol=1e-3)

        actions = np.array([
            float(policy_rows[-1][f"action_{i:02d}"])
            for i in range(NUM_ACTUATORS)
        ])
        assert np.all(np.isfinite(actions))
        assert np.max(np.abs(actions)) <= 1.0
        print("sim hardware log bridge contract passed")


if __name__ == "__main__":
    main()

"""
Smoke test for raw hardware sensor log -> deployable 80-D observation CSV.
"""

import csv
import os
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import (  # noqa: E402
    convert_raw_csv,
    neutral_raw_row,
    raw_columns,
)
from validate_hardware_log_v6 import validate_csv  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CMD_VEL_RANGE,
    NUM_ACTUATORS,
    NUM_SLIDES,
    OBS_DIM,
    SLIDE_RANGE_VAL,
)


def write_raw(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerows(rows)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "flat_mixed_raw.csv")
        out_path = os.path.join(tmp, "flat_mixed_policy.csv")
        row0 = neutral_raw_row()
        row0.update({
            "terrain": "flat",
            "mode": "mixed",
            "cmd_vel_m_s": CMD_VEL_RANGE[1],
            "gait_blend": 0.5,
            "slide_pos_m_00": -SLIDE_RANGE_VAL * 0.25,
            "action_00": 0.25,
            "action_06": -0.5,
        })
        row1 = neutral_raw_row()
        row1.update({
            "time_s": 0.1,
            "terrain": "flat",
            "mode": "mixed",
            "cmd_vel_m_s": CMD_VEL_RANGE[1] * 0.5,
            "gait_blend": 0.5,
            "slide_pos_m_00": -SLIDE_RANGE_VAL * 0.25,
            "action_00": -0.25,
            "action_06": 0.5,
        })
        write_raw(raw_path, [row0, row1])
        result = convert_raw_csv(raw_path, out_path)
        assert result["rows"] == 2
        metrics = validate_csv(out_path)
        assert metrics["obs_dim"] == OBS_DIM

        with open(out_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert np.isclose(float(rows[0]["cmd_vel_norm"]), 1.0)
        assert np.isclose(float(rows[0]["joint_pos_00"]), -0.25)
        assert np.isclose(float(rows[0]["previous_action_00"]), 0.0)
        assert np.isclose(float(rows[1]["previous_action_00"]), 0.25)
        assert np.isclose(float(rows[1][f"previous_action_{NUM_SLIDES:02d}"]), -0.5)
        actions = np.array([
            float(rows[1][f"action_{i:02d}"]) for i in range(NUM_ACTUATORS)
        ])
        assert np.all(np.isfinite(actions))
        print("hardware observation builder contract passed")


if __name__ == "__main__":
    main()

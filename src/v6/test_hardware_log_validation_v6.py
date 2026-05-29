"""
Smoke tests for strict hardware validation log rules.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from validate_hardware_log_v6 import (  # noqa: E402
    neutral_example_row,
    validate_csv,
    write_csv,
)
from audit_paper_goal_v6 import valid_hardware_log  # noqa: E402


def make_rows(terrain="flat", video_file="demo.mp4", count=5):
    rows = []
    for i in range(count):
        row = neutral_example_row()
        row["time_s"] = i * 0.05
        row["terrain"] = terrain
        row["mode"] = "mixed"
        row["video_file"] = video_file
        rows.append(row)
    return rows


def main():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "flat_demo.csv")
        open(os.path.join(tmp, "demo.mp4"), "wb").close()
        write_csv(csv_path, make_rows())

        metrics = validate_csv(
            csv_path,
            expected_terrain="flat",
            require_video=True,
            min_rows=5,
            min_duration_s=0.1,
            verbose=False)
        assert metrics["rows"] == 5
        assert metrics["duration_s"] >= 0.1
        assert metrics["terrains"] == ["flat"]
        assert metrics["video_files"] == ["demo.mp4"]
        ok, reasons = valid_hardware_log(csv_path, "flat")
        assert ok, reasons

        try:
            validate_csv(
                csv_path,
                expected_terrain="sand",
                require_video=True,
                min_rows=5,
                min_duration_s=0.1,
                verbose=False)
            raise AssertionError("terrain mismatch should fail")
        except ValueError as exc:
            assert "Invalid hardware log rows" in str(exc)

        bad_video_path = os.path.join(tmp, "sand_demo.csv")
        write_csv(bad_video_path, make_rows(terrain="sand", video_file="missing.mp4"))
        ok, _ = valid_hardware_log(bad_video_path, "sand")
        assert not ok
        try:
            validate_csv(
                bad_video_path,
                expected_terrain="sand",
                require_video=True,
                min_rows=5,
                min_duration_s=0.1,
                verbose=False)
            raise AssertionError("unresolved video should fail")
        except ValueError as exc:
            assert "Unresolved video_file" in str(exc)

    print("hardware log validation contract passed")


if __name__ == "__main__":
    main()

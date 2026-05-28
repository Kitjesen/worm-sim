"""
Smoke test for hardware validation summary artifacts.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from summarize_hardware_validation_v6 import (  # noqa: E402
    build_summary,
    write_summary,
)
from validate_hardware_log_v6 import neutral_example_row, write_csv  # noqa: E402


def write_policy(path, video_file, terrain="flat", rows=5):
    data = []
    for i in range(rows):
        row = neutral_example_row()
        row["time_s"] = i * 0.05
        row["terrain"] = terrain
        row["mode"] = "random"
        row["gait_blend"] = 0.5
        row["video_file"] = video_file
        row["velocity_estimate_m_s"] = 0.01
        data.append(row)
    write_csv(path, data)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        summary = build_summary(project_root=tmp)
        assert not summary["complete"]
        assert summary["aggregate"]["validated_terrains"] == 0
        assert len(summary["rows"]) == 3
        assert all(row["evidence_level"] == "pending_real_run"
                   for row in summary["rows"])

        video = os.path.join(
            tmp, "record", "v6", "videos",
            "flat_random_hardware_demo.mp4")
        os.makedirs(os.path.dirname(video), exist_ok=True)
        open(video, "wb").close()
        policy = os.path.join(
            tmp, "record", "v6", "hardware", "flat_random_20260528.csv")
        write_policy(policy, video)

        summary = build_summary(project_root=tmp)
        flat = next(row for row in summary["rows"]
                    if row["terrain"] == "flat")
        assert flat["policy_valid"]
        assert flat["evidence_level"] == "validated_hardware"
        assert flat["policy_rows"] == 5
        assert flat["mean_velocity_estimate_mm_s"] == 10.0
        assert summary["aggregate"]["validated_terrains"] == 1
        assert not summary["complete"]

        json_path = os.path.join(tmp, "summary.json")
        md_path = os.path.join(tmp, "summary.md")
        csv_path = os.path.join(tmp, "summary.csv")
        written = write_summary(
            json_path=json_path,
            md_path=md_path,
            csv_path=csv_path,
            project_root=tmp,
        )
        assert written["aggregate"]["validated_terrains"] == 1
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        assert os.path.exists(csv_path)
        assert "Hardware Validation Summary" in open(
            md_path, "r", encoding="utf-8").read()

    print("hardware validation summary contract passed")


if __name__ == "__main__":
    main()

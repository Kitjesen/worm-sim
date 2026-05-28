"""
Smoke test for flat/sand/slope hardware trial status diagnostics.
"""

import csv
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import neutral_raw_row, raw_columns  # noqa: E402
from hardware_trial_status_v6 import status_payload  # noqa: E402
from validate_hardware_log_v6 import (  # noqa: E402
    neutral_example_row,
    write_csv,
)


def write_raw(path, terrain="flat", rows=5):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = []
    for i in range(rows):
        row = neutral_raw_row()
        row["time_s"] = i * 0.05
        row["terrain"] = terrain
        row["mode"] = "random"
        row["gait_blend"] = 0.5
        data.append(row)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerows(data)


def write_policy(path, video_file, terrain="flat", rows=5):
    data = []
    for i in range(rows):
        row = neutral_example_row()
        row["time_s"] = i * 0.05
        row["terrain"] = terrain
        row["mode"] = "random"
        row["gait_blend"] = 0.5
        row["video_file"] = video_file
        data.append(row)
    write_csv(path, data)


def find(payload, terrain):
    return next(row for row in payload["terrains"] if row["terrain"] == terrain)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        best = os.path.join(tmp, "best_blend_by_terrain.csv")
        with open(best, "w", encoding="utf-8", newline="") as f:
            f.write("terrain,policy_mode,gait_blend\n")
            f.write("flat,random,0.25\n")

        payload = status_payload(project_root=tmp, best_blend_csv=best)
        assert not payload["complete"]
        assert find(payload, "flat")["status"] == "needs_raw_csv"
        assert "process_hardware_trial_v6.py" in find(
            payload, "flat")["process_command"]
        assert "check_controller_stream_v6.py" in find(
            payload, "flat")["stream_check_command"]
        assert "--bundle-dir record\\v6\\deploy_bundles\\flat_random" in find(
            payload, "flat")["stream_check_command"]
        assert "capture_hardware_stream_v6.py" in find(
            payload, "flat")["next_command"]

        raw = os.path.join(
            tmp, "record", "v6", "hardware", "field_trials",
            "current", "flat", "flat_random_raw.csv")
        write_raw(raw)
        payload = status_payload(project_root=tmp, best_blend_csv=best)
        flat = find(payload, "flat")
        assert flat["status"] == "needs_video"
        assert "record/v6/videos/flat_random_hardware_demo.mp4" in (
            flat["next_command"])
        assert flat["raw_rows"] == 5
        assert flat["recommended_gait_blend"] == 0.25

        video = os.path.join(
            tmp, "record", "v6", "videos", "flat_random_hardware_demo.mp4")
        os.makedirs(os.path.dirname(video), exist_ok=True)
        open(video, "wb").close()
        payload = status_payload(project_root=tmp, best_blend_csv=best)
        flat = find(payload, "flat")
        assert flat["status"] == "ready_to_import"
        assert "process_hardware_trial_v6.py" in flat["process_command"]
        assert "import_hardware_trial_v6.py" in flat["import_command"]
        assert "import_hardware_trial_v6.py" in flat["next_command"]

        policy = os.path.join(
            tmp, "record", "v6", "hardware", "flat_random_20260528.csv")
        write_policy(policy, video)
        payload = status_payload(project_root=tmp, best_blend_csv=best)
        flat = find(payload, "flat")
        assert flat["status"] == "complete"
        assert flat["policy_valid"]
        assert flat["policy_metrics"]["rows"] == 5
        assert not payload["complete"]

    print("hardware trial status contract passed")


if __name__ == "__main__":
    main()

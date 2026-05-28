"""
Smoke test for capturing controller JSONL into raw hardware CSV.
"""

import csv
import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import neutral_raw_row, raw_columns  # noqa: E402
from capture_hardware_stream_v6 import (  # noqa: E402
    capture_jsonl,
    required_stream_columns,
    metadata_overrides,
    write_json_schema,
    write_jsonl_example,
)
from hardware_trial_status_v6 import status_payload  # noqa: E402


def sensor_json_row(time_s):
    row = neutral_raw_row()
    row["time_s"] = time_s
    for key in (
            "terrain", "mode", "video_file", "gait_blend",
            "cmd_vel_m_s", "cmd_yaw_rad_s",
            "velocity_estimate_m_s", "yaw_rate_estimate_rad_s"):
        row.pop(key, None)
    for i in range(11):
        row.pop(f"action_{i:02d}", None)
    return row


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def find(payload, terrain):
    return next(row for row in payload["terrains"] if row["terrain"] == terrain)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = os.path.join(tmp, "flat_stream.jsonl")
        raw_csv = os.path.join(
            tmp, "record", "v6", "hardware", "field_trials",
            "current", "flat", "flat_random_raw.csv")
        video = os.path.join(
            tmp, "record", "v6", "videos", "flat_random_hardware_demo.mp4")
        example_jsonl = os.path.join(tmp, "controller_stream_example.jsonl")
        schema_json = os.path.join(tmp, "controller_stream_schema.json")
        os.makedirs(os.path.dirname(video), exist_ok=True)
        open(video, "wb").close()

        example = write_jsonl_example(
            example_jsonl,
            rows=2,
            terrain="flat",
            mode="random",
            video_file=video,
            gait_blend=0.25,
        )
        schema = write_json_schema(schema_json)
        assert os.path.exists(example_jsonl)
        assert os.path.exists(schema_json)
        assert "terrain" not in example["required_with_cli_overrides"]
        assert "action_00" not in example["required_with_cli_overrides"]
        assert required_stream_columns() == schema[
            "required_with_cli_overrides"]

        write_jsonl(jsonl_path, [
            sensor_json_row(0.0),
            sensor_json_row(0.05),
            sensor_json_row(0.10),
            sensor_json_row(0.15),
            sensor_json_row(0.20),
        ])
        result = capture_jsonl(
            jsonl_path,
            raw_csv,
            overrides=metadata_overrides(
                terrain="flat",
                mode="random",
                video_file=video,
                gait_blend=0.25,
                cmd_vel=0.025,
                cmd_yaw=0.0,
            ),
        )
        assert result["rows"] == 5
        assert result["duration_s"] >= 0.1
        with open(raw_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == raw_columns()
            rows = list(reader)
        assert rows[0]["terrain"] == "flat"
        assert rows[0]["mode"] == "random"
        assert rows[0]["video_file"] == video
        assert float(rows[0]["gait_blend"]) == 0.25
        assert float(rows[0]["action_00"]) == 0.0

        payload = status_payload(project_root=tmp)
        flat = find(payload, "flat")
        assert flat["status"] == "ready_to_import"

        bad_jsonl = os.path.join(tmp, "bad.jsonl")
        bad_row = sensor_json_row(0.0)
        bad_row.pop("slide_pos_m_00")
        write_jsonl(bad_jsonl, [bad_row])
        try:
            capture_jsonl(
                bad_jsonl,
                os.path.join(tmp, "bad.csv"),
                overrides=metadata_overrides(
                    terrain="flat",
                    mode="random",
                    video_file=video,
                    gait_blend=0.25,
                    cmd_vel=0.025,
                    cmd_yaw=0.0,
                ),
            )
            raise AssertionError("missing sensor field should fail")
        except ValueError as exc:
            assert "slide_pos_m_00" in str(exc)

    print("hardware stream capture contract passed")


if __name__ == "__main__":
    main()

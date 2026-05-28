"""
Smoke test for the one-command Worm V6 hardware trial processor.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from capture_hardware_stream_v6 import write_jsonl_example  # noqa: E402
from process_hardware_trial_v6 import process_trial  # noqa: E402
from test_import_hardware_trial_v6 import build_fake_bundle  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        input_jsonl = os.path.join(tmp, "controller_stream.jsonl")
        raw_csv = os.path.join(
            tmp, "record", "v6", "hardware", "field_trials", "current",
            "flat", "flat_random_raw.csv")
        video_source = os.path.join(tmp, "flat_demo_source.mp4")
        bundle_dir = os.path.join(
            tmp, "record", "v6", "deploy_bundles", "flat_random")
        hardware_dir = os.path.join(tmp, "record", "v6", "hardware")
        field_dir = os.path.join(
            tmp, "record", "v6", "hardware", "field_trials", "current")
        videos_dir = os.path.join(tmp, "record", "v6", "videos")
        best_blend_csv = os.path.join(
            tmp, "record", "v6", "paper_results",
            "best_blend_by_terrain.csv")
        status_json = os.path.join(
            tmp, "record", "v6", "hardware", "hardware_trial_status.json")
        status_md = os.path.join(
            tmp, "record", "v6", "hardware", "hardware_trial_status.md")

        write_jsonl_example(input_jsonl, rows=5, terrain="flat",
                            mode="random", gait_blend=0.5)
        open(video_source, "wb").close()
        build_fake_bundle(bundle_dir)
        os.makedirs(os.path.dirname(best_blend_csv), exist_ok=True)
        with open(best_blend_csv, "w", encoding="utf-8", newline="") as f:
            f.write("terrain,policy_mode,gait_blend\n")
            f.write("flat,random,0.5\n")

        result = process_trial(
            terrain="flat",
            mode="random",
            input_jsonl=input_jsonl,
            raw_csv=raw_csv,
            video_file=video_source,
            date_stamp="20260528",
            bundle_dir=bundle_dir,
            hardware_dir=hardware_dir,
            field_dir=field_dir,
            videos_dir=videos_dir,
            best_blend_csv=best_blend_csv,
            copy_video=True,
            status_json=status_json,
            status_md=status_md,
            project_root=tmp,
        )

        assert result["capture"]["rows"] == 5
        assert result["import"]["metrics"]["rows"] == 5
        assert result["terrain_status"]["status"] == "complete"
        assert result["terrain_status"]["policy_valid"] is True
        assert os.path.exists(raw_csv)
        assert os.path.exists(status_json)
        assert os.path.exists(status_md)

    print("hardware trial process contract passed")


if __name__ == "__main__":
    main()

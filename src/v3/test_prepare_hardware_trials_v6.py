"""
Smoke test for hardware trial package generation.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from prepare_hardware_trials_v6 import prepare_trials  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        best = os.path.join(tmp, "best.csv")
        with open(best, "w", encoding="utf-8", newline="") as f:
            f.write("terrain,policy_mode,gait_blend\n")
            f.write("flat,random,0.25\n")
            f.write("sand,random,1.0\n")
            f.write("slope,random,0.0\n")

        manifest_path, manifest = prepare_trials(
            os.path.join(tmp, "trial"),
            best_blend_csv=best,
        )
        assert os.path.exists(manifest_path)
        assert len(manifest["terrains"]) == 3
        by_terrain = {entry["terrain"]: entry for entry in manifest["terrains"]}
        assert by_terrain["flat"]["recommended_gait_blend"] == 0.25
        assert by_terrain["sand"]["recommended_gait_blend"] == 1.0
        assert by_terrain["slope"]["recommended_gait_blend"] == 0.0

        for entry in manifest["terrains"]:
            for key in ("raw_template", "raw_example", "policy_template",
                        "policy_example", "readme"):
                local = os.path.join(
                    tmp, "trial", entry["terrain"],
                    os.path.basename(entry[key]))
                assert os.path.exists(local), entry[key]
            assert len(entry["commands"]) == 11
            commands = "\n".join(cmd["command"] for cmd in entry["commands"])
            assert "preflight_hardware_deploy_v6.py" in commands
            assert "check_controller_stream_v6.py" in commands
            assert "capture_hardware_stream_v6.py" in commands
            assert "process_hardware_trial_v6.py" in commands
            assert "import_hardware_trial_v6.py" in commands
            assert "hardware_trial_status_v6.py" in commands
            assert "--input-jsonl -" in commands
            assert "--max-action-delta 0.2" in commands

    print("hardware trial preparation contract passed")


if __name__ == "__main__":
    main()

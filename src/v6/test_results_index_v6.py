"""
Smoke test for the Worm V6 result index generator.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_results_index_v6 import build_index_text, write_index  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "results_index.md")
        text = build_index_text(out)
        assert "# Worm V6 Training Result Index" in text
        assert "Fixed-Mode RL Results" in text
        assert "Best Continuous gait_blend Scan" in text
        assert "Deployable Policy Bundles" in text
        assert "Hardware Trial Status" in text
        assert "Hardware Deploy Preflight" in text
        assert "Hardware Validation Summary" in text
        assert "Observation Source Audit" in text
        assert "Forbidden policy observation hits" in text
        assert "hardware validation summary" in text
        assert "hardware deploy preflight" in text
        assert "deployable observation contract" in text
        assert "observation source audit" in text
        assert "representative simulation videos" in text
        assert "One-command hardware processing" in text
        assert "Controller stream self-check" in text
        assert "check_controller_stream_v6.py" in text
        assert "process_hardware_trial_v6.py" in text
        assert "training_arena_1280x720.mp4" in text

        write_index(out)
        assert os.path.exists(out)
        saved = open(out, "r", encoding="utf-8").read()
        assert saved == text

    print("results index contract passed")


if __name__ == "__main__":
    main()

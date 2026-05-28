"""
Smoke test for representative paper video planning/manifest generation.
"""

import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from render_paper_videos_v6 import (  # noqa: E402
    PAPER_TERRAINS,
    render_videos,
    write_manifest,
)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        best = os.path.join(tmp, "best_blend_by_terrain.csv")
        with open(best, "w", encoding="utf-8", newline="") as f:
            f.write("terrain,policy_mode,gait_blend\n")
            f.write("flat,random,0.25\n")
            f.write("sand,random,1.0\n")
            f.write("slope,random,0.0\n")

        payload = render_videos(
            episodes=1,
            time_s=0.2,
            best_blend_csv=best,
            dry_run=True,
        )
        assert payload["complete"] is False
        assert [row["terrain"] for row in payload["records"]] == list(
            PAPER_TERRAINS)
        assert payload["records"][0]["gait_blend"] == 0.25
        assert "eval_v6.py" in payload["records"][0]["command"]
        assert "--video" in payload["records"][0]["command"]

        json_out = os.path.join(tmp, "paper_video_manifest.json")
        md_out = os.path.join(tmp, "paper_video_manifest.md")
        write_manifest(payload, json_out=json_out, md_out=md_out)
        assert os.path.exists(json_out)
        assert os.path.exists(md_out)
        with open(json_out, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved["records"]) == 3
        assert "Representative Simulation Videos" in open(
            md_out, "r", encoding="utf-8").read()

    print("paper video manifest contract passed")


if __name__ == "__main__":
    main()

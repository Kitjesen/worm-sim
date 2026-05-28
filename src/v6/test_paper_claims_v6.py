"""
Smoke test for paper claim analysis from fixed-mode and gait_blend summaries.
"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from summarize_paper_results_v6 import (  # noqa: E402
    PAPER_MODES,
    PAPER_TERRAINS,
    best_scan_rows,
    build_claim_report,
    value_domain,
    write_claims,
)


def fixed_rows():
    rows = []
    speed = {
        "worm": [1.0, 1.0, 1.0],
        "snake": [2.0, 2.0, 2.0],
        "mixed": [0.0, 0.0, 0.0],
    }
    for terrain_i, terrain in enumerate(PAPER_TERRAINS):
        for mode in PAPER_MODES:
            rows.append({
                "terrain": terrain,
                "mode": mode,
                "rl_speed_mm_s": speed[mode][terrain_i],
                "rl_success_rate": 0.5,
                "rl_termination_rate": 0.0,
            })
    return rows


def scan_rows():
    rows = []
    blends = [0.0, 0.5, 1.0]
    for terrain in PAPER_TERRAINS:
        for blend in blends:
            rows.append({
                "terrain": terrain,
                "policy_mode": "random",
                "gait_blend": blend,
                "mean_speed_mm_s": 3.0 if blend == 0.5 else 1.0,
                "success_rate": 0.6,
                "termination_rate": 0.0,
            })
    return rows


def main():
    lo, hi = value_domain([-2.0, 3.0])
    assert lo < 0.0 < hi

    fixed = fixed_rows()
    scan = scan_rows()
    best = best_scan_rows(scan)
    report = build_claim_report(fixed, scan, best)
    assert report["goal3_status"] == "supported"
    assert len(report["terrain_mode_selection"]) == len(PAPER_TERRAINS)
    assert report["adaptive_best_blend_average"]["avg_speed_mm_s"] == 3.0
    stability_claims = [
        claim for claim in report["claim_assessments"]
        if "termination rate" in claim["claim"]
    ]
    assert stability_claims
    assert stability_claims[0]["status"] == "supported"

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "paper_claims.json")
        md_path = os.path.join(tmp, "paper_claims.md")
        write_claims(json_path, md_path, report)
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

    print("paper claim analysis contract passed")


if __name__ == "__main__":
    main()

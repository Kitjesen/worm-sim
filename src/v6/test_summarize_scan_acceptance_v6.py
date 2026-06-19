import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_scan_acceptance_v6 import render_markdown, summarize_item  # noqa: E402


def _write_pair(tmp_path, accepted):
    scan = tmp_path / "scan.json"
    analysis = tmp_path / "analysis.json"
    scan.write_text(json.dumps({
        "gait_prior_scale": 1.0,
        "policy_residual_scale": 0.35,
        "gait_blend": None,
        "action_adapter": {"version": "v37"},
        "summary": {
            "num_commands": 35,
            "planar_rmse_m_s": 0.05,
            "yaw_rmse_rad_s": 0.02,
            "wrong_planar_sign_count": 0 if accepted else 1,
            "wrong_yaw_sign_count": 0,
            "planar_error_exceed_count": 2,
            "off_axis_exceed_count": 1,
            "fixed_lateral_strict_gate_passed": True,
        },
        "commands": [{
            "cmd_vx_m_s": 0.0,
            "cmd_vy_m_s": -0.0375,
            "cmd_yaw_rad_s": 0.0,
            "body_vx_m_s": 0.001,
            "body_vy_m_s": -0.025,
            "yaw_rate_rad_s": -0.09,
            "planar_sign_ok": accepted,
            "mean_prior_component_l2": 1.1,
            "mean_residual_component_l2": 0.03,
        }, {
            "cmd_vx_m_s": 0.05,
            "cmd_vy_m_s": 0.075,
            "cmd_yaw_rad_s": 0.0,
            "body_vx_m_s": -0.04,
            "body_vy_m_s": 0.004,
            "yaw_rate_rad_s": 0.01,
            "planar_error_m_s": 0.11,
            "yaw_error_rad_s": 0.01,
            "planar_sign_ok": accepted,
        }],
    }), encoding="utf-8")
    analysis.write_text(json.dumps({
        "acceptance": {
            "accepted": accepted,
            "failed_conditions": [] if accepted else ["wrong_planar_sign_count"],
            "measured": {
                "planar_rmse_m_s": 0.05,
                "yaw_rmse_rad_s": 0.02,
                "wrong_planar_sign_count": 0 if accepted else 1,
                "wrong_yaw_sign_count": 0,
            },
        },
        "dominant_failure_group": None if accepted else "mixed_vx_vy",
    }), encoding="utf-8")
    return scan, analysis


def test_summarize_item_combines_scan_and_acceptance(tmp_path):
    scan, analysis = _write_pair(tmp_path, accepted=False)

    row = summarize_item("case", str(scan), str(analysis))

    assert row["label"] == "case"
    assert row["accepted"] is False
    assert row["failed_conditions"] == "wrong_planar_sign_count"
    assert row["counter_cmd_vx_m_s"] == 0.05
    assert row["counter_body_vx_m_s"] == -0.04
    assert row["counter_mixed_component_sign_ok"] is False


def test_render_markdown_includes_counter_sign(tmp_path):
    scan, analysis = _write_pair(tmp_path, accepted=True)
    row = summarize_item("case", str(scan), str(analysis))

    markdown = render_markdown([row], "Title")

    assert "# Title" in markdown
    assert "`case`" in markdown
    assert "Selected command" in markdown
    assert "Wrong mixed comp" in markdown
    assert "pass" in markdown
    assert "fail" in markdown

"""
Smoke tests for Worm V6 body-segment trajectory plotting.

The key contract is semantic: absolute-position plots must preserve the
physical segment locations at t=0, while relative-motion plots may normalize
each segment to its own start point.
"""

import os
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_gait_trajectories_v6 import (  # noqa: E402
    SEGMENT_LABELS,
    direction_summary,
    plot_absolute_mode,
    plot_relative_mode,
    simulate_mode,
    write_markdown_report,
    write_trajectory_csv,
)


def main():
    times, pos = simulate_mode(
        mode="combined",
        terrain="flat",
        duration_s=0.08,
        sample_hz=25.0,
    )
    assert pos.shape[1:] == (len(SEGMENT_LABELS), 3)
    assert len(times) >= 2

    initial_x_mm = pos[0, :, 0] * 1000.0
    assert np.all(np.isfinite(initial_x_mm))
    assert np.ptp(initial_x_mm) > 900.0
    assert not np.allclose(initial_x_mm, 0.0)
    assert np.all(np.diff(initial_x_mm) < -100.0)

    summary = direction_summary("combined", times, pos)
    assert summary["direction_label"] in {
        "forward (-X)",
        "backward (+X)",
        "lateral/no X motion",
    }

    with tempfile.TemporaryDirectory() as tmp:
        abs_plot = os.path.join(tmp, "combined_absolute_positions.png")
        rel_plot = os.path.join(tmp, "combined_relative_motion.png")
        csv_path = os.path.join(tmp, "combined_segment_trajectories.csv")
        report_path = os.path.join(tmp, "trajectory_report.md")

        plot_absolute_mode(
            abs_plot, "combined", "flat", 0.08, times, pos, summary)
        plot_relative_mode(
            rel_plot, "combined", "flat", 0.08, times, pos, summary)
        write_trajectory_csv(csv_path, "combined", times, pos)
        write_markdown_report(
            report_path,
            "flat",
            0.08,
            [{
                "mode": "combined",
                "absolute_plot": "combined_absolute_positions.png",
                "relative_plot": "combined_relative_motion.png",
                "csv": "combined_segment_trajectories.csv",
                **summary,
            }],
        )

        assert os.path.getsize(abs_plot) > 0
        assert os.path.getsize(rel_plot) > 0
        assert os.path.getsize(csv_path) > 0
        report = open(report_path, "r", encoding="utf-8").read()
        assert "actual world position time history" in report
        assert "physical length" in report
        assert "relative to its own initial position" in report

    print("gait trajectory plot contract passed")


if __name__ == "__main__":
    main()

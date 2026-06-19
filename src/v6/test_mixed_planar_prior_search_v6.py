"""
Contract checks for the V58 mixed-planar prior search diagnostic.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from search_mixed_planar_prior_v6 import (  # noqa: E402
    DEFAULT_X0,
    NUM_ACTUATORS,
    primitive_action,
    score_metrics,
)


def main():
    action = primitive_action(DEFAULT_X0, phase=0.3)
    assert action.shape == (NUM_ACTUATORS,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    cmd_vx, cmd_vy = 0.25, 0.15
    good = {
        "body_vx_m_s": 0.12,
        "body_vy_m_s": 0.06,
        "yaw_rate_rad_s": 0.02,
        "action_energy": 0.5,
        "terminated": False,
    }
    weak_or_wrong = {
        "body_vx_m_s": 0.06,
        "body_vy_m_s": -0.02,
        "yaw_rate_rad_s": 0.02,
        "action_energy": 0.5,
        "terminated": False,
    }
    good_score = score_metrics(good, cmd_vx, cmd_vy)
    weak_score = score_metrics(weak_or_wrong, cmd_vx, cmd_vy)

    assert good_score["score"] > weak_score["score"]
    assert good_score["vx_progress_m_s"] > weak_score["vx_progress_m_s"]
    assert good_score["vy_progress_m_s"] > weak_score["vy_progress_m_s"]
    assert weak_score["vy_deficit"] > good_score["vy_deficit"]
    print("mixed-planar prior search checks passed")


if __name__ == "__main__":
    main()

"""
Regression check for slope CMA-ES baseline evaluation.

The open-loop optimizer must grade uphill failures as negative/poor motion,
not collapse all slope candidates into an identical invalid penalty.
"""

import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import optimize_speed as opt  # noqa: E402


def main():
    model, data, slide_ids, yaw_ids, head_id = opt.build_model("slope")
    fitness = opt.evaluate(
        np.array(opt.X0, dtype=float),
        model,
        data,
        slide_ids,
        yaw_ids,
        head_id,
        mode="peristaltic",
        terrain="slope",
    )
    assert math.isfinite(fitness), fitness
    assert fitness < opt.PENALTY_FITNESS, fitness
    print("cmaes slope eval contract passed")


if __name__ == "__main__":
    main()

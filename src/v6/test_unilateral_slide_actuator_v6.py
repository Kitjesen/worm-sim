"""
Smoke tests for the unilateral servo-rope slide actuator model.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from motor_contract_v6 import (  # noqa: E402
    NUM_SLIDES,
    SLIDE_FORCE_LIMIT_N,
    SLIDE_TARGET_SCALE_M,
)
from unilateral_slide_actuator_v6 import (  # noqa: E402
    compute_unilateral_slide_forces,
    normalized_slide_action_to_compression,
    unilateral_slide_model_contract,
)


def main():
    contract = unilateral_slide_model_contract()
    assert contract["slide_count"] == NUM_SLIDES
    assert contract["active_pull"]["can_push_extension"] is False
    assert contract["contract_fingerprint"]

    desired = normalized_slide_action_to_compression(
        np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(
        desired,
        np.array([
            SLIDE_TARGET_SCALE_M,
            0.5 * SLIDE_TARGET_SCALE_M,
            0.0,
            0.0,
        ], dtype=np.float32),
    )

    force_q, tension, _ = compute_unilateral_slide_forces(
        np.array([0.0], dtype=np.float32),
        np.array([-0.03], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )
    assert float(tension[0]) == 0.0
    assert float(force_q[0]) > 0.0

    force_q, tension, _ = compute_unilateral_slide_forces(
        np.array([-1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )
    assert 0.0 < float(tension[0]) <= SLIDE_FORCE_LIMIT_N
    assert float(force_q[0]) < 0.0

    force_q, tension, _ = compute_unilateral_slide_forces(
        np.array([-0.2], dtype=np.float32),
        np.array([-0.04], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )
    assert float(tension[0]) == 0.0
    assert float(force_q[0]) > 0.0

    print("unilateral slide actuator contract passed")


if __name__ == "__main__":
    main()

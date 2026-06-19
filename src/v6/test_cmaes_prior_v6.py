"""
Regression tests for the deployable CMA-ES gait prior.

The paper baseline video `gait_comparison_4k.mp4` is generated from the
CMA-ES optimized peristaltic, serpentine, and full-combined gaits. The
deployable residual RL policy must therefore start from those anchors instead
of the older hand-written sinusoidal prior.
"""

import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from action_adapter_v6 import (  # noqa: E402
    ACTION_ADAPTER_VERSION,
    action_adapter_contract,
    compose_deployable_action,
    gait_prior_from_phase,
)
from motor_contract_v6 import (  # noqa: E402
    NUM_ACTUATORS,
    NUM_SLIDES,
    NUM_YAWS,
    SLIDE_TARGET_SCALE_M,
    YAW_TARGET_SCALE_RAD,
)


FULL_CMAES_PARAMS = np.array([
    0.04655121251582109,
    0.12480818933935475,
    2.3746972019013683,
    0.7167330261210002,
    0.890250887482642,
    0.5899389214089962,
    -0.15881747481015873,
    -0.061281246575806136,
    -1.5688957936234118,
    2.8738159759408957,
    2.190641561690949,
    1.6817450467151565,
    0.6146709498011067,
    -0.12037408667883753,
], dtype=np.float64)


def expected_full_anchor(phase):
    t = float(phase) / (2.0 * math.pi)
    p = FULL_CMAES_PARAMS
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    for j in range(NUM_SLIDES):
        joint_phase = (
            2.0 * math.pi * (t * p[1] - p[2] * j / NUM_SLIDES)
            + p[6 + j]
        )
        action[j] = -(p[0] / SLIDE_TARGET_SCALE_M) * (
            1.0 + math.sin(joint_phase))
    for j in range(NUM_YAWS):
        joint_phase = (
            2.0 * math.pi * p[4] * t
            + 2.0 * math.pi * p[5] * j / NUM_YAWS
            + p[13] * 2.0 * math.pi * t * p[1]
        )
        action[NUM_SLIDES + j] = (
            p[3] / YAW_TARGET_SCALE_RAD) * math.sin(joint_phase)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def main():
    contract = action_adapter_contract()
    assert contract["version"] == ACTION_ADAPTER_VERSION
    assert contract["policy_action_dim"] == NUM_ACTUATORS + 1
    assert contract["gait_blend_source"] == "policy action gate"
    assert contract["gait_anchors"]["mixed"]["source"] == "cmaes_full"

    phase = 0.37 * 2.0 * math.pi
    mixed_prior = gait_prior_from_phase(phase, gait_blend=0.5)
    expected = expected_full_anchor(phase)
    assert mixed_prior.shape == (NUM_ACTUATORS,)
    assert np.allclose(mixed_prior, expected, atol=1e-6), (
        mixed_prior, expected)

    worm_prior = gait_prior_from_phase(phase, gait_blend=0.0)
    snake_prior = gait_prior_from_phase(phase, gait_blend=1.0)
    assert np.max(np.abs(worm_prior[NUM_SLIDES:])) < 1e-6
    assert np.max(np.abs(snake_prior[:NUM_SLIDES])) < 1e-6
    assert np.any(worm_prior[:NUM_SLIDES] < -0.1)
    assert np.any(np.abs(snake_prior[NUM_SLIDES:]) > 0.1)

    zero_residual = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    deployed = compose_deployable_action(
        zero_residual, phase, gait_blend=0.5)
    assert np.allclose(deployed, mixed_prior, atol=1e-6)

    print("cmaes deployable prior checks passed")


if __name__ == "__main__":
    main()

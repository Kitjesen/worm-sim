import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worm_env_v6 import (
    CMD_VY_RANGE,
    COMMAND_CURRICULA,
    FEASIBLE_MIXED_VX_ABS_RANGE,
    FEASIBLE_MIXED_VY_ABS_RANGE,
    FEASIBLE_MIXED_YAW_ABS_RANGE,
    ROBUST_FORWARD_LEFT_VX_RANGE,
    ROBUST_FORWARD_LEFT_VY_RANGE,
    WormEnvV6,
    reward_contract,
)


def main():
    assert "robust_forward_left_diagonal_repair" in COMMAND_CURRICULA
    contract = reward_contract()
    robust_sampling = contract["normalization"][
        "robust_forward_left_diagonal_repair_sampling"]
    assert robust_sampling["vx_range_m_s"] == ROBUST_FORWARD_LEFT_VX_RANGE
    assert robust_sampling["vy_range_m_s"] == ROBUST_FORWARD_LEFT_VY_RANGE
    assert robust_sampling["target_case"] == "cmd=(+0.05,+0.075,0)"

    env = object.__new__(WormEnvV6)
    env.np_random = np.random.default_rng(73)
    env._fixed_cmd_vx = None
    env._fixed_cmd_vy = None
    env._fixed_cmd_yaw = None
    env.command_curriculum = "robust_forward_left_diagonal_repair"

    samples = np.array([env._sample_command() for _ in range(960)])
    assert np.all(np.abs(samples[:, 0]) <= FEASIBLE_MIXED_VX_ABS_RANGE[1])
    assert np.all(np.abs(samples[:, 1]) <= FEASIBLE_MIXED_VY_ABS_RANGE[1])
    assert np.all(np.abs(samples[:, 2]) <= FEASIBLE_MIXED_YAW_ABS_RANGE[1])

    forward_left_local = samples[
        (samples[:, 0] >= ROBUST_FORWARD_LEFT_VX_RANGE[0])
        & (samples[:, 0] <= ROBUST_FORWARD_LEFT_VX_RANGE[1])
        & (samples[:, 1] >= ROBUST_FORWARD_LEFT_VY_RANGE[0])
        & (samples[:, 1] <= ROBUST_FORWARD_LEFT_VY_RANGE[1])
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    assert len(forward_left_local) >= 240

    forward_right = samples[
        (samples[:, 0] > 1e-9)
        & (samples[:, 1] < -1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    generic_forward_left = samples[
        (samples[:, 0] > 1e-9)
        & (samples[:, 1] > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    reverse_diagonal = samples[
        (samples[:, 0] < -1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    forward_yaw = samples[
        (samples[:, 0] > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    pure_yaw = samples[
        (np.abs(samples[:, 0]) <= 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    pure_lateral_left = samples[
        (np.abs(samples[:, 0]) <= 1e-9)
        & (samples[:, 1] > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]

    assert len(forward_right) >= 80
    assert len(generic_forward_left) >= len(forward_left_local)
    assert len(reverse_diagonal) >= 40
    assert len(forward_yaw) >= 40
    assert len(pure_yaw) >= 20
    assert len(pure_lateral_left) >= 40
    assert np.all(pure_lateral_left[:, 1] <= CMD_VY_RANGE[1])

    print("robust forward-left curriculum checks passed")


if __name__ == "__main__":
    main()

"""
Smoke test for the Worm V6 deployable actuator/motor contract.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from motor_contract_v6 import (  # noqa: E402
    NUM_ACTUATORS,
    NUM_SLIDES,
    SLIDE_TARGET_MAX_M,
    SLIDE_TARGET_MIN_M,
    YAW_TARGET_MAX_RAD,
    YAW_TARGET_MIN_RAD,
    action_mapping_config,
    action_mapping_matches,
    motor_contract,
    normalized_action_to_ctrl,
    normalized_action_to_targets,
)
from worm_env_v6 import WormEnvV6  # noqa: E402
from worm_v6 import (  # noqa: E402
    SLIDE_FORCE,
    SLIDE_KP,
    SLIDE_RANGE_VAL,
    YAW_FORCE,
    YAW_KP,
    YAW_RANGE_VAL,
)


def main():
    contract = motor_contract()
    assert contract["actuator_count"] == NUM_ACTUATORS
    assert contract["contract_fingerprint"]
    assert action_mapping_matches(contract["action_mapping"])
    assert action_mapping_matches(action_mapping_config())

    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    action[0] = -1.0
    action[1] = 0.0
    action[2] = 1.0
    action[NUM_SLIDES] = -1.0
    action[NUM_SLIDES + 1] = 1.0
    slide, yaw = normalized_action_to_targets(action)
    assert np.isclose(slide[0], SLIDE_TARGET_MIN_M)
    assert np.isclose(slide[1], 0.0)
    assert np.isclose(slide[2], SLIDE_TARGET_MAX_M)
    assert np.isclose(yaw[0], YAW_TARGET_MIN_RAD)
    assert np.isclose(yaw[1], YAW_TARGET_MAX_RAD)

    ctrl = normalized_action_to_ctrl(np.ones(NUM_ACTUATORS, dtype=np.float32))
    assert np.all(ctrl[:NUM_SLIDES] <= SLIDE_TARGET_MAX_M + 1e-9)
    assert np.all(ctrl[:NUM_SLIDES] >= SLIDE_TARGET_MIN_M - 1e-9)
    assert np.all(ctrl[NUM_SLIDES:] <= YAW_TARGET_MAX_RAD + 1e-9)

    assert SLIDE_RANGE_VAL == contract["action_mapping"]["slide_range_m"]
    assert YAW_RANGE_VAL == contract["action_mapping"]["yaw_range_rad"]
    assert SLIDE_KP == contract["actuator_groups"]["slide"][
        "mujoco_position_kp_n_per_m"]
    assert SLIDE_FORCE == contract["actuator_groups"]["slide"][
        "mujoco_force_limit_n"]
    assert YAW_KP == contract["actuator_groups"]["yaw"][
        "mujoco_position_kp_nm_per_rad"]
    assert YAW_FORCE == contract["actuator_groups"]["yaw"][
        "mujoco_torque_limit_nm"]

    env = WormEnvV6(terrain="flat", gait_mode="worm")
    try:
        env.reset(seed=123)
        env.step(np.ones(NUM_ACTUATORS, dtype=np.float32))
        assert np.all(env.data.ctrl[:NUM_SLIDES] <= SLIDE_TARGET_MAX_M + 1e-9)
        assert np.all(env.data.ctrl[:NUM_SLIDES] >= SLIDE_TARGET_MIN_M - 1e-9)
    finally:
        env.close()

    print("motor contract passed")


if __name__ == "__main__":
    main()

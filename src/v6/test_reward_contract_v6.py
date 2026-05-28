"""
Smoke tests for the V6 reward contract.

The policy may only observe deployable sensors, but training reward may use
simulator velocity. This test keeps that reward aligned with the paper metric:
positive forward displacement must be much better than stalling or reversing.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_v6 import training_config_compatible  # noqa: E402
from training_contract_v6 import residual_exploration_contract  # noqa: E402
from action_adapter_v6 import (  # noqa: E402
    action_adapter_contract,
    compose_deployable_action,
    gait_prior_from_phase,
)
from worm_env_v6 import (  # noqa: E402
    CMD_VEL_RANGE,
    NUM_ACTUATORS,
    OBS_DIM,
    WormEnvV6,
    reward_contract,
)


class DummyModel:
    nu = 0


class DummyData:
    def __init__(self):
        self.qvel = np.zeros(6, dtype=np.float64)
        self.ctrl = np.zeros(0, dtype=np.float64)
        self.xpos = np.zeros((1, 3), dtype=np.float64)


def reward_for(forward_speed, lateral_speed=0.0, cmd_vel=None,
               action=None, residual_action=None):
    env = object.__new__(WormEnvV6)
    env._cmd_vel = CMD_VEL_RANGE[1] if cmd_vel is None else cmd_vel
    env._cmd_yaw = 0.0
    env._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_residual_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._act_qvel_idx = []
    env.model = DummyModel()
    env.data = DummyData()
    env.data.qvel[0] = -forward_speed
    env.data.qvel[1] = lateral_speed
    env.data.qvel[5] = 0.0
    if action is None:
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    return WormEnvV6._compute_reward(
        env, action, residual_action=residual_action)


def reward_for_displacement(forward_delta_m):
    env = object.__new__(WormEnvV6)
    env._cmd_vel = CMD_VEL_RANGE[1]
    env._cmd_yaw = 0.0
    env._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_residual_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._act_qvel_idx = []
    env._root_body_id = 0
    env._last_root_pos = np.zeros(3, dtype=np.float64)
    env.model = DummyModel()
    env.data = DummyData()
    env.data.xpos[0, 0] = -forward_delta_m
    return WormEnvV6._compute_reward(
        env, np.zeros(NUM_ACTUATORS, dtype=np.float32))


def config_stub(contract):
    return {
        "terrain": "sand",
        "gait_mode": "snake",
        "gait_blend": 1.0,
        "obs_dim": OBS_DIM,
        "num_actuators": NUM_ACTUATORS,
        "num_imus": 7,
        "sensor_robustness": {},
        "control_timing": {},
        "reward_contract": contract,
        "action_adapter": action_adapter_contract(),
        "residual_exploration": residual_exploration_contract(),
        "eval_command": {
            "cmd_vel_m_s": CMD_VEL_RANGE[1],
            "cmd_yaw_rad_s": 0.0,
            "command_resample_prob": 0.0,
        },
    }


def main():
    assert CMD_VEL_RANGE[1] >= 0.20, CMD_VEL_RANGE
    forward = reward_for(CMD_VEL_RANGE[1])
    slow = reward_for(0.025, cmd_vel=CMD_VEL_RANGE[1])
    stalled = reward_for(0.0)
    backward = reward_for(-0.005)
    lateral = reward_for(CMD_VEL_RANGE[1], lateral_speed=CMD_VEL_RANGE[1])
    prior_action = np.ones(NUM_ACTUATORS, dtype=np.float32)
    residual_smooth = reward_for(
        CMD_VEL_RANGE[1],
        action=prior_action,
        residual_action=np.zeros(NUM_ACTUATORS, dtype=np.float32))
    residual_jump = reward_for(
        CMD_VEL_RANGE[1],
        action=prior_action,
        residual_action=np.ones(NUM_ACTUATORS, dtype=np.float32))
    displacement_reward = reward_for_displacement(
        CMD_VEL_RANGE[1] * 0.02)

    assert forward > 8.0, forward
    assert forward > slow + 4.0, (forward, slow)
    assert stalled < 0.0, stalled
    assert backward < stalled, (backward, stalled)
    assert lateral < forward, (lateral, forward)
    assert residual_smooth > residual_jump, (residual_smooth, residual_jump)
    assert displacement_reward > stalled + 4.0, (
        displacement_reward, stalled)

    contract = reward_contract()
    assert contract["version"] == "high_speed_directional_v1"
    assert contract["normalization"]["positive_forward_required_for_vel_track"]
    prior = gait_prior_from_phase(0.0, gait_blend=0.0)
    assert prior.shape == (NUM_ACTUATORS,)
    assert np.any(prior[:6] < -0.1)
    residual = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    action = compose_deployable_action(residual, 0.0, gait_blend=0.0)
    assert np.allclose(action, prior)

    env = object.__new__(WormEnvV6)
    env.np_random = np.random.default_rng(0)
    env._fixed_cmd_vel = 0.012
    env._fixed_cmd_yaw = 0.1
    assert env._sample_cmd_vel() == 0.012
    assert env._sample_cmd_yaw() == 0.1

    ok, reasons = training_config_compatible(
        config_stub(contract), config_stub(contract))
    assert ok, reasons

    old = dict(contract)
    old["version"] = "old"
    ok, reasons = training_config_compatible(
        config_stub(old), config_stub(contract))
    assert not ok
    assert "reward_contract" in reasons

    stale_exploration = config_stub(contract)
    stale_exploration["residual_exploration"] = {
        "version": "high_noise_old",
    }
    ok, reasons = training_config_compatible(
        stale_exploration, config_stub(contract))
    assert not ok
    assert "residual_exploration" in reasons

    print("reward contract checks passed")


if __name__ == "__main__":
    main()

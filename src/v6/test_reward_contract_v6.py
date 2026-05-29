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

from train_v6 import (  # noqa: E402
    best_eval_schedule,
    best_eval_schedule_fingerprint,
    best_selection_contract,
    directional_eval_summary,
    yaw_direction_status,
    training_config_compatible,
)
from training_contract_v6 import residual_exploration_contract  # noqa: E402
from action_adapter_v6 import (  # noqa: E402
    action_adapter_contract,
    compose_deployable_action,
    gait_prior_from_phase,
    policy_action_to_residual_and_gait_blend,
)
from worm_env_v6 import (  # noqa: E402
    CMD_VX_RANGE,
    CMD_VY_RANGE,
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
        self.xmat = np.eye(3, dtype=np.float64).reshape(1, 9)


def reward_for(body_vx, body_vy=0.0, yaw_rate=0.0,
               cmd_vx=None, cmd_vy=0.0, cmd_yaw=0.0, action=None,
               residual_action=None):
    env = object.__new__(WormEnvV6)
    env._cmd_vx = CMD_VX_RANGE[1] if cmd_vx is None else cmd_vx
    env._cmd_vy = cmd_vy
    env._cmd_yaw = cmd_yaw
    env._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_residual_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._act_qvel_idx = []
    env.model = DummyModel()
    env.data = DummyData()
    env.data.qvel[0] = -body_vx
    env.data.qvel[1] = body_vy
    env.data.qvel[5] = yaw_rate
    if action is None:
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    return WormEnvV6._compute_reward(
        env, action, residual_action=residual_action)


def reward_for_displacement(forward_delta_m):
    env = object.__new__(WormEnvV6)
    env._cmd_vx = CMD_VX_RANGE[1]
    env._cmd_vy = 0.0
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
            "cmd_vx_m_s": CMD_VX_RANGE[1],
            "cmd_vy_m_s": 0.0,
            "cmd_yaw_rad_s": 0.0,
            "command_resample_prob": 0.0,
        },
        "best_selection_contract": best_selection_contract(),
    }


def main():
    assert CMD_VX_RANGE[1] >= 0.20, CMD_VX_RANGE
    assert CMD_VY_RANGE[1] > 0.0, CMD_VY_RANGE
    forward = reward_for(CMD_VX_RANGE[1])
    slow = reward_for(0.025, cmd_vx=CMD_VX_RANGE[1])
    stalled = reward_for(0.0)
    backward = reward_for(-0.005)
    lateral = reward_for(CMD_VX_RANGE[1], body_vy=CMD_VX_RANGE[1])
    sideways_correct = reward_for(
        0.0, body_vy=CMD_VY_RANGE[1], cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1])
    sideways_wrong = reward_for(
        0.0, body_vy=-CMD_VY_RANGE[1], cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1])
    backward_correct = reward_for(
        -CMD_VX_RANGE[1], cmd_vx=-CMD_VX_RANGE[1], cmd_vy=0.0)
    backward_wrong = reward_for(
        CMD_VX_RANGE[1], cmd_vx=-CMD_VX_RANGE[1], cmd_vy=0.0)
    prior_action = np.ones(NUM_ACTUATORS, dtype=np.float32)
    residual_smooth = reward_for(
        CMD_VX_RANGE[1],
        action=prior_action,
        residual_action=np.zeros(NUM_ACTUATORS, dtype=np.float32))
    residual_jump = reward_for(
        CMD_VX_RANGE[1],
        action=prior_action,
        residual_action=np.ones(NUM_ACTUATORS, dtype=np.float32))
    displacement_reward = reward_for_displacement(
        CMD_VX_RANGE[1] * 0.02)

    assert forward > 8.0, forward
    assert forward > slow + 4.0, (forward, slow)
    assert stalled < 0.0, stalled
    assert backward < stalled, (backward, stalled)
    assert lateral < forward, (lateral, forward)
    assert sideways_correct > sideways_wrong + 4.0, (
        sideways_correct, sideways_wrong)
    assert backward_correct > backward_wrong + 4.0, (
        backward_correct, backward_wrong)
    assert residual_smooth > residual_jump, (residual_smooth, residual_jump)
    assert displacement_reward > stalled + 4.0, (
        displacement_reward, stalled)

    contract = reward_contract()
    assert contract["version"] == "omni_auto_gate_v4"
    assert contract["normalization"]["body_frame_vx_vy_command_tracking"]
    assert contract["normalization"]["off_axis_penalty_tapers_with_planar_command"]
    assert contract["normalization"]["signed_yaw_alignment_reward"]
    assert contract["normalization"]["gait_blend_is_policy_gate"]
    left_correct = reward_for(
        CMD_VX_RANGE[1], yaw_rate=0.5, cmd_yaw=0.5)
    left_wrong = reward_for(
        CMD_VX_RANGE[1], yaw_rate=-0.5, cmd_yaw=0.5)
    assert left_correct > left_wrong + 6.0, (left_correct, left_wrong)
    prior = gait_prior_from_phase(0.0, gait_blend=0.0)
    assert prior.shape == (NUM_ACTUATORS,)
    assert np.any(prior[:6] < -0.1)
    residual = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    action = compose_deployable_action(residual, 0.0, gait_blend=0.0)
    assert np.allclose(action, prior)
    policy_action = np.zeros(NUM_ACTUATORS + 1, dtype=np.float32)
    policy_action[-1] = 1.0
    residual, blend = policy_action_to_residual_and_gait_blend(policy_action)
    assert residual.shape == (NUM_ACTUATORS,)
    assert blend == 1.0

    env = object.__new__(WormEnvV6)
    env.np_random = np.random.default_rng(0)
    env._fixed_cmd_vx = 0.012
    env._fixed_cmd_vy = -0.034
    env._fixed_cmd_yaw = 0.1
    assert env._sample_cmd_vx() == 0.012
    assert env._sample_cmd_vy() == -0.034
    assert env._sample_cmd_yaw() == 0.1

    random_schedule = best_eval_schedule("random")
    assert len(random_schedule) == 3
    assert {c["gait_blend"] for c in random_schedule} == {None}
    assert {c["cmd_yaw_rad_s"] for c in random_schedule} == {-0.5, 0.0, 0.5}
    assert len(best_eval_schedule("worm")) == 3
    assert best_eval_schedule_fingerprint(random_schedule)
    assert yaw_direction_status(0.5, 0.2)["status"] == "correct"
    assert yaw_direction_status(0.5, -0.2)["status"] == "wrong_sign"
    assert yaw_direction_status(-0.5, -0.2)["status"] == "correct"
    assert yaw_direction_status(0.0, 0.3)["status"] == "straight_drift"
    compact_schedule = [
        {"gait_blend": 0.5, "cmd_yaw_rad_s": 0.5},
        {"gait_blend": 0.5, "cmd_yaw_rad_s": -0.5},
        {"gait_blend": 0.5, "cmd_yaw_rad_s": 0.0},
    ]
    good_direction = directional_eval_summary(
        compact_schedule,
        episode_rewards=[1000.0, 1000.0, 1000.0],
        yaw_deltas_rad=[0.2, -0.2, 0.02],
    )
    bad_direction = directional_eval_summary(
        compact_schedule,
        episode_rewards=[1300.0, 1300.0, 1300.0],
        yaw_deltas_rad=[-0.2, -0.2, 0.3],
    )
    assert good_direction["direction_gate_passed"]
    assert not bad_direction["direction_gate_passed"]
    assert bad_direction["wrong_sign_count"] == 1
    assert bad_direction["straight_violation_count"] == 1
    assert bad_direction["selection_score"] < good_direction["selection_score"]

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

    stale_selection = config_stub(contract)
    stale_selection["best_selection_contract"] = {
        "version": "mean_reward_only",
    }
    ok, reasons = training_config_compatible(
        stale_selection, config_stub(contract))
    assert not ok
    assert "best_selection_contract" in reasons

    print("reward contract checks passed")


if __name__ == "__main__":
    main()

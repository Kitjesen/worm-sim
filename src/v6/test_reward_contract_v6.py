"""
Smoke tests for the V6 reward contract.

The policy may only observe deployable sensors, but training reward may use
simulator velocity. This test keeps that reward aligned with the paper metric:
positive forward displacement must be much better than stalling or reversing.
"""

import os
import sys
import math

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_v6 import (  # noqa: E402
    best_eval_schedule,
    best_eval_schedule_fingerprint,
    best_selection_contract,
    directional_eval_summary,
    directional_eval_steps_from_seconds,
    make_run_dirs,
    yaw_direction_status,
    training_config_compatible,
)
from training_contract_v6 import (  # noqa: E402
    MAX_STRAIGHT_VIOLATION_COUNT,
    REQUIRED_PLANAR_SUCCESS_RATE,
    REQUIRED_YAW_SUCCESS_RATE,
    residual_exploration_contract,
)
from action_adapter_v6 import (  # noqa: E402
    action_adapter_contract,
    axial_gate_center_from_speed,
    command_conditioned_gate_center,
    command_activity_scale,
    command_conditioned_prior_scale,
    command_prior_scale_floor,
    command_directional_prior_transform,
    compose_deployable_action,
    directional_gait_prior_from_phase,
    gait_prior_from_phase,
    inplace_yaw_prior_from_phase,
    policy_action_to_residual_and_gait_blend,
)
from worm_env_v6 import (  # noqa: E402
    CMD_YAW_RANGE,
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    COMMAND_CURRICULA,
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
    forward_yaw_stable = reward_for(
        CMD_VX_RANGE[1], yaw_rate=0.0, cmd_vx=CMD_VX_RANGE[1], cmd_yaw=0.0)
    forward_yaw_drift = reward_for(
        CMD_VX_RANGE[1], yaw_rate=0.5, cmd_vx=CMD_VX_RANGE[1], cmd_yaw=0.0)
    lateral_yaw_stable = reward_for(
        0.0, body_vy=CMD_VY_RANGE[1], yaw_rate=0.0,
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    lateral_yaw_drift = reward_for(
        0.0, body_vy=CMD_VY_RANGE[1], yaw_rate=0.5,
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    lateral_clean = reward_for(
        0.0, body_vy=CMD_VY_RANGE[1],
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    lateral_forward_drift = reward_for(
        0.08, body_vy=CMD_VY_RANGE[1],
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    yaw_only_stationary = reward_for(
        0.0, body_vy=0.0, yaw_rate=0.10,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.5)
    yaw_only_with_planar_drift = reward_for(
        0.06, body_vy=0.0, yaw_rate=0.10,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.5)
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
    assert forward_yaw_stable > forward_yaw_drift + 6.0, (
        forward_yaw_stable, forward_yaw_drift)
    assert lateral_yaw_stable > lateral_yaw_drift + 6.0, (
        lateral_yaw_stable, lateral_yaw_drift)
    assert lateral_clean > lateral_forward_drift + 6.0, (
        lateral_clean, lateral_forward_drift)
    assert yaw_only_stationary > yaw_only_with_planar_drift + 6.0, (
        yaw_only_stationary, yaw_only_with_planar_drift)
    assert residual_smooth > residual_jump, (residual_smooth, residual_jump)
    assert displacement_reward > stalled + 4.0, (
        displacement_reward, stalled)

    contract = reward_contract()
    assert contract["version"] == "omni_directional_offaxis_yaw_v18"
    assert contract["normalization"]["body_frame_vx_vy_command_tracking"]
    assert contract["normalization"]["off_axis_penalty_tapers_with_planar_command"]
    assert contract["normalization"]["strong_off_axis_suppression"]
    assert contract["normalization"]["signed_yaw_alignment_reward"]
    assert contract["normalization"]["zero_yaw_integrated_drift_penalty"]
    assert contract["normalization"]["zero_yaw_translation_uses_reset_body_axes"]
    assert contract["normalization"]["zero_yaw_heading_hold_weight_boost"]
    assert contract["normalization"]["yaw_only_stationary_speed_penalty"]
    assert contract["normalization"]["pure_lateral_forward_drift_penalty"]
    assert contract["normalization"]["continuous_omni_repair_oversampling"]
    assert contract["normalization"][
        "continuous_omni_mixed_yaw_repair_sampling"]
    assert contract["normalization"]["gait_blend_is_policy_gate"]
    assert contract["normalization"][
        "command_conditioned_gait_gate_regularizer"]["enabled"]
    assert contract["weights"]["yaw_drift"] >= 6.0
    assert contract["weights"]["yaw_stationary"] >= 3.0
    assert contract["weights"]["lateral_only_forward_drift"] > 0.0
    assert contract["weights"]["gait_gate_target"] > 0.0
    selection_contract = best_selection_contract()
    assert selection_contract["version"] == "omni_tracking_scan_v3"
    assert selection_contract["requires_continuous_tracking_metrics"]
    assert selection_contract["required_planar_success_rate"] == (
        REQUIRED_PLANAR_SUCCESS_RATE)
    assert selection_contract["required_yaw_success_rate"] == (
        REQUIRED_YAW_SUCCESS_RATE)
    assert selection_contract["max_straight_violation_count"] == (
        MAX_STRAIGHT_VIOLATION_COUNT)
    left_correct = reward_for(
        CMD_VX_RANGE[1], yaw_rate=0.5, cmd_yaw=0.5)
    left_wrong = reward_for(
        CMD_VX_RANGE[1], yaw_rate=-0.5, cmd_yaw=0.5)
    assert left_correct > left_wrong + 6.0, (left_correct, left_wrong)
    prior = gait_prior_from_phase(0.0, gait_blend=0.0)
    assert prior.shape == (NUM_ACTUATORS,)
    assert np.any(prior[:6] < -0.1)
    adapter_contract = action_adapter_contract()
    assert adapter_contract["version"].endswith("_v23")
    assert adapter_contract["gait_gate_mapping"]["raw_action_index"] == (
        NUM_ACTUATORS)
    centers = adapter_contract["gait_gate_mapping"]["command_centers"]
    assert centers["axial_translation"] < centers["mixed_or_stop"]
    assert centers["axial_translation_slow"] < centers["axial_translation_fast"]
    assert centers["axial_translation_fast"] == centers["mixed_or_stop"]
    assert centers["lateral_translation"] == centers["mixed_or_stop"]
    assert centers["yaw"] >= centers["lateral_translation"]
    assert command_conditioned_gate_center((0.25, 0.0, 0.0)) < (
        command_conditioned_gate_center((1.0, 0.0, 0.0)))
    assert np.isclose(
        command_conditioned_gate_center((1.0, 0.0, 0.0)),
        centers["axial_translation_fast"])
    assert np.isclose(
        axial_gate_center_from_speed(1.0),
        centers["axial_translation_fast"])
    assert adapter_contract["command_directional_prior_transform"][
        "inplace_yaw_prior"]["enabled"]
    residual = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    action = compose_deployable_action(residual, 0.0, gait_blend=0.0)
    assert np.allclose(action, prior)
    policy_action = np.zeros(NUM_ACTUATORS + 1, dtype=np.float32)
    policy_action[-1] = 1.0
    residual, blend = policy_action_to_residual_and_gait_blend(policy_action)
    assert residual.shape == (NUM_ACTUATORS,)
    assert blend == 1.0
    reverse_tf = command_directional_prior_transform(-1.0, 0.0, 0.0)
    assert reverse_tf["phase_sign"] == -1.0
    assert reverse_tf["phase_offset_rad"] == math.pi
    assert reverse_tf["yaw_scale"] > 1.0
    assert reverse_tf["yaw_trim"] == 0.0
    forward_tf = command_directional_prior_transform(1.0, 0.0, 0.0)
    assert forward_tf["phase_offset_rad"] == math.pi
    assert forward_tf["yaw_scale"] > 1.0
    left_tf = command_directional_prior_transform(0.0, 1.0, 0.0)
    right_tf = command_directional_prior_transform(0.0, -1.0, 0.0)
    assert left_tf["phase_offset_rad"] > 0.0
    assert right_tf["phase_offset_rad"] == left_tf["phase_offset_rad"]
    assert left_tf["yaw_sign"] == 1.0
    assert right_tf["yaw_sign"] == -1.0
    assert 0.0 < left_tf["yaw_scale"] < 1.0
    assert right_tf["yaw_scale"] == left_tf["yaw_scale"]
    assert left_tf["yaw_trim"] > 0.0
    assert right_tf["yaw_trim"] < 0.0
    yaw_left_tf = command_directional_prior_transform(0.0, 0.0, 1.0)
    yaw_right_tf = command_directional_prior_transform(0.0, 0.0, -1.0)
    assert yaw_left_tf["uses_inplace_yaw_prior"]
    assert yaw_right_tf["uses_inplace_yaw_prior"]
    assert yaw_left_tf["yaw_sign"] == 1.0
    assert yaw_right_tf["yaw_sign"] == -1.0
    assert yaw_right_tf["yaw_scale"] == yaw_left_tf["yaw_scale"]
    base_prior = gait_prior_from_phase(0.3, gait_blend=1.0)
    reverse_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=1.0, command=(-1.0, 0.0, 0.0))
    reverse_base = gait_prior_from_phase(-0.3 + math.pi, gait_blend=1.0)
    assert np.allclose(reverse_prior[:6], reverse_base[:6])
    assert np.allclose(
        reverse_prior[6:],
        np.clip(reverse_tf["yaw_scale"] * reverse_base[6:], -1.0, 1.0),
    )
    yaw_left_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=0.5, command=(0.0, 0.0, 1.0))
    yaw_right_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=0.5, command=(0.0, 0.0, -1.0))
    yaw_open_loop = inplace_yaw_prior_from_phase(0.3, 1.0)
    assert np.allclose(yaw_left_prior, yaw_open_loop)
    assert np.allclose(yaw_left_prior[:6], yaw_right_prior[:6])
    assert np.allclose(yaw_left_prior[6:], -yaw_right_prior[6:])
    assert command_conditioned_prior_scale(1.0, 0.0, 0.0) == 1.0
    assert command_activity_scale(0.0, 0.0, 0.0) == 0.0
    assert np.isclose(command_activity_scale(0.5, 0.0, 0.0), 0.6)
    assert command_activity_scale(1.0, 0.0, 0.0) == 1.0
    assert np.isclose(command_prior_scale_floor(-1.0, 0.0, 0.0), 0.4)
    assert np.isclose(command_prior_scale_floor(0.0, 1.0, 0.0), 0.8)
    assert np.isclose(command_prior_scale_floor(0.0, 0.0, 1.0), 1.0)
    assert np.isclose(
        command_conditioned_prior_scale(-1.0, 0.0, 0.0), 0.4)
    assert np.isclose(
        command_conditioned_prior_scale(0.0, 1.0, 0.0), 0.8)
    assert np.isclose(
        command_conditioned_prior_scale(0.0, 0.0, 1.0), 1.0)
    zero_command_action = compose_deployable_action(
        np.ones(NUM_ACTUATORS, dtype=np.float32),
        0.0,
        gait_blend=0.5,
        command=(0.0, 0.0, 0.0),
    )
    assert np.allclose(zero_command_action, 0.0)
    mixed_prior_scale = command_conditioned_prior_scale(0.6, 0.0, 1.0)
    assert 0.55 < mixed_prior_scale < 0.8

    env = object.__new__(WormEnvV6)
    env.np_random = np.random.default_rng(0)
    env._fixed_cmd_vx = 0.012
    env._fixed_cmd_vy = -0.034
    env._fixed_cmd_yaw = 0.1
    assert env._sample_cmd_vx() == 0.012
    assert env._sample_cmd_vy() == -0.034
    assert env._sample_cmd_yaw() == 0.1
    env = object.__new__(WormEnvV6)
    env.np_random = np.random.default_rng(1)
    env._fixed_cmd_vx = None
    env._fixed_cmd_vy = None
    env._fixed_cmd_yaw = None
    env.command_curriculum = "lateral"
    vx, vy, yaw = env._sample_command()
    assert vx == 0.0
    assert abs(vy) >= 0.3 * CMD_VY_RANGE[1]
    assert yaw == 0.0
    env.command_curriculum = "lateral_right"
    vx, vy, yaw = env._sample_command()
    assert vx == 0.0
    assert vy < 0.0
    assert yaw == 0.0
    env.command_curriculum = "yaw_right"
    vx, vy, yaw = env._sample_command()
    assert vx == 0.0
    assert vy == 0.0
    assert yaw < 0.0
    env.command_curriculum = "heading_hold"
    vx, vy, yaw = env._sample_command()
    assert yaw == 0.0
    assert (abs(vx) > 0.0) ^ (abs(vy) > 0.0)
    assert abs(vx) >= 0.5 * CMD_VX_RANGE[1] or abs(vy) >= 0.5 * CMD_VY_RANGE[1]
    assert "right_recovery" in COMMAND_CURRICULA
    assert "heading_hold" in COMMAND_CURRICULA
    assert "heading_omni" in COMMAND_CURRICULA
    assert "continuous_omni" in COMMAND_CURRICULA

    env.command_curriculum = "continuous_omni"
    samples = np.array([env._sample_command() for _ in range(240)])
    normalized = np.column_stack((
        samples[:, 0] / CMD_VX_RANGE[1],
        samples[:, 1] / CMD_VY_RANGE[1],
        samples[:, 2] / CMD_YAW_RANGE[1],
    ))
    command_mag = np.max(np.abs(normalized), axis=1)
    nonzero_dims = np.count_nonzero(np.abs(samples) > 1e-9, axis=1)
    assert np.any(command_mag == 0.0)
    assert np.any((command_mag > 0.0) & (command_mag <= 0.25))
    assert np.any(nonzero_dims >= 2)
    mixed_yaw_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    assert len(mixed_yaw_samples) >= 8
    assert np.any((mixed_yaw_samples[:, 0] > 0.0)
                  & (mixed_yaw_samples[:, 2] > 0.0))
    assert np.any((mixed_yaw_samples[:, 0] > 0.0)
                  & (mixed_yaw_samples[:, 2] < 0.0))
    assert np.any((mixed_yaw_samples[:, 0] < 0.0)
                  & (mixed_yaw_samples[:, 2] > 0.0))
    assert np.any((mixed_yaw_samples[:, 0] < 0.0)
                  & (mixed_yaw_samples[:, 2] < 0.0))

    random_schedule = best_eval_schedule("random")
    expected_cases = {
        "stop",
        "slow_forward",
        "slow_reverse",
        "forward",
        "reverse",
        "lateral_left",
        "lateral_right",
        "mixed_forward_left",
        "mixed_forward_right",
        "mixed_reverse_left",
        "mixed_reverse_right",
        "yaw_left",
        "yaw_right",
        "slow_yaw_left",
        "slow_yaw_right",
        "forward_yaw_left",
        "forward_yaw_right",
    }
    assert expected_cases.issubset(
        {case["case_name"] for case in random_schedule})
    assert {c["gait_blend"] for c in random_schedule} == {None}
    assert any(c["cmd_vx_m_s"] < 0.0 for c in random_schedule)
    assert any(c["cmd_vy_m_s"] > 0.0 for c in random_schedule)
    assert any(c["cmd_vy_m_s"] < 0.0 for c in random_schedule)
    assert {c["cmd_yaw_rad_s"] for c in random_schedule} >= {-0.5, 0.0, 0.5}
    assert len(best_eval_schedule("worm")) == len(random_schedule)
    assert best_eval_schedule_fingerprint(random_schedule)
    assert yaw_direction_status(0.5, 0.2)["status"] == "correct"
    assert yaw_direction_status(0.5, -0.2)["status"] == "wrong_sign"
    assert yaw_direction_status(-0.5, -0.2)["status"] == "correct"
    assert yaw_direction_status(0.0, 0.3)["status"] == "straight_drift"
    compact_schedule = [
        {
            "case_name": "forward",
            "gait_blend": 0.5,
            "cmd_vx_m_s": 0.10,
            "cmd_vy_m_s": 0.0,
            "cmd_yaw_rad_s": 0.0,
        },
        {
            "case_name": "reverse",
            "gait_blend": 0.5,
            "cmd_vx_m_s": -0.10,
            "cmd_vy_m_s": 0.0,
            "cmd_yaw_rad_s": 0.0,
        },
        {
            "case_name": "lateral_left",
            "gait_blend": 0.5,
            "cmd_vx_m_s": 0.0,
            "cmd_vy_m_s": 0.10,
            "cmd_yaw_rad_s": 0.0,
        },
        {
            "case_name": "yaw_left",
            "gait_blend": 0.5,
            "cmd_vx_m_s": 0.0,
            "cmd_vy_m_s": 0.0,
            "cmd_yaw_rad_s": 0.5,
        },
    ]
    good_direction = directional_eval_summary(
        compact_schedule,
        episode_rewards=[1000.0, 1000.0, 1000.0, 1000.0],
        yaw_deltas_rad=[0.02, 0.02, 0.02, 0.2],
        body_deltas_m=[(0.20, 0.0), (-0.20, 0.0), (0.0, 0.20), (0.0, 0.0)],
        elapsed_s=[2.0, 2.0, 2.0, 2.0],
    )
    bad_direction = directional_eval_summary(
        compact_schedule,
        episode_rewards=[1300.0, 1300.0, 1300.0, 1300.0],
        yaw_deltas_rad=[0.02, 0.02, 0.02, -0.2],
        body_deltas_m=[(0.20, 0.0), (0.20, 0.0), (0.0, -0.20), (0.0, 0.0)],
        elapsed_s=[2.0, 2.0, 2.0, 2.0],
    )
    assert good_direction["direction_gate_passed"]
    assert "tracking_gate_passed" in good_direction
    assert "planar_velocity_rmse_m_s" in good_direction
    assert "yaw_rate_rmse_rad_s" in good_direction
    assert good_direction["planar_success_rate"] == 1.0
    assert not bad_direction["direction_gate_passed"]
    assert bad_direction["wrong_planar_sign_count"] == 2
    assert bad_direction["wrong_yaw_sign_count"] == 1
    assert bad_direction["selection_score"] < good_direction["selection_score"]
    full_schedule = best_eval_schedule("random")
    elapsed = [2.0] * len(full_schedule)
    good_body_deltas = [
        (
            case["cmd_vx_m_s"] * elapsed[idx],
            case["cmd_vy_m_s"] * elapsed[idx],
        )
        for idx, case in enumerate(full_schedule)
    ]
    one_straight_drift_yaws = []
    two_straight_drift_yaws = []
    drift_used = 0
    for case in full_schedule:
        cmd_yaw = case["cmd_yaw_rad_s"]
        if cmd_yaw > 0.0:
            one_straight_drift_yaws.append(0.2)
            two_straight_drift_yaws.append(0.2)
        elif cmd_yaw < 0.0:
            one_straight_drift_yaws.append(-0.2)
            two_straight_drift_yaws.append(-0.2)
        elif drift_used == 0:
            one_straight_drift_yaws.append(0.25)
            two_straight_drift_yaws.append(0.25)
            drift_used += 1
        elif drift_used == 1:
            one_straight_drift_yaws.append(0.0)
            two_straight_drift_yaws.append(-0.25)
            drift_used += 1
        else:
            one_straight_drift_yaws.append(0.0)
            two_straight_drift_yaws.append(0.0)
    one_straight_drift = directional_eval_summary(
        full_schedule,
        episode_rewards=[1000.0] * len(full_schedule),
        yaw_deltas_rad=one_straight_drift_yaws,
        body_deltas_m=good_body_deltas,
        elapsed_s=elapsed,
    )
    two_straight_drifts = directional_eval_summary(
        full_schedule,
        episode_rewards=[1000.0] * len(full_schedule),
        yaw_deltas_rad=two_straight_drift_yaws,
        body_deltas_m=good_body_deltas,
        elapsed_s=elapsed,
    )
    assert one_straight_drift["straight_violation_count"] == 1
    assert one_straight_drift["direction_gate_passed"]
    assert two_straight_drifts["straight_violation_count"] == 2
    assert not two_straight_drifts["direction_gate_passed"]
    perfect_tracking = directional_eval_summary(
        compact_schedule,
        episode_rewards=[1000.0, 1000.0, 1000.0, 1000.0],
        yaw_deltas_rad=[0.0, 0.0, 0.0, 1.0],
        body_deltas_m=[(0.20, 0.0), (-0.20, 0.0), (0.0, 0.20), (0.0, 0.0)],
        elapsed_s=[2.0, 2.0, 2.0, 2.0],
    )
    assert perfect_tracking["tracking_gate_passed"]
    assert perfect_tracking["planar_velocity_rmse_m_s"] == 0.0
    assert perfect_tracking["yaw_rate_rmse_rad_s"] == 0.0
    assert directional_eval_steps_from_seconds(None) is None
    assert directional_eval_steps_from_seconds(2.0) == 100
    assert directional_eval_steps_from_seconds(0.001) == 1

    ok, reasons = training_config_compatible(
        config_stub(contract), config_stub(contract))
    assert ok, reasons

    omni_curriculum = config_stub(contract)
    omni_curriculum["command_curriculum"] = "omni"
    planar_curriculum = config_stub(contract)
    planar_curriculum["command_curriculum"] = "planar"
    ok, reasons = training_config_compatible(
        omni_curriculum, planar_curriculum)
    assert not ok
    assert "command_curriculum" in reasons
    try:
        ok, reasons = training_config_compatible(
            omni_curriculum,
            planar_curriculum,
            allow_command_curriculum_mismatch=True,
        )
    except TypeError as exc:
        assert False, (
            "training_config_compatible must expose an explicit "
            f"curriculum-resume override: {exc}")
    assert ok, reasons

    old = dict(contract)
    old["version"] = "old"
    ok, reasons = training_config_compatible(
        config_stub(old), config_stub(contract))
    assert not ok
    assert "reward_contract" in reasons
    ok, reasons = training_config_compatible(
        config_stub(old),
        config_stub(contract),
        allow_experimental_contract_mismatch=True,
    )
    assert ok, reasons

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
    ok, reasons = training_config_compatible(
        stale_selection,
        config_stub(contract),
        allow_experimental_contract_mismatch=True,
    )
    assert ok, reasons

    run_dir, log_dir, ckpt_dir = make_run_dirs(
        "flat", "random", run_label="flat_random_omni_smoke")
    assert run_dir.endswith(
        os.path.join("runs", "worm_v6_ppo_flat_random_omni_smoke"))
    assert log_dir == os.path.join(run_dir, "logs")
    assert ckpt_dir == os.path.join(run_dir, "checkpoints")

    print("reward contract checks passed")


if __name__ == "__main__":
    main()

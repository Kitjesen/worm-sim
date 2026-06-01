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

import action_adapter_v6 as action_adapter  # noqa: E402
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
    command_conditioned_prior_authority_scale,
    command_conditioned_prior_scale,
    command_conditioned_residual_scale,
    command_prior_scale_floor,
    command_directional_prior_transform,
    compose_deployable_action,
    directional_gait_prior_from_phase,
    gait_prior_from_phase,
    inplace_yaw_prior_from_phase,
    lateral_primitive_action_from_phase,
    policy_action_to_residual_and_gait_blend,
    split_channel_mixed_planar_gait_prior_from_phase,
)
from worm_env_v6 import (  # noqa: E402
    CMD_YAW_RANGE,
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    COMMAND_CURRICULA,
    FEASIBLE_MIXED_VX_ABS_RANGE,
    FEASIBLE_MIXED_VY_ABS_RANGE,
    FEASIBLE_MIXED_YAW_ABS_RANGE,
    LOW_YAW_ENVELOPE_AXIAL_ABS_RANGE,
    LOW_YAW_ENVELOPE_YAW_ABS_RANGE,
    NUM_ACTUATORS,
    NUM_SLIDES,
    OBS_DIM,
    WormEnvV6,
    reward_contract,
)
from action_adapter_v6 import (  # noqa: E402
    LATERAL_PHASE_OFFSET_RAD,
    LATERAL_PRIOR_SCALE_FLOOR,
    MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED,
    MIXED_PLANAR_DOMINANT_PRIOR_SCALE_MULT,
    MIXED_PLANAR_REBALANCED_RESIDUAL_SCALE_MULT,
    MIXED_PLANAR_RESIDUAL_SCALE_MULT,
    MIXED_PLANAR_PRIOR_SCALE_FLOOR,
    MIXED_PLANAR_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE,
    MIXED_PLANAR_SPLIT_PRIOR_ENABLED,
    MIXED_YAW_RESIDUAL_SCALE_MULT,
    MIXED_YAW_PRIOR_SCALE_FLOOR,
    YAW_ONLY_RESIDUAL_SCALE_MULT,
    YAW_ONLY_SLIDE_PRIOR_SCALE,
    YAW_ONLY_YAW_PRIOR_SCALE,
)


class DummyModel:
    nu = 0


class DummyData:
    def __init__(self, num_bodies=1):
        self.qvel = np.zeros(6, dtype=np.float64)
        self.ctrl = np.zeros(0, dtype=np.float64)
        self.xpos = np.zeros((num_bodies, 3), dtype=np.float64)
        self.xmat = np.tile(
            np.eye(3, dtype=np.float64).reshape(1, 9),
            (num_bodies, 1),
        )


def reward_for(body_vx, body_vy=0.0, yaw_rate=0.0,
               cmd_vx=None, cmd_vy=0.0, cmd_yaw=0.0, action=None,
               residual_action=None, prior_component=None,
               residual_component=None, pre_clip_action=None):
    env = object.__new__(WormEnvV6)
    env._cmd_vx = CMD_VX_RANGE[1] if cmd_vx is None else cmd_vx
    env._cmd_vy = cmd_vy
    env._cmd_yaw = cmd_yaw
    env._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_residual_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_prior_component = (
        np.zeros(NUM_ACTUATORS, dtype=np.float32)
        if prior_component is None
        else np.asarray(prior_component, dtype=np.float32))
    env._last_residual_component = (
        np.zeros(NUM_ACTUATORS, dtype=np.float32)
        if residual_component is None
        else np.asarray(residual_component, dtype=np.float32))
    env._last_pre_clip_action = (
        env._last_prior_component + env._last_residual_component
        if pre_clip_action is None
        else np.asarray(pre_clip_action, dtype=np.float32))
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


def reward_terms_for(body_vx, body_vy=0.0, yaw_rate=0.0,
                     cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.0):
    env = object.__new__(WormEnvV6)
    env._cmd_vx = cmd_vx
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
    reward = WormEnvV6._compute_reward(
        env, np.zeros(NUM_ACTUATORS, dtype=np.float32))
    return reward, dict(env._last_reward_terms)


def body_shape_reference(segment_xy):
    positions = np.asarray(segment_xy, dtype=np.float64)
    link_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    xy_min = np.min(positions, axis=0)
    xy_max = np.max(positions, axis=0)
    return {
        "arc_m": float(np.sum(link_lengths)),
        "extent_m": float(np.linalg.norm(xy_max - xy_min)),
    }


def reward_terms_for_body_shape(segment_xy, cmd_yaw=CMD_YAW_RANGE[1],
                                cmd_vx=0.0, cmd_vy=0.0):
    segment_xy = np.asarray(segment_xy, dtype=np.float64)
    straight_xy = np.column_stack((
        -0.18 * np.arange(len(segment_xy), dtype=np.float64),
        np.zeros(len(segment_xy), dtype=np.float64),
    ))
    reference = body_shape_reference(straight_xy)

    env = object.__new__(WormEnvV6)
    env._cmd_vx = cmd_vx
    env._cmd_vy = cmd_vy
    env._cmd_yaw = cmd_yaw
    env._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_residual_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_prior_component = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_residual_component = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._last_pre_clip_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    env._act_qvel_idx = []
    env._root_body_id = 0
    env._seg_ids = list(range(len(segment_xy)))
    env._last_root_pos = np.zeros(3, dtype=np.float64)
    env._start_body_extent_m = reference["extent_m"]
    env._start_body_arc_m = reference["arc_m"]
    env.model = DummyModel()
    env.data = DummyData(num_bodies=len(segment_xy))
    env.data.xpos[:, :2] = segment_xy
    env.data.qvel[5] = cmd_yaw
    reward = WormEnvV6._compute_reward(
        env, np.zeros(NUM_ACTUATORS, dtype=np.float32))
    return reward, dict(env._last_reward_terms)


def sampled_command_classes(curriculum, count=240):
    env = object.__new__(WormEnvV6)
    env.command_curriculum = curriculum
    env._fixed_cmd_vx = None
    env._fixed_cmd_vy = None
    env._fixed_cmd_yaw = None
    env.np_random = np.random.default_rng(20260531)
    counts = {
        "stop": 0,
        "pure": 0,
        "mixed_vx_vy": 0,
        "mixed_vx_yaw": 0,
        "other_mixed": 0,
    }
    for _ in range(count):
        vx, vy, yaw = WormEnvV6._sample_command(env)
        has_vx = abs(vx) > 1e-9
        has_vy = abs(vy) > 1e-9
        has_yaw = abs(yaw) > 1e-9
        if not has_vx and not has_vy and not has_yaw:
            counts["stop"] += 1
        elif has_vx and has_vy and not has_yaw:
            counts["mixed_vx_vy"] += 1
        elif has_vx and not has_vy and has_yaw:
            counts["mixed_vx_yaw"] += 1
        elif sum([has_vx, has_vy, has_yaw]) == 1:
            counts["pure"] += 1
        else:
            counts["other_mixed"] += 1
    return counts


def test_split_channel_mixed_planar_prior_decouples_axes():
    assert MIXED_PLANAR_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE
    assert not MIXED_PLANAR_SPLIT_PRIOR_ENABLED

    phase = 0.37
    gait_blend = 0.5
    prior = split_channel_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, (1.0, 1.0, 0.0))
    axial = directional_gait_prior_from_phase(
        phase, gait_blend, (1.0, 0.0, 0.0))
    lateral = lateral_primitive_action_from_phase("left", phase)

    np.testing.assert_allclose(
        prior[:NUM_SLIDES],
        axial[:NUM_SLIDES],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        prior[NUM_SLIDES:],
        lateral[NUM_SLIDES:],
        atol=1e-6,
    )


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
    assert CMD_VX_RANGE[1] == FEASIBLE_MIXED_VX_ABS_RANGE[1], CMD_VX_RANGE
    assert CMD_VY_RANGE[1] == FEASIBLE_MIXED_VY_ABS_RANGE[1], CMD_VY_RANGE
    assert CMD_YAW_RANGE[1] == FEASIBLE_MIXED_YAW_ABS_RANGE[1], CMD_YAW_RANGE
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
    lateral_slow = reward_for(
        0.0, body_vy=0.01,
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    lateral_forward_drift = reward_for(
        0.08, body_vy=CMD_VY_RANGE[1],
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    lateral_left_reward, lateral_left_terms = reward_terms_for(
        0.08, body_vy=CMD_VY_RANGE[1],
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    lateral_right_reward, lateral_right_terms = reward_terms_for(
        0.08, body_vy=CMD_VY_RANGE[0],
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[0], cmd_yaw=0.0)
    lateral_wrong_reward, lateral_wrong_terms = reward_terms_for(
        0.0, body_vy=-CMD_VY_RANGE[1],
        cmd_vx=0.0, cmd_vy=CMD_VY_RANGE[1], cmd_yaw=0.0)
    yaw_only_stationary = reward_for(
        0.0, body_vy=0.0, yaw_rate=0.10,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.5)
    yaw_only_with_planar_drift = reward_for(
        0.06, body_vy=0.0, yaw_rate=0.10,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.5)
    _, yaw_only_medium_drift_terms = reward_terms_for(
        0.06, body_vy=0.0, yaw_rate=0.10,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=CMD_YAW_RANGE[1])
    _, yaw_only_high_drift_terms = reward_terms_for(
        0.10, body_vy=0.0, yaw_rate=0.10,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=CMD_YAW_RANGE[1])
    straight_shape_xy = np.column_stack((
        -0.18 * np.arange(7, dtype=np.float64),
        np.zeros(7, dtype=np.float64),
    ))
    compact_shape_xy = np.array([
        [0.00, 0.00],
        [0.05, 0.02],
        [0.02, 0.05],
        [-0.03, 0.04],
        [-0.05, 0.00],
        [-0.02, -0.04],
        [0.03, -0.03],
    ], dtype=np.float64)
    yaw_straight_shape_reward, yaw_straight_shape_terms = (
        reward_terms_for_body_shape(straight_shape_xy))
    yaw_compact_shape_reward, yaw_compact_shape_terms = (
        reward_terms_for_body_shape(compact_shape_xy))
    forward_compact_shape_reward, forward_compact_shape_terms = (
        reward_terms_for_body_shape(
            compact_shape_xy,
            cmd_yaw=0.0,
            cmd_vx=CMD_VX_RANGE[1],
        ))
    prior_action = np.ones(NUM_ACTUATORS, dtype=np.float32)
    residual_smooth = reward_for(
        CMD_VX_RANGE[1],
        action=prior_action,
        residual_action=np.zeros(NUM_ACTUATORS, dtype=np.float32))
    residual_jump = reward_for(
        CMD_VX_RANGE[1],
        action=prior_action,
        residual_action=np.ones(NUM_ACTUATORS, dtype=np.float32))
    axial_prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    axial_prior[:6] = 0.8
    residual_aligned = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    residual_aligned[:6] = 0.1
    residual_cancel = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    residual_cancel[:6] = -0.4
    axial_preserved = reward_for(
        CMD_VX_RANGE[1],
        cmd_vx=CMD_VX_RANGE[1],
        prior_component=axial_prior,
        residual_component=residual_aligned)
    axial_cancelled = reward_for(
        CMD_VX_RANGE[1],
        cmd_vx=CMD_VX_RANGE[1],
        prior_component=axial_prior,
        residual_component=residual_cancel)
    mixed_planar_clean = reward_for(
        0.12, body_vy=0.07, cmd_vx=0.12, cmd_vy=0.07)
    mixed_planar_wrong_vy = reward_for(
        0.12, body_vy=-0.07, cmd_vx=0.12, cmd_vy=0.07)
    full_diag_good = reward_for(
        CMD_VX_RANGE[1] * 0.92,
        body_vy=CMD_VY_RANGE[1] * 0.92,
        cmd_vx=CMD_VX_RANGE[1],
        cmd_vy=CMD_VY_RANGE[1])
    full_diag_underpowered = reward_for(
        CMD_VX_RANGE[1] * 0.30,
        body_vy=CMD_VY_RANGE[1] * 0.30,
        cmd_vx=CMD_VX_RANGE[1],
        cmd_vy=CMD_VY_RANGE[1])
    _, full_diag_underpowered_terms = reward_terms_for(
        CMD_VX_RANGE[1] * 0.30,
        body_vy=CMD_VY_RANGE[1] * 0.30,
        cmd_vx=CMD_VX_RANGE[1],
        cmd_vy=CMD_VY_RANGE[1])
    _, full_diag_good_terms = reward_terms_for(
        CMD_VX_RANGE[1] * 0.92,
        body_vy=CMD_VY_RANGE[1] * 0.92,
        cmd_vx=CMD_VX_RANGE[1],
        cmd_vy=CMD_VY_RANGE[1])
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
    assert lateral_clean > lateral_slow + 2.0, (
        lateral_clean, lateral_slow)
    assert lateral_clean > lateral_forward_drift + 6.0, (
        lateral_clean, lateral_forward_drift)
    assert lateral_left_terms[
        "reward_lateral_only_speed_deficit_penalty"] == 0.0
    assert lateral_right_terms[
        "reward_lateral_only_speed_deficit_penalty"] == 0.0
    assert lateral_wrong_terms[
        "reward_lateral_only_speed_deficit_penalty"] > 0.0
    assert lateral_left_reward > lateral_wrong_reward
    assert lateral_right_reward > lateral_wrong_reward
    assert yaw_only_stationary > yaw_only_with_planar_drift + 6.0, (
        yaw_only_stationary, yaw_only_with_planar_drift)
    assert yaw_only_high_drift_terms[
        "reward_yaw_stationary_penalty"] > yaw_only_medium_drift_terms[
            "reward_yaw_stationary_penalty"]
    assert yaw_straight_shape_terms[
        "reward_yaw_body_compactness_penalty"] == 0.0
    assert yaw_compact_shape_terms[
        "reward_yaw_body_compactness_penalty"] > 0.5
    assert yaw_straight_shape_reward > yaw_compact_shape_reward + 5.0, (
        yaw_straight_shape_reward, yaw_compact_shape_reward)
    assert forward_compact_shape_terms[
        "reward_yaw_body_compactness_penalty"] == 0.0
    assert np.isclose(
        forward_compact_shape_terms["body_extent_ratio"],
        yaw_compact_shape_terms["body_extent_ratio"])
    assert residual_smooth > residual_jump, (residual_smooth, residual_jump)
    assert axial_preserved > axial_cancelled, (
        axial_preserved, axial_cancelled)
    assert mixed_planar_clean > mixed_planar_wrong_vy + 3.0, (
        mixed_planar_clean, mixed_planar_wrong_vy)
    assert full_diag_good > full_diag_underpowered + 4.0, (
        full_diag_good, full_diag_underpowered)
    assert full_diag_underpowered_terms["mixed_planar_fullscale_gate"] == 1.0
    assert full_diag_underpowered_terms[
        "reward_mixed_planar_fullscale_deficit_penalty"] > 0.5
    assert full_diag_good_terms[
        "reward_mixed_planar_fullscale_deficit_penalty"] == 0.0
    assert displacement_reward > stalled + 4.0, (
        displacement_reward, stalled)

    contract = reward_contract()
    assert contract["version"] == "omni_directional_offaxis_yaw_v29"
    assert contract["normalization"]["body_frame_vx_vy_command_tracking"]
    assert contract["normalization"]["off_axis_penalty_tapers_with_planar_command"]
    assert contract["normalization"]["strong_off_axis_suppression"]
    assert contract["normalization"]["signed_yaw_alignment_reward"]
    assert contract["normalization"]["zero_yaw_integrated_drift_penalty"]
    assert contract["normalization"]["zero_yaw_translation_uses_reset_body_axes"]
    assert contract["normalization"]["zero_yaw_heading_hold_weight_boost"]
    assert contract["normalization"]["yaw_only_stationary_speed_penalty"]
    assert contract["normalization"]["yaw_stationary_penalty_clip"] == 6.0
    assert contract["normalization"]["yaw_only_body_compactness_penalty"]
    assert contract["normalization"]["yaw_body_min_extent_ratio"] > 0.0
    assert contract["normalization"]["yaw_body_min_head_tail_ratio"] > 0.0
    assert contract["normalization"]["yaw_body_min_arc_ratio"] > 0.0
    assert contract["normalization"]["pure_lateral_forward_drift_penalty"]
    assert contract["normalization"]["pure_lateral_speed_deficit_penalty"]
    assert contract["normalization"]["pure_lateral_progress_target_m_s"] > 0.0
    assert contract["normalization"]["continuous_omni_repair_oversampling"]
    assert contract["normalization"][
        "continuous_omni_mixed_yaw_repair_sampling"]
    assert contract["normalization"]["mixed_planar_repair_sampling"]
    assert contract["normalization"]["mixed_planar_yaw_preserve_repair_sampling"]
    assert contract["normalization"]["feasible_mixed_low_speed_sampling"][
        "vx_abs_range_m_s"] == FEASIBLE_MIXED_VX_ABS_RANGE
    assert contract["normalization"]["feasible_mixed_low_speed_sampling"][
        "vy_abs_range_m_s"] == FEASIBLE_MIXED_VY_ABS_RANGE
    assert contract["normalization"]["feasible_mixed_low_speed_sampling"][
        "yaw_abs_range_rad_s"] == FEASIBLE_MIXED_YAW_ABS_RANGE
    assert contract["normalization"][
        "feasible_forward_diagonal_repair_sampling"]
    assert "mixed_composition_repair" in COMMAND_CURRICULA
    mixed_counts = sampled_command_classes("mixed_composition_repair")
    assert mixed_counts["mixed_vx_vy"] >= 80, mixed_counts
    assert mixed_counts["mixed_vx_yaw"] >= 80, mixed_counts
    assert mixed_counts["pure"] >= 20, mixed_counts
    balanced_counts = sampled_command_classes(
        "mixed_planar_yaw_preserve_repair")
    assert balanced_counts["mixed_vx_vy"] >= 70, balanced_counts
    assert balanced_counts["mixed_vx_yaw"] >= 45, balanced_counts
    assert balanced_counts["pure"] >= 55, balanced_counts
    assert contract["normalization"]["axis_separation_curriculum"]
    assert contract["normalization"]["yaw_only_prior_scaling_applied"]
    assert contract["normalization"]["gait_blend_is_policy_gate"]
    assert contract["normalization"][
        "command_conditioned_gait_gate_regularizer"]["enabled"]
    assert contract["weights"]["yaw_drift"] >= 6.0
    assert contract["weights"]["yaw_stationary"] >= 3.0
    assert contract["weights"]["yaw_body_compactness"] > 0.0
    assert contract["weights"]["lateral_only_forward_drift"] > 0.0
    assert contract["weights"]["lateral_only_speed_deficit"] > 0.0
    assert contract["weights"]["planar_component_deficit"] > 0.0
    assert contract["weights"]["gait_gate_target"] > 0.0
    assert contract["weights"]["axial_prior_preserve"] > 0.0
    assert contract["weights"]["component_tracking"] > 0.0
    assert contract["weights"]["mixed_planar_component_tracking"] > 0.0
    assert contract["weights"]["mixed_planar_sign"] > 0.0
    assert contract["weights"]["mixed_planar_fullscale_deficit"] > 0.0
    assert contract["normalization"]["signed_planar_component_deficit_penalty"]
    assert contract["normalization"]["pure_axial_residual_cancellation_penalty"]
    assert contract["normalization"][
        "pure_axial_slide_wave_preservation_penalty"]
    assert contract["normalization"]["componentwise_vx_vy_yaw_tracking_cost"]
    assert contract["normalization"]["mixed_planar_component_tracking_cost"]
    assert contract["normalization"]["mixed_planar_sign_penalty"]
    assert contract["normalization"]["mixed_planar_fullscale_deficit_penalty"]
    assert contract["normalization"]["mixed_planar_fullscale_threshold"] == 0.75
    assert contract["normalization"]["component_tracking_cost_scale"] == {
        "vx": CMD_VX_RANGE[1],
        "vy": CMD_VY_RANGE[1],
        "yaw": CMD_YAW_RANGE[1],
    }
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
    assert adapter_contract["version"].endswith("_v36")
    assert adapter_contract["gait_gate_mapping"]["raw_action_index"] == (
        NUM_ACTUATORS)
    assert "|cmd_yaw_norm|" in adapter_contract[
        "command_activity_scale"]["formula"]
    residual_authority = adapter_contract[
        "command_conditioned_residual_authority"]
    assert residual_authority["enabled"]
    assert "mixed vx/vy" in residual_authority["formula"]
    prior_authority = adapter_contract[
        "command_conditioned_prior_authority"]
    assert prior_authority["available"]
    assert not prior_authority["enabled"]
    assert not MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED
    assert "mixed vx/vy" in prior_authority["formula"]
    assert MIXED_PLANAR_REBALANCED_RESIDUAL_SCALE_MULT > (
        MIXED_PLANAR_RESIDUAL_SCALE_MULT)
    assert MIXED_PLANAR_DOMINANT_PRIOR_SCALE_MULT < 1.0
    assert command_conditioned_prior_authority_scale(0.5, 0.5, 0.0) == 1.0
    assert command_conditioned_prior_authority_scale(1.0, 0.0, 0.0) == 1.0
    assert command_conditioned_prior_authority_scale(0.0, 1.0, 0.0) == 1.0
    assert command_conditioned_prior_authority_scale(0.0, 0.0, 1.0) == 1.0
    assert command_conditioned_prior_authority_scale(1.0, 0.0, 1.0) == 1.0
    centers = adapter_contract["gait_gate_mapping"]["command_centers"]
    assert centers["axial_translation"] < centers["mixed_or_stop"]
    assert centers["axial_translation_slow"] < centers["axial_translation_fast"]
    assert centers["axial_translation_fast"] == centers["mixed_or_stop"]
    assert centers["lateral_translation"] < centers["mixed_or_stop"]
    assert centers["lateral_translation"] == 0.0
    assert centers["yaw"] >= centers["mixed_or_stop"]
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
    assert adapter_contract["command_directional_prior_transform"][
        "lateral_primitives"]["enabled"]
    assert adapter_contract["command_directional_prior_transform"][
        "lateral_primitives"]["runtime_scales"]["left"] == 1.0
    assert adapter_contract["command_directional_prior_transform"][
        "lateral_primitives"]["runtime_scales"]["right"] == 0.75
    mixed_composition_contract = adapter_contract[
        "command_directional_prior_transform"][
        "mixed_command_component_composition"]
    assert mixed_composition_contract["available"]
    assert not mixed_composition_contract["enabled"]
    assert mixed_composition_contract["enable_env"] == (
        "WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1")
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
    assert left_tf["phase_offset_rad"] == LATERAL_PHASE_OFFSET_RAD
    assert left_tf["phase_offset_rad"] < 0.0
    assert right_tf["phase_offset_rad"] == left_tf["phase_offset_rad"]
    assert left_tf["uses_lateral_primitive"]
    assert right_tf["uses_lateral_primitive"]
    assert left_tf["yaw_sign"] == 1.0
    assert right_tf["yaw_sign"] == -1.0
    assert 0.0 < left_tf["yaw_scale"] < 1.0
    assert right_tf["yaw_scale"] == left_tf["yaw_scale"]
    assert left_tf["yaw_trim"] > 0.0
    assert right_tf["yaw_trim"] < 0.0
    yaw_left_tf = command_directional_prior_transform(0.0, 0.0, 1.0)
    yaw_right_tf = command_directional_prior_transform(0.0, 0.0, -1.0)
    mixed_yaw_left_tf = command_directional_prior_transform(0.5, 0.0, 1.0)
    mixed_yaw_right_tf = command_directional_prior_transform(0.5, 0.0, -1.0)
    assert yaw_left_tf["uses_inplace_yaw_prior"]
    assert yaw_right_tf["uses_inplace_yaw_prior"]
    assert yaw_left_tf["yaw_sign"] == 1.0
    assert yaw_right_tf["yaw_sign"] == -1.0
    assert not mixed_yaw_left_tf["uses_inplace_yaw_prior"]
    assert not mixed_yaw_right_tf["uses_inplace_yaw_prior"]
    assert mixed_yaw_left_tf["yaw_sign"] == 1.0
    assert mixed_yaw_right_tf["yaw_sign"] == -1.0
    assert yaw_right_tf["yaw_scale"] == yaw_left_tf["yaw_scale"]
    assert yaw_left_tf["slide_scale"] == YAW_ONLY_SLIDE_PRIOR_SCALE
    assert yaw_left_tf["yaw_scale"] == YAW_ONLY_YAW_PRIOR_SCALE
    base_prior = gait_prior_from_phase(0.3, gait_blend=1.0)
    reverse_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=1.0, command=(-1.0, 0.0, 0.0))
    reverse_base = gait_prior_from_phase(-0.3 + math.pi, gait_blend=1.0)
    assert np.allclose(reverse_prior[:6], reverse_base[:6])
    assert np.allclose(
        reverse_prior[6:],
        np.clip(reverse_tf["yaw_scale"] * reverse_base[6:], -1.0, 1.0),
    )
    lateral_left_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=0.0, command=(0.0, 1.0, 0.0))
    lateral_right_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=0.0, command=(0.0, -1.0, 0.0))
    assert np.allclose(
        lateral_left_prior, lateral_primitive_action_from_phase("left", 0.3))
    assert np.allclose(
        lateral_right_prior, lateral_primitive_action_from_phase("right", 0.3))
    assert not np.allclose(lateral_left_prior, lateral_right_prior)
    yaw_left_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=0.5, command=(0.0, 0.0, 1.0))
    yaw_right_prior = directional_gait_prior_from_phase(
        0.3, gait_blend=0.5, command=(0.0, 0.0, -1.0))
    yaw_open_loop = inplace_yaw_prior_from_phase(0.3, 1.0)
    scaled_yaw_open_loop = yaw_open_loop.copy()
    scaled_yaw_open_loop[:6] *= yaw_left_tf["slide_scale"]
    scaled_yaw_open_loop[6:] *= yaw_left_tf["yaw_scale"]
    scaled_yaw_open_loop = np.clip(scaled_yaw_open_loop, -1.0, 1.0)
    assert np.allclose(yaw_left_prior, scaled_yaw_open_loop)
    assert np.allclose(yaw_left_prior[:6], yaw_right_prior[:6])
    assert np.allclose(yaw_left_prior[6:], -yaw_right_prior[6:])
    old_mixed_composition_enabled = (
        action_adapter.MIXED_COMMAND_COMPOSITION_ENABLED)
    action_adapter.MIXED_COMMAND_COMPOSITION_ENABLED = True
    try:
        mixed_planar_prior = action_adapter.directional_gait_prior_from_phase(
            0.3, gait_blend=0.5, command=(0.5, 0.5, 0.0))
        forward_prior = directional_gait_prior_from_phase(
            0.3, gait_blend=0.5, command=(1.0, 0.0, 0.0))
        lateral_left_prior = directional_gait_prior_from_phase(
            0.3, gait_blend=0.5, command=(0.0, 1.0, 0.0))
        assert not np.allclose(mixed_planar_prior, forward_prior)
        assert not np.allclose(mixed_planar_prior, lateral_left_prior)
        assert np.linalg.norm(mixed_planar_prior[:6]) >= (
            0.75 * np.linalg.norm(forward_prior[:6]))
        assert np.linalg.norm(mixed_planar_prior[6:]) >= (
            0.60 * np.linalg.norm(lateral_left_prior[6:]))
        mixed_yaw_prior = action_adapter.directional_gait_prior_from_phase(
            0.3, gait_blend=0.5, command=(0.5, 0.0, 1.0))
        assert np.linalg.norm(mixed_yaw_prior[:6]) >= (
            0.40 * np.linalg.norm(forward_prior[:6]))
        assert np.linalg.norm(mixed_yaw_prior[6:]) >= (
            0.70 * np.linalg.norm(yaw_left_prior[6:]))
        assert np.isclose(
            command_prior_scale_floor(0.5, 0.5, 0.0),
            MIXED_PLANAR_PRIOR_SCALE_FLOOR)
        assert np.isclose(
            command_prior_scale_floor(0.5, 0.0, 1.0),
            MIXED_YAW_PRIOR_SCALE_FLOOR)
    finally:
        action_adapter.MIXED_COMMAND_COMPOSITION_ENABLED = (
            old_mixed_composition_enabled)
    assert command_conditioned_prior_scale(1.0, 0.0, 0.0) == 1.0
    assert command_activity_scale(0.0, 0.0, 0.0) == 0.0
    assert np.isclose(command_activity_scale(0.5, 0.0, 0.0), 0.6)
    assert command_activity_scale(1.0, 0.0, 0.0) == 1.0
    assert np.isclose(command_activity_scale(0.0, 0.0, 0.5), 0.125)
    assert np.isclose(command_activity_scale(0.0, 0.0, 1.0), 0.25)
    assert command_conditioned_residual_scale(1.0, 0.0, 0.0) == 1.0
    assert command_conditioned_residual_scale(0.0, 1.0, 0.0) == 1.0
    assert np.isclose(
        command_conditioned_residual_scale(0.0, 0.0, 1.0),
        YAW_ONLY_RESIDUAL_SCALE_MULT)
    assert np.isclose(
        command_conditioned_residual_scale(0.5, 0.5, 0.0),
        MIXED_PLANAR_RESIDUAL_SCALE_MULT)
    assert np.isclose(
        command_conditioned_residual_scale(0.5, 0.0, 1.0),
        MIXED_YAW_RESIDUAL_SCALE_MULT)
    assert np.isclose(command_prior_scale_floor(-1.0, 0.0, 0.0), 0.4)
    assert np.isclose(
        command_prior_scale_floor(0.0, 1.0, 0.0),
        LATERAL_PRIOR_SCALE_FLOOR)
    assert np.isclose(command_prior_scale_floor(0.0, 0.0, 1.0), 1.0)
    assert np.isclose(command_prior_scale_floor(0.5, 0.5, 0.0), 0.4)
    assert np.isclose(command_prior_scale_floor(0.5, 0.0, 1.0), 0.4)
    assert np.isclose(
        command_conditioned_prior_scale(-1.0, 0.0, 0.0), 0.4)
    assert np.isclose(
        command_conditioned_prior_scale(0.0, 1.0, 0.0),
        LATERAL_PRIOR_SCALE_FLOOR)
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
    assert 0.4 < mixed_prior_scale < 1.0

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
    assert "mixed_planar_repair" in COMMAND_CURRICULA
    assert "mixed_planar_hardcase_repair" in COMMAND_CURRICULA
    assert "mixed_planar_yaw_preserve_repair" in COMMAND_CURRICULA
    assert "low_yaw_envelope" in COMMAND_CURRICULA
    assert "feasible_mixed_low_speed" in COMMAND_CURRICULA
    assert "feasible_forward_diagonal_repair" in COMMAND_CURRICULA
    assert "axis_separation" in COMMAND_CURRICULA

    env.command_curriculum = "low_yaw_envelope"
    samples = np.array([env._sample_command() for _ in range(480)])
    nonzero_dims = np.count_nonzero(np.abs(samples) > 1e-9, axis=1)
    stop_samples = samples[nonzero_dims == 0]
    pure_yaw_samples = samples[
        (np.abs(samples[:, 0]) <= 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    slow_axial_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    forward_yaw_samples = samples[
        (samples[:, 0] > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    assert len(stop_samples) >= 80
    assert len(pure_yaw_samples) >= 120
    assert len(slow_axial_samples) >= 50
    assert len(forward_yaw_samples) >= 20
    assert np.all(np.abs(samples[:, 1]) <= 1e-9)
    assert np.any(pure_yaw_samples[:, 2] > 0.0)
    assert np.any(pure_yaw_samples[:, 2] < 0.0)
    assert np.all(
        np.abs(pure_yaw_samples[:, 2]) >= LOW_YAW_ENVELOPE_YAW_ABS_RANGE[0])
    assert np.all(
        np.abs(pure_yaw_samples[:, 2]) <= LOW_YAW_ENVELOPE_YAW_ABS_RANGE[1])
    assert np.any(slow_axial_samples[:, 0] > 0.0)
    assert np.any(slow_axial_samples[:, 0] < 0.0)
    assert np.all(
        np.abs(slow_axial_samples[:, 0])
        >= LOW_YAW_ENVELOPE_AXIAL_ABS_RANGE[0])
    assert np.all(
        np.abs(slow_axial_samples[:, 0])
        <= LOW_YAW_ENVELOPE_AXIAL_ABS_RANGE[1])

    env.command_curriculum = "feasible_mixed_low_speed"
    samples = np.array([env._sample_command() for _ in range(640)])
    nonzero_dims = np.count_nonzero(np.abs(samples) > 1e-9, axis=1)
    stop_samples = samples[nonzero_dims == 0]
    pure_axial_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    pure_lateral_samples = samples[
        (np.abs(samples[:, 0]) <= 1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    pure_yaw_samples = samples[
        (np.abs(samples[:, 0]) <= 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    diagonal_planar_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    axial_yaw_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    assert len(stop_samples) >= 40
    assert len(pure_axial_samples) >= 40
    assert len(pure_lateral_samples) >= 40
    assert len(pure_yaw_samples) >= 40
    assert len(diagonal_planar_samples) >= 80
    assert len(axial_yaw_samples) >= 80
    assert np.all(np.abs(samples[:, 0]) <= FEASIBLE_MIXED_VX_ABS_RANGE[1])
    assert np.all(np.abs(samples[:, 1]) <= FEASIBLE_MIXED_VY_ABS_RANGE[1])
    assert np.all(np.abs(samples[:, 2]) <= FEASIBLE_MIXED_YAW_ABS_RANGE[1])
    assert np.all(
        np.abs(pure_axial_samples[:, 0])
        >= FEASIBLE_MIXED_VX_ABS_RANGE[0])
    assert np.all(
        np.abs(pure_lateral_samples[:, 1])
        >= FEASIBLE_MIXED_VY_ABS_RANGE[0])
    assert np.all(
        np.abs(pure_yaw_samples[:, 2])
        >= FEASIBLE_MIXED_YAW_ABS_RANGE[0])

    env.command_curriculum = "feasible_forward_diagonal_repair"
    samples = np.array([env._sample_command() for _ in range(640)])
    forward_diagonal_samples = samples[
        (samples[:, 0] > 1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    forward_left_samples = forward_diagonal_samples[
        forward_diagonal_samples[:, 1] > 1e-9]
    forward_right_samples = forward_diagonal_samples[
        forward_diagonal_samples[:, 1] < -1e-9]
    forward_yaw_samples = samples[
        (samples[:, 0] > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    reverse_diagonal_samples = samples[
        (samples[:, 0] < -1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    assert len(forward_diagonal_samples) >= 180
    assert len(forward_left_samples) >= 70
    assert len(forward_right_samples) >= 70
    assert len(forward_yaw_samples) >= 40
    assert len(reverse_diagonal_samples) >= 40
    assert np.all(np.abs(samples[:, 0]) <= FEASIBLE_MIXED_VX_ABS_RANGE[1])
    assert np.all(np.abs(samples[:, 1]) <= FEASIBLE_MIXED_VY_ABS_RANGE[1])
    assert np.all(np.abs(samples[:, 2]) <= FEASIBLE_MIXED_YAW_ABS_RANGE[1])

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

    env.command_curriculum = "mixed_planar_repair"
    samples = np.array([env._sample_command() for _ in range(240)])
    nonzero_dims = np.count_nonzero(np.abs(samples) > 1e-9, axis=1)
    mixed_planar_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    assert np.any(nonzero_dims == 0)
    assert len(mixed_planar_samples) >= 80
    assert np.any((mixed_planar_samples[:, 0] > 0.0)
                  & (mixed_planar_samples[:, 1] > 0.0))
    assert np.any((mixed_planar_samples[:, 0] > 0.0)
                  & (mixed_planar_samples[:, 1] < 0.0))
    assert np.any((mixed_planar_samples[:, 0] < 0.0)
                  & (mixed_planar_samples[:, 1] > 0.0))
    assert np.any((mixed_planar_samples[:, 0] < 0.0)
                  & (mixed_planar_samples[:, 1] < 0.0))

    env.command_curriculum = "mixed_planar_hardcase_repair"
    samples = np.array([env._sample_command() for _ in range(240)])
    hard_mixed_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    reverse_diagonal_samples = hard_mixed_samples[
        hard_mixed_samples[:, 0] < 0.0
    ]
    fullscale_hard_samples = hard_mixed_samples[
        (np.abs(hard_mixed_samples[:, 0]) >= 0.75 * CMD_VX_RANGE[1])
        & (np.abs(hard_mixed_samples[:, 1]) >= 0.75 * CMD_VY_RANGE[1])
    ]
    assert len(hard_mixed_samples) >= 120
    assert len(reverse_diagonal_samples) >= 70
    assert len(fullscale_hard_samples) >= 80
    assert np.any((hard_mixed_samples[:, 0] > 0.0)
                  & (hard_mixed_samples[:, 1] > 0.0))
    assert np.any((hard_mixed_samples[:, 0] > 0.0)
                  & (hard_mixed_samples[:, 1] < 0.0))
    assert np.any((hard_mixed_samples[:, 0] < 0.0)
                  & (hard_mixed_samples[:, 1] > 0.0))
    assert np.any((hard_mixed_samples[:, 0] < 0.0)
                  & (hard_mixed_samples[:, 1] < 0.0))

    env.command_curriculum = "mixed_planar_yaw_preserve_repair"
    samples = np.array([env._sample_command() for _ in range(240)])
    yaw_samples = samples[np.abs(samples[:, 2]) > 1e-9]
    mixed_planar_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) > 1e-9)
        & (np.abs(samples[:, 2]) <= 1e-9)
    ]
    mixed_yaw_samples = samples[
        (np.abs(samples[:, 0]) > 1e-9)
        & (np.abs(samples[:, 1]) <= 1e-9)
        & (np.abs(samples[:, 2]) > 1e-9)
    ]
    assert len(yaw_samples) >= 60
    assert len(mixed_planar_samples) >= 70
    assert len(mixed_yaw_samples) >= 45
    assert np.any(yaw_samples[:, 2] > 0.0)
    assert np.any(yaw_samples[:, 2] < 0.0)

    env.command_curriculum = "axis_separation"
    samples = np.array([env._sample_command() for _ in range(240)])
    nonzero_dims = np.count_nonzero(np.abs(samples) > 1e-9, axis=1)
    assert np.all(nonzero_dims <= 1)
    assert np.any(nonzero_dims == 0)
    assert np.any(samples[:, 0] > 0.0)
    assert np.any(samples[:, 0] < 0.0)
    assert np.any(samples[:, 1] > 0.0)
    assert np.any(samples[:, 1] < 0.0)
    assert np.any(samples[:, 2] > 0.0)
    assert np.any(samples[:, 2] < 0.0)

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
        "mixed_full_forward_left",
        "mixed_full_forward_right",
        "mixed_full_reverse_left",
        "mixed_full_reverse_right",
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
    expected_yaw_set = {float(CMD_YAW_RANGE[0]), 0.0, float(CMD_YAW_RANGE[1])}
    assert {c["cmd_yaw_rad_s"] for c in random_schedule} >= expected_yaw_set
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

    stale_eval_command = config_stub(contract)
    stale_eval_command["eval_command"] = {
        "cmd_vx_m_s": 0.25,
        "cmd_vy_m_s": 0.0,
        "cmd_yaw_rad_s": 0.0,
        "command_resample_prob": 0.0,
    }
    ok, reasons = training_config_compatible(
        stale_eval_command, config_stub(contract))
    assert not ok
    assert "eval_command" in reasons
    ok, reasons = training_config_compatible(
        stale_eval_command,
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

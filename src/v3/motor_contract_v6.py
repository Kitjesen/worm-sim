"""
Actuator and motor-limit contract for deployable Worm V6 policies.

This is the single source of truth for quantities that must match between
MuJoCo, policy export, hardware replay, and preflight checks. The values below
are still a MuJoCo position-servo abstraction, not a vendor motor datasheet.
"""

import hashlib
import json

import numpy as np


NUM_SLIDES = 6
NUM_YAWS = 5
NUM_ACTUATORS = NUM_SLIDES + NUM_YAWS

SLIDE_JOINT_NAMES = tuple(f"back{i}" for i in range(1, NUM_SLIDES + 1))
YAW_JOINT_NAMES = tuple(f"front{i}" for i in range(2, NUM_YAWS + 2))

# The prismatic slide joints in the URDF/MuJoCo model compress from 0 to -50 mm.
SLIDE_TARGET_MIN_M = -0.05
SLIDE_TARGET_MAX_M = 0.0
SLIDE_TARGET_SCALE_M = 0.05

# Yaw joints are symmetric revolute servos.
YAW_TARGET_MIN_RAD = -1.57
YAW_TARGET_MAX_RAD = 1.57
YAW_TARGET_SCALE_RAD = 1.57

# MuJoCo position-actuator abstraction.
SLIDE_POSITION_KP_N_PER_M = 800.0
SLIDE_FORCE_LIMIT_N = 50.0
YAW_POSITION_KP_NM_PER_RAD = 200.0
YAW_TORQUE_LIMIT_NM = 20.0

# Joint passive parameters used in the generated MJCF.
SLIDE_JOINT_DAMPING = 10.0
SLIDE_JOINT_STIFFNESS = 300.0
YAW_JOINT_DAMPING = 5.0

ACTION_MIN = -1.0
ACTION_MAX = 1.0


def _stable_hash(payload):
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def action_mapping_config():
    return {
        "slide_indices": [0, NUM_SLIDES],
        "yaw_indices": [NUM_SLIDES, NUM_ACTUATORS],
        "slide_target_m": (
            "clip(action[i] * slide_range_m, "
            "slide_min_m, slide_max_m)"),
        "yaw_target_rad": (
            "clip(action[i] * yaw_range_rad, "
            "yaw_min_rad, yaw_max_rad)"),
        "slide_range_m": SLIDE_TARGET_SCALE_M,
        "slide_min_m": SLIDE_TARGET_MIN_M,
        "slide_max_m": SLIDE_TARGET_MAX_M,
        "yaw_range_rad": YAW_TARGET_SCALE_RAD,
        "yaw_min_rad": YAW_TARGET_MIN_RAD,
        "yaw_max_rad": YAW_TARGET_MAX_RAD,
    }


def normalized_action_to_targets(action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != NUM_ACTUATORS:
        raise ValueError(
            f"action last dimension {action.shape[-1]} != {NUM_ACTUATORS}")
    clipped = np.clip(action, ACTION_MIN, ACTION_MAX)
    slide_targets = np.clip(
        clipped[..., :NUM_SLIDES] * SLIDE_TARGET_SCALE_M,
        SLIDE_TARGET_MIN_M,
        SLIDE_TARGET_MAX_M,
    )
    yaw_targets = np.clip(
        clipped[..., NUM_SLIDES:] * YAW_TARGET_SCALE_RAD,
        YAW_TARGET_MIN_RAD,
        YAW_TARGET_MAX_RAD,
    )
    return slide_targets.astype(np.float32), yaw_targets.astype(np.float32)


def normalized_action_to_ctrl(action):
    slide_targets, yaw_targets = normalized_action_to_targets(action)
    return np.concatenate([slide_targets, yaw_targets], axis=-1).astype(
        np.float32)


def neutral_normalized_action():
    return np.zeros(NUM_ACTUATORS, dtype=np.float32)


def motor_contract():
    payload = {
        "format_version": 1,
        "name": "worm_v6_actuator_motor_contract",
        "actuator_count": NUM_ACTUATORS,
        "normalized_action_range": [ACTION_MIN, ACTION_MAX],
        "action_mapping": action_mapping_config(),
        "actuator_groups": {
            "slide": {
                "indices": [0, NUM_SLIDES],
                "joint_names": list(SLIDE_JOINT_NAMES),
                "joint_type": "prismatic_position_servo",
                "target_unit": "m",
                "mechanical_range_m": [
                    SLIDE_TARGET_MIN_M,
                    SLIDE_TARGET_MAX_M,
                ],
                "normalized_command_scale_m": SLIDE_TARGET_SCALE_M,
                "mujoco_position_kp_n_per_m": SLIDE_POSITION_KP_N_PER_M,
                "mujoco_force_limit_n": SLIDE_FORCE_LIMIT_N,
                "joint_damping": SLIDE_JOINT_DAMPING,
                "joint_stiffness": SLIDE_JOINT_STIFFNESS,
                "hardware_rule": (
                    "Real controller must clip slide targets to "
                    "[-0.05, 0.0] m before sending commands."),
            },
            "yaw": {
                "indices": [NUM_SLIDES, NUM_ACTUATORS],
                "joint_names": list(YAW_JOINT_NAMES),
                "joint_type": "revolute_position_servo",
                "target_unit": "rad",
                "mechanical_range_rad": [
                    YAW_TARGET_MIN_RAD,
                    YAW_TARGET_MAX_RAD,
                ],
                "normalized_command_scale_rad": YAW_TARGET_SCALE_RAD,
                "mujoco_position_kp_nm_per_rad": (
                    YAW_POSITION_KP_NM_PER_RAD),
                "mujoco_torque_limit_nm": YAW_TORQUE_LIMIT_NM,
                "joint_damping": YAW_JOINT_DAMPING,
                "hardware_rule": (
                    "Real controller must clip yaw targets to "
                    "[-1.57, 1.57] rad before sending commands."),
            },
        },
        "not_yet_modeled_in_sim": [
            "motor speed limit",
            "current and voltage limits",
            "thermal derating",
            "gear backlash",
            "deadband",
            "vendor controller internal PID",
        ],
    }
    payload["contract_fingerprint"] = _stable_hash(payload)
    return payload


def motor_contract_fingerprint():
    return motor_contract()["contract_fingerprint"]


def action_mapping_matches(mapping, tol=1e-9):
    expected = action_mapping_config()
    if not isinstance(mapping, dict):
        return False
    for key, expected_value in expected.items():
        actual = mapping.get(key)
        if isinstance(expected_value, float):
            try:
                if abs(float(actual) - expected_value) > tol:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected_value:
            return False
    return True

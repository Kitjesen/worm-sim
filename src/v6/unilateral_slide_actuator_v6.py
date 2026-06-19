"""
Reduced unilateral slide actuator model for Worm V6.

The real slide mechanism contracts by servo-rope pulling and releases through
the passive spring-steel return. This module keeps the model independent from
MuJoCo or Isaac Lab so both simulators can share the same force convention.

Coordinate convention:
    q = slide joint position in meters, where q=0 is relaxed and q<0 is
        contracted.
    c = -q is compression in meters.
    positive generalized force increases q and therefore extends the segment.
"""

import hashlib
import json

import numpy as np

from motor_contract_v6 import (
    NUM_SLIDES,
    SLIDE_FORCE_LIMIT_N,
    SLIDE_JOINT_DAMPING,
    SLIDE_JOINT_STIFFNESS,
    SLIDE_TARGET_SCALE_M,
)


SLIDE_PULL_KP_N_PER_M = 1200.0
SLIDE_PULL_KD_N_PER_MPS = 40.0
ROPE_SLACK_DEADBAND_M = 0.001


def normalized_slide_action_to_compression(action):
    """Map normalized slide action to desired compression.

    The existing V6 normalized convention uses negative values for
    contraction. Values above zero mean release and map to zero desired
    compression.
    """
    action = np.asarray(action, dtype=np.float32)
    return np.clip(-action, 0.0, 1.0) * SLIDE_TARGET_SCALE_M


def compute_unilateral_slide_forces(
    normalized_slide_action,
    slide_qpos_m,
    slide_qvel_mps,
    spring_k_n_per_m=SLIDE_JOINT_STIFFNESS,
    spring_d_n_per_mps=SLIDE_JOINT_DAMPING,
    pull_kp_n_per_m=SLIDE_PULL_KP_N_PER_M,
    pull_kd_n_per_mps=SLIDE_PULL_KD_N_PER_MPS,
    max_tension_n=SLIDE_FORCE_LIMIT_N,
    slack_deadband_m=ROPE_SLACK_DEADBAND_M,
):
    """Compute generalized slide forces for unilateral rope pulling.

    Returns:
        force_q: generalized force in the slide coordinate. Positive extends,
            negative contracts.
        tension: rope tension, always non-negative.
        desired_compression: command after clipping to the one-sided stroke.
    """
    q = np.asarray(slide_qpos_m, dtype=np.float32)
    qd = np.asarray(slide_qvel_mps, dtype=np.float32)
    desired_c = normalized_slide_action_to_compression(
        normalized_slide_action)
    current_c = np.clip(-q, 0.0, SLIDE_TARGET_SCALE_M)
    current_cd = -qd

    pull_error = desired_c - current_c - float(slack_deadband_m)
    raw_tension = (
        float(pull_kp_n_per_m) * pull_error
        - float(pull_kd_n_per_mps) * current_cd)
    tension = np.clip(raw_tension, 0.0, float(max_tension_n))

    spring_force_q = (
        float(spring_k_n_per_m) * current_c
        + float(spring_d_n_per_mps) * current_cd)
    force_q = spring_force_q - tension
    return (
        np.asarray(force_q, dtype=np.float32),
        np.asarray(tension, dtype=np.float32),
        np.asarray(desired_c, dtype=np.float32),
    )


def unilateral_slide_model_contract():
    payload = {
        "format_version": 1,
        "name": "worm_v6_unilateral_slide_actuator_model",
        "slide_count": NUM_SLIDES,
        "coordinate": {
            "q_m": "slide joint position; q=0 relaxed, q<0 contracted",
            "compression_m": "c=-q, clipped to [0, stroke]",
            "positive_force_q": "extension/passive return",
            "negative_force_q": "contraction by rope tension",
        },
        "normalized_action": (
            "desired_compression_m = clip(-action, 0, 1) * stroke_m"),
        "stroke_m": SLIDE_TARGET_SCALE_M,
        "passive_return": {
            "spring_k_n_per_m": SLIDE_JOINT_STIFFNESS,
            "damping_n_per_mps": SLIDE_JOINT_DAMPING,
        },
        "active_pull": {
            "pull_kp_n_per_m": SLIDE_PULL_KP_N_PER_M,
            "pull_kd_n_per_mps": SLIDE_PULL_KD_N_PER_MPS,
            "max_tension_n": SLIDE_FORCE_LIMIT_N,
            "slack_deadband_m": ROPE_SLACK_DEADBAND_M,
            "can_push_extension": False,
        },
        "force_law": (
            "Q_q = k_s*c + d_s*c_dot - clamp("
            "k_p*(c_des-c-deadband)-k_d*c_dot, 0, T_max)"),
    }
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["contract_fingerprint"] = hashlib.sha256(
        text.encode("utf-8")).hexdigest()
    return payload


def neutral_slide_forces():
    action = np.zeros(NUM_SLIDES, dtype=np.float32)
    q = np.zeros(NUM_SLIDES, dtype=np.float32)
    qd = np.zeros(NUM_SLIDES, dtype=np.float32)
    return compute_unilateral_slide_forces(action, q, qd)

"""
Deployable action adapter for Worm V6 residual policies.

The PPO actor outputs a residual action. A deterministic gait prior generated
from the deployable phase clock and gait_blend is added before sending final
normalized joint targets to MuJoCo or hardware.
"""

import math

import numpy as np

from motor_contract_v6 import (
    NUM_ACTUATORS,
    NUM_SLIDES,
    NUM_YAWS,
    YAW_TARGET_SCALE_RAD,
)
from worm_v6 import SNAKE_AMP


ACTION_ADAPTER_VERSION = "gait_prior_residual_v1"
DEFAULT_GAIT_PRIOR_SCALE = 1.0
DEFAULT_POLICY_RESIDUAL_SCALE = 0.35
SNAKE_AMP_NORMALIZED = SNAKE_AMP / YAW_TARGET_SCALE_RAD


def action_adapter_contract(
        gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
        policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
    return {
        "version": ACTION_ADAPTER_VERSION,
        "policy_output": "normalized_residual_action",
        "deployed_action": "clip(gait_prior + residual, -1, 1)",
        "gait_prior_scale": float(gait_prior_scale),
        "policy_residual_scale": float(policy_residual_scale),
        "phase_source": "deployable phase_clock observation",
        "gait_blend_source": "command observation",
        "slide_prior": (
            "-0.5 * (1 + sin(phase + 2*pi*j/6)) * (1 - gait_blend)"),
        "yaw_prior": (
            "(snake_amp_rad/yaw_scale_rad) * "
            "sin(phase + 2*pi*1.5*j/5) * gait_blend"),
        "snake_amp_rad": float(SNAKE_AMP),
        "snake_amp_normalized": float(SNAKE_AMP_NORMALIZED),
    }


def gait_prior_from_phase(phase, gait_blend):
    phase = float(phase)
    gait_blend = float(np.clip(gait_blend, 0.0, 1.0))
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    for j in range(NUM_SLIDES):
        p = phase + 2.0 * math.pi * j / max(NUM_SLIDES, 1)
        action[j] = -0.5 * (1.0 + math.sin(p)) * (1.0 - gait_blend)
    for j in range(NUM_YAWS):
        p = phase + 2.0 * math.pi * 1.5 * j / max(NUM_YAWS, 1)
        action[NUM_SLIDES + j] = (
            SNAKE_AMP_NORMALIZED * math.sin(p) * gait_blend)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def phase_from_clock(phase_sin, phase_cos):
    return math.atan2(float(phase_sin), float(phase_cos))


def compose_deployable_action(
        residual_action,
        phase,
        gait_blend,
        gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
        policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
    residual = np.clip(
        np.asarray(residual_action, dtype=np.float32),
        -1.0,
        1.0,
    )
    if residual.shape != (NUM_ACTUATORS,):
        raise ValueError(
            f"residual action shape {residual.shape} != {(NUM_ACTUATORS,)}")
    prior = gait_prior_from_phase(phase, gait_blend)
    action = (
        float(gait_prior_scale) * prior
        + float(policy_residual_scale) * residual)
    return np.clip(action, -1.0, 1.0).astype(np.float32)

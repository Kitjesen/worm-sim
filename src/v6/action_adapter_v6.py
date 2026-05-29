"""
Deployable action adapter for Worm V6 residual policies.

The PPO actor outputs an 11-D residual action plus a 1-D gait gate. A
deterministic gait prior generated from the deployable phase clock and the
learned gait gate is added before sending final normalized joint targets to
MuJoCo or hardware.

The prior uses the CMA-ES anchor gaits that generated the strong comparison
video, projected onto the deployable 1 s phase clock so it remains a function
of real controller state instead of hidden simulator time.
"""

import math

import numpy as np

from motor_contract_v6 import (
    NUM_ACTUATORS,
    NUM_SLIDES,
    NUM_YAWS,
    SLIDE_TARGET_SCALE_M,
    YAW_TARGET_SCALE_RAD,
)


ACTION_ADAPTER_VERSION = "cmaes_tri_anchor_auto_gate_v2"
POLICY_ACTION_DIM = NUM_ACTUATORS + 1
DEFAULT_GAIT_PRIOR_SCALE = 1.0
DEFAULT_POLICY_RESIDUAL_SCALE = 0.35
TWO_PI = 2.0 * math.pi

CMAES_PARAM_NAMES = (
    "slide_amp",
    "slide_freq",
    "slide_wave_n",
    "yaw_amp",
    "yaw_freq",
    "yaw_wave_n",
    "slide_bias_0",
    "slide_bias_1",
    "slide_bias_2",
    "slide_bias_3",
    "slide_bias_4",
    "slide_bias_5",
    "step_duration",
    "yaw_slide_coupling",
)

CMAES_ANCHORS = {
    "peristaltic": {
        "source": "cmaes_peristaltic",
        "best_speed_mm_s": 64.90,
        "params": (
            0.0499967771750052,
            0.99995922729403,
            0.5053235731460374,
            0.004576804060495781,
            0.4269881104325972,
            2.989422198070506,
            2.471448492284372,
            1.8269679928597355,
            2.4449428584901423,
            2.856862382888375,
            -2.9748880169626144,
            3.108692589704268,
            0.7476788282885293,
            0.23400636325858382,
        ),
    },
    "full": {
        "source": "cmaes_full",
        "best_speed_mm_s": 247.97,
        "params": (
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
        ),
    },
    "serpentine": {
        "source": "cmaes_serpentine",
        "best_speed_mm_s": 90.38,
        "params": (
            0.03558827382644502,
            0.4244779404305604,
            2.702074806396634,
            0.7150887433302788,
            0.9861203877623932,
            0.6086064803771956,
            -0.8997403641893227,
            -2.0634603041134065,
            -1.8685883687077762,
            0.13412045251706095,
            -1.2500211112855877,
            0.08771217797686415,
            0.6234570692098784,
            -0.4580381766374815,
        ),
    },
}


def _anchor_contract(name):
    anchor = CMAES_ANCHORS[name]
    return {
        "source": anchor["source"],
        "best_speed_mm_s": float(anchor["best_speed_mm_s"]),
        "param_names": list(CMAES_PARAM_NAMES),
        "params": [float(v) for v in anchor["params"]],
    }


def action_adapter_contract(
        gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
        policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
    return {
        "version": ACTION_ADAPTER_VERSION,
        "policy_output": "normalized_residual_action_11d_plus_gait_gate_1d",
        "deployed_action": "clip(gait_prior + residual, -1, 1)",
        "policy_action_dim": POLICY_ACTION_DIM,
        "deployed_action_dim": NUM_ACTUATORS,
        "gait_prior_scale": float(gait_prior_scale),
        "policy_residual_scale": float(policy_residual_scale),
        "phase_source": "deployable phase_clock observation",
        "gait_blend_source": "policy action gate",
        "phase_projection": (
            "phase_cycle_s = (phase mod 2*pi) / (2*pi); "
            "CMA-ES frequencies are evaluated on this deployable 1 s clock"),
        "blend_rule": (
            "0.0=peristaltic anchor, 0.5=full combined anchor, "
            "1.0=serpentine anchor; intermediate values are piecewise-linear"),
        "gait_anchors": {
            "worm": _anchor_contract("peristaltic"),
            "mixed": _anchor_contract("full"),
            "snake": _anchor_contract("serpentine"),
        },
    }


def _phase_cycle_seconds(phase):
    return (float(phase) % TWO_PI) / TWO_PI


def cmaes_anchor_action(anchor_name, phase):
    if anchor_name not in CMAES_ANCHORS:
        raise ValueError(
            f"Unknown CMA-ES anchor {anchor_name!r}; expected one of "
            f"{tuple(CMAES_ANCHORS)}")
    p = CMAES_ANCHORS[anchor_name]["params"]
    t = _phase_cycle_seconds(phase)
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)

    slide_amp = float(p[0])
    yaw_amp = float(p[3])
    if anchor_name == "serpentine":
        slide_amp = 0.0
    elif anchor_name == "peristaltic":
        yaw_amp = 0.0

    for j in range(NUM_SLIDES):
        joint_phase = (
            TWO_PI * (t * float(p[1]) - float(p[2]) * j / NUM_SLIDES)
            + float(p[6 + j])
        )
        target_m = -slide_amp * (1.0 + math.sin(joint_phase))
        action[j] = target_m / SLIDE_TARGET_SCALE_M

    for j in range(NUM_YAWS):
        joint_phase = (
            TWO_PI * float(p[4]) * t
            + TWO_PI * float(p[5]) * j / NUM_YAWS
            + float(p[13]) * TWO_PI * t * float(p[1])
        )
        target_rad = yaw_amp * math.sin(joint_phase)
        action[NUM_SLIDES + j] = target_rad / YAW_TARGET_SCALE_RAD

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def gait_prior_from_phase(phase, gait_blend):
    gait_blend = float(np.clip(gait_blend, 0.0, 1.0))
    worm = cmaes_anchor_action("peristaltic", phase)
    full = cmaes_anchor_action("full", phase)
    snake = cmaes_anchor_action("serpentine", phase)
    if gait_blend <= 0.5:
        alpha = gait_blend / 0.5
        action = (1.0 - alpha) * worm + alpha * full
    else:
        alpha = (gait_blend - 0.5) / 0.5
        action = (1.0 - alpha) * full + alpha * snake
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def phase_from_clock(phase_sin, phase_cos):
    return math.atan2(float(phase_sin), float(phase_cos))


def policy_action_to_residual_and_gait_blend(policy_action):
    action = np.clip(
        np.asarray(policy_action, dtype=np.float32),
        -1.0,
        1.0,
    )
    if action.shape != (POLICY_ACTION_DIM,):
        raise ValueError(
            f"policy action shape {action.shape} != {(POLICY_ACTION_DIM,)}")
    residual = action[:NUM_ACTUATORS]
    gait_blend = float(np.clip(0.5 * (action[-1] + 1.0), 0.0, 1.0))
    return residual, gait_blend


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

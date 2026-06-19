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
import os

import numpy as np

from motor_contract_v6 import (
    NUM_ACTUATORS,
    NUM_SLIDES,
    NUM_YAWS,
    SLIDE_TARGET_SCALE_M,
    YAW_TARGET_SCALE_RAD,
)


ACTION_ADAPTER_VERSION = "cmaes_tri_anchor_auto_gate_directional_v40"
USE_CONTINUOUS_VECTOR_PRIOR_BLEND = False


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


MIXED_COMMAND_COMPOSITION_EXPERIMENTAL_AVAILABLE = True
MIXED_COMMAND_COMPOSITION_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION", default=False)
MIXED_PLANAR_CONTINUOUS_GATE_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_CONTINUOUS_GATE_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE", default=False)
MIXED_PLANAR_HARDCASE_GATE_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_HARDCASE_GATE_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE", default=False)
POLICY_ACTION_DIM = NUM_ACTUATORS + 1
GAIT_GATE_ACTION_GAIN = 3.0
COMMAND_GATE_CENTER_RESIDUAL_RANGE = 0.35
COMMAND_GATE_WORM_CENTER_SLOW = 0.35
COMMAND_GATE_WORM_CENTER_FAST = 0.50
COMMAND_GATE_WORM_FAST_THRESHOLD = 0.80
COMMAND_GATE_WORM_CENTER = COMMAND_GATE_WORM_CENTER_SLOW
COMMAND_GATE_MIXED_CENTER = 0.50
COMMAND_GATE_LATERAL_CENTER = 0.00
COMMAND_GATE_YAW_CENTER = 0.85
MIXED_PLANAR_GATE_AXIAL_SHARE_FLOOR = 0.25
MIXED_PLANAR_HARDCASE_GATE_CENTER = 0.28
MIXED_PLANAR_HARDCASE_MIN_VX_NORM = 0.40
MIXED_PLANAR_HARDCASE_MAX_VX_NORM = 0.70
MIXED_PLANAR_HARDCASE_MIN_VY_NORM = 0.85
DEFAULT_GAIT_PRIOR_SCALE = 1.0
DEFAULT_POLICY_RESIDUAL_SCALE = 0.35
COMMAND_CONDITIONED_RESIDUAL_AUTHORITY_ENABLED = True
MIXED_PLANAR_AUTHORITY_REBALANCE_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE", default=False)
MIXED_PLANAR_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_SPLIT_PRIOR_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR", default=False)
MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR", default=False)
SLOPE_MIXED_PLANAR_PRIMITIVE_EXPERIMENTAL_AVAILABLE = True
SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED = _env_flag(
    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE", default=False)
MIXED_PLANAR_RESIDUAL_SCALE_MULT = 1.80
MIXED_PLANAR_REBALANCED_RESIDUAL_SCALE_MULT = 2.50
MIXED_YAW_RESIDUAL_SCALE_MULT = 1.60
YAW_ONLY_RESIDUAL_SCALE_MULT = 1.25
MIXED_PLANAR_DOMINANT_PRIOR_SCALE_MULT = 0.65
MIXED_PLANAR_AUTHORITY_PROFILE_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT = _env_float(
    "WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT",
    MIXED_PLANAR_DOMINANT_PRIOR_SCALE_MULT)
MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT = _env_float(
    "WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT",
    MIXED_PLANAR_REBALANCED_RESIDUAL_SCALE_MULT)
SLOPE_FORWARD_AXIS_PROFILE_EXPERIMENTAL_AVAILABLE = True
SLOPE_FORWARD_AXIS_PROFILE_ENABLED = _env_flag(
    "WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE", default=False)
SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT = _env_float(
    "WORM_V6_SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT", 1.50)
SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR = _env_float(
    "WORM_V6_SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR", 0.50)
SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD = _env_float(
    "WORM_V6_SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD", math.pi)
REVERSE_PRIOR_SCALE_FLOOR = 0.40
LATERAL_PRIOR_SCALE_FLOOR = 1.00
YAW_ONLY_PRIOR_SCALE_FLOOR = 1.00
YAW_ONLY_SLIDE_PRIOR_SCALE = 0.80
YAW_ONLY_YAW_PRIOR_SCALE = 5.00
YAW_RIGHT_ONLY_YAW_PRIOR_SCALE = 5.00
MIXED_PLANAR_PRIOR_SCALE_FLOOR = 1.00
MIXED_YAW_PRIOR_SCALE_FLOOR = 0.85
MIXED_AXIAL_SLIDE_GAIN = 1.00
MIXED_LATERAL_SLIDE_GAIN = 0.60
MIXED_YAW_SLIDE_GAIN = 0.00
MIXED_AXIAL_YAW_GAIN = 0.45
MIXED_LATERAL_YAW_GAIN = 1.50
MIXED_YAW_YAW_GAIN = 1.00
MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM = True
SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN", 1.00)
SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN", 0.85)
SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN", 1.35)
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_EXPERIMENTAL_AVAILABLE = True
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED = _env_flag(
    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW",
    default=False)
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN", 0.45)
SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT", 1.00)
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_EXPERIMENTAL_AVAILABLE = True
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED = _env_flag(
    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE",
    default=False)
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD",
    SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD)
SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN = _env_float(
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN", 1.00)
LATERAL_PHASE_OFFSET_RAD = -0.5 * math.pi
LATERAL_LEFT_PRIMITIVE_SCALE = 1.00
LATERAL_RIGHT_PRIMITIVE_SCALE = 0.75
SLOW_RIGHT_LATERAL_PRIOR_NORM_MAX = 0.65
ZERO_YAW_FORWARD_PHASE_OFFSET_RAD = math.pi
ZERO_YAW_REVERSE_PHASE_OFFSET_RAD = math.pi
ZERO_YAW_FORWARD_YAW_PRIOR_SCALE = 1.15
ZERO_YAW_REVERSE_YAW_PRIOR_SCALE = 1.25
ZERO_YAW_LATERAL_YAW_PRIOR_SCALE = 0.65
ZERO_YAW_LATERAL_YAW_TRIM = 0.24
COMMAND_ACTIVITY_MIN_ACTIVE_SCALE = 0.20
YAW_ONLY_ACTIVITY_MIN_ACTIVE_SCALE = 0.00
YAW_ONLY_ACTIVITY_GAIN = 0.25
NON_FORWARD_PRIOR_SCALE_FLOOR = REVERSE_PRIOR_SCALE_FLOOR
DIRECTIONAL_PRIOR_THRESHOLD = 0.20
TWO_PI = 2.0 * math.pi
INPLACE_YAW_SLIDE_BIAS = 0.573
INPLACE_YAW_SLIDE_AMP = 0.410
INPLACE_YAW_SLIDE_FREQ = 1.188
INPLACE_YAW_SLIDE_WAVE_N = 1.259
INPLACE_YAW_YAW_AMP = 0.612
INPLACE_YAW_YAW_FREQ = 0.352
INPLACE_YAW_YAW_WAVE_N = -0.091
INPLACE_YAW_YAW_PHASE_RAD = 0.623
INPLACE_YAW_PRIOR_PARAM_NAMES = (
    "slide_bias",
    "slide_amp",
    "slide_freq",
    "slide_wave_n",
    "yaw_amp",
    "yaw_freq",
    "yaw_wave_n",
    "yaw_phase_rad",
)
INPLACE_YAW_PRIOR_PARAMS = (
    INPLACE_YAW_SLIDE_BIAS,
    INPLACE_YAW_SLIDE_AMP,
    INPLACE_YAW_SLIDE_FREQ,
    INPLACE_YAW_SLIDE_WAVE_N,
    INPLACE_YAW_YAW_AMP,
    INPLACE_YAW_YAW_FREQ,
    INPLACE_YAW_YAW_WAVE_N,
    INPLACE_YAW_YAW_PHASE_RAD,
)

LATERAL_PRIMITIVE_PARAM_NAMES = (
    "slide_bias",
    "slide_amp",
    "slide_freq",
    "slide_wave_n",
    "slide_phase",
    "yaw_amp",
    "yaw_freq",
    "yaw_wave_n",
    "yaw_phase",
    "yaw_bias",
    "yaw_trim_gradient",
)
LATERAL_PRIMITIVE_ANCHORS = {
    "left": {
        "source": "flat_omni_v41_lateral_prior_search_left6s",
        "validation": {
            "signed_lateral_m_s": 0.07789418867807911,
            "body_vx_m_s": -0.013611432358963315,
            "yaw_rate_rad_s": 0.12322999117731077,
            "seconds": 6.0,
            "accepted": True,
        },
        "params": (
            -0.16168439069782814,
            0.22441459381006515,
            1.081554091014169,
            0.5406491247057135,
            -2.5410827041471515,
            0.2148520770425268,
            0.8036597351620074,
            -0.8284909199581207,
            2.041947989004318,
            0.12645026605709297,
            0.21854809007045828,
        ),
    },
    "right": {
        "source": "flat_omni_v41_lateral_prior_search_right6s",
        "validation": {
            "signed_lateral_m_s": 0.12631675565734124,
            "body_vx_m_s": 0.010188427029385095,
            "yaw_rate_rad_s": -0.11304522310354237,
            "seconds": 6.0,
            "accepted": True,
        },
        "params": (
            -0.5054361187954457,
            0.0004378962640689175,
            1.22782626619309,
            2.2511784615350194,
            -1.7471121550631785,
            0.6417442419638648,
            1.0357562708917787,
            -0.40805273913296425,
            -0.688387111640751,
            0.008020713598054452,
            -0.3785763291478317,
        ),
    },
    "right_slow": {
        "source": "flat_omni_v71_slow_right_prior_search_multiseed",
        "validation": {
            "target_m_s": 0.0375,
            "activity_scale": 0.60,
            "mean_signed_lateral_m_s": 0.03710144785075189,
            "min_signed_lateral_m_s": 0.027842330059581446,
            "mean_abs_body_vx_m_s": 0.004838955215654884,
            "mean_abs_yaw_rate_rad_s": 0.10735149045257755,
            "validation_seed_count": 10,
            "accepted_seed_count": 8,
        },
        "params": (
            -0.8887450885461599,
            0.04387732204684349,
            1.166506059967546,
            1.7177987000592116,
            -0.3871971664770437,
            0.5051193932806227,
            0.810264361188886,
            -2.974397331956491,
            0.0896514070847898,
            0.3636877072067817,
            -0.246068835251783,
        ),
    },
}

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
        "command_conditioned_residual_authority": {
            "enabled": COMMAND_CONDITIONED_RESIDUAL_AUTHORITY_ENABLED,
            "source": "deployable normalized command obs[0:3]",
            "formula": (
                "residual_scale = policy_residual_scale * m(cmd); "
                "m=1 for pure vx/vy, "
                f"m={YAW_ONLY_RESIDUAL_SCALE_MULT:.2f} for pure yaw, "
                f"m={MIXED_PLANAR_RESIDUAL_SCALE_MULT:.2f} for mixed vx/vy, "
                f"m={MIXED_YAW_RESIDUAL_SCALE_MULT:.2f} for mixed yaw; "
                f"optional V54 ablation uses "
                f"m={MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT:.2f} "
                "for mixed vx/vy"),
            "reason": (
                "V50 strict scans showed hand-composed mixed priors degraded "
                "sign reliability, while telemetry showed mixed-command "
                "residual authority was tiny compared with the prior. V29 "
                "keeps the safe V49/V41 prior path and gives the learned "
                "residual more command-conditioned authority only where "
                "continuous tracking needs correction."),
        },
        "command_conditioned_prior_authority": {
            "available": (
                MIXED_PLANAR_AUTHORITY_REBALANCE_EXPERIMENTAL_AVAILABLE),
            "enabled": MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED,
            "source": "deployable normalized command obs[0:3]",
            "enable_env": (
                "WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE=1"),
            "formula": (
                "prior_scale *= a(cmd); "
                f"a={MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT:.2f} for "
                "mixed vx/vy with zero yaw when enabled; "
                f"optional slope-forward profile uses "
                f"a={SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT:.2f} for "
                "positive vx with zero yaw; a=1 otherwise"),
            "reason": (
                "V51/V52 raised residual authority but the dominant mixed "
                "planar prior still overwhelmed the learned correction. V54 "
                "tested reducing the dominant mixed-planar prior while "
                "increasing residual authority, but no-retrain and short "
                "continuation scans regressed planar RMSE. The ablation is "
                "kept available for comparison and disabled by default."),
            "terrain_profile_ablation": {
                "available": (
                    MIXED_PLANAR_AUTHORITY_PROFILE_EXPERIMENTAL_AVAILABLE),
                "default_prior_authority_mult": (
                    MIXED_PLANAR_DOMINANT_PRIOR_SCALE_MULT),
                "default_residual_scale_mult": (
                    MIXED_PLANAR_REBALANCED_RESIDUAL_SCALE_MULT),
                "prior_authority_mult": (
                    MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT),
                "residual_scale_mult": (
                    MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT),
                "prior_env": (
                    "WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT"),
                "residual_env": (
                    "WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT"),
                "reason": (
                    "V82/V83 showed gate and full-channel split-prior changes "
                    "did not repair slope mixed-vx/vy signs. These default-off "
                    "numeric overrides allow sand/slope deployment profiles to "
                    "test prior-vs-residual authority without changing the "
                    "80D observation or 12D action ABI."),
            },
            "slope_forward_axis_profile": {
                "available": (
                    SLOPE_FORWARD_AXIS_PROFILE_EXPERIMENTAL_AVAILABLE),
                "enabled": SLOPE_FORWARD_AXIS_PROFILE_ENABLED,
                "default": "disabled",
                "enable_env": (
                    "WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE=1"),
                "prior_authority_mult": (
                    SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT),
                "prior_authority_env": (
                    "WORM_V6_SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT"),
                "gait_blend_floor": SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR,
                "gait_blend_floor_env": (
                    "WORM_V6_SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR"),
                "phase_offset_rad": SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD,
                "phase_offset_env": (
                    "WORM_V6_SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD"),
                "target_command": {
                    "cmd_vx_norm": "positive and active",
                    "cmd_yaw_norm_max": DIRECTIONAL_PRIOR_THRESHOLD,
                    "intended_terrain": "slope",
                },
                "reason": (
                    "V84 authority-only profiles reduced slope RMSE but did "
                    "not fix positive-vx wrong signs, while prior-only "
                    "slope probes showed positive forward motion only around "
                    "gait_blend 0.5-0.75 with stronger prior scale. This "
                    "default-off V85 profile tests that structural forward "
                    "axis correction without changing the observation or "
                    "action ABI."),
            },
        },
        "mixed_planar_split_prior": {
            "available": MIXED_PLANAR_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE,
            "enabled": MIXED_PLANAR_SPLIT_PRIOR_ENABLED,
            "default": "disabled",
            "enable_env": "WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR=1",
            "formula": (
                "for mixed vx/vy with zero yaw: slide actuators come from "
                "the signed axial prior and yaw actuators come from the "
                "signed lateral primitive, with per-axis command gains"),
            "reason": (
                "V53/V54 strict scans show the dominant-prior path alternates "
                "between axial-only and lateral-only behavior. V55 tests a "
                "split-channel prior so mixed planar commands can express "
                "axial contraction and lateral steering at the same time "
                "without changing the 80D observation or 12D action ABI."),
            "full_channel_ablation": {
                "available": (
                    MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE),
                "enabled": MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED,
                "default": "disabled",
                "enable_env": (
                    "WORM_V6_ENABLE_MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR=1"),
                "formula": (
                    "for mixed vx/vy with zero yaw: retain both axial and "
                    "lateral primitive components on slide and yaw channels, "
                    "using the mixed-composition gains and axis-norm "
                    "normalization"),
                "reason": (
                    "V81/V82 strict scans showed the split-channel prior still "
                    "fails mixed-vx/vy signs: gate ablations reduced some "
                    "slope off-axis failures but did not repair planar sign. "
                    "This V83 ablation tests whether the split prior is too "
                    "under-expressive by giving both planar axes authority on "
                    "both actuator groups while preserving the action ABI."),
            },
            "slope_mixed_planar_primitive": {
                "available": (
                    SLOPE_MIXED_PLANAR_PRIMITIVE_EXPERIMENTAL_AVAILABLE),
                "enabled": SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED,
                "default": "disabled",
                "enable_env": (
                    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE=1"),
                "formula": (
                    "for slope mixed vx/vy with zero yaw: slide actuators "
                    "compose signed axial slides plus signed lateral slides; "
                    "yaw actuators keep only signed lateral yaw authority"),
                "gains": {
                    "axial_slide": SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN,
                    "lateral_slide": SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN,
                    "lateral_yaw": SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN,
                    "axis_norm": MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM,
                    "positive_vx_axial_yaw_ablation": {
                        "available": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_EXPERIMENTAL_AVAILABLE),
                        "enabled": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED),
                        "default": "disabled",
                        "enable_env": (
                            "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW=1"),
                        "axial_yaw_gain": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN),
                        "axial_yaw_gain_env": (
                            "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN"),
                        "lateral_yaw_mult": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT),
                        "lateral_yaw_mult_env": (
                            "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT"),
                    },
                    "positive_vx_axial_phase_ablation": {
                        "available": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_EXPERIMENTAL_AVAILABLE),
                        "enabled": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED),
                        "default": "disabled",
                        "enable_env": (
                            "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE=1"),
                        "phase_offset_rad": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD),
                        "phase_offset_env": (
                            "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD"),
                        "slide_sign": (
                            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN),
                        "slide_sign_env": (
                            "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN"),
                    },
                },
                "reason": (
                    "V86/V87 profile and phase grids saturated at mixed-vx/vy "
                    "wrong-planar signs, and V88 short retraining under the "
                    "best phase profiles regressed to wrong>=6. This "
                    "default-off V89 structural probe removes the axial yaw "
                    "component that induces diagonal yaw drift while keeping "
                    "both planar contact primitives available."),
            },
        },
        "command_conditioned_prior_scale": {
            "enabled": True,
            "source": "deployable normalized command obs[0:3]",
            "forward_only_scale": 1.0,
            "reverse_floor": REVERSE_PRIOR_SCALE_FLOOR,
            "lateral_floor": LATERAL_PRIOR_SCALE_FLOOR,
            "yaw_only_floor": YAW_ONLY_PRIOR_SCALE_FLOOR,
            "reason": (
                "The CMA-ES anchors are strong forward priors. Reverse, "
                "and pure yaw commands reduce the prior so the residual "
                "policy does not first need to cancel forward drift. "
                "Dominant lateral commands now use dedicated V41 lateral "
                "primitives and keep full prior scale."),
        },
        "command_activity_scale": {
            "enabled": True,
            "source": "deployable normalized command obs[0:3]",
            "formula": (
                "s_c = 0 if max(|cmd|)=0; "
                f"s_c = {YAW_ONLY_ACTIVITY_GAIN:.2f} * |cmd_yaw_norm| "
                "for pure yaw commands; "
                "otherwise "
                "clip(0.20 + 0.80 * max(|cmd|), 0, 1)"),
            "applied_to": "composed prior-plus-residual action",
            "reason": (
                "Discrete direction gates can move in six primitive "
                "directions, but continuous velocity tracking also needs "
                "zero and low-speed commands to reduce motor amplitude. This "
                "keeps the action ABI fixed while making stop and slow "
                "commands deployable. V60/V61/V62 low-yaw scans showed the "
                "pure yaw primitive was overdriven for small commands, so "
                "V34 uses calibrated linear pure-yaw activity to match the "
                "feasible low-yaw envelope. V35 halves that pure-yaw "
                "activity gain after the global yaw command range is narrowed "
                "from +/-0.25 rad/s to +/-0.125 rad/s, keeping physical "
                "turning activity close to the previous feasible envelope "
                "instead of overdriving pure-yaw commands."),
        },
        "command_directional_prior_transform": {
            "enabled": True,
            "source": "deployable normalized command obs[0:3]",
            "threshold": DIRECTIONAL_PRIOR_THRESHOLD,
            "continuous_vector_prior_blend": {
                "enabled": USE_CONTINUOUS_VECTOR_PRIOR_BLEND,
                "status": "experimental_disabled",
                "reason": (
                    "A V14 short run blended signed primitive priors for "
                    "mixed vx/vy/yaw commands, but the 35-command scan "
                    "worsened planar RMSE and planar sign rate. Keep V13 as "
                    "the default accepted adapter until a longer curriculum "
                    "or model-selection change proves the blend helps."),
            },
            "mixed_command_component_composition": {
                "available": MIXED_COMMAND_COMPOSITION_EXPERIMENTAL_AVAILABLE,
                "enabled": MIXED_COMMAND_COMPOSITION_ENABLED,
                "version": "v28",
                "enable_env": "WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION=1",
                "status": (
                    "experimental_default_off_after_v50_smoke_rejected"
                    if not MIXED_COMMAND_COMPOSITION_ENABLED
                    else "experimental_enabled"),
                "reason": (
                    "V48/V49 strict scans showed that dominant-direction "
                    "selection preserves signs but cannot compose mixed "
                    "vx/vy or vx/yaw commands. V28 keeps pure-command "
                    "priors unchanged and combines active axis primitives "
                    "componentwise for mixed commands: axial commands "
                    "primarily own slides, lateral commands own lateral "
                    "yaw structure, and yaw commands inject the stronger "
                    "in-place yaw primitive. V50 no-retrain and short "
                    "continuation scans did not pass the strict gate, so "
                    "the componentwise path is kept behind an explicit "
                    "experiment switch instead of becoming the default."),
                "gains": {
                    "axial_slide": MIXED_AXIAL_SLIDE_GAIN,
                    "lateral_slide": MIXED_LATERAL_SLIDE_GAIN,
                    "yaw_slide": MIXED_YAW_SLIDE_GAIN,
                    "axial_yaw": MIXED_AXIAL_YAW_GAIN,
                    "lateral_yaw": MIXED_LATERAL_YAW_GAIN,
                    "yaw_yaw": MIXED_YAW_YAW_GAIN,
                },
                "prior_scale_floors": {
                    "mixed_planar": MIXED_PLANAR_PRIOR_SCALE_FLOOR,
                    "mixed_yaw": MIXED_YAW_PRIOR_SCALE_FLOOR,
                },
                "normalization": {
                    "axis_norm": MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM,
                    "reason": (
                        "Mixed commands should preserve multiple primitive "
                        "components without doubling motor amplitude when two "
                        "full-strength axes are active."),
                },
            },
            "mixed_planar_hardcase_gate": {
                "available": MIXED_PLANAR_HARDCASE_GATE_EXPERIMENTAL_AVAILABLE,
                "enabled": MIXED_PLANAR_HARDCASE_GATE_ENABLED,
                "default": "disabled",
                "enable_env": "WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1",
                "target_command": {
                    "cmd_vx_norm_range": [
                        MIXED_PLANAR_HARDCASE_MIN_VX_NORM,
                        MIXED_PLANAR_HARDCASE_MAX_VX_NORM,
                    ],
                    "cmd_vy_norm_min": MIXED_PLANAR_HARDCASE_MIN_VY_NORM,
                    "cmd_yaw_norm_max": DIRECTIONAL_PRIOR_THRESHOLD,
                    "signed_vx": "positive_or_negative",
                },
                "gate_center": MIXED_PLANAR_HARDCASE_GATE_CENTER,
                "reason": (
                    "V74/V76 diagnostics show cmd=(+0.05,+0.075,0) is "
                    "sampled but still mapped to the pure lateral gate "
                    "center, suppressing visible axial peristaltic action "
                    "and yielding negative body vx. V96 strict scans then "
                    "showed reverse mixed-vx/vy commands also fail the "
                    "component-sign gate. This narrow ablation restores a "
                    "small axial gate for slow-axial, full-lateral planar "
                    "commands in both vx signs without changing the 80D "
                    "observation or 12D action ABI."),
            },
            "slope_forward_axis_profile": {
                "available": (
                    SLOPE_FORWARD_AXIS_PROFILE_EXPERIMENTAL_AVAILABLE),
                "enabled": SLOPE_FORWARD_AXIS_PROFILE_ENABLED,
                "default": "disabled",
                "gait_blend_floor": SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR,
                "phase_offset_rad": SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD,
                "target_command": {
                    "cmd_vx_norm": "positive and active",
                    "cmd_yaw_norm_max": DIRECTIONAL_PRIOR_THRESHOLD,
                },
            },
            "reverse": "dominant negative vx reverses phase",
            "lateral": (
                "dominant vy uses dedicated left/right open-loop lateral "
                "primitives searched on the deployable 1 s phase clock"),
            "yaw": (
                "pure yaw commands use an independent in-place yaw prior; "
                "mixed yaw commands use the commanded yaw sign on the yaw "
                "anchor"),
            "inplace_yaw_prior": {
                "enabled": True,
                "source": "open-loop MuJoCo search constrained to deployable phase clock",
                "param_names": list(INPLACE_YAW_PRIOR_PARAM_NAMES),
                "params": [float(v) for v in INPLACE_YAW_PRIOR_PARAMS],
                "left_right_rule": (
                    "cmd_yaw sign multiplies only the yaw joints; slide "
                    "compression wave is shared for both turn directions"),
            },
            "lateral_primitives": {
                "enabled": True,
                "source": "open-loop MuJoCo lateral primitive search constrained to deployable phase clock",
                "runtime_scales": {
                    "left": LATERAL_LEFT_PRIMITIVE_SCALE,
                    "right": LATERAL_RIGHT_PRIMITIVE_SCALE,
                },
                "slow_right_norm_max": SLOW_RIGHT_LATERAL_PRIOR_NORM_MAX,
                "param_names": list(LATERAL_PRIMITIVE_PARAM_NAMES),
                "anchors": {
                    side: {
                        "source": anchor["source"],
                        "validation": dict(anchor["validation"]),
                        "params": [float(v) for v in anchor["params"]],
                    }
                    for side, anchor in LATERAL_PRIMITIVE_ANCHORS.items()
                },
                "acceptance": {
                    "signed_lateral_m_s_min": 0.03,
                    "abs_body_vx_m_s_max": 0.08,
                    "abs_yaw_rate_rad_s_max": 0.20,
                    "validation_seconds": 6.0,
                },
            },
            "lateral_right_yaw_flip": (
                "negative vy flips only the yaw-anchor sign; the slide phase "
                "offset is shared because the V35/V36 prior-only diagnosis "
                "showed mixed-centered +pi/2 lateral motion was dominated by "
                "forward off-axis speed"),
            "lateral_phase_offset_rad": LATERAL_PHASE_OFFSET_RAD,
            "yaw_only_slide_scale": YAW_ONLY_SLIDE_PRIOR_SCALE,
            "yaw_only_yaw_scale": YAW_ONLY_YAW_PRIOR_SCALE,
            "yaw_right_only_yaw_scale": YAW_RIGHT_ONLY_YAW_PRIOR_SCALE,
            "zero_yaw_forward_phase_offset_rad": (
                ZERO_YAW_FORWARD_PHASE_OFFSET_RAD),
            "zero_yaw_reverse_phase_offset_rad": (
                ZERO_YAW_REVERSE_PHASE_OFFSET_RAD),
            "zero_yaw_forward_yaw_scale": ZERO_YAW_FORWARD_YAW_PRIOR_SCALE,
            "zero_yaw_reverse_yaw_scale": ZERO_YAW_REVERSE_YAW_PRIOR_SCALE,
            "zero_yaw_lateral_yaw_scale": ZERO_YAW_LATERAL_YAW_PRIOR_SCALE,
            "zero_yaw_lateral_yaw_trim": ZERO_YAW_LATERAL_YAW_TRIM,
            "v10_reason": (
                "Flat fixed-command diagnostics showed lateral floor 0.80 and "
                "extra pure-right-yaw authority reach all six 6 s threshold "
                "directions without changing the deployable observation or "
                "policy-action ABI. PPO still has to learn yaw-drift "
                "suppression for the formal direction gate."),
            "v11_reason": (
                "Heading-hold training showed the V10 prior still injects too "
                "much yaw during zero-yaw translation. V11 keeps the direction "
                "transform but attenuates yaw-prior authority for zero-yaw "
                "forward/reverse/lateral commands so translation and turning "
                "are easier to decouple."),
            "v12_reason": (
                "Flat V11 diagnostics showed zero-yaw lateral translation "
                "needs yaw authority for thrust but also a small heading trim; "
                "reverse translation needs an axial phase offset rather than a "
                "weak yaw prior. V12 adds command-conditioned trims without "
                "changing the 80D observation or 12D policy-action ABI."),
            "v13_reason": (
                "Continuous command scans showed primitive signs are solved "
                "but zero and low-speed commands still over-actuate. V13 adds "
                "a command-magnitude activity scale to the composed action so "
                "the same 12D policy can represent stop and slow commands."),
            "v14_reason": (
                "Continuous command scans showed mixed vx/vy/yaw commands "
                "were still routed through one dominant primitive. V14 keeps "
                "the primitive transforms unchanged for pure commands but "
                "linearly blends the six signed primitive priors according to "
                "the normalized command vector for arbitrary velocity "
                "commands."),
            "v16_reason": (
                "The first V15 run under the stronger tracking scan kept yaw "
                "signs separated but under-produced yaw magnitude. V16 keeps "
                "the deployable 80D/12D ABI fixed and raises only the "
                "yaw-only prior authority, with symmetric left/right yaw "
                "scales, so PPO starts from a usable turn-rate primitive "
                "instead of spending early training on basic yaw amplitude."),
            "v18_reason": (
                "V17 showed yaw signs were learned but pure yaw still "
                "translated forward and under-produced turn rate because the "
                "yaw-only primitive was a transformed forward CMA-ES anchor. "
                "V18 replaces pure yaw commands with an independently "
                "searched in-place yaw prior while preserving the 80D "
                "observation and 12D policy-action ABI."),
            "v26_reason": (
                "V40 still failed fixed lateral speed gates. V26 replaces "
                "dominant lateral command priors with independently searched "
                "left/right lateral primitives that pass 6 s prior-only "
                "lateral acceptance while preserving the deployable 80D "
                "observation and 12D residual-plus-gate action ABI."),
            "v27_reason": (
                "Strict V48 scan telemetry showed mixed vx+yaw commands had "
                "correct primitive signs elsewhere but collapsed or flipped "
                "the yaw component. V27 keeps the same ABI and prior family "
                "but makes mixed-yaw anchor yaw_sign follow the commanded yaw "
                "sign instead of the legacy inverted transform."),
            "v28_reason": (
                "V49 improved mixed-yaw sign but left the dominant mixed_vx_vy "
                "failure unchanged. V28 adds a componentwise mixed-command "
                "prior with axis-norm normalization so mixed planar "
                "commands can retain both axial slide authority and "
                "lateral yaw structure, while mixed yaw commands inherit "
                "stronger in-place yaw authority without removing axial "
                "slides or doubling total motor authority."),
            "v29_reason": (
                "V50 rejected the stronger componentwise mixed-prior path. "
                "V29 leaves that path behind an explicit experiment switch "
                "and instead increases residual authority for mixed planar, "
                "mixed yaw, and pure yaw commands so PPO can learn corrective "
                "components without changing the deployable 80D observation "
                "or 12D action ABI."),
            "v30_reason": (
                "V53 curriculum continuation preserved signs and improved yaw "
                "RMSE, but mixed vx/vy planar RMSE remained above the strict "
                "gate. V30 exposes the V54 mixed-planar authority rebalance "
                "as a default-off ablation because short scans regressed "
                "planar RMSE and sometimes planar sign reliability."),
            "v31_reason": (
                "V55 adds a default-off split-channel mixed planar prior "
                "candidate. Slides follow the axial primitive while yaw joints "
                "follow the lateral primitive, addressing the observed "
                "dominant-prior failure where mixed vx/vy commands express "
                "only one planar component at a time."),
            "v32_reason": (
                "V60 low-yaw envelope training separated yaw signs but "
                "over-shot small pure-yaw commands by applying full in-place "
                "yaw activity even when |cmd_yaw_norm| was small. V32 keeps "
                "the same 80D observation and 12D residual-plus-gate action "
                "ABI, but makes pure-yaw action activity command-magnitude "
                "scaled so low yaw rates can be tracked without saturating "
                "the yaw prior."),
            "v33_reason": (
                "V61 no-retrain and trained scans showed the nonzero yaw "
                "activity floor still over-shot low yaw commands and did not "
                "remove yaw-only planar drift. V33 makes pure-yaw activity "
                "linear in |cmd_yaw_norm| with no minimum floor, preserving "
                "full authority at max yaw while reducing small-command "
                "overdrive."),
            "v34_reason": (
                "V62 gain ablations showed the in-place yaw primitive still "
                "over-produced yaw rate with s_c=|cmd_yaw_norm|. V34 adds a "
                "0.50 pure-yaw activity gain; a no-retrain 13-command "
                "low-yaw scan reached yaw RMSE near 0.02 rad/s and pure-yaw "
                "planar drift near 0.03 m/s while preserving yaw sign."),
            "v35_reason": (
                "The deployable command envelope was narrowed to the "
                "first-stage feasible range vx +/-0.10 m/s, vy +/-0.075 m/s, "
                "and yaw +/-0.125 rad/s. Because normalized pure-yaw commands "
                "now map to half the old physical yaw range, V35 reduces the "
                "pure-yaw activity gain from 0.50 to 0.25 to avoid the robot "
                "curling under yaw-only commands while preserving the 80D "
                "observation and 12D residual-plus-gate action ABI."),
            "v36_reason": (
                "A V35 no-retrain feasible-envelope scan showed yaw was fixed "
                "but the right-lateral primitive remained calibrated to the "
                "old wider lateral command range, producing about -0.16 m/s "
                "for a -0.075 m/s command. V36 keeps the left primitive "
                "unchanged and applies a conservative 0.75 runtime scale to "
                "the right lateral primitive, reducing overshoot without "
                "removing the signed right-lateral contact pattern before "
                "PPO continuation."),
            "v37_reason": (
                "V70 strict scans isolated the remaining counterexample to "
                "cmd=(0,-0.0375,0): the full-speed right-lateral primitive "
                "moves right at full authority but flips left when command "
                "activity scales it to the slow-command amplitude. V37 adds a "
                "separately searched slow-right lateral primitive for "
                "dominant negative-vy commands with normalized magnitude up "
                f"to {SLOW_RIGHT_LATERAL_PRIOR_NORM_MAX:.2f}."),
        },
        "phase_source": "deployable phase_clock observation",
        "gait_blend_source": "policy action gate",
        "gait_gate_mapping": {
            "raw_action_index": NUM_ACTUATORS,
            "formula": (
                "learned_gate = clip(0.5 + 0.5 * "
                f"{GAIT_GATE_ACTION_GAIN:.1f} * raw_gate, 0, 1); "
                "gait_blend = clip(command_center + "
                f"{2.0 * COMMAND_GATE_CENTER_RESIDUAL_RANGE:.2f} * "
                "(learned_gate - 0.5), 0, 1)"),
            "command_centers": {
                "axial_translation": COMMAND_GATE_WORM_CENTER,
                "axial_translation_slow": COMMAND_GATE_WORM_CENTER_SLOW,
                "axial_translation_fast": COMMAND_GATE_WORM_CENTER_FAST,
                "axial_fast_threshold_norm": (
                    COMMAND_GATE_WORM_FAST_THRESHOLD),
                "mixed_or_stop": COMMAND_GATE_MIXED_CENTER,
                "mixed_planar_continuous_gate": {
                    "available": (
                        MIXED_PLANAR_CONTINUOUS_GATE_EXPERIMENTAL_AVAILABLE),
                    "enabled": MIXED_PLANAR_CONTINUOUS_GATE_ENABLED,
                    "enable_env": (
                        "WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE=1"),
                    "axial_share_floor": (
                        MIXED_PLANAR_GATE_AXIAL_SHARE_FLOOR),
                },
                "lateral_translation": COMMAND_GATE_LATERAL_CENTER,
                "yaw": COMMAND_GATE_YAW_CENTER,
            },
            "reason": (
                "V20/V21 showed the pure latent gate stayed near 0.5 because "
                "the low-noise residual PPO policy barely explored the gate "
                "dimension. V19 made the gate more sensitive. V20 adds a "
                "deployable command-conditioned gate center with a learned "
                "residual so axial commands visibly recover worm-like "
                "peristalsis while lateral/yaw commands keep snake-like "
                "authority. Earlier centered-gate runs restored visual mode "
                "separation but reduced forward speed and flipped lateral "
                "signs. V22 keeps yaw snake-like but moves lateral back to "
                "the mixed anchor and axial closer to the high-speed "
                "combined anchor. V23 makes the axial "
                "center speed-dependent: slow axial commands stay visibly "
                "peristaltic, while full-speed axial commands use the mixed "
                "center so tracking is not capped by the slower worm anchor. "
                "V25 moves dominant lateral commands back to the worm anchor "
                "after prior-only search showed this reduces forward "
                "off-axis drift. V26 then replaces that transformed anchor "
                "with dedicated searched lateral primitives, so the command "
                "center can still make lateral motion visibly worm-like while "
                "the prior has enough side-slip authority. The V75 continuous "
                "mixed-planar gate is kept as an experiment switch rather "
                "than a default because no-retrain and early retrain scans "
                "regressed nominal and robust mixed-planar sign reliability."),
        },
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


def is_yaw_only_command(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
    abs_vx = abs(float(cmd_vx_norm))
    abs_vy = abs(float(cmd_vy_norm))
    abs_yaw = abs(float(cmd_yaw_norm))
    return (
        abs_yaw >= DIRECTIONAL_PRIOR_THRESHOLD
        and max(abs_vx, abs_vy) < DIRECTIONAL_PRIOR_THRESHOLD)


def inplace_yaw_prior_from_phase(phase, cmd_yaw_norm):
    """Normalized open-loop primitive for near-stationary yaw commands."""
    yaw_sign = 1.0 if float(cmd_yaw_norm) >= 0.0 else -1.0
    t = _phase_cycle_seconds(phase)
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)

    for j in range(NUM_SLIDES):
        joint_phase = (
            TWO_PI
            * (INPLACE_YAW_SLIDE_FREQ * t
               - INPLACE_YAW_SLIDE_WAVE_N * j / NUM_SLIDES)
        )
        action[j] = -(
            INPLACE_YAW_SLIDE_BIAS
            + INPLACE_YAW_SLIDE_AMP
            * (0.5 + 0.5 * math.sin(joint_phase))
        )

    for j in range(NUM_YAWS):
        joint_phase = (
            TWO_PI
            * (INPLACE_YAW_YAW_FREQ * t
               + INPLACE_YAW_YAW_WAVE_N * j / NUM_YAWS)
            + INPLACE_YAW_YAW_PHASE_RAD
        )
        action[NUM_SLIDES + j] = (
            yaw_sign * INPLACE_YAW_YAW_AMP * math.sin(joint_phase))

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def lateral_primitive_side_for_command(cmd_vy_norm):
    """Select the lateral primitive for a normalized lateral command."""
    cmd_vy_norm = float(cmd_vy_norm)
    if cmd_vy_norm >= 0.0:
        return "left"
    if abs(cmd_vy_norm) <= SLOW_RIGHT_LATERAL_PRIOR_NORM_MAX:
        return "right_slow"
    return "right"


def lateral_primitive_action_from_phase(side, phase):
    """Normalized open-loop primitive for dominant lateral commands."""
    if side not in LATERAL_PRIMITIVE_ANCHORS:
        raise ValueError(
            f"Unknown lateral primitive side {side!r}; expected one of "
            f"{tuple(LATERAL_PRIMITIVE_ANCHORS)}")
    direction_sign = 1.0 if side == "left" else -1.0
    p = LATERAL_PRIMITIVE_ANCHORS[side]["params"]
    t = _phase_cycle_seconds(phase)
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    slide_bias = float(p[0])
    slide_amp = float(p[1])
    slide_freq = float(p[2])
    slide_wave_n = float(p[3])
    slide_phase = float(p[4])
    yaw_amp = float(p[5])
    yaw_freq = float(p[6])
    yaw_wave_n = float(p[7])
    yaw_phase = float(p[8])
    yaw_bias = float(p[9])
    yaw_trim_gradient = float(p[10])

    for j in range(NUM_SLIDES):
        joint_phase = (
            TWO_PI * (t * slide_freq - slide_wave_n * j / NUM_SLIDES)
            + slide_phase
        )
        action[j] = (
            slide_bias
            - slide_amp * (0.5 + 0.5 * math.sin(joint_phase)))

    center = 0.5 * (NUM_YAWS - 1)
    for j in range(NUM_YAWS):
        joint_phase = (
            TWO_PI * (t * yaw_freq + yaw_wave_n * j / NUM_YAWS)
            + yaw_phase
        )
        body_gradient = 0.0 if center <= 0.0 else (j - center) / center
        action[NUM_SLIDES + j] = direction_sign * (
            yaw_bias
            + yaw_trim_gradient * body_gradient
            + yaw_amp * math.sin(joint_phase))

    runtime_scale = (
        LATERAL_LEFT_PRIMITIVE_SCALE
        if side == "left" else LATERAL_RIGHT_PRIMITIVE_SCALE)
    action *= float(runtime_scale)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def command_directional_prior_transform(
        cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
    cmd_vx_norm = float(cmd_vx_norm)
    cmd_vy_norm = float(cmd_vy_norm)
    cmd_yaw_norm = float(cmd_yaw_norm)
    abs_vx = abs(cmd_vx_norm)
    abs_vy = abs(cmd_vy_norm)
    abs_yaw = abs(cmd_yaw_norm)
    threshold = DIRECTIONAL_PRIOR_THRESHOLD

    phase_sign = 1.0
    phase_offset = 0.0
    yaw_sign = 1.0
    slide_scale = 1.0
    yaw_scale = 1.0
    yaw_trim = 0.0
    uses_inplace_yaw_prior = False
    uses_lateral_primitive = False

    reverse = max(-cmd_vx_norm, 0.0)
    if (reverse >= threshold
            and reverse >= abs_vy
            and reverse >= abs_yaw):
        phase_sign = -1.0

    lateral_dominant = (
        abs_vy >= threshold
        and abs_vy > abs_vx
        and abs_vy >= abs_yaw)
    if lateral_dominant:
        phase_offset = LATERAL_PHASE_OFFSET_RAD
        uses_lateral_primitive = True

    yaw_only_command = is_yaw_only_command(
        cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm)
    if yaw_only_command:
        uses_inplace_yaw_prior = True
        yaw_sign = 1.0 if cmd_yaw_norm >= 0.0 else -1.0
        slide_scale = YAW_ONLY_SLIDE_PRIOR_SCALE
        yaw_scale = (
            YAW_ONLY_YAW_PRIOR_SCALE
            if yaw_sign > 0.0 else YAW_RIGHT_ONLY_YAW_PRIOR_SCALE)
    elif abs_yaw >= threshold:
        yaw_sign = 1.0 if cmd_yaw_norm >= 0.0 else -1.0
    if lateral_dominant and cmd_vy_norm < 0.0:
        yaw_sign = -1.0
    if (not yaw_only_command
          and abs_yaw < threshold
          and max(abs_vx, abs_vy) >= threshold):
        if lateral_dominant:
            yaw_scale = ZERO_YAW_LATERAL_YAW_PRIOR_SCALE
            yaw_trim = ZERO_YAW_LATERAL_YAW_TRIM * np.sign(cmd_vy_norm)
        elif cmd_vx_norm < -threshold:
            phase_offset = ZERO_YAW_REVERSE_PHASE_OFFSET_RAD
            yaw_scale = ZERO_YAW_REVERSE_YAW_PRIOR_SCALE
        elif cmd_vx_norm > threshold:
            phase_offset = (
                SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD
                if is_slope_forward_axis_profile_command(
                    cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm)
                else ZERO_YAW_FORWARD_PHASE_OFFSET_RAD)
            yaw_scale = ZERO_YAW_FORWARD_YAW_PRIOR_SCALE
        else:
            yaw_scale = 1.0

    return {
        "phase_sign": phase_sign,
        "phase_offset_rad": phase_offset,
        "yaw_sign": yaw_sign,
        "slide_scale": slide_scale,
        "yaw_scale": yaw_scale,
        "yaw_trim": yaw_trim,
        "uses_inplace_yaw_prior": uses_inplace_yaw_prior,
        "uses_lateral_primitive": uses_lateral_primitive,
    }


def _dominant_directional_gait_prior_from_phase(phase, gait_blend, command):
    if command is None:
        return gait_prior_from_phase(phase, gait_blend)
    transform = command_directional_prior_transform(*command)
    if transform["uses_lateral_primitive"]:
        side = lateral_primitive_side_for_command(command[1])
        return lateral_primitive_action_from_phase(side, phase)
    if transform["uses_inplace_yaw_prior"]:
        prior = inplace_yaw_prior_from_phase(phase, command[2])
        prior = prior.copy()
        prior[:NUM_SLIDES] *= transform["slide_scale"]
        prior[NUM_SLIDES:] *= transform["yaw_scale"]
        return np.clip(prior, -1.0, 1.0).astype(np.float32)
    prior = gait_prior_from_phase(
        transform["phase_sign"] * float(phase)
        + transform["phase_offset_rad"],
        gait_blend,
    )
    prior = prior.copy()
    prior[:NUM_SLIDES] *= transform["slide_scale"]
    prior[NUM_SLIDES:] *= transform["yaw_sign"] * transform["yaw_scale"]
    prior[NUM_SLIDES:] += transform["yaw_trim"]
    return np.clip(prior, -1.0, 1.0).astype(np.float32)


def _active_axis_gain(value, max_axis):
    value = abs(float(value))
    if value < DIRECTIONAL_PRIOR_THRESHOLD:
        return 0.0
    return float(np.clip(value / max(max_axis, 1e-9), 0.0, 1.0))


def is_mixed_command(command):
    if command is None:
        return False
    active_axes = sum(
        1
        for value in command
        if abs(float(value)) >= DIRECTIONAL_PRIOR_THRESHOLD
    )
    return active_axes >= 2


def is_mixed_planar_command(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
    abs_vx = abs(float(cmd_vx_norm))
    abs_vy = abs(float(cmd_vy_norm))
    abs_yaw = abs(float(cmd_yaw_norm))
    threshold = DIRECTIONAL_PRIOR_THRESHOLD
    return (
        abs_vx >= threshold
        and abs_vy >= threshold
        and abs_yaw < threshold)


def is_slope_forward_axis_profile_command(
        cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
    del cmd_vy_norm
    if not SLOPE_FORWARD_AXIS_PROFILE_ENABLED:
        return False
    return (
        float(cmd_vx_norm) >= DIRECTIONAL_PRIOR_THRESHOLD
        and abs(float(cmd_yaw_norm)) < DIRECTIONAL_PRIOR_THRESHOLD)


def _componentwise_mixed_gait_prior_from_phase(phase, gait_blend, command):
    cmd_vx, cmd_vy, cmd_yaw = [float(v) for v in command]
    abs_vx = abs(cmd_vx)
    abs_vy = abs(cmd_vy)
    abs_yaw = abs(cmd_yaw)
    max_axis = max(abs_vx, abs_vy, abs_yaw)
    if max_axis < DIRECTIONAL_PRIOR_THRESHOLD:
        return gait_prior_from_phase(phase, gait_blend)

    vx_gain = _active_axis_gain(cmd_vx, max_axis)
    vy_gain = _active_axis_gain(cmd_vy, max_axis)
    yaw_gain = _active_axis_gain(cmd_yaw, max_axis)
    if (vx_gain > 0.0) + (vy_gain > 0.0) + (yaw_gain > 0.0) < 2:
        return _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, command)

    axial_prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    lateral_prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    yaw_prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)

    if vx_gain > 0.0:
        axial_prior = _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, (np.sign(cmd_vx), 0.0, 0.0))
    if vy_gain > 0.0:
        lateral_prior = _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, (0.0, np.sign(cmd_vy), 0.0))
    if yaw_gain > 0.0:
        yaw_prior = _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, (0.0, 0.0, np.sign(cmd_yaw)))

    prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    prior[:NUM_SLIDES] = (
        MIXED_AXIAL_SLIDE_GAIN * vx_gain * axial_prior[:NUM_SLIDES]
        + MIXED_LATERAL_SLIDE_GAIN * vy_gain * lateral_prior[:NUM_SLIDES]
        + MIXED_YAW_SLIDE_GAIN * yaw_gain * yaw_prior[:NUM_SLIDES]
    )
    prior[NUM_SLIDES:] = (
        MIXED_AXIAL_YAW_GAIN * vx_gain * axial_prior[NUM_SLIDES:]
        + MIXED_LATERAL_YAW_GAIN * vy_gain * lateral_prior[NUM_SLIDES:]
        + MIXED_YAW_YAW_GAIN * yaw_gain * yaw_prior[NUM_SLIDES:]
    )
    if MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM:
        axis_norm = math.sqrt(vx_gain ** 2 + vy_gain ** 2 + yaw_gain ** 2)
        prior /= max(1.0, axis_norm)
    return np.clip(prior, -1.0, 1.0).astype(np.float32)


def split_channel_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, command):
    """Compose mixed vx/vy with either split or full-channel authority."""
    cmd_vx, cmd_vy, cmd_yaw = [float(v) for v in command]
    if not is_mixed_planar_command(cmd_vx, cmd_vy, cmd_yaw):
        return _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, command)

    abs_vx = abs(cmd_vx)
    abs_vy = abs(cmd_vy)
    max_axis = max(abs_vx, abs_vy, DIRECTIONAL_PRIOR_THRESHOLD)
    vx_gain = float(np.clip(abs_vx / max_axis, 0.0, 1.0))
    vy_gain = float(np.clip(abs_vy / max_axis, 0.0, 1.0))

    axial_prior = _dominant_directional_gait_prior_from_phase(
        phase, gait_blend, (np.sign(cmd_vx), 0.0, 0.0))
    side = lateral_primitive_side_for_command(cmd_vy)
    lateral_prior = lateral_primitive_action_from_phase(side, phase)

    prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    if MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED:
        prior[:NUM_SLIDES] = (
            MIXED_AXIAL_SLIDE_GAIN * vx_gain * axial_prior[:NUM_SLIDES]
            + MIXED_LATERAL_SLIDE_GAIN * vy_gain
            * lateral_prior[:NUM_SLIDES]
        )
        prior[NUM_SLIDES:] = (
            MIXED_AXIAL_YAW_GAIN * vx_gain * axial_prior[NUM_SLIDES:]
            + MIXED_LATERAL_YAW_GAIN * vy_gain
            * lateral_prior[NUM_SLIDES:]
        )
        if MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM:
            axis_norm = math.sqrt(vx_gain ** 2 + vy_gain ** 2)
            prior /= max(1.0, axis_norm)
    else:
        prior[:NUM_SLIDES] = vx_gain * axial_prior[:NUM_SLIDES]
        prior[NUM_SLIDES:] = vy_gain * lateral_prior[NUM_SLIDES:]
    return np.clip(prior, -1.0, 1.0).astype(np.float32)


def slope_mixed_planar_gait_prior_from_phase(phase, gait_blend, command):
    """Slope mixed-planar primitive that avoids axial yaw contamination."""
    cmd_vx, cmd_vy, cmd_yaw = [float(v) for v in command]
    if not is_mixed_planar_command(cmd_vx, cmd_vy, cmd_yaw):
        return _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, command)

    abs_vx = abs(cmd_vx)
    abs_vy = abs(cmd_vy)
    max_axis = max(abs_vx, abs_vy, DIRECTIONAL_PRIOR_THRESHOLD)
    vx_gain = float(np.clip(abs_vx / max_axis, 0.0, 1.0))
    vy_gain = float(np.clip(abs_vy / max_axis, 0.0, 1.0))

    axial_prior = _dominant_directional_gait_prior_from_phase(
        phase, gait_blend, (np.sign(cmd_vx), 0.0, 0.0))
    if (SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED
            and cmd_vx >= DIRECTIONAL_PRIOR_THRESHOLD):
        axial_prior = gait_prior_from_phase(
            float(phase)
            + SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD,
            gait_blend,
        )
        axial_prior = axial_prior.copy()
        axial_prior[:NUM_SLIDES] *= (
            SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN)
        axial_prior[NUM_SLIDES:] *= ZERO_YAW_FORWARD_YAW_PRIOR_SCALE
    side = lateral_primitive_side_for_command(cmd_vy)
    lateral_prior = lateral_primitive_action_from_phase(side, phase)

    prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    prior[:NUM_SLIDES] = (
        SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN
        * vx_gain
        * axial_prior[:NUM_SLIDES]
        + SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN
        * vy_gain
        * lateral_prior[:NUM_SLIDES]
    )
    lateral_yaws = (
        SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN
        * vy_gain
        * lateral_prior[NUM_SLIDES:]
    )
    if (SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED
            and cmd_vx >= DIRECTIONAL_PRIOR_THRESHOLD):
        lateral_yaws = (
            SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT
            * lateral_yaws)
        prior[NUM_SLIDES:] = (
            lateral_yaws
            + SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN
            * vx_gain
            * axial_prior[NUM_SLIDES:]
        )
    else:
        prior[NUM_SLIDES:] = lateral_yaws
    if MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM:
        axis_norm = math.sqrt(vx_gain ** 2 + vy_gain ** 2)
        prior /= max(1.0, axis_norm)
    return np.clip(prior, -1.0, 1.0).astype(np.float32)


def directional_gait_prior_from_phase(phase, gait_blend, command=None):
    if command is None:
        return gait_prior_from_phase(phase, gait_blend)
    if (SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED
            and is_mixed_planar_command(*command)):
        return slope_mixed_planar_gait_prior_from_phase(
            phase, gait_blend, command)
    if (MIXED_PLANAR_SPLIT_PRIOR_ENABLED
            and is_mixed_planar_command(*command)):
        return split_channel_mixed_planar_gait_prior_from_phase(
            phase, gait_blend, command)
    if MIXED_COMMAND_COMPOSITION_ENABLED and is_mixed_command(command):
        return _componentwise_mixed_gait_prior_from_phase(
            phase, gait_blend, command)
    if not USE_CONTINUOUS_VECTOR_PRIOR_BLEND:
        return _dominant_directional_gait_prior_from_phase(
            phase, gait_blend, command)

    cmd_vx, cmd_vy, cmd_yaw = [float(v) for v in command]
    weights = (
        max(cmd_vx, 0.0),
        max(-cmd_vx, 0.0),
        max(cmd_vy, 0.0),
        max(-cmd_vy, 0.0),
        max(cmd_yaw, 0.0),
        max(-cmd_yaw, 0.0),
    )
    total_weight = float(sum(weights))
    if total_weight <= 1e-9:
        return gait_prior_from_phase(phase, gait_blend)

    primitive_commands = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    prior = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    for weight, primitive_command in zip(weights, primitive_commands):
        if weight <= 0.0:
            continue
        prior += float(weight) * _dominant_directional_gait_prior_from_phase(
            phase,
            gait_blend,
            primitive_command,
        )
    prior /= total_weight
    return np.clip(prior, -1.0, 1.0).astype(np.float32)


def command_prior_scale_floor(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
    cmd_vx_norm = float(cmd_vx_norm)
    cmd_vy_norm = float(cmd_vy_norm)
    cmd_yaw_norm = float(cmd_yaw_norm)
    abs_vx = abs(cmd_vx_norm)
    abs_vy = abs(cmd_vy_norm)
    abs_yaw = abs(cmd_yaw_norm)
    reverse = max(-cmd_vx_norm, 0.0)
    threshold = DIRECTIONAL_PRIOR_THRESHOLD
    mixed_planar = (
        abs_vx >= threshold
        and abs_vy >= threshold
        and abs_yaw < threshold)
    mixed_yaw = (
        abs_yaw >= threshold
        and max(abs_vx, abs_vy) >= threshold)
    if (abs_yaw >= threshold
            and max(abs_vx, abs_vy) < threshold):
        return YAW_ONLY_PRIOR_SCALE_FLOOR
    if MIXED_COMMAND_COMPOSITION_ENABLED and mixed_yaw:
        return MIXED_YAW_PRIOR_SCALE_FLOOR
    if MIXED_COMMAND_COMPOSITION_ENABLED and mixed_planar:
        return MIXED_PLANAR_PRIOR_SCALE_FLOOR
    if (abs_vy >= threshold
            and abs_vy > abs_vx
            and abs_vy >= abs_yaw):
        return LATERAL_PRIOR_SCALE_FLOOR
    if reverse >= threshold:
        return REVERSE_PRIOR_SCALE_FLOOR
    return REVERSE_PRIOR_SCALE_FLOOR


def command_conditioned_prior_scale(
        cmd_vx_norm,
        cmd_vy_norm,
        cmd_yaw_norm,
        non_forward_floor=None):
    forward = max(float(cmd_vx_norm), 0.0)
    non_forward = max(
        max(-float(cmd_vx_norm), 0.0),
        abs(float(cmd_vy_norm)),
        abs(float(cmd_yaw_norm)),
    )
    if non_forward <= 1e-9:
        return 1.0
    forward_share = forward / max(forward + non_forward, 1e-9)
    if non_forward_floor is None:
        non_forward_floor = command_prior_scale_floor(
            cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm)
    floor = float(np.clip(non_forward_floor, 0.0, 1.0))
    return float(floor + (1.0 - floor) * forward_share)


def command_conditioned_residual_scale(
        cmd_vx_norm,
        cmd_vy_norm,
        cmd_yaw_norm):
    if not COMMAND_CONDITIONED_RESIDUAL_AUTHORITY_ENABLED:
        return 1.0
    cmd_vx_norm = float(cmd_vx_norm)
    cmd_vy_norm = float(cmd_vy_norm)
    cmd_yaw_norm = float(cmd_yaw_norm)
    abs_vx = abs(cmd_vx_norm)
    abs_vy = abs(cmd_vy_norm)
    abs_yaw = abs(cmd_yaw_norm)
    threshold = DIRECTIONAL_PRIOR_THRESHOLD
    yaw_only = (
        abs_yaw >= threshold
        and max(abs_vx, abs_vy) < threshold)
    mixed_planar = (
        abs_vx >= threshold
        and abs_vy >= threshold
        and abs_yaw < threshold)
    mixed_yaw = (
        abs_yaw >= threshold
        and max(abs_vx, abs_vy) >= threshold)
    if mixed_planar:
        if MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED:
            return MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT
        return MIXED_PLANAR_RESIDUAL_SCALE_MULT
    if mixed_yaw:
        return MIXED_YAW_RESIDUAL_SCALE_MULT
    if yaw_only:
        return YAW_ONLY_RESIDUAL_SCALE_MULT
    return 1.0


def command_conditioned_prior_authority_scale(
        cmd_vx_norm,
        cmd_vy_norm,
        cmd_yaw_norm):
    if is_slope_forward_axis_profile_command(
            cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
        return SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT
    if not MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED:
        return 1.0
    if is_mixed_planar_command(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
        return MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT
    return 1.0


def command_activity_scale(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
    command_mag = max(
        abs(float(cmd_vx_norm)),
        abs(float(cmd_vy_norm)),
        abs(float(cmd_yaw_norm)),
    )
    if command_mag <= 1e-9:
        return 0.0
    if is_yaw_only_command(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm):
        abs_yaw = abs(float(cmd_yaw_norm))
        return float(np.clip(
            YAW_ONLY_ACTIVITY_GAIN
            * (YAW_ONLY_ACTIVITY_MIN_ACTIVE_SCALE
               + (1.0 - YAW_ONLY_ACTIVITY_MIN_ACTIVE_SCALE) * abs_yaw),
            0.0,
            1.0,
        ))
    return float(np.clip(
        COMMAND_ACTIVITY_MIN_ACTIVE_SCALE
        + (1.0 - COMMAND_ACTIVITY_MIN_ACTIVE_SCALE) * command_mag,
        0.0,
        1.0,
    ))


def phase_from_clock(phase_sin, phase_cos):
    return math.atan2(float(phase_sin), float(phase_cos))


def gait_blend_from_policy_gate(policy_gate_action):
    return float(np.clip(
        0.5 + 0.5 * GAIT_GATE_ACTION_GAIN * float(policy_gate_action),
        0.0,
        1.0,
    ))


def axial_gate_center_from_speed(cmd_vx_norm):
    axial_mag = abs(float(cmd_vx_norm))
    denom = max(
        COMMAND_GATE_WORM_FAST_THRESHOLD - DIRECTIONAL_PRIOR_THRESHOLD,
        1e-6,
    )
    alpha = float(np.clip(
        (axial_mag - DIRECTIONAL_PRIOR_THRESHOLD) / denom,
        0.0,
        1.0,
    ))
    return float(
        COMMAND_GATE_WORM_CENTER_SLOW
        + alpha
        * (COMMAND_GATE_WORM_CENTER_FAST - COMMAND_GATE_WORM_CENTER_SLOW)
    )


def command_conditioned_gate_center(command):
    if command is None:
        return COMMAND_GATE_MIXED_CENTER
    signed_vx, signed_vy, signed_yaw = [float(v) for v in command]
    cmd_vx, cmd_vy, cmd_yaw = [
        abs(signed_vx), abs(signed_vy), abs(signed_yaw)]
    cmd_mag = max(cmd_vx, cmd_vy, cmd_yaw)
    if cmd_mag <= 1e-6:
        return COMMAND_GATE_MIXED_CENTER
    if cmd_yaw >= DIRECTIONAL_PRIOR_THRESHOLD and max(cmd_vx, cmd_vy) < DIRECTIONAL_PRIOR_THRESHOLD:
        return COMMAND_GATE_YAW_CENTER
    if (MIXED_PLANAR_HARDCASE_GATE_ENABLED
            and MIXED_PLANAR_HARDCASE_MIN_VX_NORM <= cmd_vx
            <= MIXED_PLANAR_HARDCASE_MAX_VX_NORM
            and cmd_vy >= MIXED_PLANAR_HARDCASE_MIN_VY_NORM
            and cmd_yaw < DIRECTIONAL_PRIOR_THRESHOLD):
        return MIXED_PLANAR_HARDCASE_GATE_CENTER
    if (MIXED_PLANAR_CONTINUOUS_GATE_ENABLED
            and cmd_vx >= DIRECTIONAL_PRIOR_THRESHOLD
            and cmd_vy >= DIRECTIONAL_PRIOR_THRESHOLD
            and cmd_yaw < DIRECTIONAL_PRIOR_THRESHOLD):
        axial_share = cmd_vx / max(cmd_vx + cmd_vy, 1e-9)
        if axial_share < MIXED_PLANAR_GATE_AXIAL_SHARE_FLOOR:
            return COMMAND_GATE_LATERAL_CENTER
        alpha = float(np.clip(
            (axial_share - MIXED_PLANAR_GATE_AXIAL_SHARE_FLOOR)
            / max(1.0 - MIXED_PLANAR_GATE_AXIAL_SHARE_FLOOR, 1e-9),
            0.0,
            1.0,
        ))
        axial_center = axial_gate_center_from_speed(cmd_vx)
        return float(
            COMMAND_GATE_LATERAL_CENTER
            + alpha * (axial_center - COMMAND_GATE_LATERAL_CENTER))
    if (cmd_vy >= DIRECTIONAL_PRIOR_THRESHOLD
            and cmd_vy > cmd_vx
            and cmd_vy >= cmd_yaw):
        return COMMAND_GATE_LATERAL_CENTER
    if (cmd_vx >= DIRECTIONAL_PRIOR_THRESHOLD
            and cmd_vy < DIRECTIONAL_PRIOR_THRESHOLD
            and cmd_yaw < DIRECTIONAL_PRIOR_THRESHOLD):
        return axial_gate_center_from_speed(cmd_vx)
    return COMMAND_GATE_MIXED_CENTER


def command_conditioned_gait_blend(policy_gait_blend, command):
    center = command_conditioned_gate_center(command)
    residual = (
        2.0
        * COMMAND_GATE_CENTER_RESIDUAL_RANGE
        * (float(policy_gait_blend) - 0.5)
    )
    gait_blend = float(np.clip(center + residual, 0.0, 1.0))
    if command is not None and is_slope_forward_axis_profile_command(*command):
        gait_blend = max(
            gait_blend,
            float(np.clip(SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR, 0.0, 1.0)),
        )
    return gait_blend


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
    gait_blend = gait_blend_from_policy_gate(action[-1])
    return residual, gait_blend


def compose_deployable_action(
        residual_action,
        phase,
        gait_blend,
        command=None,
        gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
        policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE,
        activity_floor=0.0):
    residual = np.clip(
        np.asarray(residual_action, dtype=np.float32),
        -1.0,
        1.0,
    )
    if residual.shape != (NUM_ACTUATORS,):
        raise ValueError(
            f"residual action shape {residual.shape} != {(NUM_ACTUATORS,)}")
    prior = directional_gait_prior_from_phase(phase, gait_blend, command)
    prior_scale = float(gait_prior_scale)
    residual_scale = float(policy_residual_scale)
    if command is not None:
        prior_scale *= command_conditioned_prior_scale(*command)
        prior_scale *= command_conditioned_prior_authority_scale(*command)
        residual_scale *= command_conditioned_residual_scale(*command)
    activity_scale = (
        1.0 if command is None else command_activity_scale(*command))
    activity_scale = max(activity_scale, float(activity_floor))
    action = (
        prior_scale * prior
        + residual_scale * residual)
    action *= activity_scale
    return np.clip(action, -1.0, 1.0).astype(np.float32)

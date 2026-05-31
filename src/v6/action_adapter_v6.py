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


ACTION_ADAPTER_VERSION = "cmaes_tri_anchor_auto_gate_directional_v29"
USE_CONTINUOUS_VECTOR_PRIOR_BLEND = False


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


MIXED_COMMAND_COMPOSITION_EXPERIMENTAL_AVAILABLE = True
MIXED_COMMAND_COMPOSITION_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION", default=False)
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
DEFAULT_GAIT_PRIOR_SCALE = 1.0
DEFAULT_POLICY_RESIDUAL_SCALE = 0.35
COMMAND_CONDITIONED_RESIDUAL_AUTHORITY_ENABLED = True
MIXED_PLANAR_RESIDUAL_SCALE_MULT = 1.80
MIXED_YAW_RESIDUAL_SCALE_MULT = 1.60
YAW_ONLY_RESIDUAL_SCALE_MULT = 1.25
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
LATERAL_PHASE_OFFSET_RAD = -0.5 * math.pi
ZERO_YAW_FORWARD_PHASE_OFFSET_RAD = math.pi
ZERO_YAW_REVERSE_PHASE_OFFSET_RAD = math.pi
ZERO_YAW_FORWARD_YAW_PRIOR_SCALE = 1.15
ZERO_YAW_REVERSE_YAW_PRIOR_SCALE = 1.25
ZERO_YAW_LATERAL_YAW_PRIOR_SCALE = 0.65
ZERO_YAW_LATERAL_YAW_TRIM = 0.24
COMMAND_ACTIVITY_MIN_ACTIVE_SCALE = 0.20
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
                f"m={MIXED_YAW_RESIDUAL_SCALE_MULT:.2f} for mixed yaw"),
            "reason": (
                "V50 strict scans showed hand-composed mixed priors degraded "
                "sign reliability, while telemetry showed mixed-command "
                "residual authority was tiny compared with the prior. V29 "
                "keeps the safe V49/V41 prior path and gives the learned "
                "residual more command-conditioned authority only where "
                "continuous tracking needs correction."),
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
                "s_c = 1 for pure yaw commands; otherwise "
                "clip(0.20 + 0.80 * max(|cmd|), 0, 1)"),
            "applied_to": "composed prior-plus-residual action",
            "reason": (
                "Discrete direction gates can move in six primitive "
                "directions, but continuous velocity tracking also needs "
                "zero and low-speed commands to reduce motor amplitude. This "
                "keeps the action ABI fixed while making stop and slow "
                "commands deployable. Pure yaw commands keep full activity "
                "because the in-place yaw prior is not linearly "
                "speed-proportional at small turn rates."),
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
                "the prior has enough side-slip authority."),
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


def lateral_primitive_action_from_phase(side, phase):
    """Normalized open-loop primitive for dominant lateral commands."""
    if side not in LATERAL_PRIMITIVE_ANCHORS:
        raise ValueError(
            f"Unknown lateral primitive side {side!r}; expected left/right")
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
            phase_offset = ZERO_YAW_FORWARD_PHASE_OFFSET_RAD
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
        side = "left" if command[1] >= 0.0 else "right"
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


def directional_gait_prior_from_phase(phase, gait_blend, command=None):
    if command is None:
        return gait_prior_from_phase(phase, gait_blend)
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
        return MIXED_PLANAR_RESIDUAL_SCALE_MULT
    if mixed_yaw:
        return MIXED_YAW_RESIDUAL_SCALE_MULT
    if yaw_only:
        return YAW_ONLY_RESIDUAL_SCALE_MULT
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
        return 1.0
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
    cmd_vx, cmd_vy, cmd_yaw = [abs(float(v)) for v in command]
    cmd_mag = max(cmd_vx, cmd_vy, cmd_yaw)
    if cmd_mag <= 1e-6:
        return COMMAND_GATE_MIXED_CENTER
    if cmd_yaw >= DIRECTIONAL_PRIOR_THRESHOLD and max(cmd_vx, cmd_vy) < DIRECTIONAL_PRIOR_THRESHOLD:
        return COMMAND_GATE_YAW_CENTER
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
    return float(np.clip(center + residual, 0.0, 1.0))


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
        policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
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
        residual_scale *= command_conditioned_residual_scale(*command)
    activity_scale = (
        1.0 if command is None else command_activity_scale(*command))
    action = (
        prior_scale * prior
        + residual_scale * residual)
    action *= activity_scale
    return np.clip(action, -1.0, 1.0).astype(np.float32)

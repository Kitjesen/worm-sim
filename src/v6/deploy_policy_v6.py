"""
Export and replay a deployable Worm V6 policy.

The export path freezes the deterministic PPO actor plus VecNormalize
observation statistics into a TorchScript module. The replay path consumes a
hardware CSV with the 80-D deployable observation columns and writes normalized
11-D actions plus physical joint targets, so the hardware log schema can be
tested without MuJoCo or SB3.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from motor_contract_v6 import (  # noqa: E402
    NUM_SLIDES,
    NUM_YAWS,
    SLIDE_TARGET_SCALE_M,
    YAW_TARGET_SCALE_RAD,
    action_mapping_config,
    motor_contract,
)
from action_adapter_v6 import (  # noqa: E402
    CMAES_ANCHORS,
    COMMAND_ACTIVITY_MIN_ACTIVE_SCALE,
    COMMAND_CONDITIONED_RESIDUAL_AUTHORITY_ENABLED,
    DEFAULT_GAIT_PRIOR_SCALE,
    DEFAULT_POLICY_RESIDUAL_SCALE,
    DIRECTIONAL_PRIOR_THRESHOLD,
    COMMAND_GATE_CENTER_RESIDUAL_RANGE,
    COMMAND_GATE_LATERAL_CENTER,
    COMMAND_GATE_MIXED_CENTER,
    COMMAND_GATE_WORM_CENTER_FAST,
    COMMAND_GATE_WORM_CENTER_SLOW,
    COMMAND_GATE_WORM_FAST_THRESHOLD,
    COMMAND_GATE_YAW_CENTER,
    GAIT_GATE_ACTION_GAIN,
    INPLACE_YAW_SLIDE_AMP,
    INPLACE_YAW_SLIDE_BIAS,
    INPLACE_YAW_SLIDE_FREQ,
    INPLACE_YAW_SLIDE_WAVE_N,
    INPLACE_YAW_YAW_AMP,
    INPLACE_YAW_YAW_FREQ,
    INPLACE_YAW_YAW_PHASE_RAD,
    INPLACE_YAW_YAW_WAVE_N,
    LATERAL_PHASE_OFFSET_RAD,
    LATERAL_LEFT_PRIMITIVE_SCALE,
    LATERAL_PRIMITIVE_ANCHORS,
    LATERAL_PRIOR_SCALE_FLOOR,
    LATERAL_RIGHT_PRIMITIVE_SCALE,
    MIXED_AXIAL_SLIDE_GAIN,
    MIXED_AXIAL_YAW_GAIN,
    MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM,
    MIXED_COMMAND_COMPOSITION_ENABLED,
    MIXED_LATERAL_SLIDE_GAIN,
    MIXED_LATERAL_YAW_GAIN,
    MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED,
    MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT,
    MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT,
    MIXED_PLANAR_RESIDUAL_SCALE_MULT,
    MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED,
    MIXED_PLANAR_PRIOR_SCALE_FLOOR,
    MIXED_PLANAR_SPLIT_PRIOR_ENABLED,
    MIXED_YAW_RESIDUAL_SCALE_MULT,
    MIXED_YAW_PRIOR_SCALE_FLOOR,
    MIXED_YAW_SLIDE_GAIN,
    MIXED_YAW_YAW_GAIN,
    POLICY_ACTION_DIM,
    REVERSE_PRIOR_SCALE_FLOOR,
    SLOW_RIGHT_LATERAL_PRIOR_NORM_MAX,
    SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR,
    SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD,
    SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT,
    SLOPE_FORWARD_AXIS_PROFILE_ENABLED,
    SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN,
    SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN,
    SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN,
    SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED,
    SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN,
    SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED,
    SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD,
    SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN,
    SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT,
    SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED,
    USE_CONTINUOUS_VECTOR_PRIOR_BLEND,
    YAW_ONLY_PRIOR_SCALE_FLOOR,
    YAW_ONLY_RESIDUAL_SCALE_MULT,
    YAW_ONLY_SLIDE_PRIOR_SCALE,
    YAW_ONLY_YAW_PRIOR_SCALE,
    YAW_RIGHT_ONLY_YAW_PRIOR_SCALE,
    ZERO_YAW_FORWARD_PHASE_OFFSET_RAD,
    ZERO_YAW_FORWARD_YAW_PRIOR_SCALE,
    ZERO_YAW_LATERAL_YAW_PRIOR_SCALE,
    ZERO_YAW_LATERAL_YAW_TRIM,
    ZERO_YAW_REVERSE_PHASE_OFFSET_RAD,
    ZERO_YAW_REVERSE_YAW_PRIOR_SCALE,
    action_adapter_contract,
    command_conditioned_gait_blend,
    compose_deployable_action,
    phase_from_clock,
    policy_action_to_residual_and_gait_blend,
)


class DeployablePPOActor(torch.nn.Module):
    def __init__(
            self, policy, obs_mean, obs_var, epsilon, clip_obs,
            gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
            policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.register_buffer(
            "obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer(
            "obs_var", torch.as_tensor(obs_var, dtype=torch.float32))
        self.epsilon = float(epsilon)
        self.clip_obs = float(clip_obs)
        self.gait_prior_scale = float(gait_prior_scale)
        self.policy_residual_scale = float(policy_residual_scale)
        self.register_buffer(
            "slide_fraction",
            torch.arange(NUM_SLIDES, dtype=torch.float32) / float(NUM_SLIDES))
        self.register_buffer(
            "yaw_fraction",
            torch.arange(NUM_YAWS, dtype=torch.float32) / float(NUM_YAWS))
        self.register_buffer(
            "peristaltic_params",
            torch.tensor(CMAES_ANCHORS["peristaltic"]["params"],
                         dtype=torch.float32))
        self.register_buffer(
            "full_params",
            torch.tensor(CMAES_ANCHORS["full"]["params"],
                         dtype=torch.float32))
        self.register_buffer(
            "serpentine_params",
            torch.tensor(CMAES_ANCHORS["serpentine"]["params"],
                         dtype=torch.float32))
        self.register_buffer(
            "lateral_left_params",
            torch.tensor(LATERAL_PRIMITIVE_ANCHORS["left"]["params"],
                         dtype=torch.float32))
        self.register_buffer(
            "lateral_right_params",
            torch.tensor(LATERAL_PRIMITIVE_ANCHORS["right"]["params"],
                         dtype=torch.float32))
        self.register_buffer(
            "lateral_right_slow_params",
            torch.tensor(LATERAL_PRIMITIVE_ANCHORS["right_slow"]["params"],
                         dtype=torch.float32))
        yaw_center = max(0.5 * (NUM_YAWS - 1), 1.0)
        self.register_buffer(
            "yaw_centered_fraction",
            (torch.arange(NUM_YAWS, dtype=torch.float32)
             - 0.5 * float(NUM_YAWS - 1)) / float(yaw_center))
        self.slide_target_scale_m = float(SLIDE_TARGET_SCALE_M)
        self.yaw_target_scale_rad = float(YAW_TARGET_SCALE_RAD)

    def _anchor_prior(self, phase_cycle_s, params, slide_enable, yaw_enable):
        slide_phase = (
            2.0 * torch.pi
            * (phase_cycle_s * params[1]
               - params[2] * self.slide_fraction.unsqueeze(0))
            + params[6:12].unsqueeze(0))
        slide_prior = (
            -slide_enable
            * (params[0] / self.slide_target_scale_m)
            * (1.0 + torch.sin(slide_phase)))

        yaw_phase = (
            2.0 * torch.pi * params[4] * phase_cycle_s
            + 2.0 * torch.pi * params[5] * self.yaw_fraction.unsqueeze(0)
            + params[13] * 2.0 * torch.pi * phase_cycle_s * params[1])
        yaw_prior = (
            yaw_enable
            * (params[3] / self.yaw_target_scale_rad)
            * torch.sin(yaw_phase))
        return torch.clamp(
            torch.cat([slide_prior, yaw_prior], dim=1), -1.0, 1.0)

    def _base_gait_prior(self, phase, gait_blend):
        phase_cycle_s = torch.remainder(
            phase, 2.0 * torch.pi) / (2.0 * torch.pi)
        worm_prior = self._anchor_prior(
            phase_cycle_s, self.peristaltic_params, 1.0, 0.0)
        full_prior = self._anchor_prior(
            phase_cycle_s, self.full_params, 1.0, 1.0)
        snake_prior = self._anchor_prior(
            phase_cycle_s, self.serpentine_params, 0.0, 1.0)
        low_alpha = torch.clamp(gait_blend / 0.5, 0.0, 1.0)
        high_alpha = torch.clamp((gait_blend - 0.5) / 0.5, 0.0, 1.0)
        low_prior = (1.0 - low_alpha) * worm_prior + low_alpha * full_prior
        high_prior = (1.0 - high_alpha) * full_prior + high_alpha * snake_prior
        use_low = (gait_blend <= 0.5).to(dtype=torch.float32)
        return torch.clamp(
            use_low * low_prior + (1.0 - use_low) * high_prior,
            -1.0,
            1.0,
        )

    def _inplace_yaw_prior(self, phase, cmd_yaw):
        phase_cycle_s = torch.remainder(
            phase, 2.0 * torch.pi) / (2.0 * torch.pi)
        slide_phase = (
            2.0 * torch.pi
            * (phase_cycle_s * float(INPLACE_YAW_SLIDE_FREQ)
               - float(INPLACE_YAW_SLIDE_WAVE_N)
               * self.slide_fraction.unsqueeze(0)))
        slide_prior = -(
            float(INPLACE_YAW_SLIDE_BIAS)
            + float(INPLACE_YAW_SLIDE_AMP)
            * (0.5 + 0.5 * torch.sin(slide_phase)))

        yaw_phase = (
            2.0 * torch.pi
            * (phase_cycle_s * float(INPLACE_YAW_YAW_FREQ)
               + float(INPLACE_YAW_YAW_WAVE_N)
               * self.yaw_fraction.unsqueeze(0))
            + float(INPLACE_YAW_YAW_PHASE_RAD))
        yaw_sign = torch.where(
            cmd_yaw >= 0.0,
            torch.ones_like(cmd_yaw),
            -torch.ones_like(cmd_yaw),
        )
        yaw_prior = (
            yaw_sign
            * float(INPLACE_YAW_YAW_AMP)
            * torch.sin(yaw_phase))
        return torch.clamp(
            torch.cat([slide_prior, yaw_prior], dim=1), -1.0, 1.0)

    def _lateral_prior_from_params(self, phase, params, direction_sign):
        phase_cycle_s = torch.remainder(
            phase, 2.0 * torch.pi) / (2.0 * torch.pi)
        slide_phase = (
            2.0 * torch.pi
            * (phase_cycle_s * params[2]
               - params[3] * self.slide_fraction.unsqueeze(0))
            + params[4])
        slide_prior = (
            params[0]
            - params[1] * (0.5 + 0.5 * torch.sin(slide_phase)))

        yaw_phase = (
            2.0 * torch.pi
            * (phase_cycle_s * params[6]
               + params[7] * self.yaw_fraction.unsqueeze(0))
            + params[8])
        yaw_prior = direction_sign * (
            params[9]
            + params[10] * self.yaw_centered_fraction.unsqueeze(0)
            + params[5] * torch.sin(yaw_phase))
        return torch.clamp(
            torch.cat([slide_prior, yaw_prior], dim=1), -1.0, 1.0)

    def _lateral_prior(self, phase, cmd_vy):
        left_prior = self._lateral_prior_from_params(
            phase, self.lateral_left_params, 1.0) * float(
                LATERAL_LEFT_PRIMITIVE_SCALE)
        right_prior = self._lateral_prior_from_params(
            phase, self.lateral_right_params, -1.0) * float(
                LATERAL_RIGHT_PRIMITIVE_SCALE)
        slow_right_prior = self._lateral_prior_from_params(
            phase, self.lateral_right_slow_params, -1.0) * float(
                LATERAL_RIGHT_PRIMITIVE_SCALE)
        right_slow_mask = (
            (cmd_vy < 0.0)
            & (torch.abs(cmd_vy)
               <= float(SLOW_RIGHT_LATERAL_PRIOR_NORM_MAX)))
        right_prior = torch.where(right_slow_mask, slow_right_prior,
                                  right_prior)
        return torch.where(cmd_vy >= 0.0, left_prior, right_prior)

    def _dominant_directional_prior(
            self, phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw):
        abs_vx = torch.abs(cmd_vx)
        abs_vy = torch.abs(cmd_vy)
        abs_yaw = torch.abs(cmd_yaw)
        threshold = float(DIRECTIONAL_PRIOR_THRESHOLD)
        reverse = torch.clamp(-cmd_vx, min=0.0)
        reverse_mask = (
            (reverse >= threshold)
            & (reverse >= abs_vy)
            & (reverse >= abs_yaw))
        lateral_mask = (
            (abs_vy >= threshold)
            & (abs_vy > abs_vx)
            & (abs_vy >= abs_yaw))
        yaw_only_mask = (
            (abs_yaw >= threshold)
            & (torch.maximum(abs_vx, abs_vy) < threshold))
        zero_yaw_planar_mask = (
            (abs_yaw < threshold)
            & (torch.maximum(abs_vx, abs_vy) >= threshold))
        zero_yaw_forward_mask = (
            zero_yaw_planar_mask
            & (~lateral_mask)
            & (cmd_vx > threshold))
        zero_yaw_reverse_mask = (
            zero_yaw_planar_mask
            & (~lateral_mask)
            & (cmd_vx < -threshold))
        ones = torch.ones_like(cmd_vx)
        zeros = torch.zeros_like(cmd_vx)
        phase_sign = torch.where(reverse_mask, -ones, ones)
        phase_offset = torch.where(
            lateral_mask,
            ones * float(LATERAL_PHASE_OFFSET_RAD),
            zeros,
        )
        forward_phase_offset = (
            float(SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD)
            if bool(SLOPE_FORWARD_AXIS_PROFILE_ENABLED)
            else float(ZERO_YAW_FORWARD_PHASE_OFFSET_RAD)
        )
        phase_offset = torch.where(
            zero_yaw_forward_mask,
            ones * forward_phase_offset,
            phase_offset,
        )
        phase_offset = torch.where(
            zero_yaw_reverse_mask,
            ones * float(ZERO_YAW_REVERSE_PHASE_OFFSET_RAD),
            phase_offset,
        )
        yaw_sign = torch.where(
            abs_yaw >= threshold,
            torch.where(cmd_yaw >= 0.0, ones, -ones),
            ones,
        )
        yaw_sign = torch.where(
            lateral_mask & (cmd_vy < 0.0),
            -ones,
            yaw_sign,
        )
        slide_scale = torch.where(
            yaw_only_mask,
            ones * float(YAW_ONLY_SLIDE_PRIOR_SCALE),
            ones,
        )
        yaw_scale = torch.where(
            yaw_only_mask,
            ones,
            ones,
        )
        yaw_scale = torch.where(
            zero_yaw_planar_mask & lateral_mask,
            ones * float(ZERO_YAW_LATERAL_YAW_PRIOR_SCALE),
            yaw_scale,
        )
        yaw_scale = torch.where(
            zero_yaw_forward_mask,
            ones * float(ZERO_YAW_FORWARD_YAW_PRIOR_SCALE),
            yaw_scale,
        )
        yaw_scale = torch.where(
            zero_yaw_reverse_mask,
            ones * float(ZERO_YAW_REVERSE_YAW_PRIOR_SCALE),
            yaw_scale,
        )
        yaw_trim = torch.where(
            zero_yaw_planar_mask & lateral_mask,
            ones * float(ZERO_YAW_LATERAL_YAW_TRIM) * torch.sign(cmd_vy),
            zeros,
        )
        prior = self._base_gait_prior(phase_sign * phase + phase_offset,
                                      gait_blend)
        prior = torch.cat(
            [
                prior[:, :NUM_SLIDES] * slide_scale,
                prior[:, NUM_SLIDES:] * yaw_sign * yaw_scale + yaw_trim,
            ],
            dim=1,
        )
        prior = torch.clamp(prior, -1.0, 1.0)
        inplace_yaw_prior = self._inplace_yaw_prior(phase, cmd_yaw)
        yaw_only_yaw_scale = torch.where(
            cmd_yaw >= 0.0,
            ones * float(YAW_ONLY_YAW_PRIOR_SCALE),
            ones * float(YAW_RIGHT_ONLY_YAW_PRIOR_SCALE),
        )
        inplace_yaw_prior = torch.cat(
            [
                inplace_yaw_prior[:, :NUM_SLIDES]
                * float(YAW_ONLY_SLIDE_PRIOR_SCALE),
                inplace_yaw_prior[:, NUM_SLIDES:] * yaw_only_yaw_scale,
            ],
            dim=1,
        )
        inplace_yaw_prior = torch.clamp(inplace_yaw_prior, -1.0, 1.0)
        lateral_prior = self._lateral_prior(phase, cmd_vy)
        prior = torch.where(lateral_mask, lateral_prior, prior)
        return torch.where(yaw_only_mask, inplace_yaw_prior, prior)

    def _continuous_directional_prior(
            self, phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw):
        ones = torch.ones_like(cmd_vx)
        zeros = torch.zeros_like(cmd_vx)
        forward_w = torch.clamp(cmd_vx, min=0.0)
        reverse_w = torch.clamp(-cmd_vx, min=0.0)
        lateral_left_w = torch.clamp(cmd_vy, min=0.0)
        lateral_right_w = torch.clamp(-cmd_vy, min=0.0)
        yaw_left_w = torch.clamp(cmd_yaw, min=0.0)
        yaw_right_w = torch.clamp(-cmd_yaw, min=0.0)
        total_w = (
            forward_w + reverse_w + lateral_left_w + lateral_right_w
            + yaw_left_w + yaw_right_w)
        blended = (
            forward_w * self._dominant_directional_prior(
                phase, gait_blend, ones, zeros, zeros)
            + reverse_w * self._dominant_directional_prior(
                phase, gait_blend, -ones, zeros, zeros)
            + lateral_left_w * self._dominant_directional_prior(
                phase, gait_blend, zeros, ones, zeros)
            + lateral_right_w * self._dominant_directional_prior(
                phase, gait_blend, zeros, -ones, zeros)
            + yaw_left_w * self._dominant_directional_prior(
                phase, gait_blend, zeros, zeros, ones)
            + yaw_right_w * self._dominant_directional_prior(
                phase, gait_blend, zeros, zeros, -ones))
        blended = blended / torch.clamp(total_w, min=1e-9)
        base = self._base_gait_prior(phase, gait_blend)
        return torch.where(total_w > 1e-9, blended, base)

    def _componentwise_mixed_directional_prior(
            self, phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw):
        ones = torch.ones_like(cmd_vx)
        zeros = torch.zeros_like(cmd_vx)
        abs_vx = torch.abs(cmd_vx)
        abs_vy = torch.abs(cmd_vy)
        abs_yaw = torch.abs(cmd_yaw)
        max_axis = torch.clamp(
            torch.maximum(torch.maximum(abs_vx, abs_vy), abs_yaw),
            min=1e-9,
        )
        threshold = float(DIRECTIONAL_PRIOR_THRESHOLD)
        vx_gain = torch.where(
            abs_vx >= threshold,
            torch.clamp(abs_vx / max_axis, 0.0, 1.0),
            zeros,
        )
        vy_gain = torch.where(
            abs_vy >= threshold,
            torch.clamp(abs_vy / max_axis, 0.0, 1.0),
            zeros,
        )
        yaw_gain = torch.where(
            abs_yaw >= threshold,
            torch.clamp(abs_yaw / max_axis, 0.0, 1.0),
            zeros,
        )
        forward_prior = self._dominant_directional_prior(
            phase, gait_blend, ones, zeros, zeros)
        reverse_prior = self._dominant_directional_prior(
            phase, gait_blend, -ones, zeros, zeros)
        axial_prior = torch.where(cmd_vx >= 0.0, forward_prior, reverse_prior)
        lateral_prior = self._lateral_prior(phase, cmd_vy)
        yaw_left_prior = self._dominant_directional_prior(
            phase, gait_blend, zeros, zeros, ones)
        yaw_right_prior = self._dominant_directional_prior(
            phase, gait_blend, zeros, zeros, -ones)
        yaw_prior = torch.where(cmd_yaw >= 0.0, yaw_left_prior, yaw_right_prior)
        slides = (
            float(MIXED_AXIAL_SLIDE_GAIN)
            * vx_gain
            * axial_prior[:, :NUM_SLIDES]
            + float(MIXED_LATERAL_SLIDE_GAIN)
            * vy_gain
            * lateral_prior[:, :NUM_SLIDES]
            + float(MIXED_YAW_SLIDE_GAIN)
            * yaw_gain
            * yaw_prior[:, :NUM_SLIDES]
        )
        yaws = (
            float(MIXED_AXIAL_YAW_GAIN)
            * vx_gain
            * axial_prior[:, NUM_SLIDES:]
            + float(MIXED_LATERAL_YAW_GAIN)
            * vy_gain
            * lateral_prior[:, NUM_SLIDES:]
            + float(MIXED_YAW_YAW_GAIN)
            * yaw_gain
            * yaw_prior[:, NUM_SLIDES:]
        )
        prior = torch.cat([slides, yaws], dim=1)
        if bool(MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM):
            axis_norm = torch.sqrt(
                vx_gain * vx_gain + vy_gain * vy_gain + yaw_gain * yaw_gain)
            prior = prior / torch.clamp(axis_norm, min=1.0)
        return torch.clamp(prior, -1.0, 1.0)

    def _split_channel_mixed_planar_prior(
            self, phase, gait_blend, cmd_vx, cmd_vy):
        ones = torch.ones_like(cmd_vx)
        zeros = torch.zeros_like(cmd_vx)
        abs_vx = torch.abs(cmd_vx)
        abs_vy = torch.abs(cmd_vy)
        max_axis = torch.clamp(
            torch.maximum(abs_vx, abs_vy),
            min=float(DIRECTIONAL_PRIOR_THRESHOLD),
        )
        vx_gain = torch.clamp(abs_vx / max_axis, 0.0, 1.0)
        vy_gain = torch.clamp(abs_vy / max_axis, 0.0, 1.0)
        forward_prior = self._dominant_directional_prior(
            phase, gait_blend, ones, zeros, zeros)
        reverse_prior = self._dominant_directional_prior(
            phase, gait_blend, -ones, zeros, zeros)
        axial_prior = torch.where(cmd_vx >= 0.0, forward_prior, reverse_prior)
        lateral_prior = self._lateral_prior(phase, cmd_vy)
        if bool(MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED):
            slides = (
                float(MIXED_AXIAL_SLIDE_GAIN)
                * vx_gain
                * axial_prior[:, :NUM_SLIDES]
                + float(MIXED_LATERAL_SLIDE_GAIN)
                * vy_gain
                * lateral_prior[:, :NUM_SLIDES]
            )
            yaws = (
                float(MIXED_AXIAL_YAW_GAIN)
                * vx_gain
                * axial_prior[:, NUM_SLIDES:]
                + float(MIXED_LATERAL_YAW_GAIN)
                * vy_gain
                * lateral_prior[:, NUM_SLIDES:]
            )
            prior = torch.cat([slides, yaws], dim=1)
            if bool(MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM):
                axis_norm = torch.sqrt(vx_gain * vx_gain + vy_gain * vy_gain)
                prior = prior / torch.clamp(axis_norm, min=1.0)
            return torch.clamp(prior, -1.0, 1.0)
        return torch.clamp(
            torch.cat(
                [
                    vx_gain * axial_prior[:, :NUM_SLIDES],
                    vy_gain * lateral_prior[:, NUM_SLIDES:],
                ],
                dim=1,
            ),
            -1.0,
            1.0,
        )

    def _slope_mixed_planar_prior(self, phase, gait_blend, cmd_vx, cmd_vy):
        abs_vx = torch.abs(cmd_vx)
        abs_vy = torch.abs(cmd_vy)
        max_axis = torch.clamp(
            torch.maximum(abs_vx, abs_vy),
            min=float(DIRECTIONAL_PRIOR_THRESHOLD),
        )
        vx_gain = torch.clamp(abs_vx / max_axis, 0.0, 1.0)
        vy_gain = torch.clamp(abs_vy / max_axis, 0.0, 1.0)
        ones = torch.ones_like(cmd_vx)
        zeros = torch.zeros_like(cmd_vx)
        forward_prior = self._dominant_directional_prior(
            phase, gait_blend, ones, zeros, zeros)
        reverse_prior = self._dominant_directional_prior(
            phase, gait_blend, -ones, zeros, zeros)
        axial_prior = torch.where(cmd_vx >= 0.0, forward_prior, reverse_prior)
        if bool(SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED):
            positive_vx_mask = cmd_vx >= float(DIRECTIONAL_PRIOR_THRESHOLD)
            positive_vx_axial_prior = self._base_gait_prior(
                phase
                + float(
                    SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD),
                gait_blend,
            )
            positive_vx_axial_prior = torch.cat(
                [
                    positive_vx_axial_prior[:, :NUM_SLIDES]
                    * float(
                        SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN),
                    positive_vx_axial_prior[:, NUM_SLIDES:]
                    * float(ZERO_YAW_FORWARD_YAW_PRIOR_SCALE),
                ],
                dim=1,
            )
            axial_prior = torch.where(
                positive_vx_mask, positive_vx_axial_prior, axial_prior)
        lateral_prior = self._lateral_prior(phase, cmd_vy)
        slides = (
            float(SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN)
            * vx_gain
            * axial_prior[:, :NUM_SLIDES]
            + float(SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN)
            * vy_gain
            * lateral_prior[:, :NUM_SLIDES]
        )
        yaws = (
            float(SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN)
            * vy_gain
            * lateral_prior[:, NUM_SLIDES:]
        )
        if bool(SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED):
            positive_vx_mask = cmd_vx >= float(DIRECTIONAL_PRIOR_THRESHOLD)
            positive_vx_yaws = (
                float(SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT)
                * yaws
                + float(SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN)
                * vx_gain
                * axial_prior[:, NUM_SLIDES:]
            )
            yaws = torch.where(positive_vx_mask, positive_vx_yaws, yaws)
        prior = torch.cat([slides, yaws], dim=1)
        if bool(MIXED_COMPOSITION_NORMALIZE_BY_AXIS_NORM):
            axis_norm = torch.sqrt(vx_gain * vx_gain + vy_gain * vy_gain)
            prior = prior / torch.clamp(axis_norm, min=1.0)
        return torch.clamp(prior, -1.0, 1.0)

    def forward(self, raw_obs):
        if raw_obs.dim() == 1:
            raw_obs = raw_obs.unsqueeze(0)
        raw_obs = raw_obs.to(dtype=torch.float32)
        obs = (raw_obs - self.obs_mean) * torch.rsqrt(
            self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        policy_action = torch.clamp(self.action_net(latent_pi), -1.0, 1.0)
        residual = policy_action[:, :NUM_SLIDES + NUM_YAWS]
        learned_gait_blend = torch.clamp(
            0.5
            + 0.5
            * float(GAIT_GATE_ACTION_GAIN)
            * policy_action[:, NUM_SLIDES + NUM_YAWS:],
            0.0,
            1.0,
        )
        cmd_vx = raw_obs[:, 0:1]
        cmd_vy = raw_obs[:, 1:2]
        cmd_yaw = raw_obs[:, 2:3]
        abs_vx = torch.abs(cmd_vx)
        abs_vy = torch.abs(cmd_vy)
        abs_yaw = torch.abs(cmd_yaw)
        ones = torch.ones_like(cmd_vx)
        threshold = float(DIRECTIONAL_PRIOR_THRESHOLD)
        slope_forward_axis_mask = (
            (cmd_vx >= threshold)
            & (abs_yaw < threshold)
        )
        mixed_center = ones * float(COMMAND_GATE_MIXED_CENTER)
        axial_alpha = torch.clamp(
            (abs_vx - threshold)
            / max(
                float(COMMAND_GATE_WORM_FAST_THRESHOLD) - threshold,
                1e-6,
            ),
            0.0,
            1.0,
        )
        worm_center = (
            float(COMMAND_GATE_WORM_CENTER_SLOW)
            + axial_alpha
            * (
                float(COMMAND_GATE_WORM_CENTER_FAST)
                - float(COMMAND_GATE_WORM_CENTER_SLOW)
            )
        )
        lateral_center = ones * float(COMMAND_GATE_LATERAL_CENTER)
        yaw_center = ones * float(COMMAND_GATE_YAW_CENTER)
        yaw_center_mask = (
            (abs_yaw >= threshold)
            & (torch.maximum(abs_vx, abs_vy) < threshold)
        )
        lateral_center_mask = (
            (abs_vy >= threshold)
            & (abs_vy > abs_vx)
            & (abs_vy >= abs_yaw)
        )
        axial_center_mask = (
            (abs_vx >= threshold)
            & (abs_vy < threshold)
            & (abs_yaw < threshold)
        )
        command_center = torch.where(axial_center_mask, worm_center, mixed_center)
        command_center = torch.where(
            lateral_center_mask, lateral_center, command_center)
        command_center = torch.where(yaw_center_mask, yaw_center, command_center)
        gait_blend = torch.clamp(
            command_center
            + 2.0
            * float(COMMAND_GATE_CENTER_RESIDUAL_RANGE)
            * (learned_gait_blend - 0.5),
            0.0,
            1.0,
        )
        if bool(SLOPE_FORWARD_AXIS_PROFILE_ENABLED):
            gait_blend = torch.where(
                slope_forward_axis_mask,
                torch.maximum(
                    gait_blend,
                    ones * float(SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR),
                ),
                gait_blend,
            )
        abs_vx = torch.abs(cmd_vx)
        abs_vy = torch.abs(cmd_vy)
        abs_yaw = torch.abs(cmd_yaw)
        threshold = float(DIRECTIONAL_PRIOR_THRESHOLD)
        lateral_mask = (
            (abs_vy >= threshold)
            & (abs_vy > abs_vx)
            & (abs_vy >= abs_yaw))
        yaw_only_mask = (
            (abs_yaw >= threshold)
            & (torch.maximum(abs_vx, abs_vy) < threshold))
        mixed_composition_mask = (
            (
                (abs_vx >= threshold).to(dtype=torch.int32)
                + (abs_vy >= threshold).to(dtype=torch.int32)
                + (abs_yaw >= threshold).to(dtype=torch.int32)
            )
            >= 2
        )
        mixed_planar_mask = (
            (abs_vx >= threshold)
            & (abs_vy >= threshold)
            & (abs_yaw < threshold)
        )
        ones = torch.ones_like(cmd_vx)
        phase = torch.atan2(raw_obs[:, 78:79], raw_obs[:, 79:80])
        if bool(MIXED_PLANAR_SPLIT_PRIOR_ENABLED):
            dominant_prior = self._dominant_directional_prior(
                phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw)
            if bool(MIXED_COMMAND_COMPOSITION_ENABLED):
                mixed_prior = self._componentwise_mixed_directional_prior(
                    phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw)
                prior = torch.where(mixed_composition_mask, mixed_prior,
                                    dominant_prior)
            else:
                prior = dominant_prior
            split_prior = self._split_channel_mixed_planar_prior(
                phase, gait_blend, cmd_vx, cmd_vy)
            prior = torch.where(mixed_planar_mask, split_prior, prior)
        elif bool(MIXED_COMMAND_COMPOSITION_ENABLED):
            dominant_prior = self._dominant_directional_prior(
                phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw)
            mixed_prior = self._componentwise_mixed_directional_prior(
                phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw)
            prior = torch.where(mixed_composition_mask, mixed_prior,
                                dominant_prior)
        elif bool(USE_CONTINUOUS_VECTOR_PRIOR_BLEND):
            prior = self._continuous_directional_prior(
                phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw)
        else:
            prior = self._dominant_directional_prior(
                phase, gait_blend, cmd_vx, cmd_vy, cmd_yaw)
        if bool(SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED):
            slope_mixed_prior = self._slope_mixed_planar_prior(
                phase, gait_blend, cmd_vx, cmd_vy)
            prior = torch.where(mixed_planar_mask, slope_mixed_prior, prior)
        forward = torch.clamp(cmd_vx, min=0.0)
        non_forward = torch.maximum(
            torch.maximum(torch.clamp(-cmd_vx, min=0.0), torch.abs(cmd_vy)),
            torch.abs(cmd_yaw),
        )
        forward_share = forward / torch.clamp(
            forward + non_forward, min=1e-9)
        floor = torch.where(
            lateral_mask,
            ones * float(LATERAL_PRIOR_SCALE_FLOOR),
            ones * float(REVERSE_PRIOR_SCALE_FLOOR),
        )
        floor = torch.where(
            yaw_only_mask,
            ones * float(YAW_ONLY_PRIOR_SCALE_FLOOR),
            floor,
        )
        mixed_yaw_mask = (
            (abs_yaw >= threshold)
            & (torch.maximum(abs_vx, abs_vy) >= threshold)
        )
        if bool(MIXED_COMMAND_COMPOSITION_ENABLED):
            floor = torch.where(
                mixed_planar_mask,
                ones * float(MIXED_PLANAR_PRIOR_SCALE_FLOOR),
                floor,
            )
            floor = torch.where(
                mixed_yaw_mask,
                ones * float(MIXED_YAW_PRIOR_SCALE_FLOOR),
                floor,
            )
        conditioned_prior_scale = torch.where(
            non_forward <= 1e-9,
            torch.ones_like(forward_share),
            floor + (1.0 - floor) * forward_share,
        )
        if not bool(MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED):
            prior_authority_multiplier = ones
        else:
            prior_authority_multiplier = torch.where(
                mixed_planar_mask,
                ones * float(MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT),
                ones,
            )
        if bool(SLOPE_FORWARD_AXIS_PROFILE_ENABLED):
            prior_authority_multiplier = torch.where(
                slope_forward_axis_mask,
                ones * float(SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT),
                prior_authority_multiplier,
            )
        residual_multiplier = torch.ones_like(cmd_vx)
        if bool(COMMAND_CONDITIONED_RESIDUAL_AUTHORITY_ENABLED):
            residual_multiplier = torch.where(
                yaw_only_mask,
                ones * float(YAW_ONLY_RESIDUAL_SCALE_MULT),
                residual_multiplier,
            )
            residual_multiplier = torch.where(
                mixed_planar_mask,
                ones * float(
                    MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT
                    if bool(MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED)
                    else MIXED_PLANAR_RESIDUAL_SCALE_MULT),
                residual_multiplier,
            )
            residual_multiplier = torch.where(
                mixed_yaw_mask,
                ones * float(MIXED_YAW_RESIDUAL_SCALE_MULT),
                residual_multiplier,
            )
        command_mag = torch.maximum(
            torch.maximum(torch.abs(cmd_vx), torch.abs(cmd_vy)),
            torch.abs(cmd_yaw),
        )
        command_activity_scale = torch.where(
            command_mag <= 1e-9,
            torch.zeros_like(command_mag),
            torch.clamp(
                float(COMMAND_ACTIVITY_MIN_ACTIVE_SCALE)
                + (1.0 - float(COMMAND_ACTIVITY_MIN_ACTIVE_SCALE))
                * command_mag,
                0.0,
                1.0,
            ),
        )
        command_activity_scale = torch.where(
            yaw_only_mask,
            torch.ones_like(command_activity_scale),
            command_activity_scale,
        )
        actions = (
            self.gait_prior_scale
            * conditioned_prior_scale
            * prior_authority_multiplier
            * prior
            + self.policy_residual_scale * residual_multiplier * residual)
        actions = command_activity_scale * actions
        return torch.clamp(actions, -1.0, 1.0)


def default_run_dir(terrain, gait_mode):
    return os.path.join(PROJECT_ROOT, "runs",
                        f"worm_v6_ppo_{terrain}_{gait_mode}")


def default_model_path(run_dir):
    for name in ("best_model.zip", "final_model.zip"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(run_dir, "best_model.zip")


def find_vecnormalize(model_path):
    candidates = [
        model_path.replace(".zip", "_vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_path), "best_model_vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_path), "final_model_vecnormalize.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def gait_blend_for(mode, override):
    if override is not None:
        return float(np.clip(override, 0.0, 1.0))
    if mode == "worm":
        return 0.0
    if mode == "snake":
        return 1.0
    if mode == "mixed":
        return 0.5
    return 0.5


def layout_to_json(layout):
    return {key: [value.start, value.stop] for key, value in layout.items()}


def load_vecnormalize_stats(vec_path, terrain, gait_mode, gait_blend, obs_dim):
    if vec_path is None:
        return {
            "mean": np.zeros(obs_dim, dtype=np.float32),
            "var": np.ones(obs_dim, dtype=np.float32),
            "epsilon": 1e-8,
            "clip_obs": 10.0,
            "path": None,
        }

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from worm_env_v6 import WormEnvV6

    vec_env = DummyVecEnv([lambda: WormEnvV6(
        terrain=terrain, gait_mode=gait_mode, gait_blend=gait_blend)])
    vec_norm = VecNormalize.load(vec_path, vec_env)
    try:
        return {
            "mean": vec_norm.obs_rms.mean.astype(np.float32),
            "var": vec_norm.obs_rms.var.astype(np.float32),
            "epsilon": float(vec_norm.epsilon),
            "clip_obs": float(vec_norm.clip_obs),
            "path": vec_path,
        }
    finally:
        vec_norm.close()


def write_config(path, config):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_policy(args):
    sys.path.insert(0, SCRIPT_DIR)
    from stable_baselines3 import PPO
    from observation_contract_v6 import attach_contract_to_config
    from validate_hardware_log_v6 import observation_columns
    from worm_env_v6 import (
        CMD_VX_RANGE,
        CMD_VY_RANGE,
        CMD_YAW_RANGE,
        CTRL_DT,
        NUM_ACTUATORS,
        NUM_SLIDES,
        OBS_DIM,
        OBS_LAYOUT,
        PERISTALTIC_ACTUATION_PERIOD_S,
        PHASE_FREQ,
    )

    run_dir = args.run_dir or default_run_dir(args.terrain, args.gait_mode)
    model_path = args.model or default_model_path(run_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "record", "v6", "deploy_bundles",
        f"{args.terrain}_{args.gait_mode}")
    gait_blend = gait_blend_for(args.gait_mode, args.gait_blend)
    vec_path = args.vecnormalize or find_vecnormalize(model_path)
    norm = load_vecnormalize_stats(
        vec_path, args.terrain, args.gait_mode, gait_blend, OBS_DIM)

    model = PPO.load(model_path, device="cpu")
    model.policy.eval()

    actor = DeployablePPOActor(
        model.policy,
        norm["mean"],
        norm["var"],
        norm["epsilon"],
        norm["clip_obs"],
    )
    actor.eval()

    dummy_raw = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        actor_action = actor(dummy_raw).cpu().numpy()
        dummy_norm = (
            (dummy_raw.cpu().numpy() - norm["mean"]) /
            np.sqrt(norm["var"] + norm["epsilon"]))
        dummy_norm = np.clip(dummy_norm, -norm["clip_obs"], norm["clip_obs"])
        sb3_action, _ = model.predict(dummy_norm, deterministic=True)
        residual, learned_gait_blend = policy_action_to_residual_and_gait_blend(
            sb3_action[0])
        command = dummy_raw[0, 0:3].cpu().numpy()
        gait_blend = command_conditioned_gait_blend(
            learned_gait_blend, command)
        expected_action = compose_deployable_action(
            residual,
            phase=phase_from_clock(dummy_raw[0, 78], dummy_raw[0, 79]),
            gait_blend=gait_blend,
            command=command,
        )[None, :]
        max_diff = float(np.max(np.abs(actor_action - expected_action)))
        if max_diff > args.max_export_diff:
            raise RuntimeError(
                f"Exported deployable actor mismatch: max_diff={max_diff:.3g}")

    os.makedirs(out_dir, exist_ok=True)
    actor_path = os.path.join(out_dir, "policy_actor.pt")
    traced = torch.jit.trace(actor, dummy_raw, check_trace=True)
    traced.save(actor_path)

    action_columns = [f"action_{i:02d}" for i in range(NUM_ACTUATORS)]
    config = {
        "format_version": 1,
        "model_type": "sb3_ppo_deterministic_actor_torchscript",
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": gait_blend,
        "obs_dim": OBS_DIM,
        "obs_layout": layout_to_json(OBS_LAYOUT),
        "observation_columns": observation_columns(),
        "action_dim": NUM_ACTUATORS,
        "policy_action_dim": POLICY_ACTION_DIM,
        "action_columns": action_columns,
        "action_range": [-1.0, 1.0],
        "action_adapter": action_adapter_contract(),
        "action_mapping": action_mapping_config(),
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "command_ranges": {
            "cmd_vx_m_s": list(CMD_VX_RANGE),
            "cmd_vy_m_s": list(CMD_VY_RANGE),
            "cmd_yaw_rad_s": list(CMD_YAW_RANGE),
        },
        "control_timing": {
            "control_dt_s": CTRL_DT,
            "control_rate_hz": 1.0 / CTRL_DT,
            "peristaltic_actuation_period_s": (
                PERISTALTIC_ACTUATION_PERIOD_S),
            "phase_freq_hz": PHASE_FREQ,
        },
        "normalization": {
            "mean": norm["mean"].tolist(),
            "var": norm["var"].tolist(),
            "epsilon": norm["epsilon"],
            "clip_obs": norm["clip_obs"],
            "source": norm["path"],
        },
        "source_model": model_path,
        "torchscript_actor": actor_path,
        "export_validation": {
            "dummy_obs_max_abs_diff_vs_sb3_predict": max_diff,
        },
    }
    config = attach_contract_to_config(config)
    config_path = os.path.join(out_dir, "deploy_config.json")
    write_config(config_path, config)

    result = {
        "bundle_dir": out_dir,
        "actor": actor_path,
        "config": config_path,
        "source_model": model_path,
        "vecnormalize": norm["path"],
        "max_export_diff": max_diff,
    }
    print(json.dumps(result, indent=2))
    return result


def default_actions_path(input_csv):
    root, ext = os.path.splitext(input_csv)
    return f"{root}_actions{ext or '.csv'}"


def action_mapping_fields(config):
    action_dim = int(config["action_dim"])
    defaults = action_mapping_config()
    action_mapping = {**defaults, **config.get("action_mapping", {})}
    slide_start, slide_stop = action_mapping.get("slide_indices", [0, 0])
    yaw_start, yaw_stop = action_mapping.get("yaw_indices", [slide_stop, action_dim])
    slide_range_m = float(action_mapping.get("slide_range_m", 1.0))
    yaw_range_rad = float(action_mapping.get("yaw_range_rad", 1.0))
    slide_min_m = float(action_mapping.get("slide_min_m", -slide_range_m))
    slide_max_m = float(action_mapping.get("slide_max_m", slide_range_m))
    yaw_min_rad = float(action_mapping.get("yaw_min_rad", -yaw_range_rad))
    yaw_max_rad = float(action_mapping.get("yaw_max_rad", yaw_range_rad))
    slide_target_cols = [
        f"slide_target_m_{i - slide_start:02d}"
        for i in range(slide_start, slide_stop)
    ]
    yaw_target_cols = [
        f"yaw_target_rad_{i - yaw_start:02d}"
        for i in range(yaw_start, yaw_stop)
    ]
    return {
        "slide_start": slide_start,
        "slide_stop": slide_stop,
        "yaw_start": yaw_start,
        "yaw_stop": yaw_stop,
        "slide_range_m": slide_range_m,
        "slide_min_m": slide_min_m,
        "slide_max_m": slide_max_m,
        "yaw_range_rad": yaw_range_rad,
        "yaw_min_rad": yaw_min_rad,
        "yaw_max_rad": yaw_max_rad,
        "slide_target_cols": slide_target_cols,
        "yaw_target_cols": yaw_target_cols,
    }


def action_targets(config, action):
    mapping = action_mapping_fields(config)
    action = np.asarray(action, dtype=np.float32)
    return {
        "slide_targets_m": np.clip(
            action[mapping["slide_start"]:mapping["slide_stop"]] *
            mapping["slide_range_m"],
            mapping["slide_min_m"],
            mapping["slide_max_m"],
        ).astype(np.float32),
        "yaw_targets_rad": np.clip(
            action[mapping["yaw_start"]:mapping["yaw_stop"]] *
            mapping["yaw_range_rad"],
            mapping["yaw_min_rad"],
            mapping["yaw_max_rad"],
        ).astype(np.float32),
        "mapping": mapping,
    }


def replay_csv(args):
    config_path = args.config or os.path.join(args.bundle_dir, "deploy_config.json")
    actor_path = args.actor or os.path.join(args.bundle_dir, "policy_actor.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"TorchScript actor not found: {actor_path}")

    config = read_config(config_path)
    obs_cols = config["observation_columns"]
    action_cols = config["action_columns"]
    obs_dim = int(config["obs_dim"])
    action_dim = int(config["action_dim"])
    output_csv = args.output_csv or default_actions_path(args.input_csv)
    mapping = action_mapping_fields(config)
    slide_target_cols = mapping["slide_target_cols"]
    yaw_target_cols = mapping["yaw_target_cols"]

    actor = torch.jit.load(actor_path, map_location="cpu")
    actor.eval()

    rows = []
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        missing = [col for col in obs_cols if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing observation columns: {missing}")
        for row_index, row in enumerate(reader):
            obs = np.array([float(row[col]) for col in obs_cols], dtype=np.float32)
            if obs.shape != (obs_dim,) or not np.all(np.isfinite(obs)):
                raise ValueError(f"Invalid observation at row {row_index + 1}")
            rows.append((row_index, row, obs))

    if not rows:
        raise ValueError("Input CSV contains no data rows")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    metadata_cols = [
        col for col in ("time_s", "terrain", "mode", "video_file")
        if col in rows[0][1]
    ]
    fieldnames = [
        "row_index", *metadata_cols, *action_cols,
        *slide_target_cols, *yaw_target_cols,
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start:start + args.batch_size]
                obs_batch = np.stack([item[2] for item in chunk], axis=0)
                actions = actor(torch.from_numpy(obs_batch)).cpu().numpy()
                if actions.shape != (len(chunk), action_dim):
                    raise RuntimeError(
                        f"Actor returned shape {actions.shape}, "
                        f"expected {(len(chunk), action_dim)}")
                if not np.all(np.isfinite(actions)):
                    raise RuntimeError("Actor returned non-finite actions")
                for (row_index, row, _), action in zip(chunk, actions):
                    targets = action_targets(config, action)
                    out = {"row_index": row_index}
                    for col in metadata_cols:
                        out[col] = row[col]
                    for col, value in zip(action_cols, action):
                        out[col] = f"{float(value):.8f}"
                    for col, value in zip(
                            slide_target_cols, targets["slide_targets_m"]):
                        out[col] = f"{float(value):.8f}"
                    for col, value in zip(
                            yaw_target_cols, targets["yaw_targets_rad"]):
                        out[col] = f"{float(value):.8f}"
                    writer.writerow(out)

    metrics = {
        "input_csv": args.input_csv,
        "output_csv": output_csv,
        "rows": len(rows),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "slide_target_columns": slide_target_cols,
        "yaw_target_columns": yaw_target_cols,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export or replay a deployable Worm V6 policy")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Export PPO policy to a deploy bundle")
    export.add_argument("--model", default=None, help="Path to PPO .zip model")
    export.add_argument("--vecnormalize", default=None,
                        help="Path to VecNormalize .pkl stats")
    export.add_argument("--run-dir", default=None, help="Run directory override")
    export.add_argument("--terrain", default="flat",
                        choices=["flat", "sand", "slope", "rough", "steps",
                                 "channel"])
    export.add_argument("--gait-mode", default="random",
                        choices=["worm", "snake", "mixed", "random"])
    export.add_argument("--gait-blend", type=float, default=None)
    export.add_argument("--out-dir", default=None)
    export.add_argument("--max-export-diff", type=float, default=1e-5)

    replay = sub.add_parser(
        "replay", help="Replay hardware observation CSV through an export")
    replay.add_argument("--bundle-dir", default=os.path.join(
        PROJECT_ROOT, "record", "v6", "deploy_bundles", "flat_random"))
    replay.add_argument("--config", default=None,
                        help="deploy_config.json override")
    replay.add_argument("--actor", default=None,
                        help="policy_actor.pt override")
    replay.add_argument("--input-csv", required=True,
                        help="Hardware CSV with 80-D observation columns")
    replay.add_argument("--output-csv", default=None,
                        help="Output action CSV")
    replay.add_argument("--batch-size", type=int, default=256)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "export":
        export_policy(args)
    elif args.command == "replay":
        replay_csv(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

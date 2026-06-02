"""
Smoke tests for command-frame V6 evaluation metrics.

All-direction policies must be judged in the commanded body-frame direction.
World -X distance alone is not a valid success metric for reverse or lateral
commands.
"""

import os
import sys
import tempfile
from argparse import Namespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eval_v6 import command_tracking_metrics, find_vecnormalize  # noqa: E402
from scan_command_tracking_v6 import (  # noqa: E402
    build_commands,
    evaluate_command,
    robust_env_kwargs,
    sensor_noise_summary,
    summarize,
    summarize_step_infos,
)


def main():
    lateral = command_tracking_metrics(
        cmd_vx=0.0,
        cmd_vy=0.10,
        cmd_yaw=0.0,
        body_delta_x=0.0,
        body_delta_y=0.20,
        yaw_delta=0.0,
        elapsed_s=2.0,
        success_distance=0.05,
    )
    assert lateral["commanded_planar_distance_m"] == 0.20
    assert lateral["body_vy_m_s"] == 0.10
    assert lateral["planar_success"]
    assert lateral["success"]

    reverse_wrong = command_tracking_metrics(
        cmd_vx=-0.10,
        cmd_vy=0.0,
        cmd_yaw=0.0,
        body_delta_x=0.20,
        body_delta_y=0.0,
        yaw_delta=0.0,
        elapsed_s=2.0,
        success_distance=0.05,
    )
    assert reverse_wrong["commanded_planar_distance_m"] < 0.0
    assert not reverse_wrong["planar_success"]
    assert not reverse_wrong["success"]

    yaw_left = command_tracking_metrics(
        cmd_vx=0.0,
        cmd_vy=0.0,
        cmd_yaw=0.5,
        body_delta_x=0.0,
        body_delta_y=0.0,
        yaw_delta=0.2,
        elapsed_s=2.0,
        success_distance=0.05,
    )
    assert yaw_left["yaw_success"]
    assert yaw_left["success"]

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "worm_v6_ppo_220224_steps.zip")
        vecnorm = os.path.join(
            tmp, "worm_v6_ppo_vecnormalize_220224_steps.pkl")
        open(checkpoint, "wb").close()
        open(vecnorm, "wb").close()
        assert find_vecnormalize(checkpoint) == vecnorm

    assert "policy_residual_scale" in evaluate_command.__code__.co_varnames
    assert "env_kwargs" in evaluate_command.__code__.co_varnames

    robust_args = Namespace(
        encoder_pos_noise=0.01,
        encoder_vel_noise=0.02,
        imu_gravity_noise=0.03,
        imu_gyro_noise=0.04,
        action_delay_steps=1,
        action_saturation=0.90,
        zero_command_activity_floor=0.0,
    )
    env_kwargs = robust_env_kwargs(robust_args)
    assert env_kwargs == {
        "encoder_pos_noise_std": 0.01,
        "encoder_vel_noise_std": 0.02,
        "imu_gravity_noise_std": 0.03,
        "imu_gyro_noise_std": 0.04,
        "action_delay_steps": 1,
        "action_saturation": 0.90,
        "zero_command_activity_floor": 0.0,
    }
    assert sensor_noise_summary(env_kwargs) == {
        "encoder_pos_noise_std": 0.01,
        "encoder_vel_noise_std": 0.02,
        "imu_gravity_noise_std": 0.03,
        "imu_gyro_noise_std": 0.04,
    }

    default_scan_args = Namespace(
        vx_values="-0.10,-0.05,0,0.05,0.10",
        vy_values="-0.075,-0.0375,0,0.0375,0.075",
        yaw_values="-0.125,-0.10,0,0.10,0.125",
        include_forward_yaw=True,
        forward_yaw_vx_values="-0.10,0.05,0.10",
        forward_yaw_values="-0.10,0.10",
    )
    assert len(build_commands(default_scan_args)) == 35

    rows = [
        {
            "cmd_vx_m_s": 0.0,
            "cmd_vy_m_s": 0.15,
            "cmd_yaw_rad_s": 0.0,
            "body_vx_m_s": 0.02,
            "body_vy_m_s": 0.04,
            "yaw_rate_rad_s": 0.05,
            "vx_error_m_s": 0.02,
            "vy_error_m_s": -0.11,
            "planar_error_m_s": 0.11,
            "yaw_error_rad_s": 0.05,
            "off_axis_speed_m_s": 0.02,
            "planar_sign_ok": True,
            "yaw_sign_ok": True,
        },
        {
            "cmd_vx_m_s": 0.0,
            "cmd_vy_m_s": -0.15,
            "cmd_yaw_rad_s": 0.0,
            "body_vx_m_s": 0.01,
            "body_vy_m_s": -0.04,
            "yaw_rate_rad_s": -0.04,
            "vx_error_m_s": 0.01,
            "vy_error_m_s": 0.11,
            "planar_error_m_s": 0.11,
            "yaw_error_rad_s": 0.04,
            "off_axis_speed_m_s": 0.01,
            "planar_sign_ok": True,
            "yaw_sign_ok": True,
        },
        {
            "cmd_vx_m_s": 0.0,
            "cmd_vy_m_s": 0.0,
            "cmd_yaw_rad_s": 0.0,
            "body_vx_m_s": 0.0,
            "body_vy_m_s": 0.0,
            "yaw_rate_rad_s": 0.0,
            "vx_error_m_s": 0.0,
            "vy_error_m_s": 0.0,
            "planar_error_m_s": 0.0,
            "yaw_error_rad_s": 0.0,
            "off_axis_speed_m_s": 0.0,
            "planar_sign_ok": True,
            "yaw_sign_ok": True,
        },
    ]
    scan_summary = summarize(rows)
    assert scan_summary["fixed_lateral_left_body_vy_m_s"] == 0.04
    assert scan_summary["fixed_lateral_right_body_vy_m_s"] == -0.04
    assert scan_summary["fixed_lateral_speed_gate_passed"]
    assert scan_summary["fixed_lateral_strict_gate_passed"]
    assert scan_summary["vy_error_exceed_count"] == 2
    assert scan_summary["planar_error_exceed_count"] == 2
    assert scan_summary["wrong_planar_sign_count"] == 0

    rows[1]["body_vy_m_s"] = -0.02
    failed_summary = summarize(rows)
    assert not failed_summary["fixed_lateral_speed_gate_passed"]
    assert not failed_summary["fixed_lateral_strict_gate_passed"]

    telemetry = summarize_step_infos([
        {
            "gait_blend": 0.20,
            "learned_gait_blend": 0.30,
            "prior_component_l2": 1.0,
            "residual_component_l2": 0.2,
            "reward_component_tracking_cost": 3.0,
        },
        {
            "gait_blend": 0.40,
            "learned_gait_blend": 0.50,
            "prior_component_l2": 2.0,
            "residual_component_l2": 0.6,
            "reward_component_tracking_cost": 1.0,
        },
    ])
    assert telemetry["mean_gait_blend"] == 0.30
    assert telemetry["final_gait_blend"] == 0.40
    assert telemetry["mean_prior_component_l2"] == 1.50
    assert telemetry["mean_residual_component_l2"] == 0.40
    assert telemetry["mean_reward_component_tracking_cost"] == 2.00

    print("omni eval metrics checks passed")


if __name__ == "__main__":
    main()

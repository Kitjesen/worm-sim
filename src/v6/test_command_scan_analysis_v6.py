import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from analyze_command_scan_v6 import analyze_scan_payload, classify_command  # noqa: E402


def row(cmd_vx, cmd_vy, cmd_yaw, body_vx, body_vy, yaw_rate):
    return {
        "cmd_vx_m_s": cmd_vx,
        "cmd_vy_m_s": cmd_vy,
        "cmd_yaw_rad_s": cmd_yaw,
        "body_vx_m_s": body_vx,
        "body_vy_m_s": body_vy,
        "yaw_rate_rad_s": yaw_rate,
        "vx_error_m_s": body_vx - cmd_vx,
        "vy_error_m_s": body_vy - cmd_vy,
        "planar_error_m_s": (
            (body_vx - cmd_vx) ** 2 + (body_vy - cmd_vy) ** 2) ** 0.5,
        "yaw_error_rad_s": abs(yaw_rate - cmd_yaw),
        "off_axis_speed_m_s": abs(body_vy) if abs(cmd_vx) > 0 else abs(body_vx),
        "mean_gait_blend": 0.5,
        "mean_prior_component_l2": 1.0,
        "mean_residual_component_l2": 0.1,
        "planar_sign_ok": True,
        "yaw_sign_ok": True,
    }


def main():
    assert classify_command(row(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == "stop"
    assert classify_command(row(0.1, 0.0, 0.0, 0.1, 0.0, 0.0)) == "pure_vx"
    assert classify_command(row(0.0, -0.1, 0.0, 0.0, -0.1, 0.0)) == "pure_vy"
    assert classify_command(row(0.0, 0.0, 0.3, 0.0, 0.0, 0.3)) == "pure_yaw"
    assert classify_command(row(0.1, 0.1, 0.0, 0.1, 0.1, 0.0)) == "mixed_vx_vy"
    assert classify_command(row(0.1, 0.0, 0.3, 0.1, 0.0, 0.3)) == "mixed_vx_yaw"
    assert classify_command(row(0.0, 0.1, -0.3, 0.0, 0.1, -0.3)) == "mixed_vy_yaw"
    assert classify_command(row(0.1, 0.1, 0.3, 0.1, 0.1, 0.3)) == "full_mixed"

    payload = {
        "summary": {
            "planar_rmse_m_s": 0.12,
            "yaw_rmse_rad_s": 0.10,
            "wrong_planar_sign_count": 0,
            "wrong_yaw_sign_count": 0,
            "zero_command_mean_speed_m_s": 0.0,
            "yaw_only_mean_planar_speed_m_s": 0.01,
        },
        "eval_condition": "robust",
        "sensor_noise": {
            "encoder_pos_noise_std": 0.01,
            "encoder_vel_noise_std": 0.02,
            "imu_gravity_noise_std": 0.01,
            "imu_gyro_noise_std": 0.01,
        },
        "action_delay_steps": 1,
        "action_saturation": 0.90,
        "commands": [
            row(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            row(0.25, 0.0, 0.0, 0.18, 0.0, 0.0),
            row(0.0, 0.15, 0.0, 0.0, 0.11, 0.0),
            row(0.0, 0.0, 0.5, 0.0, 0.0, 0.42),
            row(0.125, -0.075, 0.0, 0.02, 0.04, 0.0),
            row(0.125, 0.0, -0.25, 0.02, 0.0, 0.05),
        ],
    }
    analysis = analyze_scan_payload(payload)

    assert analysis["acceptance"]["accepted"] is False
    assert "planar_rmse_m_s" in analysis["acceptance"]["failed_conditions"]
    assert analysis["groups"]["mixed_vx_vy"]["planar_error_exceed_count"] == 1
    assert analysis["groups"]["mixed_vx_yaw"]["yaw_error_exceed_count"] == 1
    assert analysis["groups"]["mixed_vx_yaw"]["telemetry_means"][
        "mean_gait_blend"] == 0.5
    assert analysis["groups"]["mixed_vx_yaw"]["telemetry_means"][
        "mean_residual_component_l2"] == 0.1
    assert analysis["source_scan"]["eval_condition"] == "robust"
    assert analysis["source_scan"]["action_delay_steps"] == 1
    assert analysis["source_scan"]["action_saturation"] == 0.90
    assert analysis["source_scan"]["sensor_noise"]["encoder_pos_noise_std"] == 0.01
    assert analysis["dominant_failure_group"] == "mixed_vx_yaw"
    assert analysis["worst_commands"][0]["class"] == "mixed_vx_yaw"

    print("command scan analysis checks passed")


if __name__ == "__main__":
    main()

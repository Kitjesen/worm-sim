"""
Training-side contracts for the V6 deployable residual policy.

The deployed actor outputs deterministic joint commands. PPO training is still
stochastic, so its exploration scale is part of the experimental contract:
too much residual noise overwhelms the gait prior and makes train-episode
rewards look catastrophically negative even when the deterministic policy walks.
"""

import math


RESIDUAL_EXPLORATION_CONTRACT_VERSION = "low_noise_residual_v1"
DEFAULT_ENT_COEF = 0.005
DEFAULT_LOG_STD_INIT = -2.5
DIRECTIONAL_SELECTION_CONTRACT_VERSION = "omni_tracking_scan_v5"
DIRECTION_MIN_TURN_DELTA_RAD = 0.05
DIRECTION_STRAIGHT_TOLERANCE_RAD = 0.20
DIRECTION_FAILED_SCORE_OFFSET = 1_000_000.0
PLANAR_MIN_PROGRESS_M = 0.02
PLANAR_STATIONARY_TOLERANCE_M = 0.30
REQUIRED_PLANAR_SUCCESS_RATE = 0.85
REQUIRED_YAW_SUCCESS_RATE = 0.70
MAX_STRAIGHT_VIOLATION_COUNT = 1
PLANAR_TRACKING_RMSE_TARGET_M_S = 0.10
YAW_TRACKING_RMSE_TARGET_RAD_S = 0.20
MEAN_OFF_AXIS_SPEED_TARGET_M_S = 0.08
ZERO_COMMAND_SPEED_TARGET_M_S = 0.02
YAW_ONLY_MEAN_PLANAR_SPEED_TARGET_M_S = 0.05
MIXED_COMPONENT_SIGN_MIN_SPEED_M_S = 0.005


def residual_exploration_contract(
        ent_coef=DEFAULT_ENT_COEF,
        log_std_init=DEFAULT_LOG_STD_INIT):
    return {
        "version": RESIDUAL_EXPLORATION_CONTRACT_VERSION,
        "policy": "PPO diagonal Gaussian over normalized residual action",
        "reason": (
            "Residual exploration must stay small enough that the deployable "
            "gait prior remains the dominant initial behavior."),
        "ent_coef": float(ent_coef),
        "log_std_init": float(log_std_init),
        "initial_std": float(math.exp(float(log_std_init))),
    }


def best_selection_contract():
    return {
        "version": DIRECTIONAL_SELECTION_CONTRACT_VERSION,
        "requires_direction_gate": True,
        "min_turn_delta_rad": DIRECTION_MIN_TURN_DELTA_RAD,
        "straight_tolerance_rad": DIRECTION_STRAIGHT_TOLERANCE_RAD,
        "planar_min_progress_m": PLANAR_MIN_PROGRESS_M,
        "planar_stationary_tolerance_m": PLANAR_STATIONARY_TOLERANCE_M,
        "required_planar_success_rate": REQUIRED_PLANAR_SUCCESS_RATE,
        "required_yaw_success_rate": REQUIRED_YAW_SUCCESS_RATE,
        "max_straight_violation_count": MAX_STRAIGHT_VIOLATION_COUNT,
        "requires_continuous_tracking_metrics": True,
        "best_model_requires_tracking_gate": True,
        "progress_best_artifacts": True,
        "planar_tracking_rmse_target_m_s": PLANAR_TRACKING_RMSE_TARGET_M_S,
        "yaw_tracking_rmse_target_rad_s": YAW_TRACKING_RMSE_TARGET_RAD_S,
        "mean_off_axis_speed_target_m_s": MEAN_OFF_AXIS_SPEED_TARGET_M_S,
        "zero_command_speed_target_m_s": ZERO_COMMAND_SPEED_TARGET_M_S,
        "yaw_only_mean_planar_speed_target_m_s": (
            YAW_ONLY_MEAN_PLANAR_SPEED_TARGET_M_S),
        "requires_mixed_vx_vy_component_sign_gate": True,
        "mixed_component_sign_min_speed_m_s": (
            MIXED_COMPONENT_SIGN_MIN_SPEED_M_S),
        "failed_score_offset": DIRECTION_FAILED_SCORE_OFFSET,
    }

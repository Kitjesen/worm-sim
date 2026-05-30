"""
Worm Robot V6 — RL Training Script
====================================
Train the longworm2 robot with PPO using Stable-Baselines3.

Usage:
    # Quick self-test (CPU, 10k steps)
    python train_v6.py --test

    # Full local training (CPU is usually faster for SB3 MLP-PPO)
    python train_v6.py --timesteps 1000000 --device cpu

    # Resume from checkpoint
    python train_v6.py --timesteps 1000000 --device cpu --resume runs/worm_v6_ppo/best_model.zip
"""

import os
import argparse
import json
import shutil
import time
import math
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from training_contract_v6 import (
    DEFAULT_ENT_COEF,
    DEFAULT_LOG_STD_INIT,
    DIRECTION_FAILED_SCORE_OFFSET,
    DIRECTION_MIN_TURN_DELTA_RAD,
    DIRECTION_STRAIGHT_TOLERANCE_RAD,
    DIRECTIONAL_SELECTION_CONTRACT_VERSION,
    MAX_STRAIGHT_VIOLATION_COUNT,
    MEAN_OFF_AXIS_SPEED_TARGET_M_S,
    PLANAR_MIN_PROGRESS_M,
    PLANAR_STATIONARY_TOLERANCE_M,
    PLANAR_TRACKING_RMSE_TARGET_M_S,
    REQUIRED_PLANAR_SUCCESS_RATE,
    REQUIRED_YAW_SUCCESS_RATE,
    YAW_ONLY_MEAN_PLANAR_SPEED_TARGET_M_S,
    YAW_TRACKING_RMSE_TARGET_RAD_S,
    ZERO_COMMAND_SPEED_TARGET_M_S,
    best_selection_contract,
    residual_exploration_contract,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
PAPER_TERRAINS = ("flat", "sand", "slope")
GAIT_MODES = ("worm", "snake", "mixed", "random")
FIXED_GAIT_BLEND_BY_MODE = {
    "worm": 0.0,
    "mixed": 0.5,
    "snake": 1.0,
}
RANDOM_POLICY_EVAL_BLENDS = (0.0, 0.5, 1.0)
DEFAULT_POLICY_NET_ARCH = (512, 256, 128)
DEFAULT_VALUE_NET_ARCH = (512, 256, 128)


def parse_net_arch_arg(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    parts = [p.strip() for p in str(value).replace(";", ",").split(",")]
    arch = [int(p) for p in parts if p]
    if not arch or any(width <= 0 for width in arch):
        raise ValueError(
            f"Invalid network architecture {value!r}; expected e.g. 256,256")
    return arch


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError(
                "--device cuda requested, but PyTorch is not importable"
            ) from exc
        return "cpu"
    cuda_ok = torch.cuda.is_available()
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError(
            "--device cuda requested, but torch.cuda.is_available() is false"
        )
    if requested == "cuda":
        return "cuda"
    return "cuda" if cuda_ok else "cpu"


def make_run_dirs(terrain='flat', gait_mode='random', run_label=None):
    label = run_label if run_label is not None else f"{terrain}_{gait_mode}"
    run_dir  = os.path.join(PROJECT_ROOT, "runs",
                            f"worm_v6_ppo_{label}")
    log_dir  = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    return run_dir, log_dir, ckpt_dir


def infer_vecnormalize_path(model_path):
    """Return the VecNormalize file paired with a final/best/checkpoint model."""
    stem, ext = os.path.splitext(model_path)
    if ext.lower() != ".zip":
        return None

    candidates = [f"{stem}_vecnormalize.pkl"]
    base = os.path.basename(model_path)
    directory = os.path.dirname(model_path)
    marker = "_steps.zip"
    if base.endswith(marker):
        step_part = base[:-len(marker)].split("_")[-1]
        prefix = base[:-len(step_part + marker)]
        candidates.append(os.path.join(
            directory, f"{prefix}vecnormalize_{step_part}_steps.pkl"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def close_float(value, expected, tol=1e-6):
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def dict_float_match(actual, expected):
    if not isinstance(actual, dict):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if not close_float(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def best_eval_schedule(gait_mode, gait_blend=None):
    """Return deterministic command cases used for best-model selection."""
    from worm_env_v6 import CMD_VX_RANGE, CMD_VY_RANGE, CMD_YAW_RANGE

    if gait_blend is not None:
        blends = (float(gait_blend),)
    elif gait_mode == "random":
        blends = (None,)
    else:
        blends = (FIXED_GAIT_BLEND_BY_MODE[gait_mode],)

    cases = (
        ("stop", 0.0, 0.0, 0.0),
        ("slow_forward", float(CMD_VX_RANGE[1]) * 0.25, 0.0, 0.0),
        ("slow_reverse", float(CMD_VX_RANGE[0]) * 0.25, 0.0, 0.0),
        ("forward", float(CMD_VX_RANGE[1]), 0.0, 0.0),
        ("reverse", float(CMD_VX_RANGE[0]), 0.0, 0.0),
        ("lateral_left", 0.0, float(CMD_VY_RANGE[1]), 0.0),
        ("lateral_right", 0.0, float(CMD_VY_RANGE[0]), 0.0),
        (
            "mixed_forward_left",
            float(CMD_VX_RANGE[1]) * 0.5,
            float(CMD_VY_RANGE[1]) * 0.5,
            0.0,
        ),
        (
            "mixed_forward_right",
            float(CMD_VX_RANGE[1]) * 0.5,
            float(CMD_VY_RANGE[0]) * 0.5,
            0.0,
        ),
        (
            "mixed_reverse_left",
            float(CMD_VX_RANGE[0]) * 0.5,
            float(CMD_VY_RANGE[1]) * 0.5,
            0.0,
        ),
        (
            "mixed_reverse_right",
            float(CMD_VX_RANGE[0]) * 0.5,
            float(CMD_VY_RANGE[0]) * 0.5,
            0.0,
        ),
        ("yaw_left", 0.0, 0.0, float(CMD_YAW_RANGE[1])),
        ("yaw_right", 0.0, 0.0, float(CMD_YAW_RANGE[0])),
        ("slow_yaw_left", 0.0, 0.0, float(CMD_YAW_RANGE[1]) * 0.5),
        ("slow_yaw_right", 0.0, 0.0, float(CMD_YAW_RANGE[0]) * 0.5),
        (
            "forward_yaw_left",
            float(CMD_VX_RANGE[1]) * 0.6,
            0.0,
            float(CMD_YAW_RANGE[1]),
        ),
        (
            "forward_yaw_right",
            float(CMD_VX_RANGE[1]) * 0.6,
            0.0,
            float(CMD_YAW_RANGE[0]),
        ),
    )
    schedule = []
    for blend in blends:
        for name, vx, vy, yaw in cases:
            schedule.append({
                "case_name": name,
                "gait_blend": None if blend is None else float(blend),
                "cmd_vx_m_s": float(vx),
                "cmd_vy_m_s": float(vy),
                "cmd_yaw_rad_s": float(yaw),
            })
    return schedule


def best_eval_schedule_fingerprint(schedule):
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directional_eval_steps_from_seconds(seconds):
    if seconds is None:
        return None
    from worm_env_v6 import CTRL_DT
    return max(1, int(math.ceil(float(seconds) / CTRL_DT)))


def wrap_angle_rad(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_direction_status(cmd_yaw_rad_s, yaw_delta_rad,
                         min_turn_delta_rad=DIRECTION_MIN_TURN_DELTA_RAD,
                         straight_tolerance_rad=DIRECTION_STRAIGHT_TOLERANCE_RAD):
    cmd_yaw_rad_s = float(cmd_yaw_rad_s)
    yaw_delta_rad = float(yaw_delta_rad)
    if abs(cmd_yaw_rad_s) < 1e-9:
        status = (
            "straight_ok"
            if abs(yaw_delta_rad) <= straight_tolerance_rad
            else "straight_drift")
    elif abs(yaw_delta_rad) < min_turn_delta_rad:
        status = "weak_turn"
    elif yaw_delta_rad * cmd_yaw_rad_s > 0.0:
        status = "correct"
    else:
        status = "wrong_sign"
    return {
        "cmd_yaw_rad_s": cmd_yaw_rad_s,
        "yaw_delta_rad": yaw_delta_rad,
        "status": status,
    }


def planar_direction_status(cmd_vx_m_s, cmd_vy_m_s, body_delta_x_m,
                            body_delta_y_m, elapsed_s,
                            min_progress_m=PLANAR_MIN_PROGRESS_M,
                            stationary_tolerance_m=PLANAR_STATIONARY_TOLERANCE_M):
    cmd_vec = np.array([cmd_vx_m_s, cmd_vy_m_s], dtype=np.float64)
    delta_vec = np.array([body_delta_x_m, body_delta_y_m], dtype=np.float64)
    cmd_speed = float(np.linalg.norm(cmd_vec))
    elapsed_s = max(float(elapsed_s), 1e-6)
    body_vel = delta_vec / elapsed_s
    velocity_error = float(np.linalg.norm(body_vel - cmd_vec))

    if cmd_speed <= 1e-9:
        drift = float(np.linalg.norm(delta_vec))
        status = (
            "stationary_ok"
            if drift <= stationary_tolerance_m
            else "stationary_drift")
        return {
            "cmd_vx_m_s": float(cmd_vx_m_s),
            "cmd_vy_m_s": float(cmd_vy_m_s),
            "body_delta_x_m": float(body_delta_x_m),
            "body_delta_y_m": float(body_delta_y_m),
            "body_vx_m_s": float(body_vel[0]),
            "body_vy_m_s": float(body_vel[1]),
            "commanded_planar_distance_m": 0.0,
            "off_axis_distance_m": drift,
            "planar_velocity_error_m_s": velocity_error,
            "status": status,
        }

    cmd_unit = cmd_vec / cmd_speed
    projected = float(np.dot(delta_vec, cmd_unit))
    off_axis_vec = delta_vec - projected * cmd_unit
    off_axis = float(np.linalg.norm(off_axis_vec))
    required_progress = min(
        float(min_progress_m),
        max(0.0, 0.25 * cmd_speed * elapsed_s),
    )
    if projected < -1e-6:
        status = "wrong_planar_sign"
    elif projected < required_progress:
        status = "weak_translation"
    else:
        status = "correct"
    return {
        "cmd_vx_m_s": float(cmd_vx_m_s),
        "cmd_vy_m_s": float(cmd_vy_m_s),
        "body_delta_x_m": float(body_delta_x_m),
        "body_delta_y_m": float(body_delta_y_m),
        "body_vx_m_s": float(body_vel[0]),
        "body_vy_m_s": float(body_vel[1]),
        "commanded_planar_distance_m": projected,
        "off_axis_distance_m": off_axis,
        "planar_velocity_error_m_s": velocity_error,
        "status": status,
    }


def directional_eval_summary(eval_schedule, episode_rewards, yaw_deltas_rad,
                             body_deltas_m=None, elapsed_s=None,
                             gait_blend_means=None,
                             learned_gait_blend_means=None):
    if len(eval_schedule) != len(episode_rewards):
        raise ValueError("eval_schedule and episode_rewards length mismatch")
    if len(eval_schedule) != len(yaw_deltas_rad):
        raise ValueError("eval_schedule and yaw_deltas_rad length mismatch")
    if body_deltas_m is None:
        body_deltas_m = [(0.0, 0.0)] * len(eval_schedule)
    if elapsed_s is None:
        elapsed_s = [1.0] * len(eval_schedule)
    if len(eval_schedule) != len(body_deltas_m):
        raise ValueError("eval_schedule and body_deltas_m length mismatch")
    if len(eval_schedule) != len(elapsed_s):
        raise ValueError("eval_schedule and elapsed_s length mismatch")
    if gait_blend_means is None:
        gait_blend_means = [None] * len(eval_schedule)
    if learned_gait_blend_means is None:
        learned_gait_blend_means = [None] * len(eval_schedule)
    if len(eval_schedule) != len(gait_blend_means):
        raise ValueError("eval_schedule and gait_blend_means length mismatch")
    if len(eval_schedule) != len(learned_gait_blend_means):
        raise ValueError(
            "eval_schedule and learned_gait_blend_means length mismatch")

    cases = []
    for (case, reward, yaw_delta, body_delta, case_elapsed_s,
         gait_blend_mean, learned_gait_blend_mean) in zip(
            eval_schedule, episode_rewards, yaw_deltas_rad,
            body_deltas_m, elapsed_s, gait_blend_means,
            learned_gait_blend_means):
        yaw_status = yaw_direction_status(case["cmd_yaw_rad_s"], yaw_delta)
        planar_status = planar_direction_status(
            case.get("cmd_vx_m_s", 0.0),
            case.get("cmd_vy_m_s", 0.0),
            body_delta[0],
            body_delta[1],
            case_elapsed_s,
        )
        cases.append({
            "case_name": case.get("case_name", "unnamed"),
            "gait_blend": (
                None if case.get("gait_blend") is None
                else float(case["gait_blend"])),
            "mean_gait_blend": (
                None if gait_blend_mean is None
                else float(gait_blend_mean)),
            "mean_learned_gait_blend": (
                None if learned_gait_blend_mean is None
                else float(learned_gait_blend_mean)),
            "cmd_vx_m_s": float(case.get("cmd_vx_m_s", 0.0)),
            "cmd_vy_m_s": float(case.get("cmd_vy_m_s", 0.0)),
            "cmd_yaw_rad_s": float(case["cmd_yaw_rad_s"]),
            "episode_reward": float(reward),
            "elapsed_s": float(case_elapsed_s),
            "body_delta_x_m": float(body_delta[0]),
            "body_delta_y_m": float(body_delta[1]),
            "body_vx_m_s": planar_status["body_vx_m_s"],
            "body_vy_m_s": planar_status["body_vy_m_s"],
            "commanded_planar_distance_m": (
                planar_status["commanded_planar_distance_m"]),
            "off_axis_distance_m": planar_status["off_axis_distance_m"],
            "planar_velocity_error_m_s": (
                planar_status["planar_velocity_error_m_s"]),
            "yaw_delta_rad": float(yaw_delta),
            "planar_status": planar_status["status"],
            "yaw_status": yaw_status["status"],
            "direction_status": (
                f"{planar_status['status']}:{yaw_status['status']}"),
        })

    for case in cases:
        case_elapsed = max(float(case["elapsed_s"]), 1e-6)
        cmd_vec = np.array([
            case["cmd_vx_m_s"],
            case["cmd_vy_m_s"],
        ], dtype=np.float64)
        body_vec = np.array([
            case["body_vx_m_s"],
            case["body_vy_m_s"],
        ], dtype=np.float64)
        cmd_speed = float(np.linalg.norm(cmd_vec))
        case["planar_cmd_speed_m_s"] = cmd_speed
        case["body_planar_speed_m_s"] = float(np.linalg.norm(body_vec))
        case["off_axis_speed_m_s"] = float(
            case["off_axis_distance_m"] / case_elapsed)
        case["yaw_rate_rad_s"] = float(case["yaw_delta_rad"] / case_elapsed)
        case["yaw_rate_error_rad_s"] = float(
            case["yaw_rate_rad_s"] - case["cmd_yaw_rad_s"])

    wrong_yaw_sign_count = sum(
        1 for case in cases if case["yaw_status"] == "wrong_sign")
    weak_turn_count = sum(
        1 for case in cases if case["yaw_status"] == "weak_turn")
    straight_violation_count = sum(
        1 for case in cases if case["yaw_status"] == "straight_drift")
    wrong_planar_sign_count = sum(
        1 for case in cases if case["planar_status"] == "wrong_planar_sign")
    weak_translation_count = sum(
        1 for case in cases if case["planar_status"] == "weak_translation")
    stationary_violation_count = sum(
        1 for case in cases if case["planar_status"] == "stationary_drift")
    yaw_passed_count = sum(
        1 for case in cases
        if case["yaw_status"] in ("correct", "straight_ok"))
    planar_passed_count = sum(
        1 for case in cases
        if case["planar_status"] in ("correct", "stationary_ok"))
    passed_count = sum(
        1 for case in cases
        if (case["yaw_status"] in ("correct", "straight_ok")
            and case["planar_status"] in ("correct", "stationary_ok")))
    planar_success_rate = float(planar_passed_count / max(len(cases), 1))
    yaw_success_rate = float(yaw_passed_count / max(len(cases), 1))
    direction_gate_passed = (
        wrong_yaw_sign_count == 0
        and wrong_planar_sign_count == 0
        and planar_success_rate >= REQUIRED_PLANAR_SUCCESS_RATE
        and yaw_success_rate >= REQUIRED_YAW_SUCCESS_RATE
        and straight_violation_count <= MAX_STRAIGHT_VIOLATION_COUNT
        and stationary_violation_count == 0)
    planar_velocity_rmse = float(np.sqrt(np.mean([
        case["planar_velocity_error_m_s"] ** 2 for case in cases
    ]))) if cases else float("inf")
    yaw_rate_rmse = float(np.sqrt(np.mean([
        case["yaw_rate_error_rad_s"] ** 2 for case in cases
    ]))) if cases else float("inf")
    off_axis_cases = [
        case for case in cases if case["planar_cmd_speed_m_s"] > 1e-9]
    mean_off_axis_speed = (
        float(np.mean([case["off_axis_speed_m_s"]
                       for case in off_axis_cases]))
        if off_axis_cases else 0.0)
    zero_command_cases = [
        case for case in cases
        if (case["planar_cmd_speed_m_s"] <= 1e-9
            and abs(case["cmd_yaw_rad_s"]) <= 1e-9)
    ]
    zero_command_mean_speed = (
        float(np.mean([case["body_planar_speed_m_s"]
                       for case in zero_command_cases]))
        if zero_command_cases else 0.0)
    yaw_only_cases = [
        case for case in cases
        if (case["planar_cmd_speed_m_s"] <= 1e-9
            and abs(case["cmd_yaw_rad_s"]) > 1e-9)
    ]
    yaw_only_mean_planar_speed = (
        float(np.mean([case["body_planar_speed_m_s"]
                       for case in yaw_only_cases]))
        if yaw_only_cases else 0.0)
    yaw_only_max_planar_speed = (
        float(max(case["body_planar_speed_m_s"]
                  for case in yaw_only_cases))
        if yaw_only_cases else 0.0)
    tracking_gate_passed = (
        direction_gate_passed
        and planar_velocity_rmse <= PLANAR_TRACKING_RMSE_TARGET_M_S
        and yaw_rate_rmse <= YAW_TRACKING_RMSE_TARGET_RAD_S
        and mean_off_axis_speed <= MEAN_OFF_AXIS_SPEED_TARGET_M_S
        and zero_command_mean_speed <= ZERO_COMMAND_SPEED_TARGET_M_S
        and yaw_only_mean_planar_speed <= (
            YAW_ONLY_MEAN_PLANAR_SPEED_TARGET_M_S))
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else -np.inf
    selection_score = (
        mean_reward
        + 5000.0 * planar_success_rate
        + 5000.0 * yaw_success_rate
        - 20000.0 * planar_velocity_rmse
        - 10000.0 * yaw_rate_rmse
        - 10000.0 * mean_off_axis_speed
        - 10000.0 * zero_command_mean_speed
        - 8000.0 * yaw_only_mean_planar_speed
    )
    if not direction_gate_passed:
        selection_score -= DIRECTION_FAILED_SCORE_OFFSET
        selection_score -= 1000.0 * wrong_yaw_sign_count
        selection_score -= 1000.0 * wrong_planar_sign_count
        selection_score -= 250.0 * weak_turn_count
        selection_score -= 250.0 * weak_translation_count
        excess_straight = max(
            0, straight_violation_count - MAX_STRAIGHT_VIOLATION_COUNT)
        selection_score -= 500.0 * excess_straight
        selection_score -= 500.0 * stationary_violation_count

    return {
        "selection_contract": best_selection_contract(),
        "mean_reward": mean_reward,
        "selection_score": float(selection_score),
        "direction_gate_passed": bool(direction_gate_passed),
        "tracking_gate_passed": bool(tracking_gate_passed),
        "planar_velocity_rmse_m_s": planar_velocity_rmse,
        "yaw_rate_rmse_rad_s": yaw_rate_rmse,
        "mean_off_axis_speed_m_s": mean_off_axis_speed,
        "zero_command_mean_speed_m_s": zero_command_mean_speed,
        "yaw_only_mean_planar_speed_m_s": yaw_only_mean_planar_speed,
        "yaw_only_max_planar_speed_m_s": yaw_only_max_planar_speed,
        "direction_success_rate": float(passed_count / max(len(cases), 1)),
        "yaw_success_rate": yaw_success_rate,
        "planar_success_rate": planar_success_rate,
        "wrong_sign_count": int(wrong_yaw_sign_count),
        "wrong_yaw_sign_count": int(wrong_yaw_sign_count),
        "wrong_planar_sign_count": int(wrong_planar_sign_count),
        "weak_turn_count": int(weak_turn_count),
        "weak_translation_count": int(weak_translation_count),
        "straight_violation_count": int(straight_violation_count),
        "stationary_violation_count": int(stationary_violation_count),
        "cases": cases,
    }


def comparable_training_fields(config):
    return {
        "terrain": config.get("terrain"),
        "gait_mode": config.get("gait_mode"),
        "gait_blend": config.get("gait_blend"),
        "obs_dim": config.get("obs_dim"),
        "num_actuators": config.get("num_actuators"),
        "policy_action_dim": config.get("policy_action_dim"),
        "num_imus": config.get("num_imus"),
    }


def training_config_compatible(
        existing, expected, allow_command_curriculum_mismatch=False,
        allow_experimental_contract_mismatch=False):
    if not isinstance(existing, dict):
        return False, ["missing training_config.json"]
    reasons = []
    if comparable_training_fields(existing) != comparable_training_fields(expected):
        reasons.append("training fields")
    if not dict_float_match(
            existing.get("sensor_robustness"),
            expected.get("sensor_robustness", {})):
        reasons.append("sensor_robustness")
    if not dict_float_match(
            existing.get("control_timing"),
            expected.get("control_timing", {})):
        reasons.append("control_timing")
    if (existing.get("reward_contract") != expected.get("reward_contract")
            and not allow_experimental_contract_mismatch):
        reasons.append("reward_contract")
    if existing.get("eval_command") != expected.get("eval_command"):
        reasons.append("eval_command")
    if (existing.get("command_curriculum") != expected.get(
            "command_curriculum")
            and not allow_command_curriculum_mismatch):
        reasons.append("command_curriculum")
    if existing.get("best_selection_contract") != expected.get(
            "best_selection_contract") and not allow_experimental_contract_mismatch:
        reasons.append("best_selection_contract")
    if (existing.get("action_adapter") != expected.get("action_adapter")
            and not allow_experimental_contract_mismatch):
        reasons.append("action_adapter")
    if (existing.get("residual_exploration")
            != expected.get("residual_exploration")):
        reasons.append("residual_exploration")
    existing_training = existing.get("training", {})
    expected_training = expected.get("training", {})
    if existing_training.get("policy_net_arch") != expected_training.get(
            "policy_net_arch"):
        reasons.append("policy_net_arch")
    expected_actuator = expected.get("actuator_contract_fingerprint")
    if (expected_actuator
            and existing.get("actuator_contract_fingerprint")
            != expected_actuator):
        reasons.append("actuator_contract")
    return not reasons, reasons


def run_dir_for_model(model_path):
    directory = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(directory) == "checkpoints":
        return os.path.dirname(directory)
    return directory


def resume_model_compatible(
        model_path, expected_config, allow_command_curriculum_mismatch=False,
        allow_experimental_contract_mismatch=False):
    config = read_json(os.path.join(
        run_dir_for_model(model_path), "training_config.json"))
    return training_config_compatible(
        config,
        expected_config,
        allow_command_curriculum_mismatch=allow_command_curriculum_mismatch,
        allow_experimental_contract_mismatch=(
            allow_experimental_contract_mismatch),
    )


def archive_existing_run_artifacts(run_dir, reason):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(
        run_dir, "incompatible_timing_archive", timestamp)
    names = [
        "best_model.zip",
        "best_model_vecnormalize.pkl",
        "final_model.zip",
        "final_model_vecnormalize.pkl",
        "training_config.json",
        "training_result.json",
        "best_eval_summary.json",
        "last_directional_eval_summary.json",
        "eval_metrics.json",
        "eval_metrics_robust.json",
        "checkpoints",
    ]
    moved = []
    for name in names:
        src = os.path.join(run_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(archive_dir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved.append(dst)
    if moved:
        marker = {
            "created_unix_time": time.time(),
            "reason": reason,
            "moved": moved,
        }
        os.makedirs(archive_dir, exist_ok=True)
        write_json(os.path.join(archive_dir, "archive_reason.json"), marker)
        print(f"  Archived incompatible previous artifacts: {archive_dir}")
    return moved


def make_env(terrain='flat', gait_mode='random', gait_blend=None,
             encoder_pos_noise_std=0.0, encoder_vel_noise_std=0.0,
             imu_gravity_noise_std=0.0, imu_gyro_noise_std=0.0,
             action_delay_steps=0, action_saturation=1.0, seed=0,
             fixed_cmd_vel=None, fixed_cmd_yaw=None,
             fixed_cmd_vx=None, fixed_cmd_vy=None,
             command_curriculum="omni",
             command_resample_prob=None, gait_prior_scale=None,
             policy_residual_scale=None):
    """Factory for creating a monitored WormEnvV6 instance."""
    def _init():
        from worm_env_v6 import WormEnvV6
        env = WormEnvV6(
            terrain=terrain, gait_mode=gait_mode, gait_blend=gait_blend,
            encoder_pos_noise_std=encoder_pos_noise_std,
            encoder_vel_noise_std=encoder_vel_noise_std,
            imu_gravity_noise_std=imu_gravity_noise_std,
            imu_gyro_noise_std=imu_gyro_noise_std,
            action_delay_steps=action_delay_steps,
            action_saturation=action_saturation,
            fixed_cmd_vel=fixed_cmd_vel,
            fixed_cmd_vx=fixed_cmd_vx,
            fixed_cmd_vy=fixed_cmd_vy,
            fixed_cmd_yaw=fixed_cmd_yaw,
            command_curriculum=command_curriculum,
            command_resample_prob=(
                command_resample_prob
                if command_resample_prob is not None else 0.005),
            **({} if gait_prior_scale is None else {
                "gait_prior_scale": gait_prior_scale}),
            **({} if policy_residual_scale is None else {
                "policy_residual_scale": policy_residual_scale}))
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def obs_layout_json(layout):
    return {key: [value.start, value.stop] for key, value in layout.items()}


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def best_eval_summary_path(run_dir):
    return os.path.join(run_dir, "best_eval_summary.json")


def has_best_model_pair(run_dir):
    return (
        os.path.exists(os.path.join(run_dir, "best_model.zip"))
        and os.path.exists(os.path.join(run_dir, "best_model_vecnormalize.pkl"))
    )


def load_persistent_best_eval(run_dir, log_dir, eval_schedule_fingerprint=None,
                              selection_contract_version=None):
    if not has_best_model_pair(run_dir):
        return -np.inf

    summary = read_json(best_eval_summary_path(run_dir))
    if eval_schedule_fingerprint is not None:
        if ((summary or {}).get("eval_schedule_fingerprint")
                != eval_schedule_fingerprint):
            return -np.inf
    if selection_contract_version is not None:
        summary_contract = (summary or {}).get("selection_contract", {})
        if summary_contract.get("version") != selection_contract_version:
            return -np.inf
    try:
        value = float((summary or {}).get(
            "selection_score", (summary or {}).get("best_mean_reward")))
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    if (eval_schedule_fingerprint is not None
            or selection_contract_version is not None):
        return -np.inf

    eval_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(eval_path):
        return -np.inf
    try:
        data = np.load(eval_path)
        results = data["results"]
        if len(results) == 0:
            return -np.inf
        return float(np.max(np.mean(results, axis=1)))
    except (KeyError, OSError, ValueError):
        return -np.inf


def write_best_eval_summary(path, mean_reward, timestep,
                            eval_schedule=None,
                            eval_schedule_fingerprint=None,
                            directional_summary=None):
    payload = {
        "format_version": 1,
        "updated_unix_time": time.time(),
        "best_mean_reward": float(mean_reward),
        "best_timestep": int(timestep),
    }
    if eval_schedule_fingerprint is not None:
        payload["eval_schedule_fingerprint"] = eval_schedule_fingerprint
    if eval_schedule is not None:
        payload["eval_schedule"] = eval_schedule
    if directional_summary is not None:
        payload.update({
            "selection_contract": directional_summary["selection_contract"],
            "selection_score": directional_summary["selection_score"],
            "direction_gate_passed": directional_summary[
                "direction_gate_passed"],
            "tracking_gate_passed": directional_summary[
                "tracking_gate_passed"],
            "planar_velocity_rmse_m_s": directional_summary[
                "planar_velocity_rmse_m_s"],
            "yaw_rate_rmse_rad_s": directional_summary[
                "yaw_rate_rmse_rad_s"],
            "mean_off_axis_speed_m_s": directional_summary[
                "mean_off_axis_speed_m_s"],
            "zero_command_mean_speed_m_s": directional_summary[
                "zero_command_mean_speed_m_s"],
            "direction_success_rate": directional_summary[
                "direction_success_rate"],
            "planar_success_rate": directional_summary[
                "planar_success_rate"],
            "yaw_success_rate": directional_summary["yaw_success_rate"],
            "wrong_sign_count": directional_summary["wrong_sign_count"],
            "wrong_yaw_sign_count": directional_summary[
                "wrong_yaw_sign_count"],
            "wrong_planar_sign_count": directional_summary[
                "wrong_planar_sign_count"],
            "weak_turn_count": directional_summary["weak_turn_count"],
            "weak_translation_count": directional_summary[
                "weak_translation_count"],
            "straight_violation_count": directional_summary[
                "straight_violation_count"],
            "stationary_violation_count": directional_summary[
                "stationary_violation_count"],
            "directional_cases": directional_summary["cases"],
        })
    write_json(path, payload)


def build_training_config(args, run_gait_label, sensor_kwargs, device):
    from motor_contract_v6 import motor_contract
    from worm_env_v6 import (
        CMD_VX_RANGE,
        CMD_VY_RANGE,
        CMD_YAW_RANGE,
        CTRL_DT,
        NUM_ACTUATORS,
        NUM_POLICY_ACTIONS,
        NUM_IMUS,
        OBS_DIM,
        OBS_LAYOUT,
        PERISTALTIC_ACTUATION_PERIOD_S,
        PHASE_FREQ,
        action_adapter_contract,
        reward_contract,
    )

    policy_net_arch = parse_net_arch_arg(
        getattr(args, "policy_net_arch", None), DEFAULT_POLICY_NET_ARCH)
    value_net_arch = parse_net_arch_arg(
        getattr(args, "value_net_arch", None), DEFAULT_VALUE_NET_ARCH)

    return {
        "format_version": 1,
        "created_unix_time": time.time(),
        "algorithm": "PPO",
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": args.gait_blend,
        "run_gait_label": run_gait_label,
        "run_label": getattr(args, "run_label", None),
        "obs_dim": OBS_DIM,
        "obs_layout": obs_layout_json(OBS_LAYOUT),
        "num_actuators": NUM_ACTUATORS,
        "policy_action_dim": NUM_POLICY_ACTIONS,
        "num_imus": NUM_IMUS,
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "action_adapter": action_adapter_contract(
            args.gait_prior_scale, args.policy_residual_scale),
        "residual_exploration": residual_exploration_contract(
            args.ent_coef, args.log_std_init),
        "reward_contract": reward_contract(),
        "deployable_observation_sources": [
            "body-frame vx command",
            "body-frame vy command",
            "yaw-rate command",
            "learned gait_blend action gate",
            "actuated joint encoder positions",
            "actuated joint encoder velocities",
            "previous applied action",
            "per-segment IMU projected gravity",
            "per-segment IMU angular velocity",
            "phase clock",
        ],
        "forbidden_policy_sources": [
            "base linear velocity",
            "global position",
            "global yaw",
            "MuJoCo freejoint state as policy input",
            "external localization as policy input",
        ],
        "command_ranges": {
            "cmd_vx_m_s": list(CMD_VX_RANGE),
            "cmd_vy_m_s": list(CMD_VY_RANGE),
            "cmd_yaw_rad_s": list(CMD_YAW_RANGE),
        },
        "command_curriculum": args.command_curriculum,
        "command_resample_prob": args.command_resample_prob,
        "eval_command": {
            "cmd_vx_m_s": CMD_VX_RANGE[1],
            "cmd_vy_m_s": 0.0,
            "cmd_yaw_rad_s": 0.0,
            "command_resample_prob": 0.0,
        },
        "best_eval_schedule": {
            "command_resample_prob": 0.0,
            "cases": best_eval_schedule(args.gait_mode, args.gait_blend),
        },
        "best_selection_contract": best_selection_contract(),
        "control_timing": {
            "control_dt_s": CTRL_DT,
            "control_rate_hz": 1.0 / CTRL_DT,
            "peristaltic_actuation_period_s": (
                PERISTALTIC_ACTUATION_PERIOD_S),
            "phase_freq_hz": PHASE_FREQ,
        },
        "sensor_robustness": sensor_kwargs,
        "training": {
            "timesteps": args.timesteps,
            "train_chunk_timesteps": args.train_chunk_timesteps,
            "n_envs": args.n_envs,
            "requested_device": getattr(args, "device", "cpu"),
            "resolved_device": device,
            "seed": 42,
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 1024,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": args.ent_coef,
            "log_std_init": args.log_std_init,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_net_arch": {
                "pi": policy_net_arch,
                "vf": value_net_arch,
            },
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0,
            "directional_eval_freq_steps": (
                args.directional_eval_freq_steps),
            "directional_eval_seconds": args.directional_eval_seconds,
        },
        "resume": args.resume,
    }


class NormSyncCallback(BaseCallback):
    """Sync VecNormalize stats from train → eval, and save alongside best model."""
    def __init__(self, train_env, eval_env, save_path, print_freq=5000):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.save_path = save_path
        self.print_freq = print_freq
        best_path = os.path.join(self.save_path, "best_model.zip")
        self._last_best = (
            os.path.getmtime(best_path) if os.path.exists(best_path) else None)

    def _on_step(self):
        # Sync obs normalization
        self.eval_env.obs_rms = self.train_env.obs_rms

        # Save VecNormalize when EvalCallback finds new best
        best_path = os.path.join(self.save_path, "best_model.zip")
        if os.path.exists(best_path):
            mtime = os.path.getmtime(best_path)
            if self._last_best is None or mtime > self._last_best:
                self._last_best = mtime
                norm_path = os.path.join(
                    self.save_path, "best_model_vecnormalize.pkl")
                self.train_env.save(norm_path)

        # Print progress
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mean_r = np.mean(ep_rewards)
                mean_l = np.mean(ep_lengths)
                print(f"  step {self.num_timesteps:>8d}  "
                      f"ep_reward={mean_r:>8.2f}  ep_len={mean_l:>6.0f}")
        return True


def raw_envs_from_vec(vec_env):
    base = vec_env
    while hasattr(base, "venv"):
        base = base.venv
    if not hasattr(base, "envs"):
        raise TypeError("directional eval requires DummyVecEnv-style envs")
    return [env.unwrapped if hasattr(env, "unwrapped") else env
            for env in base.envs]


class DirectionalPersistentBestEvalCallback(BaseCallback):
    def __init__(self, eval_env, train_env, eval_schedule, eval_freq,
                 best_model_save_path, persistent_best_score=-np.inf,
                 persistent_path=None, eval_schedule_fingerprint=None,
                 deterministic=True, eval_max_steps=None):
        super().__init__()
        self.eval_env = eval_env
        self.train_env = train_env
        self.eval_schedule = eval_schedule
        self.eval_freq = int(eval_freq)
        self.best_model_save_path = best_model_save_path
        self.best_selection_score = float(persistent_best_score)
        self.persistent_path = persistent_path
        self.eval_schedule_fingerprint = eval_schedule_fingerprint
        self.deterministic = deterministic
        self.eval_max_steps = (
            None if eval_max_steps is None else max(1, int(eval_max_steps)))

    def _evaluate_once(self):
        from worm_env_v6 import MAX_EP_STEPS

        if hasattr(self.eval_env, "obs_rms") and hasattr(self.train_env, "obs_rms"):
            self.eval_env.obs_rms = self.train_env.obs_rms
        raw_envs = raw_envs_from_vec(self.eval_env)
        obs = self.eval_env.reset()
        start_yaws = [env._root_yaw_rad() for env in raw_envs]
        start_positions = [
            env.data.xpos[env._root_body_id].copy() for env in raw_envs]
        start_forward_axes = []
        start_lateral_axes = []
        for env in raw_envs:
            root_xmat = env.data.xmat[env._root_body_id].reshape(3, 3)
            start_forward_axes.append(-root_xmat[:2, 0].copy())
            start_lateral_axes.append(root_xmat[:2, 1].copy())
        episode_rewards = np.zeros(len(self.eval_schedule), dtype=np.float64)
        yaw_deltas = np.zeros(len(self.eval_schedule), dtype=np.float64)
        body_deltas = np.zeros((len(self.eval_schedule), 2), dtype=np.float64)
        elapsed = np.zeros(len(self.eval_schedule), dtype=np.float64)
        done = np.zeros(len(self.eval_schedule), dtype=bool)
        step_counts = np.zeros(len(self.eval_schedule), dtype=np.int32)
        gait_blend_sums = np.zeros(len(self.eval_schedule), dtype=np.float64)
        learned_gait_blend_sums = np.zeros(
            len(self.eval_schedule), dtype=np.float64)
        gait_blend_counts = np.zeros(len(self.eval_schedule), dtype=np.int32)

        max_steps = (
            MAX_EP_STEPS if self.eval_max_steps is None
            else min(MAX_EP_STEPS, self.eval_max_steps))

        for _ in range(max_steps):
            actions, _ = self.model.predict(
                obs, deterministic=self.deterministic)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            for idx, reward in enumerate(rewards):
                if done[idx]:
                    continue
                step_counts[idx] += 1
                episode_rewards[idx] += float(reward)
                if "gait_blend" in infos[idx]:
                    gait_blend_sums[idx] += float(infos[idx]["gait_blend"])
                    gait_blend_counts[idx] += 1
                if "learned_gait_blend" in infos[idx]:
                    learned_gait_blend_sums[idx] += float(
                        infos[idx]["learned_gait_blend"])
                if bool(dones[idx]):
                    done[idx] = True
                    final_yaw = infos[idx].get("root_yaw_rad")
                    if final_yaw is None:
                        final_yaw = raw_envs[idx]._root_yaw_rad()
                    yaw_deltas[idx] = wrap_angle_rad(
                        float(final_yaw) - start_yaws[idx])
                    final_xy = np.array([
                        infos[idx].get(
                            "root_x_m",
                            raw_envs[idx].data.xpos[
                                raw_envs[idx]._root_body_id, 0]),
                        infos[idx].get(
                            "root_y_m",
                            raw_envs[idx].data.xpos[
                                raw_envs[idx]._root_body_id, 1]),
                    ], dtype=np.float64)
                    delta_world = final_xy - start_positions[idx][:2]
                    body_deltas[idx, 0] = float(np.dot(
                        delta_world, start_forward_axes[idx]))
                    body_deltas[idx, 1] = float(np.dot(
                        delta_world, start_lateral_axes[idx]))
                    elapsed[idx] = float(infos[idx].get(
                        "elapsed_s", step_counts[idx] * 0.02))
            if np.all(done):
                break

        for idx, is_done in enumerate(done):
            if not is_done:
                yaw_deltas[idx] = wrap_angle_rad(
                    raw_envs[idx]._root_yaw_rad() - start_yaws[idx])
                final_xy = raw_envs[idx].data.xpos[
                    raw_envs[idx]._root_body_id, :2].copy()
                delta_world = final_xy - start_positions[idx][:2]
                body_deltas[idx, 0] = float(np.dot(
                    delta_world, start_forward_axes[idx]))
                body_deltas[idx, 1] = float(np.dot(
                    delta_world, start_lateral_axes[idx]))
                elapsed[idx] = float(step_counts[idx] * 0.02)

        return directional_eval_summary(
            self.eval_schedule,
            episode_rewards=[float(v) for v in episode_rewards],
            yaw_deltas_rad=[float(v) for v in yaw_deltas],
            body_deltas_m=[tuple(v) for v in body_deltas],
            elapsed_s=[float(v) for v in elapsed],
            gait_blend_means=[
                None if count <= 0 else float(total / count)
                for total, count in zip(gait_blend_sums, gait_blend_counts)
            ],
            learned_gait_blend_means=[
                None if count <= 0 else float(total / count)
                for total, count in zip(
                    learned_gait_blend_sums, gait_blend_counts)
            ],
        )

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        summary = self._evaluate_once()
        summary["timestep"] = int(self.num_timesteps)
        if self.best_model_save_path:
            write_json(
                os.path.join(
                    self.best_model_save_path,
                    "last_directional_eval_summary.json"),
                summary,
            )

        print(
            "Eval directional: "
            f"score={summary['selection_score']:.2f} "
            f"mean_reward={summary['mean_reward']:.2f} "
            f"pass={summary['direction_gate_passed']} "
            f"track={summary['tracking_gate_passed']} "
            f"planar={summary['planar_success_rate']:.2f} "
            f"yaw={summary['yaw_success_rate']:.2f} "
            f"rmse=({summary['planar_velocity_rmse_m_s']:.3f},"
            f"{summary['yaw_rate_rmse_rad_s']:.3f}) "
            f"wrong_planar={summary['wrong_planar_sign_count']} "
            f"wrong_yaw={summary['wrong_yaw_sign_count']}")

        if summary["selection_score"] <= self.best_selection_score:
            return True

        self.best_selection_score = float(summary["selection_score"])
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
            self.model.save(os.path.join(
                self.best_model_save_path, "best_model"))
        if self.persistent_path:
            write_best_eval_summary(
                self.persistent_path,
                summary["mean_reward"],
                self.num_timesteps,
                eval_schedule=self.eval_schedule,
                eval_schedule_fingerprint=self.eval_schedule_fingerprint,
                directional_summary=summary,
            )
        if summary["tracking_gate_passed"]:
            gate_label = "accepted"
        elif summary["direction_gate_passed"]:
            gate_label = "direction-only"
        else:
            gate_label = "progress"
        print(
            f"New best omni-{gate_label} model at "
            f"{self.num_timesteps} steps")
        return True


def train(args):
    terrain = args.terrain
    gait_mode = args.gait_mode
    gait_blend = args.gait_blend
    sensor_kwargs = dict(
        encoder_pos_noise_std=args.encoder_pos_noise,
        encoder_vel_noise_std=args.encoder_vel_noise,
        imu_gravity_noise_std=args.imu_gravity_noise,
        imu_gyro_noise_std=args.imu_gyro_noise,
        action_delay_steps=args.action_delay_steps,
        action_saturation=args.action_saturation,
    )
    run_gait_label = gait_mode
    if gait_blend is not None:
        run_gait_label = f"blend_{gait_blend:.2f}".replace(".", "p")
    RUN_DIR, LOG_DIR, CKPT_DIR = make_run_dirs(
        terrain, run_gait_label, run_label=args.run_label)
    device = resolve_device(args.device)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    training_config = build_training_config(
        args, run_gait_label, sensor_kwargs, device)
    config_path = os.path.join(RUN_DIR, "training_config.json")
    requested_resume = args.resume
    if requested_resume:
        resume_ok, resume_reasons = resume_model_compatible(
            requested_resume,
            training_config,
            allow_command_curriculum_mismatch=(
                args.allow_curriculum_resume),
            allow_experimental_contract_mismatch=(
                args.allow_contract_resume),
        )
        if not resume_ok:
            print(
                "  WARNING: ignoring incompatible resume checkpoint "
                f"({', '.join(resume_reasons)}): {requested_resume}")
            args.resume = None

    existing_config = read_json(config_path)
    existing_ok, existing_reasons = training_config_compatible(
        existing_config, training_config)
    if not existing_ok:
        archive_existing_run_artifacts(
            RUN_DIR, ", ".join(existing_reasons) or "configuration mismatch")
        os.makedirs(CKPT_DIR, exist_ok=True)

    training_config = build_training_config(
        args, run_gait_label, sensor_kwargs, device)
    write_json(config_path, training_config)

    n_envs = args.n_envs
    print(f"Worm V6 RL Training — PPO (Longworm2) [{terrain}]")
    print(f"  terrain:    {terrain}")
    print(f"  gait_mode:  {gait_mode}")
    print(f"  gait_blend: {gait_blend if gait_blend is not None else 'mode/default'}")
    print(f"  sensor:     {sensor_kwargs}")
    print(f"  device:     {device} (requested: {args.device})")
    print(f"  envs:       {n_envs}")
    print(f"  timesteps:  {args.timesteps:,}")
    print(f"  run_dir:    {RUN_DIR}")
    print(f"  metadata:   {config_path}")
    policy_net_arch = parse_net_arch_arg(
        args.policy_net_arch, DEFAULT_POLICY_NET_ARCH)
    value_net_arch = parse_net_arch_arg(
        args.value_net_arch, DEFAULT_VALUE_NET_ARCH)
    print(f"  actor_arch: {policy_net_arch}")
    print(f"  critic_arch:{value_net_arch}")
    eval_max_steps = directional_eval_steps_from_seconds(
        args.directional_eval_seconds)
    if eval_max_steps is None:
        print(f"  dir_eval:   every {args.directional_eval_freq_steps:,} "
              "steps, full episode")
    else:
        print(f"  dir_eval:   every {args.directional_eval_freq_steps:,} "
              f"steps, {args.directional_eval_seconds:.2f}s")

    # ── Create vectorized environments ──
    if n_envs == 1:
        raw_vec_env = DummyVecEnv([make_env(
            terrain=terrain, gait_mode=gait_mode,
            gait_blend=gait_blend, seed=42,
            command_curriculum=args.command_curriculum,
            command_resample_prob=args.command_resample_prob,
            gait_prior_scale=args.gait_prior_scale,
            policy_residual_scale=args.policy_residual_scale,
            **sensor_kwargs)])
    else:
        raw_vec_env = SubprocVecEnv(
            [make_env(terrain=terrain, gait_mode=gait_mode,
                      gait_blend=gait_blend, seed=42 + i,
                      command_curriculum=args.command_curriculum,
                      command_resample_prob=args.command_resample_prob,
                      gait_prior_scale=args.gait_prior_scale,
                      policy_residual_scale=args.policy_residual_scale,
                      **sensor_kwargs)
             for i in range(n_envs)])

    # Observation normalization — reward normalization DISABLED
    # ── Tensorboard ──
    try:
        import tensorboard  # noqa: F401
        tb_log = LOG_DIR
        print(f"  tensorboard: {LOG_DIR}")
    except ImportError:
        tb_log = None
        print(f"  tensorboard: not installed (logging disabled)")

    # ── Create or load model ──
    start_timesteps = 0
    learn_timesteps = args.timesteps
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        norm_path = infer_vecnormalize_path(args.resume)
        if norm_path:
            vec_env = VecNormalize.load(norm_path, raw_vec_env)
            vec_env.training = True
            vec_env.norm_reward = False
            print(f"  Loaded VecNormalize from: {norm_path}")
        else:
            print("  WARNING: no paired VecNormalize file found; "
                  "observation normalization will start from defaults")
            vec_env = VecNormalize(
                raw_vec_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )
        model = PPO.load(args.resume, env=vec_env, device=device)
        start_timesteps = int(model.num_timesteps)
        learn_timesteps = max(args.timesteps - start_timesteps, 0)
        print(f"  resume_start_timesteps: {start_timesteps:,}")
        print(f"  remaining_to_target:     {learn_timesteps:,}")
    else:
        vec_env = VecNormalize(
            raw_vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=policy_net_arch, vf=value_net_arch),
                log_std_init=args.log_std_init,
            ),
            tensorboard_log=tb_log,
            verbose=0,
            device=device,
            seed=42,
        )
    if args.train_chunk_timesteps is not None:
        learn_timesteps = min(learn_timesteps, args.train_chunk_timesteps)
        print(f"  chunk_timesteps:         {learn_timesteps:,}")

    print(f"  Policy network: {model.policy}")

    # ── Callbacks ──
    eval_schedule = best_eval_schedule(gait_mode, gait_blend)
    eval_schedule_fp = best_eval_schedule_fingerprint(eval_schedule)
    print("  best_eval_schedule:")
    for case in eval_schedule:
        blend_label = (
            "auto" if case["gait_blend"] is None
            else f"{case['gait_blend']:.2f}")
        print("    "
              f"blend={blend_label} "
              f"case={case['case_name']} "
              f"cmd=({case['cmd_vx_m_s']:+.2f},"
              f"{case['cmd_vy_m_s']:+.2f},"
              f"{case['cmd_yaw_rad_s']:+.1f})")

    eval_env = DummyVecEnv([
        make_env(
            terrain=terrain, gait_mode=gait_mode,
            gait_blend=case["gait_blend"], seed=999 + idx,
            fixed_cmd_vx=case["cmd_vx_m_s"],
            fixed_cmd_vy=case["cmd_vy_m_s"],
            fixed_cmd_yaw=case["cmd_yaw_rad_s"],
            command_curriculum=args.command_curriculum,
            command_resample_prob=0.0,
            gait_prior_scale=args.gait_prior_scale,
            policy_residual_scale=args.policy_residual_scale,
            **sensor_kwargs)
        for idx, case in enumerate(eval_schedule)
    ])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False)
    eval_env.obs_rms = vec_env.obs_rms

    persistent_best = load_persistent_best_eval(
        RUN_DIR, LOG_DIR,
        eval_schedule_fingerprint=eval_schedule_fp,
        selection_contract_version=DIRECTIONAL_SELECTION_CONTRACT_VERSION)
    if np.isfinite(persistent_best):
        print(f"  persistent_best_selection_score: {persistent_best:.2f}")
    else:
        print("  persistent_best_selection_score: reset for eval/selection contract")

    eval_callback = DirectionalPersistentBestEvalCallback(
        eval_env,
        train_env=vec_env,
        eval_schedule=eval_schedule,
        eval_freq=(
            max(args.directional_eval_freq_steps // n_envs, 1)
            if args.directional_eval_freq_steps > 0 else 0),
        best_model_save_path=RUN_DIR,
        persistent_best_score=persistent_best,
        persistent_path=best_eval_summary_path(RUN_DIR),
        eval_schedule_fingerprint=eval_schedule_fp,
        deterministic=True,
        eval_max_steps=eval_max_steps,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // n_envs, 1),
        save_path=CKPT_DIR,
        name_prefix="worm_v6_ppo",
        save_vecnormalize=True,
    )

    norm_sync = NormSyncCallback(
        train_env=vec_env, eval_env=eval_env,
        save_path=RUN_DIR,
        print_freq=max(5000 // n_envs, 1),
    )

    # ── Train ──
    if learn_timesteps > 0:
        print(f"\n  Training started...")
        model.learn(
            total_timesteps=learn_timesteps,
            callback=[eval_callback, checkpoint_callback, norm_sync],
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )
    else:
        print("\n  Target timesteps already reached; saving current artifacts.")

    # ── Save ──
    final_path = os.path.join(RUN_DIR, "final_model")
    model.save(final_path)
    vec_env.save(f"{final_path}_vecnormalize.pkl")
    result_path = os.path.join(RUN_DIR, "training_result.json")
    result = dict(training_config)
    result["completed_unix_time"] = time.time()
    result["resume_start_timesteps"] = start_timesteps
    result["target_timesteps"] = int(args.timesteps)
    result["train_chunk_timesteps"] = (
        int(args.train_chunk_timesteps)
        if args.train_chunk_timesteps is not None else None)
    result["completed_timesteps"] = int(model.num_timesteps)
    result["artifacts"] = {
        "final_model": f"{final_path}.zip",
        "final_vecnormalize": f"{final_path}_vecnormalize.pkl",
        "best_model": os.path.join(RUN_DIR, "best_model.zip"),
        "best_vecnormalize": os.path.join(
            RUN_DIR, "best_model_vecnormalize.pkl"),
        "best_eval_summary": best_eval_summary_path(RUN_DIR),
        "training_config": config_path,
    }
    best_eval = read_json(best_eval_summary_path(RUN_DIR))
    if best_eval:
        result["best_eval"] = best_eval
    write_json(result_path, result)
    print(f"\n  Saved final model: {final_path}.zip")
    print(f"  Saved VecNormalize: {final_path}_vecnormalize.pkl")
    print(f"  Saved training result: {result_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train worm V6 robot with PPO")
    ap.add_argument("--terrain", type=str, default="flat",
                    choices=["flat", "sand", "slope", "rough", "steps", "channel"],
                    help="Terrain type (default: flat)")
    ap.add_argument("--gait-mode", type=str, default="random",
                    choices=GAIT_MODES,
                    help="Mode command: worm=0, mixed=0.5, snake=1, random=sample per episode")
    ap.add_argument("--gait-blend", type=float, default=None,
                    help="Override gait blend in [0, 1]")
    ap.add_argument("--run-label", type=str, default=None,
                    help="Optional run directory label after worm_v6_ppo_")
    ap.add_argument("--encoder-pos-noise", type=float, default=0.0,
                    help="Normalized encoder position noise std")
    ap.add_argument("--encoder-vel-noise", type=float, default=0.0,
                    help="Normalized encoder velocity noise std")
    ap.add_argument("--imu-gravity-noise", type=float, default=0.0,
                    help="Projected-gravity IMU noise std")
    ap.add_argument("--imu-gyro-noise", type=float, default=0.0,
                    help="Normalized gyro noise std")
    ap.add_argument("--action-delay-steps", type=int, default=0,
                    help="Integer control-step delay before action is applied")
    ap.add_argument("--action-saturation", type=float, default=1.0,
                    help="Applied action limit in [0, 1] before actuator scaling")
    ap.add_argument("--gait-prior-scale", type=float, default=1.0,
                    help="Scale for deterministic phase/gait_blend action prior")
    ap.add_argument("--policy-residual-scale", type=float, default=0.35,
                    help="Scale applied to PPO residual before adding gait prior")
    from worm_env_v6 import COMMAND_CURRICULA
    ap.add_argument("--command-curriculum", type=str, default="omni",
                    choices=COMMAND_CURRICULA,
                    help="Command sampling curriculum for train episodes")
    ap.add_argument("--command-resample-prob", type=float, default=None,
                    help="Per-control-step command resampling probability; "
                         "default keeps the environment default")
    ap.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF,
                    help="PPO entropy coefficient for residual exploration")
    ap.add_argument("--log-std-init", type=float, default=DEFAULT_LOG_STD_INIT,
                    help="Initial Gaussian log std for residual policy actions")
    ap.add_argument("--policy-net-arch", type=str, default="512,256,128",
                    help="Comma-separated actor MLP hidden widths")
    ap.add_argument("--value-net-arch", type=str, default="512,256,128",
                    help="Comma-separated critic MLP hidden widths")
    ap.add_argument("--timesteps", type=int, default=1_000_000,
                    help="Total training timesteps")
    ap.add_argument("--train-chunk-timesteps", type=int, default=None,
                    help="Train at most this many additional timesteps "
                         "while keeping --timesteps as the formal target")
    ap.add_argument("--directional-eval-freq-steps", type=int, default=5000,
                    help="Run directional best-model evaluation every N "
                         "environment steps; set 0 to disable during train")
    ap.add_argument("--directional-eval-seconds", type=float, default=None,
                    help="Limit each directional evaluation case to this many "
                         "seconds; default uses the full episode")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="Number of parallel environments")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["auto", "cpu", "cuda"],
                    help="PPO network device; MuJoCo env stepping remains CPU-bound")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to model checkpoint to resume from")
    ap.add_argument("--allow-curriculum-resume", action="store_true",
                    help="Allow --resume from a checkpoint trained with a "
                         "different command curriculum")
    ap.add_argument("--allow-contract-resume", action="store_true",
                    help="Allow --resume across reward/action-adapter "
                         "research iterations while keeping the hard "
                         "deployable ABI, timing, sensor, actuator, and "
                         "observation checks")
    ap.add_argument("--test", action="store_true",
                    help="Quick test run (10k steps, 1 env)")
    args = ap.parse_args()

    if args.test:
        args.timesteps = 10_000
        args.n_envs = 1
        print("=== TEST MODE (10k steps, 1 env) ===")

    import sys
    sys.path.insert(0, SCRIPT_DIR)

    train(args)

"""
Scan continuous body-frame command tracking for a deployable Worm V6 policy.

This is the acceptance tool for moving beyond six primitive directions. It
evaluates a grid of fixed vx/vy/yaw commands and reports tracking error,
direction signs, zero-command drift, and per-command responses.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from action_adapter_v6 import action_adapter_contract  # noqa: E402
from eval_v6 import find_vecnormalize, wrap_angle_rad  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_YAW_RANGE,
    CTRL_DT,
    WormEnvV6,
    reward_contract,
)

FIXED_LATERAL_SPEED_THRESHOLD_M_S = 0.03
FIXED_LATERAL_OFF_AXIS_THRESHOLD_M_S = 0.08
FIXED_LATERAL_YAW_THRESHOLD_RAD_S = 0.20
VX_ERROR_EXCEED_THRESHOLD_M_S = 0.10
VY_ERROR_EXCEED_THRESHOLD_M_S = 0.10
PLANAR_ERROR_EXCEED_THRESHOLD_M_S = 0.10
YAW_ERROR_EXCEED_THRESHOLD_RAD_S = 0.20
OFF_AXIS_EXCEED_THRESHOLD_M_S = 0.08
ZERO_COMMAND_SPEED_EXCEED_THRESHOLD_M_S = 0.02
YAW_ONLY_PLANAR_EXCEED_THRESHOLD_M_S = 0.08
STEP_INFO_TELEMETRY_KEYS = (
    "gait_blend",
    "learned_gait_blend",
    "raw_gait_gate_action",
    "prior_component_l2",
    "residual_component_l2",
    "applied_action_l2",
    "desired_gait_blend",
    "gait_gate_error",
    "reward_component_tracking_cost",
    "reward_planar_component_deficit_penalty",
    "mixed_planar_fullscale_gate",
    "reward_mixed_planar_fullscale_deficit_penalty",
    "reward_axial_prior_preserve_penalty",
    "axial_prior_residual_cancellation",
    "axial_slide_activity_deficit",
)


def parse_values(text):
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def default_run_dir(terrain, gait_mode):
    return os.path.join(PROJECT_ROOT, "runs", f"worm_v6_ppo_{terrain}_{gait_mode}")


def default_model_path(run_dir):
    for name in ("best_model.zip", "final_model.zip"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(run_dir, "best_model.zip")


def summarize_step_infos(infos):
    summary = {}
    for key in STEP_INFO_TELEMETRY_KEYS:
        values = [
            float(info[key])
            for info in infos
            if key in info and info[key] is not None
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[f"mean_{key}"] = float(round(float(np.mean(arr)), 10))
        summary[f"std_{key}"] = float(round(float(np.std(arr)), 10))
        summary[f"final_{key}"] = float(round(float(arr[-1]), 10))
    return summary


def robust_env_kwargs(args):
    return {
        "encoder_pos_noise_std": float(args.encoder_pos_noise),
        "encoder_vel_noise_std": float(args.encoder_vel_noise),
        "imu_gravity_noise_std": float(args.imu_gravity_noise),
        "imu_gyro_noise_std": float(args.imu_gyro_noise),
        "action_delay_steps": int(args.action_delay_steps),
        "action_saturation": float(args.action_saturation),
    }


def sensor_noise_summary(env_kwargs):
    return {
        "encoder_pos_noise_std": env_kwargs["encoder_pos_noise_std"],
        "encoder_vel_noise_std": env_kwargs["encoder_vel_noise_std"],
        "imu_gravity_noise_std": env_kwargs["imu_gravity_noise_std"],
        "imu_gyro_noise_std": env_kwargs["imu_gyro_noise_std"],
    }


def evaluate_command(model, norm_env, terrain, gait_mode, gait_blend,
                     vx, vy, yaw, seconds, seed,
                     gait_prior_scale=1.0, policy_residual_scale=0.35,
                     env_kwargs=None):
    env_kwargs = dict(env_kwargs or {})
    env = WormEnvV6(
        terrain=terrain,
        gait_mode=gait_mode,
        gait_blend=gait_blend,
        fixed_cmd_vx=vx,
        fixed_cmd_vy=vy,
        fixed_cmd_yaw=yaw,
        command_resample_prob=0.0,
        gait_prior_scale=gait_prior_scale,
        policy_residual_scale=policy_residual_scale,
        **env_kwargs,
    )
    obs, _ = env.reset(seed=seed)
    env.set_command(vx=vx, vy=vy, yaw_rate=yaw, gait_blend=gait_blend)
    obs = env._get_obs()
    start_pos = env.data.xpos[env._root_body_id].copy()
    start_yaw = env._root_yaw_rad()
    start_xmat = env.data.xmat[env._root_body_id].reshape(3, 3)
    forward_axis = -start_xmat[:2, 0].copy()
    lateral_axis = start_xmat[:2, 1].copy()
    reward = 0.0
    steps = 0
    terminated = False
    step_infos = []

    for _ in range(int(seconds / CTRL_DT)):
        obs_batch = norm_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_batch, deterministic=True)
        obs, step_reward, done, truncated, info = env.step(action[0])
        reward += float(step_reward)
        steps += 1
        step_infos.append(info)
        if done or truncated:
            terminated = bool(done)
            break

    elapsed = max(steps * CTRL_DT, CTRL_DT)
    delta_world = env.data.xpos[env._root_body_id, :2].copy() - start_pos[:2]
    body_vx = float(np.dot(delta_world, forward_axis) / elapsed)
    body_vy = float(np.dot(delta_world, lateral_axis) / elapsed)
    yaw_delta = wrap_angle_rad(env._root_yaw_rad() - start_yaw)
    yaw_rate = float(yaw_delta / elapsed)
    env.close()

    cmd_vec = np.array([vx, vy], dtype=np.float64)
    body_vec = np.array([body_vx, body_vy], dtype=np.float64)
    cmd_speed = float(np.linalg.norm(cmd_vec))
    body_planar_speed = float(np.linalg.norm(body_vec))
    vx_error = float(body_vx - vx)
    vy_error = float(body_vy - vy)
    planar_error = float(np.linalg.norm(body_vec - cmd_vec))
    if cmd_speed > 1e-9:
        projected_speed = float(np.dot(body_vec, cmd_vec / cmd_speed))
        off_axis_speed = float(np.linalg.norm(
            body_vec - projected_speed * cmd_vec / cmd_speed))
        planar_sign_ok = projected_speed >= -1e-6
    else:
        projected_speed = 0.0
        off_axis_speed = float(np.linalg.norm(body_vec))
        planar_sign_ok = off_axis_speed <= 0.02
    if abs(yaw) > 1e-9:
        yaw_sign_ok = yaw_rate * yaw >= 0.0
    else:
        yaw_sign_ok = abs(yaw_rate) <= 0.05

    row = {
        "cmd_vx_m_s": float(vx),
        "cmd_vy_m_s": float(vy),
        "cmd_yaw_rad_s": float(yaw),
        "body_vx_m_s": body_vx,
        "body_vy_m_s": body_vy,
        "yaw_rate_rad_s": yaw_rate,
        "elapsed_s": elapsed,
        "reward": reward,
        "terminated": terminated,
        "command_planar_speed_m_s": cmd_speed,
        "body_planar_speed_m_s": body_planar_speed,
        "vx_error_m_s": vx_error,
        "vy_error_m_s": vy_error,
        "planar_error_m_s": planar_error,
        "yaw_error_rad_s": float(abs(yaw_rate - yaw)),
        "projected_speed_m_s": projected_speed,
        "off_axis_speed_m_s": off_axis_speed,
        "planar_sign_ok": bool(planar_sign_ok),
        "yaw_sign_ok": bool(yaw_sign_ok),
    }
    row.update(summarize_step_infos(step_infos))
    return row


def build_commands(args):
    commands = []
    for vx in parse_values(args.vx_values):
        for vy in parse_values(args.vy_values):
            commands.append((vx, vy, 0.0))
    for yaw in parse_values(args.yaw_values):
        commands.append((0.0, 0.0, yaw))
    if args.include_forward_yaw:
        for vx in parse_values(args.forward_yaw_vx_values):
            for yaw in parse_values(args.forward_yaw_values):
                commands.append((vx, 0.0, yaw))
    deduped = []
    seen = set()
    for command in commands:
        key = tuple(round(v, 6) for v in command)
        if key not in seen:
            deduped.append(command)
            seen.add(key)
    return deduped


def summarize(rows):
    planar_rows = [
        r for r in rows
        if abs(r["cmd_yaw_rad_s"]) <= 1e-9
    ]
    yaw_rows = [
        r for r in rows
        if (abs(r["cmd_vx_m_s"]) <= 1e-9
            and abs(r["cmd_vy_m_s"]) <= 1e-9)
    ]
    zero_rows = [
        r for r in rows
        if (abs(r["cmd_vx_m_s"]) <= 1e-9
            and abs(r["cmd_vy_m_s"]) <= 1e-9
            and abs(r["cmd_yaw_rad_s"]) <= 1e-9)
    ]
    yaw_only_rows = [
        r for r in rows
        if (abs(r["cmd_vx_m_s"]) <= 1e-9
            and abs(r["cmd_vy_m_s"]) <= 1e-9
            and abs(r["cmd_yaw_rad_s"]) > 1e-9)
    ]

    def rmse(values, key):
        if not values:
            return None
        return float(np.sqrt(np.mean([r[key] ** 2 for r in values])))

    def count(values, predicate):
        return int(sum(1 for r in values if predicate(r)))

    def planar_speed(row):
        return float(np.linalg.norm([row["body_vx_m_s"], row["body_vy_m_s"]]))

    def strongest_pure_lateral(sign):
        candidates = [
            r for r in rows
            if (abs(r["cmd_vx_m_s"]) <= 1e-9
                and abs(r["cmd_yaw_rad_s"]) <= 1e-9
                and r["cmd_vy_m_s"] * sign > 1e-9)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: abs(r["cmd_vy_m_s"]))

    def lateral_fields(label, row, sign):
        if row is None:
            return {
                f"fixed_lateral_{label}_cmd_vy_m_s": None,
                f"fixed_lateral_{label}_body_vx_m_s": None,
                f"fixed_lateral_{label}_body_vy_m_s": None,
                f"fixed_lateral_{label}_yaw_rate_rad_s": None,
                f"fixed_lateral_{label}_speed_passed": False,
                f"fixed_lateral_{label}_strict_passed": False,
            }
        speed_passed = (
            row["body_vy_m_s"] * sign >= FIXED_LATERAL_SPEED_THRESHOLD_M_S
        )
        strict_passed = (
            speed_passed
            and abs(row["body_vx_m_s"]) <= FIXED_LATERAL_OFF_AXIS_THRESHOLD_M_S
            and abs(row["yaw_rate_rad_s"]) <= FIXED_LATERAL_YAW_THRESHOLD_RAD_S
        )
        return {
            f"fixed_lateral_{label}_cmd_vy_m_s": float(row["cmd_vy_m_s"]),
            f"fixed_lateral_{label}_body_vx_m_s": float(row["body_vx_m_s"]),
            f"fixed_lateral_{label}_body_vy_m_s": float(row["body_vy_m_s"]),
            f"fixed_lateral_{label}_yaw_rate_rad_s": float(row["yaw_rate_rad_s"]),
            f"fixed_lateral_{label}_speed_passed": bool(speed_passed),
            f"fixed_lateral_{label}_strict_passed": bool(strict_passed),
        }

    lateral_left = strongest_pure_lateral(sign=1.0)
    lateral_right = strongest_pure_lateral(sign=-1.0)
    left_fields = lateral_fields("left", lateral_left, sign=1.0)
    right_fields = lateral_fields("right", lateral_right, sign=-1.0)

    summary = {
        "num_commands": len(rows),
        "planar_rmse_m_s": rmse(planar_rows, "planar_error_m_s"),
        "yaw_rmse_rad_s": rmse(yaw_rows, "yaw_error_rad_s"),
        "vx_error_rmse_m_s": rmse(rows, "vx_error_m_s"),
        "vy_error_rmse_m_s": rmse(rows, "vy_error_m_s"),
        "planar_sign_rate": (
            float(np.mean([r["planar_sign_ok"] for r in planar_rows]))
            if planar_rows else None),
        "yaw_sign_rate": (
            float(np.mean([r["yaw_sign_ok"] for r in yaw_rows]))
            if yaw_rows else None),
        "max_planar_error_m_s": (
            float(max(r["planar_error_m_s"] for r in rows)) if rows else None),
        "max_yaw_error_rad_s": (
            float(max(r["yaw_error_rad_s"] for r in rows)) if rows else None),
        "zero_command_mean_speed_m_s": (
            float(np.mean([
                np.linalg.norm([r["body_vx_m_s"], r["body_vy_m_s"]])
                for r in zero_rows
            ])) if zero_rows else None),
        "zero_command_mean_abs_yaw_rate_rad_s": (
            float(np.mean([abs(r["yaw_rate_rad_s"]) for r in zero_rows]))
            if zero_rows else None),
        "yaw_only_mean_planar_speed_m_s": (
            float(np.mean([
                np.linalg.norm([r["body_vx_m_s"], r["body_vy_m_s"]])
                for r in yaw_only_rows
            ])) if yaw_only_rows else None),
        "yaw_only_max_planar_speed_m_s": (
            float(max([
                np.linalg.norm([r["body_vx_m_s"], r["body_vy_m_s"]])
                for r in yaw_only_rows
            ])) if yaw_only_rows else None),
        "vx_error_exceed_threshold_m_s": VX_ERROR_EXCEED_THRESHOLD_M_S,
        "vy_error_exceed_threshold_m_s": VY_ERROR_EXCEED_THRESHOLD_M_S,
        "planar_error_exceed_threshold_m_s": (
            PLANAR_ERROR_EXCEED_THRESHOLD_M_S),
        "yaw_error_exceed_threshold_rad_s": (
            YAW_ERROR_EXCEED_THRESHOLD_RAD_S),
        "off_axis_exceed_threshold_m_s": OFF_AXIS_EXCEED_THRESHOLD_M_S,
        "zero_command_speed_exceed_threshold_m_s": (
            ZERO_COMMAND_SPEED_EXCEED_THRESHOLD_M_S),
        "yaw_only_planar_exceed_threshold_m_s": (
            YAW_ONLY_PLANAR_EXCEED_THRESHOLD_M_S),
        "vx_error_exceed_count": count(
            rows,
            lambda r: abs(r["vx_error_m_s"])
            > VX_ERROR_EXCEED_THRESHOLD_M_S,
        ),
        "vy_error_exceed_count": count(
            rows,
            lambda r: abs(r["vy_error_m_s"])
            > VY_ERROR_EXCEED_THRESHOLD_M_S,
        ),
        "planar_error_exceed_count": count(
            planar_rows,
            lambda r: r["planar_error_m_s"]
            > PLANAR_ERROR_EXCEED_THRESHOLD_M_S,
        ),
        "yaw_error_exceed_count": count(
            yaw_rows,
            lambda r: r["yaw_error_rad_s"]
            > YAW_ERROR_EXCEED_THRESHOLD_RAD_S,
        ),
        "off_axis_exceed_count": count(
            rows,
            lambda r: (
                abs(r["cmd_vx_m_s"]) > 1e-9
                or abs(r["cmd_vy_m_s"]) > 1e-9)
            and r["off_axis_speed_m_s"] > OFF_AXIS_EXCEED_THRESHOLD_M_S,
        ),
        "zero_command_speed_exceed_count": count(
            zero_rows,
            lambda r: planar_speed(r)
            > ZERO_COMMAND_SPEED_EXCEED_THRESHOLD_M_S,
        ),
        "yaw_only_planar_exceed_count": count(
            yaw_only_rows,
            lambda r: planar_speed(r)
            > YAW_ONLY_PLANAR_EXCEED_THRESHOLD_M_S,
        ),
        "wrong_planar_sign_count": count(
            planar_rows,
            lambda r: not r["planar_sign_ok"],
        ),
        "wrong_yaw_sign_count": count(
            yaw_rows,
            lambda r: not r["yaw_sign_ok"],
        ),
        "fixed_lateral_speed_threshold_m_s": FIXED_LATERAL_SPEED_THRESHOLD_M_S,
        "fixed_lateral_off_axis_threshold_m_s": (
            FIXED_LATERAL_OFF_AXIS_THRESHOLD_M_S),
        "fixed_lateral_yaw_threshold_rad_s": FIXED_LATERAL_YAW_THRESHOLD_RAD_S,
    }
    summary.update(left_fields)
    summary.update(right_fields)
    summary["fixed_lateral_speed_gate_passed"] = bool(
        summary["fixed_lateral_left_speed_passed"]
        and summary["fixed_lateral_right_speed_passed"]
    )
    summary["fixed_lateral_strict_gate_passed"] = bool(
        summary["fixed_lateral_left_strict_passed"]
        and summary["fixed_lateral_right_strict_passed"]
    )
    return summary


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Scan Worm V6 continuous command tracking")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--terrain", default="flat",
                    choices=["flat", "sand", "slope", "rough", "steps", "channel"])
    ap.add_argument("--gait-mode", default="random",
                    choices=["worm", "snake", "mixed", "random"])
    ap.add_argument("--gait-blend", type=float, default=None)
    ap.add_argument("--time", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--vx-values", default="-0.10,-0.05,0,0.05,0.10")
    ap.add_argument("--vy-values", default="-0.075,-0.0375,0,0.0375,0.075")
    ap.add_argument("--yaw-values", default="-0.125,-0.10,0,0.10,0.125")
    ap.add_argument("--include-forward-yaw", action="store_true")
    ap.add_argument("--forward-yaw-vx-values", default="-0.10,0.05,0.10")
    ap.add_argument("--forward-yaw-values", default="-0.10,0.10")
    ap.add_argument("--gait-prior-scale", type=float, default=1.0)
    ap.add_argument("--policy-residual-scale", type=float, default=0.35)
    ap.add_argument("--eval-condition", default="nominal",
                    help="Free-form label, e.g. nominal or robust")
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
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()
    env_kwargs = robust_env_kwargs(args)

    run_dir = args.run_dir or default_run_dir(args.terrain, args.gait_mode)
    model_path = args.model or default_model_path(run_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = PPO.load(model_path, device="cpu")
    raw_env = DummyVecEnv([lambda: WormEnvV6(
        terrain=args.terrain,
        gait_mode=args.gait_mode,
        gait_blend=args.gait_blend,
        gait_prior_scale=args.gait_prior_scale,
        policy_residual_scale=args.policy_residual_scale,
        command_resample_prob=0.0,
        **env_kwargs,
    )])
    norm_path = find_vecnormalize(model_path)
    if norm_path is None:
        raise FileNotFoundError(f"VecNormalize not found for {model_path}")
    norm_env = VecNormalize.load(norm_path, raw_env)
    norm_env.training = False
    norm_env.norm_reward = False

    rows = []
    for idx, (vx, vy, yaw) in enumerate(build_commands(args)):
        vx = float(np.clip(vx, *CMD_VX_RANGE))
        vy = float(np.clip(vy, *CMD_VY_RANGE))
        yaw = float(np.clip(yaw, *CMD_YAW_RANGE))
        rows.append(evaluate_command(
            model=model,
            norm_env=norm_env,
            terrain=args.terrain,
            gait_mode=args.gait_mode,
            gait_blend=args.gait_blend,
            vx=vx,
            vy=vy,
            yaw=yaw,
            seconds=args.time,
            seed=args.seed + idx,
            gait_prior_scale=args.gait_prior_scale,
            policy_residual_scale=args.policy_residual_scale,
            env_kwargs=env_kwargs,
        ))

    summary = {
        "format_version": 1,
        "created_unix_time": time.time(),
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": args.gait_blend,
        "time_s": args.time,
        "model_path": os.path.abspath(model_path),
        "vecnormalize": os.path.abspath(norm_path),
        "eval_condition": args.eval_condition,
        "sensor_noise": sensor_noise_summary(env_kwargs),
        "action_delay_steps": env_kwargs["action_delay_steps"],
        "action_saturation": env_kwargs["action_saturation"],
        "gait_prior_scale": args.gait_prior_scale,
        "policy_residual_scale": args.policy_residual_scale,
        "action_adapter": action_adapter_contract(
            gait_prior_scale=args.gait_prior_scale,
            policy_residual_scale=args.policy_residual_scale,
        ),
        "reward_contract": reward_contract(),
        "summary": summarize(rows),
        "commands": rows,
    }

    json_out = args.json_out or os.path.join(
        run_dir, "command_tracking_scan.json")
    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.csv_out:
        write_csv(args.csv_out, rows)
    print(json.dumps(summary["summary"], indent=2))
    print(f"Saved scan: {json_out}")
    if args.csv_out:
        print(f"Saved CSV: {args.csv_out}")
    norm_env.close()


if __name__ == "__main__":
    main()

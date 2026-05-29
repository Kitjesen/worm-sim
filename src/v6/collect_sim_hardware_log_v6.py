"""
Collect Worm V6 MuJoCo sensor readings in the raw hardware-log format.

This is a bridge test for deployability: it records only quantities that the
real robot controller is expected to have, then reuses build_hardware_obs_v6.py
to create the 80-D policy-observation CSV consumed by deployment replay.
"""

import argparse
import csv
import json
import math
import os
import sys

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import convert_raw_csv, raw_columns  # noqa: E402
from validate_hardware_log_v6 import AXES  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_YAW_RANGE,
    CTRL_DT,
    GAIT_MODES,
    NUM_ACTUATORS,
    NUM_IMUS,
    NUM_SLIDES,
    PHASE_FREQ,
    WormEnvV6,
)


def default_raw_path(terrain, gait_mode):
    return os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        f"sim_{terrain}_{gait_mode}_raw.csv")


def default_policy_path(raw_path):
    root, ext = os.path.splitext(raw_path)
    return f"{root}_policy{ext or '.csv'}"


def write_raw_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerows(rows)


def action_for_step(source, step, gait_blend, rng):
    gate = float(np.clip(2.0 * gait_blend - 1.0, -1.0, 1.0))
    if source == "zero":
        action = np.zeros(NUM_ACTUATORS + 1, dtype=np.float32)
        action[-1] = gate
        return action
    if source == "random":
        action = rng.uniform(-1.0, 1.0, size=NUM_ACTUATORS + 1).astype(np.float32)
        action[-1] = gate
        return action

    t = step * CTRL_DT
    phase = 2.0 * math.pi * PHASE_FREQ * t
    action = np.zeros(NUM_ACTUATORS + 1, dtype=np.float32)
    slide_offsets = np.linspace(0.0, 2.0 * math.pi, NUM_SLIDES, endpoint=False)
    yaw_count = NUM_ACTUATORS - NUM_SLIDES
    yaw_offsets = np.linspace(0.0, 2.0 * math.pi, yaw_count, endpoint=False)
    action[:NUM_SLIDES] = (1.0 - gait_blend) * np.sin(phase + slide_offsets)
    action[NUM_SLIDES:NUM_ACTUATORS] = (
        gait_blend * np.sin(phase + yaw_offsets))
    action[-1] = gate
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def segment_imu_readings(env):
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gravity = np.zeros((NUM_IMUS, 3), dtype=np.float32)
    gyro = np.zeros((NUM_IMUS, 3), dtype=np.float32)
    for i, body_id in enumerate(env._imu_body_ids):
        xmat = env.data.xmat[body_id].reshape(3, 3)
        gravity[i] = (xmat.T @ gravity_world).astype(np.float32)

        local_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body_id,
            local_vel, 1)
        gyro[i] = local_vel[:3].astype(np.float32)
    return gravity, gyro


def raw_row_from_env(env, terrain, mode, video_file, action):
    row = {
        "time_s": env._step_count * CTRL_DT,
        "terrain": terrain,
        "mode": mode,
        "video_file": video_file,
        "cmd_vx_m_s": env._cmd_vx,
        "cmd_vy_m_s": env._cmd_vy,
        "cmd_yaw_rad_s": env._cmd_yaw,
        "gait_blend": env._gait_blend,
        "velocity_estimate_m_s": -float(env.data.qvel[0]),
        "yaw_rate_estimate_rad_s": float(env.data.qvel[5]),
    }

    for i in range(NUM_SLIDES):
        row[f"slide_pos_m_{i:02d}"] = float(
            env.data.qpos[env._act_qpos_idx[i]])
        row[f"slide_vel_m_s_{i:02d}"] = float(
            env.data.qvel[env._act_qvel_idx[i]])

    for i in range(NUM_ACTUATORS - NUM_SLIDES):
        src = NUM_SLIDES + i
        row[f"yaw_pos_rad_{i:02d}"] = float(
            env.data.qpos[env._act_qpos_idx[src]])
        row[f"yaw_vel_rad_s_{i:02d}"] = float(
            env.data.qvel[env._act_qvel_idx[src]])

    gravity, gyro = segment_imu_readings(env)
    for seg in range(NUM_IMUS):
        for axis_i, axis in enumerate(AXES):
            row[f"segment_gravity_{seg:02d}_{axis}"] = float(
                gravity[seg, axis_i])
            row[f"segment_gyro_rad_s_{seg:02d}_{axis}"] = float(
                gyro[seg, axis_i])

    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    for i in range(NUM_ACTUATORS):
        row[f"action_{i:02d}"] = float(action[i])
    return row


def collect_sim_hardware_log(
        output_raw,
        output_policy=None,
        terrain="flat",
        gait_mode="mixed",
        gait_blend=None,
        duration_s=0.2,
        action_source="sine",
        seed=0,
        cmd_vel=None,
        cmd_vx=None,
        cmd_vy=0.0,
        cmd_yaw=0.0,
        video_file="sim_sensor_smoke.mp4",
        validate=True):
    rng = np.random.default_rng(seed)
    env = WormEnvV6(terrain=terrain, gait_mode=gait_mode, gait_blend=gait_blend)
    rows = []
    try:
        env.reset(seed=seed)
        if cmd_vx is None:
            cmd_vx = CMD_VX_RANGE[1] * 0.5 if cmd_vel is None else cmd_vel
        env.set_command(
            vx=cmd_vx,
            vy=float(np.clip(cmd_vy, *CMD_VY_RANGE)),
            yaw_rate=cmd_yaw,
            gait_blend=gait_blend)

        steps = max(1, int(math.ceil(duration_s / CTRL_DT)))
        for step in range(steps):
            action = action_for_step(
                action_source, step, env._gait_blend, rng)
            _, _, terminated, truncated, _ = env.step(action)
            rows.append(raw_row_from_env(
                env, terrain, gait_mode, video_file, env._last_action))
            if terminated or truncated:
                break
    finally:
        env.close()

    write_raw_csv(output_raw, rows)
    result = {
        "raw": output_raw,
        "rows": len(rows),
        "terrain": terrain,
        "gait_mode": gait_mode,
        "gait_blend": rows[0]["gait_blend"] if rows else None,
        "action_source": action_source,
    }
    if output_policy:
        result["policy"] = output_policy
        result["policy_conversion"] = convert_raw_csv(
            output_raw, output_policy, validate=validate)
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Collect sim sensor data in Worm V6 raw hardware format")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--gait-mode", choices=GAIT_MODES, default="mixed")
    ap.add_argument("--gait-blend", type=float, default=None)
    ap.add_argument("--time", type=float, default=0.2)
    ap.add_argument("--action-source", choices=["zero", "sine", "random"],
                    default="sine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd-vel", type=float, default=None,
                    help="Legacy alias for --cmd-vx")
    ap.add_argument("--cmd-vx", type=float, default=None)
    ap.add_argument("--cmd-vy", type=float, default=0.0)
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ap.add_argument("--video-file", default="sim_sensor_smoke.mp4")
    ap.add_argument("--output-raw", default=None)
    ap.add_argument("--output-policy", default=None)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    output_raw = args.output_raw or default_raw_path(
        args.terrain, args.gait_mode)
    output_policy = args.output_policy
    if output_policy is None:
        output_policy = default_policy_path(output_raw)

    result = collect_sim_hardware_log(
        output_raw=output_raw,
        output_policy=output_policy,
        terrain=args.terrain,
        gait_mode=args.gait_mode,
        gait_blend=args.gait_blend,
        duration_s=args.time,
        action_source=args.action_source,
        seed=args.seed,
        cmd_vel=args.cmd_vel,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_yaw=args.cmd_yaw,
        video_file=args.video_file,
        validate=not args.no_validate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

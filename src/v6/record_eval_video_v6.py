"""
Low-overhead deterministic rollout video recorder for V6 policies.

The main evaluator can render high-resolution videos, which is slow for the
CAD-heavy worm model. This script records a small MP4 plus a metrics JSON using
the same deployable observation and action path as eval_v6.py.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import cv2
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from train_v6 import make_env  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CMD_VEL_RANGE,
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_YAW_RANGE,
    CTRL_DT,
    OBS_DIM,
    reward_contract,
)
from worm_v6 import NUM_ACTUATORS, inject_strips  # noqa: E402
from action_adapter_v6 import action_adapter_contract  # noqa: E402


SEGMENT_LABELS = ["head"] + [f"seg{i}" for i in range(1, 7)]
DISPLAY_VELOCITY_WINDOW_S = 1.50


def command_for_time(schedule, time_s, duration_s, cmd_vx, cmd_vy, cmd_yaw):
    """Return the body-frame command for fixed or dynamic video rollouts."""
    if schedule in (None, "fixed"):
        return (
            float(np.clip(cmd_vx, *CMD_VX_RANGE)),
            float(np.clip(cmd_vy, *CMD_VY_RANGE)),
            float(np.clip(cmd_yaw, *CMD_YAW_RANGE)),
        )
    if schedule != "continuous_sweep":
        raise ValueError(f"unknown dynamic command schedule: {schedule}")

    duration = max(float(duration_s), CTRL_DT)
    tau = float(np.clip(time_s / duration, 0.0, 1.0))
    # Smoothly start/finish near zero while sweeping direction and turn rate.
    envelope = np.sin(np.pi * tau) ** 0.7
    phase = 2.0 * np.pi * 1.25 * tau
    speed_mod = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(4.0 * np.pi * tau - np.pi / 2.0))
    scale = envelope * speed_mod
    vx = 0.22 * scale * np.cos(phase)
    vy = 0.13 * scale * np.sin(phase)
    yaw = 0.22 * scale * np.sin(2.0 * phase)
    return (
        float(np.clip(vx, *CMD_VX_RANGE)),
        float(np.clip(vy, *CMD_VY_RANGE)),
        float(np.clip(yaw, *CMD_YAW_RANGE)),
    )


def infer_norm_path(model_path):
    stem, ext = os.path.splitext(model_path)
    if ext.lower() != ".zip":
        return None
    candidates = [f"{stem}_vecnormalize.pkl"]
    base = os.path.basename(model_path)
    directory = os.path.dirname(model_path)
    checkpoint_match = re.fullmatch(r"worm_v6_ppo_(\d+)_steps\.zip", base)
    if checkpoint_match:
        candidates.append(os.path.join(
            directory,
            f"worm_v6_ppo_vecnormalize_{checkpoint_match.group(1)}_steps.pkl"))
    if base == "best_model.zip":
        candidates.append(os.path.join(directory, "best_model_vecnormalize.pkl"))
    if base == "final_model.zip":
        candidates.append(os.path.join(directory, "final_model_vecnormalize.pkl"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def root_yaw_rad(data, body_id):
    mat = data.xmat[body_id].reshape(3, 3)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def wrap_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def project_world_to_pixel(scene_camera, world_pos, width, height):
    """Project a world point near the camera frustum into video pixels."""
    cam_pos = np.asarray(scene_camera.pos, dtype=np.float64)
    forward = np.asarray(scene_camera.forward, dtype=np.float64)
    up = np.asarray(scene_camera.up, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    up /= max(np.linalg.norm(up), 1e-9)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)

    rel = np.asarray(world_pos, dtype=np.float64) - cam_pos
    depth = float(np.dot(rel, forward))
    if depth <= 1e-6:
        return None

    near = float(scene_camera.frustum_near)
    top = float(scene_camera.frustum_top)
    bottom = float(scene_camera.frustum_bottom)
    half_h = max((top - bottom) * 0.5, 1e-9)
    half_w = half_h * float(width) / max(float(height), 1.0)
    x_near = float(np.dot(rel, right)) * near / depth
    y_near = float(np.dot(rel, up)) * near / depth
    px = int(round((x_near + half_w) / (2.0 * half_w) * width))
    py = int(round((half_h - y_near) / (2.0 * half_h) * height))
    if px < -width or px > 2 * width or py < -height or py > 2 * height:
        return None
    return px, py


def draw_head_speed_overlay(rgb, sim_env, scene_camera, info, cmd_vx, cmd_vy,
                            cmd_yaw):
    """Draw command and measured velocity near the head in RGB frame space."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    head_pos = sim_env.data.xpos[sim_env._root_body_id].copy()
    label_pos = head_pos + np.array([0.0, 0.0, 0.18], dtype=np.float64)
    pixel = project_world_to_pixel(
        scene_camera, label_pos, frame.shape[1], frame.shape[0])
    if pixel is None:
        x, y = 18, 24
    else:
        x = int(np.clip(pixel[0] + 12, 8, max(8, frame.shape[1] - 300)))
        y = int(np.clip(pixel[1] - 58, 8, max(8, frame.shape[0] - 68)))

    body_vx = float(info.get("body_vx_m_s", 0.0))
    body_vy = float(info.get("body_vy_m_s", 0.0))
    yaw_rate = float(info.get("body_yaw_rate_rad_s", 0.0))
    speed = float(np.linalg.norm([body_vx, body_vy]))
    line1 = f"v_body=({body_vx:+.3f},{body_vy:+.3f}) | |v|={speed:.3f}"
    line2 = f"yaw={yaw_rate:+.3f} | cmd=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_yaw:+.2f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.36, min(0.55, frame.shape[1] / 1900.0))
    thickness = max(1, int(round(scale * 2.0)))
    line_h = int(round(28 * scale))
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0]
             for line in (line1, line2)]
    box_w = max(size[0] for size in sizes) + 18
    box_h = line_h * 2 + 16
    x = int(np.clip(x, 8, max(8, frame.shape[1] - box_w - 2)))
    y = int(np.clip(y, 8, max(8, frame.shape[0] - box_h - 2)))
    x2 = min(frame.shape[1] - 1, x + box_w)
    y2 = min(frame.shape[0] - 1, y + box_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0)
    cv2.putText(frame, line1, (x + 9, y + line_h),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, line2, (x + 9, y + line_h * 2),
                font, scale, (210, 245, 255), thickness, cv2.LINE_AA)

    if pixel is not None:
        hx, hy = int(np.clip(pixel[0], 0, frame.shape[1] - 1)), int(
            np.clip(pixel[1], 0, frame.shape[0] - 1))
        cv2.circle(frame, (hx, hy), 5, (0, 230, 255), -1)
        cv2.line(frame, (hx, hy), (x + 8, y + box_h // 2),
                 (0, 230, 255), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--terrain", default="flat",
                    choices=["flat", "sand", "slope", "rough", "steps", "channel"])
    ap.add_argument("--gait-mode", default="worm",
                    choices=["worm", "snake", "mixed", "random"])
    ap.add_argument("--gait-blend", type=float, default=None)
    ap.add_argument("--time", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--cmd-vel", type=float, default=CMD_VEL_RANGE[1],
                    help="Legacy alias for --cmd-vx")
    ap.add_argument("--cmd-vx", type=float, default=None)
    ap.add_argument("--cmd-vy", type=float, default=0.0)
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ap.add_argument("--dynamic-command-schedule", default="fixed",
                    choices=["fixed", "continuous_sweep"],
                    help=(
                        "Use fixed command or a smooth continuously changing "
                        "vx/vy/yaw schedule for one video."))
    ap.add_argument("--encoder-pos-noise", type=float, default=0.0)
    ap.add_argument("--encoder-vel-noise", type=float, default=0.0)
    ap.add_argument("--imu-gravity-noise", type=float, default=0.0)
    ap.add_argument("--imu-gyro-noise", type=float, default=0.0)
    ap.add_argument("--action-delay-steps", type=int, default=0)
    ap.add_argument("--action-saturation", type=float, default=1.0)
    ap.add_argument("--condition", default="nominal")
    ap.add_argument("--prior-only", action="store_true",
                    help="Record the deterministic gait prior with zero residual")
    ap.add_argument("--no-steel-strips", action="store_true",
                    help="Disable visual-only spring-steel strip rendering")
    ap.add_argument("--no-head-speed-overlay", action="store_true",
                    help="Disable the head speed text overlay")
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--trajectory-csv-out", default=None)
    ap.add_argument("--trajectory-plot-out", default=None)
    ap.add_argument("--telemetry-csv-out", default=None)
    ap.add_argument("--telemetry-plot-out", default=None)
    args = ap.parse_args()
    cmd_vx = args.cmd_vx if args.cmd_vx is not None else args.cmd_vel
    cmd_vx = float(np.clip(cmd_vx, *CMD_VX_RANGE))
    cmd_vy = float(np.clip(args.cmd_vy, *CMD_VY_RANGE))
    cmd_yaw = float(np.clip(args.cmd_yaw, *CMD_YAW_RANGE))
    initial_cmd = command_for_time(
        args.dynamic_command_schedule, 0.0, args.time,
        cmd_vx, cmd_vy, cmd_yaw)

    raw_env = DummyVecEnv([make_env(
        terrain=args.terrain,
        gait_mode=args.gait_mode,
        gait_blend=args.gait_blend,
        seed=args.seed,
        fixed_cmd_vx=initial_cmd[0],
        fixed_cmd_vy=initial_cmd[1],
        fixed_cmd_yaw=initial_cmd[2],
        command_resample_prob=0.0,
        encoder_pos_noise_std=args.encoder_pos_noise,
        encoder_vel_noise_std=args.encoder_vel_noise,
        imu_gravity_noise_std=args.imu_gravity_noise,
        imu_gyro_noise_std=args.imu_gyro_noise,
        action_delay_steps=args.action_delay_steps,
        action_saturation=args.action_saturation,
    )])
    model_path = None
    norm_path = None
    model = None
    if args.prior_only:
        env = raw_env
    else:
        model_path = args.model or os.path.join(args.run_dir, "best_model.zip")
        norm_path = infer_norm_path(model_path)
        if norm_path is None:
            raise FileNotFoundError(
                f"No VecNormalize file paired with {model_path}")
        env = VecNormalize.load(norm_path, raw_env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(model_path, env=env, device="cpu")
    base_env = (env.venv if hasattr(env, "venv") else env).envs[0]
    raw_obs, _ = base_env.reset(seed=args.seed)
    sim_env = base_env.unwrapped
    obs = raw_obs.reshape(1, -1)
    if not args.prior_only:
        obs = env.normalize_obs(obs)

    renderer = mujoco.Renderer(sim_env.model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.0
    camera.elevation = -25
    camera.azimuth = 135

    steps = int(round(args.time / CTRL_DT))
    frame_stride = max(1, int(round((1.0 / args.fps) / CTRL_DT)))
    effective_fps = 1.0 / (frame_stride * CTRL_DT)

    os.makedirs(os.path.dirname(os.path.abspath(args.video_out)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.video_out}")

    start_pos = sim_env.data.xpos[sim_env._root_body_id].copy()
    start_xmat = sim_env.data.xmat[
        sim_env._root_body_id].reshape(3, 3).copy()
    start_forward_axis = -start_xmat[:2, 0]
    start_lateral_axis = start_xmat[:2, 1]
    start_segment_pos = np.stack(
        [sim_env.data.xpos[sid].copy() for sid in sim_env._seg_ids],
        axis=0,
    )
    start_yaw = root_yaw_rad(sim_env.data, sim_env._root_body_id)
    previous_yaw = start_yaw
    cumulative_yaw = 0.0
    reward_sum = 0.0
    action_l2 = []
    action_rate_l2 = []
    last_action = None
    frames = 0
    terminated = False
    truncated = False
    first_truncated_step = None
    started = time.time()
    trajectory_times = []
    trajectory_positions = []
    telemetry_rows = []
    root_history = []

    for step in range(steps):
        step_time = step * CTRL_DT
        current_cmd_vx, current_cmd_vy, current_cmd_yaw = command_for_time(
            args.dynamic_command_schedule,
            step_time,
            args.time,
            cmd_vx,
            cmd_vy,
            cmd_yaw,
        )
        if args.dynamic_command_schedule != "fixed":
            sim_env.set_command(
                vx=current_cmd_vx,
                vy=current_cmd_vy,
                yaw_rate=current_cmd_yaw,
                gait_blend=args.gait_blend,
            )
            raw_obs = sim_env._get_obs().reshape(1, -1)
            obs = raw_obs if args.prior_only else env.normalize_obs(raw_obs)

        if args.prior_only:
            action = np.zeros((1, sim_env.action_space.shape[0]), dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, step_truncated, info = sim_env.step(action[0])
        if args.prior_only:
            obs = next_obs.reshape(1, -1)
        else:
            obs = env.normalize_obs(next_obs.reshape(1, -1))
        current_yaw = root_yaw_rad(sim_env.data, sim_env._root_body_id)
        cumulative_yaw += wrap_pi(current_yaw - previous_yaw)
        previous_yaw = current_yaw
        current_time = (step + 1) * CTRL_DT
        current_root_pos = sim_env.data.xpos[sim_env._root_body_id].copy()
        root_history.append((current_time, current_root_pos, cumulative_yaw))
        while (len(root_history) > 2
               and current_time - root_history[0][0]
               > DISPLAY_VELOCITY_WINDOW_S):
            root_history.pop(0)
        hist_t0, hist_pos0, hist_yaw0 = root_history[0]
        hist_dt = max(current_time - hist_t0, CTRL_DT)
        hist_delta = current_root_pos - hist_pos0
        window_info = dict(info)
        window_info["body_vx_m_s"] = float(
            np.dot(hist_delta[:2], start_forward_axis) / hist_dt)
        window_info["body_vy_m_s"] = float(
            np.dot(hist_delta[:2], start_lateral_axis) / hist_dt)
        window_info["body_yaw_rate_rad_s"] = float(
            (cumulative_yaw - hist_yaw0) / hist_dt)
        reward_sum += float(reward)
        trajectory_times.append(current_time)
        trajectory_positions.append(np.stack(
            [sim_env.data.xpos[sid].copy() for sid in sim_env._seg_ids],
            axis=0,
        ))
        action_vec = np.asarray(action[0], dtype=np.float64)
        action_l2.append(float(np.linalg.norm(action_vec)))
        if last_action is not None:
            action_rate_l2.append(float(np.linalg.norm(action_vec - last_action)))
        last_action = action_vec.copy()
        telemetry_rows.append(build_telemetry_row(
            current_time,
            action_vec,
            window_info,
            sim_env,
            current_cmd_vx,
            current_cmd_vy,
            current_cmd_yaw,
        ))

        if step % frame_stride == 0:
            mid = np.mean(
                [sim_env.data.xpos[sid] for sid in sim_env._seg_ids], axis=0)
            camera.lookat[:] = mid
            renderer.update_scene(sim_env.data, camera)
            if not args.no_steel_strips:
                inject_strips(
                    renderer.scene,
                    sim_env.data,
                    sim_env._slide_pairs,
                    sim_env._strip_spacings,
                )
            rgb = renderer.render()
            if args.no_head_speed_overlay:
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                frame = draw_head_speed_overlay(
                    rgb,
                    sim_env,
                    renderer.scene.camera[0],
                    window_info,
                    current_cmd_vx,
                    current_cmd_vy,
                    current_cmd_yaw,
                )
            writer.write(frame)
            frames += 1

        if step_truncated and first_truncated_step is None:
            first_truncated_step = step + 1
        if done:
            terminated = bool(done)
            break

    writer.release()
    renderer.close()
    end_pos = sim_env.data.xpos[sim_env._root_body_id].copy()
    env.close()

    elapsed_s = (step + 1) * CTRL_DT
    delta_world = end_pos - start_pos
    forward_m = float(-(end_pos[0] - start_pos[0]))
    lateral_m = float(abs(end_pos[1] - start_pos[1]))
    yaw_delta = cumulative_yaw
    metrics = {
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": (
            args.gait_blend
            if args.gait_blend is not None
            else {"worm": 0.0, "mixed": 0.5, "snake": 1.0}.get(args.gait_mode)
        ),
        "obs_dim": OBS_DIM,
        "eval_condition": args.condition,
        "prior_only": bool(args.prior_only),
        "action_metric_note": (
            "mean_action_l2 is PPO residual action norm; in prior-only mode "
            "the deployed prior action is composed inside the environment"),
        "eval_command": {
            "cmd_vx_m_s": cmd_vx,
            "cmd_vy_m_s": cmd_vy,
            "cmd_yaw_rad_s": cmd_yaw,
            "dynamic_command_schedule": args.dynamic_command_schedule,
            "command_resample_prob": 0.0,
        },
        "action_adapter": action_adapter_contract(),
        "reward_contract": reward_contract(),
        "sensor_noise": {
            "encoder_pos_noise_std": args.encoder_pos_noise,
            "encoder_vel_noise_std": args.encoder_vel_noise,
            "imu_gravity_noise_std": args.imu_gravity_noise,
            "imu_gyro_noise_std": args.imu_gyro_noise,
            "action_delay_steps": args.action_delay_steps,
            "action_saturation": args.action_saturation,
        },
        "time_s": elapsed_s,
        "frames": frames,
        "requested_fps": args.fps,
        "frame_stride_steps": frame_stride,
        "video_fps": effective_fps,
        "terminated": terminated,
        "truncated": bool(first_truncated_step is not None),
        "first_truncated_step": (
            int(first_truncated_step)
            if first_truncated_step is not None else None),
        "continued_after_time_limit": bool(first_truncated_step is not None),
        "reward": reward_sum,
        "world_delta_m": [float(v) for v in delta_world[:3]],
        "distance_mm": forward_m * 1000.0,
        "speed_mm_s": forward_m * 1000.0 / max(elapsed_s, 1e-9),
        "lateral_drift_mm": lateral_m * 1000.0,
        "root_yaw_delta_rad": yaw_delta,
        "root_yaw_rate_rad_s": yaw_delta / max(elapsed_s, 1e-9),
        "mean_action_l2_per_step": float(np.mean(action_l2)) if action_l2 else 0.0,
        "mean_action_rate_l2_per_step": (
            float(np.mean(action_rate_l2)) if action_rate_l2 else 0.0),
        "model_path": model_path or "prior_only_zero_residual",
        "vecnormalize": norm_path,
        "video": args.video_out,
        "visual_steel_strips": not args.no_steel_strips,
        "head_speed_overlay": not args.no_head_speed_overlay,
        "wall_time_s": time.time() - started,
    }
    if trajectory_positions and args.trajectory_csv_out:
        positions = np.stack(trajectory_positions, axis=0)
        write_trajectory_csv(
            args.trajectory_csv_out,
            trajectory_times,
            positions,
            start_segment_pos,
            start_forward_axis,
            start_lateral_axis,
        )
        metrics["trajectory_csv"] = args.trajectory_csv_out
    if trajectory_positions and args.trajectory_plot_out:
        positions = np.stack(trajectory_positions, axis=0)
        plot_trajectory(
            args.trajectory_plot_out,
            trajectory_times,
            positions,
            start_segment_pos,
            start_forward_axis,
            start_lateral_axis,
            args.terrain,
            cmd_vx,
            cmd_vy,
            cmd_yaw,
            command_label=args.dynamic_command_schedule,
        )
        metrics["trajectory_plot"] = args.trajectory_plot_out
    if telemetry_rows and args.telemetry_csv_out:
        write_telemetry_csv(args.telemetry_csv_out, telemetry_rows)
        metrics["telemetry_csv"] = args.telemetry_csv_out
    if telemetry_rows and args.telemetry_plot_out:
        plot_telemetry(args.telemetry_plot_out, telemetry_rows)
        metrics["telemetry_plot"] = args.telemetry_plot_out
    if telemetry_rows:
        metrics["gait_gate_telemetry"] = summarize_gate_telemetry(
            telemetry_rows)
    write_json(args.json_out, metrics)
    print(json.dumps(metrics, indent=2))


def build_telemetry_row(time_s, policy_action, info, sim_env,
                        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw=0.0):
    row = {
        "time_s": float(time_s),
        "cmd_vx_m_s": float(cmd_vx),
        "cmd_vy_m_s": float(cmd_vy),
        "cmd_yaw_rad_s": float(cmd_yaw),
        "body_vx_m_s": float(info.get("body_vx_m_s", np.nan)),
        "body_vy_m_s": float(info.get("body_vy_m_s", np.nan)),
        "body_yaw_rate_rad_s": float(
            info.get("body_yaw_rate_rad_s", np.nan)),
        "raw_gait_gate_action": float(policy_action[-1]),
        "learned_gait_blend": float(info.get("learned_gait_blend", np.nan)),
        "gait_blend": float(info.get("gait_blend", np.nan)),
        "desired_gait_blend": float(info.get("desired_gait_blend", np.nan)),
        "gait_gate_error": float(info.get("gait_gate_error", np.nan)),
        "prior_component_l2": float(info.get("prior_component_l2", np.nan)),
        "residual_component_l2": float(
            info.get("residual_component_l2", np.nan)),
        "applied_action_l2": float(info.get("applied_action_l2", np.nan)),
    }
    residual = np.asarray(
        getattr(sim_env, "_last_residual_action", np.zeros(NUM_ACTUATORS)),
        dtype=np.float64,
    )
    prior_component = np.asarray(
        getattr(sim_env, "_last_prior_component", np.zeros(NUM_ACTUATORS)),
        dtype=np.float64,
    )
    residual_component = np.asarray(
        getattr(sim_env, "_last_residual_component", np.zeros(NUM_ACTUATORS)),
        dtype=np.float64,
    )
    applied = np.asarray(
        getattr(sim_env, "_last_action", np.zeros(NUM_ACTUATORS)),
        dtype=np.float64,
    )
    for i in range(NUM_ACTUATORS):
        row[f"policy_residual_{i}"] = float(policy_action[i])
        row[f"ema_residual_{i}"] = float(residual[i])
        row[f"prior_component_{i}"] = float(prior_component[i])
        row[f"residual_component_{i}"] = float(residual_component[i])
        row[f"applied_action_{i}"] = float(applied[i])
    return row


def summarize_gate_telemetry(rows):
    def mean(key):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else None

    def span(key):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            return None
        return [float(np.min(vals)), float(np.max(vals))]

    return {
        "mean_raw_gait_gate_action": mean("raw_gait_gate_action"),
        "raw_gait_gate_action_range": span("raw_gait_gate_action"),
        "mean_learned_gait_blend": mean("learned_gait_blend"),
        "learned_gait_blend_range": span("learned_gait_blend"),
        "mean_deployed_gait_blend": mean("gait_blend"),
        "deployed_gait_blend_range": span("gait_blend"),
        "mean_desired_gait_blend": mean("desired_gait_blend"),
        "mean_gait_gate_error": mean("gait_gate_error"),
        "mean_prior_component_l2": mean("prior_component_l2"),
        "mean_residual_component_l2": mean("residual_component_l2"),
        "mean_applied_action_l2": mean("applied_action_l2"),
    }


def write_telemetry_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory_csv(path, times, positions, start_segment_pos,
                         forward_axis, lateral_axis):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time_s",
            "segment_index",
            "segment_label",
            "world_x_m",
            "world_y_m",
            "world_z_m",
            "relative_forward_m",
            "relative_lateral_m",
            "relative_z_m",
        ])
        writer.writeheader()
        for ti, t in enumerate(times):
            for si, pos in enumerate(positions[ti]):
                rel = pos - start_segment_pos[si]
                writer.writerow({
                    "time_s": float(t),
                    "segment_index": int(si),
                    "segment_label": SEGMENT_LABELS[si],
                    "world_x_m": float(pos[0]),
                    "world_y_m": float(pos[1]),
                    "world_z_m": float(pos[2]),
                    "relative_forward_m": float(np.dot(
                        rel[:2], forward_axis)),
                    "relative_lateral_m": float(np.dot(
                        rel[:2], lateral_axis)),
                    "relative_z_m": float(rel[2]),
                })


def plot_trajectory(path, times, positions, start_segment_pos,
                    forward_axis, lateral_axis, terrain,
                    cmd_vx, cmd_vy, cmd_yaw, command_label="fixed"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    times = np.asarray(times, dtype=np.float64)
    pos_mm = positions * 1000.0
    rel = positions - start_segment_pos[None, :, :]
    forward = np.einsum("tix,x->ti", rel[:, :, :2], forward_axis) * 1000.0
    lateral = np.einsum("tix,x->ti", rel[:, :, :2], lateral_axis) * 1000.0
    dz = rel[:, :, 2] * 1000.0
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, positions.shape[1]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    if command_label and command_label != "fixed":
        title_cmd = f"cmd schedule={command_label}"
    else:
        title_cmd = f"cmd=({cmd_vx:.2f}, {cmd_vy:.2f}, {cmd_yaw:.2f})"
    fig.suptitle(
        f"Worm V6 policy segment trajectories on {terrain} | {title_cmd}",
        fontsize=14)

    ax = axes[0, 0]
    for si, label in enumerate(SEGMENT_LABELS):
        ax.plot(pos_mm[:, si, 0], pos_mm[:, si, 1],
                color=colors[si], linewidth=1.5, label=label)
        ax.scatter(pos_mm[0, si, 0], pos_mm[0, si, 1],
                   color=colors[si], marker="o", s=20)
        ax.scatter(pos_mm[-1, si, 0], pos_mm[-1, si, 1],
                   color=colors[si], marker="x", s=30)
    head_delta = pos_mm[-1, 0, :2] - pos_mm[0, 0, :2]
    ax.arrow(
        pos_mm[0, 0, 0],
        pos_mm[0, 0, 1],
        head_delta[0],
        head_delta[1],
        color="black",
        width=2.0,
        length_includes_head=True,
    )
    ax.set_title("Absolute top-down XY; circles=start, x=end")
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for si, label in enumerate(SEGMENT_LABELS):
        ax.plot(times, forward[:, si], color=colors[si],
                linewidth=1.5, label=label)
    ax.set_title("Forward displacement of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("body-frame forward (mm)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    for si in range(positions.shape[1]):
        ax.plot(times, lateral[:, si], color=colors[si], linewidth=1.5)
    ax.set_title("Lateral displacement of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("body-frame lateral (mm)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    for si in range(positions.shape[1]):
        ax.plot(times, dz[:, si], color=colors[si], linewidth=1.5)
    ax.set_title("Height displacement of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Delta Z (mm)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_telemetry(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    times = np.asarray([r["time_s"] for r in rows], dtype=np.float64)
    keys = [
        "raw_gait_gate_action",
        "learned_gait_blend",
        "gait_blend",
        "desired_gait_blend",
    ]
    actuator_labels = [f"a{i}" for i in range(NUM_ACTUATORS)]
    applied = np.asarray([
        [r[f"applied_action_{i}"] for i in range(NUM_ACTUATORS)]
        for r in rows
    ], dtype=np.float64).T
    prior = np.asarray([
        [r[f"prior_component_{i}"] for i in range(NUM_ACTUATORS)]
        for r in rows
    ], dtype=np.float64).T
    residual = np.asarray([
        [r[f"residual_component_{i}"] for i in range(NUM_ACTUATORS)]
        for r in rows
    ], dtype=np.float64).T

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    ax = axes[0]
    for cmd_key, meas_key, label in [
            ("cmd_vx_m_s", "body_vx_m_s", "vx"),
            ("cmd_vy_m_s", "body_vy_m_s", "vy"),
            ("cmd_yaw_rad_s", "body_yaw_rate_rad_s", "yaw")]:
        cmd_vals = np.asarray([r[cmd_key] for r in rows], dtype=np.float64)
        meas_vals = np.asarray([r[meas_key] for r in rows], dtype=np.float64)
        ax.plot(times, cmd_vals, linewidth=1.4, label=f"cmd {label}")
        ax.plot(times, meas_vals, linewidth=1.1, linestyle="--",
                label=f"body {label}")
    ax.set_ylabel("m/s or rad/s")
    ax.set_title("Commanded vs measured body-frame velocity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    ax = axes[1]
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        ax.plot(times, vals, linewidth=1.3, label=key)
    ax.set_ylabel("gate")
    ax.set_title("Learned latent gait gate and deployed blend")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    for ax, data, title in [
            (axes[2], prior, "prior component"),
            (axes[3], residual, "residual component"),
            (axes[4], applied, "applied action")]:
        image = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            extent=[times[0], times[-1], NUM_ACTUATORS - 0.5, -0.5],
        )
        ax.set_yticks(range(NUM_ACTUATORS))
        ax.set_yticklabels(actuator_labels, fontsize=8)
        ax.set_ylabel(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, pad=0.01, fraction=0.025)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

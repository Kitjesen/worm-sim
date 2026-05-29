"""
Low-overhead deterministic rollout video recorder for V6 policies.

The main evaluator can render high-resolution videos, which is slow for the
CAD-heavy worm model. This script records a small MP4 plus a metrics JSON using
the same deployable observation and action path as eval_v6.py.
"""

import argparse
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
    CTRL_DT,
    OBS_DIM,
    reward_contract,
)
from action_adapter_v6 import action_adapter_contract  # noqa: E402


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
    ap.add_argument("--cmd-vel", type=float, default=CMD_VEL_RANGE[1])
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ap.add_argument("--encoder-pos-noise", type=float, default=0.0)
    ap.add_argument("--encoder-vel-noise", type=float, default=0.0)
    ap.add_argument("--imu-gravity-noise", type=float, default=0.0)
    ap.add_argument("--imu-gyro-noise", type=float, default=0.0)
    ap.add_argument("--action-delay-steps", type=int, default=0)
    ap.add_argument("--action-saturation", type=float, default=1.0)
    ap.add_argument("--condition", default="nominal")
    ap.add_argument("--prior-only", action="store_true",
                    help="Record the deterministic gait prior with zero residual")
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    raw_env = DummyVecEnv([make_env(
        terrain=args.terrain,
        gait_mode=args.gait_mode,
        gait_blend=args.gait_blend,
        seed=args.seed,
        fixed_cmd_vel=args.cmd_vel,
        fixed_cmd_yaw=args.cmd_yaw,
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
    obs = env.reset()
    sim_env = (env.venv if hasattr(env, "venv") else env).envs[0].unwrapped

    renderer = mujoco.Renderer(sim_env.model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.0
    camera.elevation = -25
    camera.azimuth = 135

    os.makedirs(os.path.dirname(os.path.abspath(args.video_out)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.video_out}")

    start_pos = sim_env.data.xpos[sim_env._root_body_id].copy()
    start_yaw = root_yaw_rad(sim_env.data, sim_env._root_body_id)
    reward_sum = 0.0
    action_l2 = []
    action_rate_l2 = []
    last_action = None
    frames = 0
    steps = int(round(args.time / CTRL_DT))
    frame_stride = max(1, int(round((1.0 / args.fps) / CTRL_DT)))
    terminated = False
    started = time.time()

    for step in range(steps):
        if args.prior_only:
            action = np.zeros((1, sim_env.action_space.shape[0]), dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        reward_sum += float(reward[0])
        action_vec = np.asarray(action[0], dtype=np.float64)
        action_l2.append(float(np.linalg.norm(action_vec)))
        if last_action is not None:
            action_rate_l2.append(float(np.linalg.norm(action_vec - last_action)))
        last_action = action_vec.copy()

        if step % frame_stride == 0:
            mid = np.mean(
                [sim_env.data.xpos[sid] for sid in sim_env._seg_ids], axis=0)
            camera.lookat[:] = mid
            renderer.update_scene(sim_env.data, camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frames += 1

        if bool(done[0]):
            terminated = True
            break

    writer.release()
    renderer.close()
    end_pos = sim_env.data.xpos[sim_env._root_body_id].copy()
    end_yaw = root_yaw_rad(sim_env.data, sim_env._root_body_id)
    env.close()

    elapsed_s = (step + 1) * CTRL_DT
    delta_world = end_pos - start_pos
    forward_m = float(-(end_pos[0] - start_pos[0]))
    lateral_m = float(abs(end_pos[1] - start_pos[1]))
    yaw_delta = wrap_pi(end_yaw - start_yaw)
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
            "cmd_vel_m_s": args.cmd_vel,
            "cmd_yaw_rad_s": args.cmd_yaw,
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
        "terminated": terminated,
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
        "wall_time_s": time.time() - started,
    }
    write_json(args.json_out, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

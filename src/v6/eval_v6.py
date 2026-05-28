"""
Evaluate a deployable Worm V6 policy on one terrain and one gait mode.

The policy observation is the deployable 80-D state from worm_env_v6:
encoders, per-segment IMUs, previous action, command, and phase.
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from motor_contract_v6 import motor_contract
from worm_env_v6 import (
    WormEnvV6,
    CMD_VEL_RANGE,
    CMD_YAW_RANGE,
    CTRL_DT,
    GAIT_BLENDS,
    OBS_DIM,
    PERISTALTIC_ACTUATION_PERIOD_S,
    PHASE_FREQ,
    reward_contract,
    action_adapter_contract,
)


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
    if mode == "random":
        return 0.5
    return GAIT_BLENDS[mode]


def evaluate(args):
    run_dir = args.run_dir or default_run_dir(args.terrain, args.gait_mode)
    model_path = args.model or default_model_path(run_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    gait_blend = gait_blend_for(args.gait_mode, args.gait_blend)
    model = PPO.load(model_path, device="cpu")
    sensor_kwargs = {
        "encoder_pos_noise_std": args.encoder_pos_noise,
        "encoder_vel_noise_std": args.encoder_vel_noise,
        "imu_gravity_noise_std": args.imu_gravity_noise,
        "imu_gyro_noise_std": args.imu_gyro_noise,
        "action_delay_steps": args.action_delay_steps,
        "action_saturation": args.action_saturation,
    }

    raw_env = WormEnvV6(
        terrain=args.terrain, gait_mode=args.gait_mode, gait_blend=gait_blend,
        fixed_cmd_vel=args.cmd_vel, fixed_cmd_yaw=args.cmd_yaw,
        command_resample_prob=0.0,
        **sensor_kwargs)
    norm_env = DummyVecEnv([lambda: WormEnvV6(
        terrain=args.terrain, gait_mode=args.gait_mode, gait_blend=gait_blend,
        fixed_cmd_vel=args.cmd_vel, fixed_cmd_yaw=args.cmd_yaw,
        command_resample_prob=0.0,
        **sensor_kwargs)])
    norm_path = find_vecnormalize(model_path)
    if norm_path is not None:
        norm_env = VecNormalize.load(norm_path, norm_env)
        norm_env.training = False
        norm_env.norm_reward = False
        print(f"Loaded VecNormalize: {norm_path}")
    else:
        print("WARNING: VecNormalize not found; using raw observations")

    max_steps = int(args.time / CTRL_DT)
    frames = []
    rewards = []
    speeds = []
    lateral_drifts = []
    distances = []
    action_l2_per_step = []
    action_rate_l2_per_step = []
    action_l2_per_m = []
    path_efficiencies = []
    slip_proxies = []
    propulsion_efficiencies = []
    successes = []
    terminations = 0

    for ep in range(args.episodes):
        obs, _ = raw_env.reset(seed=args.seed + ep)
        raw_env.set_command(
            velocity=args.cmd_vel, yaw_rate=args.cmd_yaw,
            gait_blend=gait_blend)
        obs = raw_env._get_obs()
        start_pos = raw_env.data.xpos[raw_env._root_body_id].copy()
        prev_pos = start_pos.copy()
        path_length = 0.0
        ep_reward = 0.0
        steps = 0
        ep_action_l2 = 0.0
        ep_action_rate_l2 = 0.0
        prev_applied_action = np.zeros(raw_env.action_space.shape, dtype=np.float32)
        episode_terminated = False

        for step in range(max_steps):
            obs_batch = obs.reshape(1, -1)
            if norm_path is not None:
                obs_batch = norm_env.normalize_obs(obs_batch)
            action, _ = model.predict(obs_batch, deterministic=True)
            obs, reward, terminated, truncated, _ = raw_env.step(action[0])
            current_pos = raw_env.data.xpos[raw_env._root_body_id].copy()
            path_length += float(np.linalg.norm(current_pos[:2] - prev_pos[:2]))
            prev_pos = current_pos
            applied_action = raw_env._last_action.copy()
            ep_action_l2 += float(np.sum(np.square(applied_action)))
            ep_action_rate_l2 += float(np.sum(
                np.square(applied_action - prev_applied_action)))
            prev_applied_action = applied_action
            ep_reward += reward
            steps += 1

            if args.video and ep == 0:
                frame = raw_env.render()
                if frame is not None:
                    frames.append(frame.copy())

            if terminated or truncated:
                terminations += int(terminated)
                episode_terminated = bool(terminated)
                break

        end_pos = raw_env.data.xpos[raw_env._root_body_id].copy()
        distance = -(end_pos[0] - start_pos[0])
        lateral = abs(end_pos[1] - start_pos[1])
        elapsed = max(steps * CTRL_DT, CTRL_DT)
        forward_distance = max(distance, 0.0)
        path_efficiency = forward_distance / max(path_length, 1e-6)
        path_efficiency = float(np.clip(path_efficiency, 0.0, 1.0))
        slip_proxy = 1.0 - path_efficiency
        propulsion_efficiency = forward_distance / max(ep_action_l2, 1e-6)
        success = (
            distance >= args.success_distance and not episode_terminated)
        rewards.append(ep_reward)
        distances.append(distance)
        speeds.append(distance / elapsed)
        lateral_drifts.append(lateral)
        action_l2_per_step.append(ep_action_l2 / max(steps, 1))
        action_rate_l2_per_step.append(ep_action_rate_l2 / max(steps, 1))
        action_l2_per_m.append(ep_action_l2 / max(distance, 1e-6))
        path_efficiencies.append(path_efficiency)
        slip_proxies.append(slip_proxy)
        propulsion_efficiencies.append(propulsion_efficiency)
        successes.append(float(success))
        print(
            f"ep {ep + 1}/{args.episodes}: "
            f"speed={speeds[-1] * 1000:.2f} mm/s "
            f"distance={distance * 1000:.1f} mm "
            f"lateral={lateral * 1000:.1f} mm "
            f"slip_proxy={slip_proxy:.2f} "
            f"success={int(success)} "
            f"action_l2/m={action_l2_per_m[-1]:.2f} reward={ep_reward:.2f}")

    metrics = {
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": gait_blend,
        "cmd_vel_m_s": args.cmd_vel,
        "cmd_yaw_rad_s": args.cmd_yaw,
        "eval_command": {
            "cmd_vel_m_s": args.cmd_vel,
            "cmd_yaw_rad_s": args.cmd_yaw,
            "command_resample_prob": 0.0,
        },
        "obs_dim": OBS_DIM,
        "eval_condition": args.eval_condition,
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "action_adapter": action_adapter_contract(),
        "reward_contract": reward_contract(),
        "control_timing": {
            "control_dt_s": CTRL_DT,
            "control_rate_hz": 1.0 / CTRL_DT,
            "peristaltic_actuation_period_s": (
                PERISTALTIC_ACTUATION_PERIOD_S),
            "phase_freq_hz": PHASE_FREQ,
        },
        "sensor_noise": sensor_kwargs,
        "episodes": args.episodes,
        "time_s": args.time,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_distance_mm": float(np.mean(distances) * 1000.0),
        "mean_speed_mm_s": float(np.mean(speeds) * 1000.0),
        "std_speed_mm_s": float(np.std(speeds) * 1000.0),
        "mean_lateral_drift_mm": float(np.mean(lateral_drifts) * 1000.0),
        "mean_action_l2_per_step": float(np.mean(action_l2_per_step)),
        "mean_action_rate_l2_per_step": float(np.mean(action_rate_l2_per_step)),
        "mean_action_l2_per_m": float(np.mean(action_l2_per_m)),
        "mean_path_efficiency": float(np.mean(path_efficiencies)),
        "mean_slip_proxy": float(np.mean(slip_proxies)),
        "mean_propulsion_efficiency_m_per_action_l2": float(
            np.mean(propulsion_efficiencies)),
        "success_distance_m": args.success_distance,
        "success_rate": float(np.mean(successes)),
        "slope_success_rate": (
            float(np.mean(successes)) if args.terrain == "slope" else None),
        "sand_slip_proxy": (
            float(np.mean(slip_proxies)) if args.terrain == "sand" else None),
        "sand_propulsion_efficiency_m_per_action_l2": (
            float(np.mean(propulsion_efficiencies))
            if args.terrain == "sand" else None),
        "termination_rate": float(terminations / max(args.episodes, 1)),
        "model_path": model_path,
    }

    json_out = args.json_out or os.path.join(run_dir, "eval_metrics.json")
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {json_out}")

    if args.video and frames:
        try:
            import mediapy
            vid_dir = os.path.join(PROJECT_ROOT, "record", "v6", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(
                vid_dir, f"eval_{args.terrain}_{args.gait_mode}.mp4")
            mediapy.write_video(vid_path, frames, fps=30)
            print(f"Saved video: {vid_path}")
        except ImportError:
            print("mediapy not installed; video not saved")

    raw_env.close()
    norm_env.close()
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate Worm V6 deployable policy")
    ap.add_argument("--model", default=None, help="Path to PPO .zip model")
    ap.add_argument("--run-dir", default=None, help="Run directory override")
    ap.add_argument("--terrain", default="flat",
                    choices=["flat", "sand", "slope", "rough", "steps", "channel"])
    ap.add_argument("--gait-mode", default="mixed",
                    choices=["worm", "snake", "mixed", "random"])
    ap.add_argument("--gait-blend", type=float, default=None)
    ap.add_argument("--cmd-vel", type=float, default=CMD_VEL_RANGE[1])
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--success-distance", type=float, default=0.05,
                    help="Minimum forward distance in meters for success")
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
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    args.cmd_vel = float(np.clip(args.cmd_vel, *CMD_VEL_RANGE))
    args.cmd_yaw = float(np.clip(args.cmd_yaw, *CMD_YAW_RANGE))
    evaluate(args)


if __name__ == "__main__":
    main()

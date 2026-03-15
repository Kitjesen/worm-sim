"""
Evaluate trained V6 worm policy — record video with different commands.
Shows: straight, left turn, right turn.
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from worm_env_v6 import WormEnvV6, CMD_VEL_RANGE, CMD_YAW_RANGE

RUN_DIR = os.path.join(PROJECT_ROOT, "runs", "worm_v6_ppo")


def evaluate_and_record():
    # Load model
    model_path = os.path.join(RUN_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(RUN_DIR, "final_model.zip")
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Raw env for rendering (no VecNormalize wrapper)
    raw_env = WormEnvV6()

    # Also need VecNormalize for obs normalization during inference
    vec_env = DummyVecEnv([lambda: WormEnvV6()])
    norm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize: {norm_path}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               training=False)
        print("WARNING: No VecNormalize found, using fresh normalization")

    # Test scenarios
    scenarios = [
        ("straight_fast", 0.025, 0.0,  "Full speed straight (25 mm/s)"),
        ("straight_slow", 0.012, 0.0,  "Half speed straight (12 mm/s)"),
        ("turn_left",     0.020, -0.25, "Forward + left turn"),
        ("turn_right",    0.020,  0.25, "Forward + right turn"),
        ("stop",          0.0,   0.0,  "Stop (zero command)"),
    ]

    vid_dir = os.path.join(PROJECT_ROOT, "record", "v6", "videos")
    os.makedirs(vid_dir, exist_ok=True)

    for name, cmd_vel, cmd_yaw, desc in scenarios:
        print(f"\n{'─'*60}")
        print(f"  Scenario: {desc}")
        print(f"  cmd_vel={cmd_vel:.3f} m/s, cmd_yaw={cmd_yaw:.2f} rad/s")

        # Reset both envs
        obs_raw, _ = raw_env.reset(seed=42)
        vec_obs = vec_env.reset()

        # Override command
        raw_env._cmd_vel = cmd_vel
        raw_env._cmd_yaw = cmd_yaw
        # Also override in vec env's underlying env
        vec_env.envs[0].unwrapped._cmd_vel = cmd_vel
        vec_env.envs[0].unwrapped._cmd_yaw = cmd_yaw

        frames = []
        total_reward = 0.0
        positions = []

        for step in range(500):  # 10 seconds
            # Get normalized obs for policy
            # We need to manually normalize the raw_env obs
            obs_for_policy = vec_env.normalize_obs(obs_raw.reshape(1, -1))
            action, _ = model.predict(obs_for_policy, deterministic=True)
            action = action.flatten()

            # Step raw env
            obs_raw, reward, terminated, truncated, _ = raw_env.step(action)
            total_reward += reward

            # Also step vec env to keep it in sync (for obs normalization)
            # Actually we just need the raw env, normalize manually

            # Record position
            root_id = raw_env._root_body_id
            pos = raw_env.data.xpos[root_id].copy()
            positions.append(pos)

            # Render frame
            frame = raw_env.render()
            if frame is not None:
                frames.append(frame.copy())

            if terminated:
                print(f"  TERMINATED at step {step}")
                break

        # Results
        p0 = positions[0]
        pf = positions[-1]
        disp_x = -(pf[0] - p0[0])  # forward is -X
        disp_y = pf[1] - p0[1]     # lateral
        speed = disp_x / (len(positions) * 0.02)

        print(f"  Steps: {len(positions)}")
        print(f"  Forward (-X): {disp_x*1000:.1f} mm  ({speed*1000:.2f} mm/s)")
        print(f"  Lateral (Y):  {disp_y*1000:.1f} mm")
        print(f"  Total reward: {total_reward:.1f}")

        # Save video
        if frames:
            try:
                import mediapy
                vid_path = os.path.join(vid_dir, f"eval_{name}.mp4")
                mediapy.write_video(vid_path, frames, fps=30)
                print(f"  Video: {vid_path} ({len(frames)} frames)")
            except ImportError:
                print("  mediapy not installed — no video saved")

    raw_env.close()
    vec_env.close()


if __name__ == "__main__":
    evaluate_and_record()

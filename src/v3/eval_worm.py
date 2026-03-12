"""
Worm Robot V5.1 — Evaluate Trained Policy
==========================================
Load a trained PPO model and run it with visualization + optional video recording.

Usage:
    # Evaluate with rendering
    python eval_worm.py runs/worm_ppo/best_model.zip

    # Record video
    python eval_worm.py runs/worm_ppo/best_model.zip --video

    # Run open-loop baseline for comparison
    python eval_worm.py --baseline snake
"""

import os
import sys
import math
import argparse
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from worm_env import WormEnv, N_SLIDES, N_YAWS, N_ACTUATORS, CTRL_DT
from worm_v5_1 import (
    inject_strips, NUM_SEGMENTS, SLIDE_RANGE, YAW_RANGE,
    GAIT, STEP_DURATION,
)


def open_loop_snake(t, amp=0.30, freq=0.5, waves=1.0):
    """Open-loop snake gait (baseline for comparison)."""
    action = np.zeros(N_ACTUATORS, dtype=np.float32)
    for j in range(N_YAWS):
        phase = 2.0 * math.pi * freq * t + 2.0 * math.pi * waves * j / N_YAWS
        action[N_SLIDES + j] = amp * math.sin(phase) / YAW_RANGE  # normalize to [-1,1]
    return action


def open_loop_worm(t, step_dur=STEP_DURATION):
    """Open-loop peristaltic gait (baseline)."""
    action = np.zeros(N_ACTUATORS, dtype=np.float32)
    step_idx = int(t / step_dur) % NUM_SEGMENTS
    for j in range(N_SLIDES):
        states = [GAIT[(seg + step_idx) % NUM_SEGMENTS]
                  for seg in range(NUM_SEGMENTS)]
        if states[j + 1] == 0:
            action[j] = 1.0      # extend
        else:
            action[j] = -1.0     # contract
    return action


def open_loop_combined(t, snake_amp=0.30, snake_freq=0.5):
    """Open-loop combined gait (baseline)."""
    a_worm = open_loop_worm(t)
    a_snake = open_loop_snake(t, amp=snake_amp, freq=snake_freq)
    action = a_worm.copy()
    action[N_SLIDES:] = a_snake[N_SLIDES:]
    return action


def _find_vecnormalize(model_path):
    """Search for VecNormalize .pkl file near the model path."""
    candidates = [
        model_path.replace(".zip", "_vecnormalize.pkl"),
        # checkpoints/ → parent run dir
        os.path.join(os.path.dirname(os.path.dirname(model_path)),
                     "final_model_vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_path), "vecnormalize.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def evaluate(model_path, args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = WormEnv()
    print(f"Worm V5.1 — Evaluation")
    print(f"  Model: {model_path}")

    # Load model and normalization
    model = PPO.load(model_path, device="cpu")
    vec_env = None
    norm_path = _find_vecnormalize(model_path)
    if norm_path:
        vec_env = DummyVecEnv([lambda: WormEnv()])
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  VecNormalize loaded from: {norm_path}")
    else:
        print(f"  WARNING: VecNormalize not found — obs may be wrong!")

    return _run_eval(env, args,
                     policy_fn=lambda obs: model.predict(obs, deterministic=True)[0],
                     vec_env=vec_env,
                     label="RL Policy")


def evaluate_baseline(mode, args):
    """Run open-loop baseline for comparison."""
    env = WormEnv()
    print(f"Worm V5.1 — Baseline: {mode}")

    baseline_fns = {
        "snake": open_loop_snake,
        "worm": open_loop_worm,
        "combined": open_loop_combined,
    }
    base_fn = baseline_fns[mode]

    def policy_fn(obs):
        t = env._step_count * CTRL_DT
        return base_fn(t)

    return _run_eval(env, args, policy_fn=policy_fn, label=f"Baseline ({mode})")


def _run_eval(env, args, policy_fn, label="Policy", vec_env=None):
    """Run evaluation loop with optional video recording."""
    n_episodes = args.episodes
    sim_time = args.time
    max_steps = int(sim_time / CTRL_DT)

    frames = []
    all_rewards = []
    all_distances = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 100)
        ep_reward = 0.0
        start_y = env.data.xpos[env._seg_ids[-1], 1]

        for step in range(max_steps):
            # Normalize obs if VecNormalize is available
            if vec_env is not None:
                obs_norm = vec_env.normalize_obs(obs)
                action = policy_fn(obs_norm)
            else:
                action = policy_fn(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            # Collect video frames
            if args.video and ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if terminated or truncated:
                break

        end_y = env.data.xpos[env._seg_ids[-1], 1]
        distance = (end_y - start_y) * 1000  # mm
        all_rewards.append(ep_reward)
        all_distances.append(distance)

        head = env.data.xpos[env._seg_ids[-1]] * 1000
        print(f"  ep {ep+1}/{n_episodes}: reward={ep_reward:>8.2f}  "
              f"distance={distance:>7.1f}mm  "
              f"head=({head[0]:.1f}, {head[1]:.1f}, {head[2]:.1f})mm")

    # Summary
    print(f"\n{'─'*50}")
    print(f"  {label} — {n_episodes} episodes, {sim_time}s each")
    print(f"  Mean reward:   {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean distance: {np.mean(all_distances):.1f} ± {np.std(all_distances):.1f} mm")
    print(f"  Mean speed:    {np.mean(all_distances)/sim_time:.2f} mm/s")
    print(f"{'─'*50}")

    # Save video
    if args.video and frames:
        try:
            import mediapy
            project_root = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
            vid_dir = os.path.join(project_root, "record", "v5_1", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"worm_v5_1_eval_{label.lower().replace(' ','_')}.mp4")
            mediapy.write_video(vid_path, frames, fps=30)
            print(f"  Video saved: {vid_path} ({len(frames)} frames)")
        except ImportError:
            print("  mediapy not found — video not saved")

    env.close()
    return np.mean(all_rewards), np.mean(all_distances)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate trained worm policy")
    ap.add_argument("model", nargs="?", default=None,
                    help="Path to trained model .zip file")
    ap.add_argument("--baseline", choices=["snake", "worm", "combined"],
                    default=None, help="Run open-loop baseline instead")
    ap.add_argument("--video", action="store_true",
                    help="Record evaluation video")
    ap.add_argument("--episodes", type=int, default=3,
                    help="Number of evaluation episodes")
    ap.add_argument("--time", type=float, default=20.0,
                    help="Simulation time per episode (seconds)")
    args = ap.parse_args()

    if args.baseline:
        evaluate_baseline(args.baseline, args)
    elif args.model:
        evaluate(args.model, args)
    else:
        print("Usage:")
        print("  python eval_worm.py <model.zip>          # evaluate trained policy")
        print("  python eval_worm.py --baseline snake     # run baseline")

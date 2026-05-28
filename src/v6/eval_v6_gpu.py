"""
Evaluate MJX-trained worm policy using CPU MuJoCo (for rendering/video).
========================================================================
Loads brax-trained params and runs scenarios on the CPU WormEnvV6 environment.
"""

import os
import sys
import pickle
import numpy as np

import jax
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from worm_env_v6 import WormEnvV6, CMD_VEL_RANGE, CMD_YAW_RANGE


def main():
    run_dir = os.path.join(PROJECT_ROOT, "runs", "worm_v6_mjx")

    # ── Load trained policy ──
    try:
        import cloudpickle
    except ImportError:
        print("ERROR: cloudpickle required. pip install cloudpickle")
        return

    inf_path = os.path.join(run_dir, "make_inference_fn.pkl")
    params_path = os.path.join(run_dir, "params.pkl")

    if not os.path.exists(inf_path) or not os.path.exists(params_path):
        print(f"ERROR: Model not found in {run_dir}")
        print(f"  Expected: {inf_path}")
        print(f"            {params_path}")
        return

    print(f"Loading model from {run_dir}")
    with open(inf_path, "rb") as f:
        make_inference_fn = cloudpickle.load(f)
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    inference_fn = make_inference_fn(params, deterministic=True)
    print("  Policy loaded OK")

    # ── CPU env for rendering ──
    raw_env = WormEnvV6()
    rng = jax.random.PRNGKey(42)

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
        print(f"\n{'─' * 60}")
        print(f"  Scenario: {desc}")
        print(f"  cmd_vel={cmd_vel:.3f} m/s, cmd_yaw={cmd_yaw:.2f} rad/s")

        obs, _ = raw_env.reset(seed=42)
        raw_env._cmd_vel = cmd_vel
        raw_env._cmd_yaw = cmd_yaw

        frames, positions = [], []
        total_reward = 0.0

        for step in range(500):
            rng, act_rng = jax.random.split(rng)

            # Policy inference (add batch dim, remove after)
            obs_jax = jnp.array(obs.reshape(1, -1))
            action_jax, _ = inference_fn(obs_jax, act_rng)
            action = np.array(action_jax[0])

            obs, reward, terminated, truncated, _ = raw_env.step(action)
            total_reward += reward
            positions.append(
                raw_env.data.xpos[raw_env._root_body_id].copy())

            frame = raw_env.render()
            if frame is not None:
                frames.append(frame.copy())

            if terminated:
                print(f"  TERMINATED at step {step}")
                break

        # Results
        p0, pf = positions[0], positions[-1]
        disp_x = -(pf[0] - p0[0])
        disp_y = pf[1] - p0[1]
        speed = disp_x / (len(positions) * 0.02)

        print(f"  Steps:   {len(positions)}")
        print(f"  Forward: {disp_x * 1000:.1f} mm  ({speed * 1000:.2f} mm/s)")
        print(f"  Lateral: {disp_y * 1000:.1f} mm")
        print(f"  Reward:  {total_reward:.1f}")

        # Save video
        if frames:
            try:
                import mediapy
                vid_path = os.path.join(vid_dir, f"eval_mjx_{name}.mp4")
                mediapy.write_video(vid_path, frames, fps=30)
                print(f"  Video:   {vid_path} ({len(frames)} frames)")
            except ImportError:
                print("  mediapy not installed — no video saved")

    raw_env.close()


if __name__ == "__main__":
    main()

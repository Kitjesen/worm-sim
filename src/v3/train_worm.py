"""
Worm Robot V5.1 — RL Training Script
======================================
Train the worm robot with PPO using Stable-Baselines3.

Usage:
    # Quick local test (CPU, few steps)
    python train_worm.py --test

    # Full local training (CPU, slow but functional)
    python train_worm.py --timesteps 500000

    # Resume from checkpoint
    python train_worm.py --resume runs/worm_ppo/best_model.zip
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
RUN_DIR     = os.path.join(PROJECT_ROOT, "runs", "worm_ppo")
LOG_DIR     = os.path.join(RUN_DIR, "logs")
CKPT_DIR    = os.path.join(RUN_DIR, "checkpoints")


def make_env(seed=0):
    """Factory for creating a monitored WormEnv instance."""
    def _init():
        from worm_env import WormEnv
        env = WormEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


class NormSyncCallback(BaseCallback):
    """Sync VecNormalize stats from train → eval, and save alongside best model."""
    def __init__(self, train_env, eval_env, save_path, print_freq=5000):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.save_path = save_path
        self.print_freq = print_freq
        self._last_best = None

    def _on_step(self):
        # Sync obs normalization every step (lightweight)
        self.eval_env.obs_rms = self.train_env.obs_rms

        # Check if EvalCallback saved a new best model → save VecNormalize too
        best_path = os.path.join(self.save_path, "best_model.zip")
        if os.path.exists(best_path):
            mtime = os.path.getmtime(best_path)
            if self._last_best is None or mtime > self._last_best:
                self._last_best = mtime
                norm_path = os.path.join(self.save_path,
                                         "best_model_vecnormalize.pkl")
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


def train(args):
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    n_envs = args.n_envs
    print(f"Worm RL Training — PPO")
    print(f"  envs:       {n_envs}")
    print(f"  timesteps:  {args.timesteps:,}")
    print(f"  run_dir:    {RUN_DIR}")

    # ── Create vectorized environments ──
    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(seed=42)])
    else:
        vec_env = SubprocVecEnv([make_env(seed=42 + i) for i in range(n_envs)])

    # Observation normalization only — reward normalization DISABLED
    # (reward norm makes the signal non-stationary, preventing learning)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # ── Tensorboard (optional) ──
    try:
        import tensorboard  # noqa: F401
        tb_log = LOG_DIR
        print(f"  tensorboard: {LOG_DIR}")
    except ImportError:
        tb_log = None
        print(f"  tensorboard: not installed (logging disabled)")

    # ── Create or load model ──
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
        # Load normalization stats if available
        norm_path = args.resume.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env)
            print(f"  Loaded VecNormalize from: {norm_path}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,          # steps per env per update
            batch_size=512,        # larger batch for stability
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,         # higher exploration for discovery
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            tensorboard_log=tb_log,
            verbose=0,
            device="cpu",          # MLP policy is faster on CPU for PPO
            seed=42,
        )

    print(f"  Policy network: {model.policy}")

    # ── Callbacks ──
    eval_env = DummyVecEnv([make_env(seed=999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)
    eval_env.obs_rms = vec_env.obs_rms  # initial sync

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=RUN_DIR,
        log_path=LOG_DIR,
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // n_envs, 1),
        save_path=CKPT_DIR,
        name_prefix="worm_ppo",
    )

    norm_sync = NormSyncCallback(
        train_env=vec_env, eval_env=eval_env,
        save_path=RUN_DIR,
        print_freq=max(5000 // n_envs, 1),
    )

    # ── Train ──
    print(f"\n  Training started...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback, norm_sync],
        progress_bar=True,
    )

    # ── Save ──
    final_path = os.path.join(RUN_DIR, "final_model")
    model.save(final_path)
    vec_env.save(f"{final_path}_vecnormalize.pkl")
    print(f"\n  Saved final model: {final_path}.zip")
    print(f"  Saved VecNormalize: {final_path}_vecnormalize.pkl")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train worm robot with PPO")
    ap.add_argument("--timesteps", type=int, default=1_000_000,
                    help="Total training timesteps")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="Number of parallel environments")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to model checkpoint to resume from")
    ap.add_argument("--test", action="store_true",
                    help="Quick test run (10k steps, 1 env)")
    args = ap.parse_args()

    if args.test:
        args.timesteps = 10_000
        args.n_envs = 1
        print("=== TEST MODE (10k steps, 1 env) ===")

    # Ensure worm_env is importable
    import sys
    sys.path.insert(0, SCRIPT_DIR)

    train(args)

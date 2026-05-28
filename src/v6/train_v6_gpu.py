"""
GPU-accelerated PPO training for Worm V6 using MJX + brax.
==========================================================
Runs thousands of parallel environments on GPU for massive throughput.

Network: [256, 256, 128] policy / [512, 256, 128] value with swish activation.

Usage:
    python train_v6_gpu.py                        # 10M steps, 2048 envs
    python train_v6_gpu.py --test                  # Quick 100k test
    python train_v6_gpu.py --timesteps 50000000    # 50M steps
"""

import os
import sys
import time
import functools
import argparse
import json
import pickle

import jax
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Worm V6 GPU training (MJX + brax PPO)")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--test", action="store_true",
                        help="Quick 100k test run with 64 envs")
    args = parser.parse_args()

    if args.test:
        args.timesteps = 100_000
        args.num_envs = 64
        args.batch_size = 64
        args.num_minibatches = 4
        args.num_evals = 5
    else:
        args.num_evals = 50

    run_dir = os.path.join(PROJECT_ROOT, "runs", "worm_v6_mjx")
    os.makedirs(run_dir, exist_ok=True)

    print("Worm V6 GPU Training — MJX + brax PPO")
    print(f"  JAX devices:  {jax.devices()}")
    print(f"  num_envs:     {args.num_envs}")
    print(f"  timesteps:    {args.timesteps:,}")
    print(f"  unroll_len:   {args.unroll_length}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  lr:           {args.lr}")
    print(f"  entropy:      {args.entropy}")
    print(f"  run_dir:      {run_dir}")

    # ── Create environment ──
    from worm_env_v6_mjx import WormMJXEnv
    env = WormMJXEnv()
    print(f"  obs_dim:      {env.observation_size}")
    print(f"  action_dim:   {env.action_size}")

    # ── Network: larger MLP with swish activation ──
    from brax.training.agents.ppo.train import train as ppo_train
    from brax.training.agents.ppo import networks as ppo_networks

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )

    # ── Progress tracking ──
    eval_metrics = []
    t0 = time.time()

    def progress_fn(num_steps, metrics):
        elapsed = time.time() - t0
        fps = num_steps / max(elapsed, 1)
        r = float(metrics.get('eval/episode_reward', 0))
        print(f"  {num_steps:>12,} steps  reward={r:>8.1f}  "
              f"fps={fps:>10,.0f}  elapsed={elapsed:.0f}s")
        eval_metrics.append({'steps': int(num_steps), 'reward': r})

    # ── Train ──
    train_fn = functools.partial(
        ppo_train,
        num_timesteps=args.timesteps,
        episode_length=1000,
        num_evals=args.num_evals,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        entropy_cost=args.entropy,
        discounting=0.99,
        seed=args.seed,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=4,
        normalize_observations=True,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        network_factory=network_factory,
        progress_fn=progress_fn,
    )

    print("\n  Training...")
    make_inference_fn, params, metrics = train_fn(environment=env)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s  ({args.timesteps / max(elapsed, 1):,.0f} steps/sec)")

    # ── Save ──
    # Policy params
    params_path = os.path.join(run_dir, "params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f)

    # Inference function (includes obs normalizer state)
    try:
        import cloudpickle
        inf_path = os.path.join(run_dir, "make_inference_fn.pkl")
        with open(inf_path, "wb") as f:
            cloudpickle.dump(make_inference_fn, f)
        print(f"  Saved inference fn: {inf_path}")
    except ImportError:
        print("  WARNING: cloudpickle not installed, inference fn not saved")
        print("  Install with: pip install cloudpickle")

    # Metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"  Saved params:  {params_path}")
    print(f"  Saved metrics: {metrics_path}")

    if eval_metrics:
        best = max(eval_metrics, key=lambda x: x['reward'])
        print(f"  Best reward:   {best['reward']:.1f} at {best['steps']:,} steps")


if __name__ == "__main__":
    main()

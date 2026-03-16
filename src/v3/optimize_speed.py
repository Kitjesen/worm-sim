"""
CMA-ES Speed Optimization — Find the fastest open-loop gait
================================================================
Optimizes periodic gait parameters (amplitudes, frequencies, phases)
to maximize forward speed using Covariance Matrix Adaptation ES.

Usage:
    python optimize_speed.py                # Full optimization (~2-4 hours)
    python optimize_speed.py --test         # Quick test (20 generations)
    python optimize_speed.py --popsize 32   # Larger population

Requirements:
    pip install cma
"""

import mujoco
import numpy as np
import math
import os
import sys
import time
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from worm_v6 import (
    build_xml, NUM_SLIDES, NUM_YAWS, NUM_ACTUATORS,
    SLIDE_RANGE_VAL, YAW_RANGE_VAL, BODY_Z,
)

# ─── Simulation Config ─────────────────────────────────────────────
SIM_TIME    = 10.0      # seconds per evaluation
SETTLE_TIME = 1.0       # seconds to settle before gait starts
PHYSICS_DT  = 0.002

# ─── Parameter Encoding ────────────────────────────────────────────
# 14-dimensional search space:
#   [0]     slide_amp       ∈ [0, SLIDE_RANGE_VAL]
#   [1]     slide_freq      ∈ [0.1, 5.0] Hz  (realistic servo limit ~3-5Hz)
#   [2]     slide_wave_n    ∈ [0.5, 3.0] wavelengths across body
#   [3]     yaw_amp         ∈ [0, YAW_RANGE_VAL]
#   [4]     yaw_freq        ∈ [0.1, 5.0] Hz
#   [5]     yaw_wave_n      ∈ [0.5, 3.0] wavelengths
#   [6:12]  slide_phase_bias per joint  ∈ [-π, π]
#   [12:13] step_duration   ∈ [0.3, 2.0] seconds
#   [13]    yaw_slide_coupling ∈ [-1, 1]  (phase coupling between slide and yaw)
N_PARAMS = 14

# Bounds (for clamping after CMA-ES proposal)
BOUNDS_LO = np.array([
    0.0,    # slide_amp
    0.1,    # slide_freq
    0.5,    # slide_wave_n
    0.0,    # yaw_amp
    0.1,    # yaw_freq
    0.5,    # yaw_wave_n
    *[-math.pi]*6,  # slide_phase_bias [6:12]
    0.3,    # step_duration
    -1.0,   # coupling
])
BOUNDS_HI = np.array([
    SLIDE_RANGE_VAL,  # slide_amp
    5.0,    # slide_freq (realistic servo limit)
    3.0,    # slide_wave_n
    YAW_RANGE_VAL,    # yaw_amp
    5.0,    # yaw_freq
    3.0,    # yaw_wave_n
    *[math.pi]*6,     # slide_phase_bias [6:12]
    2.0,    # step_duration
    1.0,    # coupling
])

# Initial guess (current combined gait)
X0 = np.array([
    SLIDE_RANGE_VAL * 0.5,  # slide_amp
    1.25,   # slide_freq (1/STEP_DURATION=0.8)
    1.0,    # slide_wave_n
    0.40,   # yaw_amp (SNAKE_AMP)
    0.40,   # yaw_freq (SNAKE_FREQ)
    1.5,    # yaw_wave_n (SNAKE_WAVES)
    *[0.0]*6,  # no per-joint bias
    0.8,    # step_duration
    0.0,    # no coupling
])
SIGMA0 = 0.3  # initial step size (in normalized space)


def build_model():
    """Build MuJoCo model once, return (model, data, actuator_ids)."""
    mesh_dir = os.path.join(PROJECT_ROOT, "meshes")
    urdf_path = os.path.join(mesh_dir, "longworm2", "longworm2.SLDASM.urdf")

    if not os.path.exists(urdf_path):
        import shutil
        src = os.path.join("D:/inovxio/3d/longworm2/longworm2.SLDASM/urdf",
                           "longworm2.SLDASM.urdf")
        if os.path.exists(src):
            os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
            shutil.copy2(src, urdf_path)

    xml_str = build_xml(mesh_dir, urdf_path)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    slide_ids, yaw_ids = [], []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith('act_back'):
            slide_ids.append(i)
        elif name.startswith('act_front'):
            yaw_ids.append(i)

    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    return model, data, slide_ids, yaw_ids, head_id


def evaluate(params, model, data, slide_ids, yaw_ids, head_id, mode='full'):
    """Simulate gait and return negative forward displacement (CMA-ES minimizes)."""
    # Clamp parameters
    p = np.clip(params, BOUNDS_LO, BOUNDS_HI)

    slide_amp    = p[0]
    slide_freq   = p[1]
    slide_wave_n = p[2]
    yaw_amp      = p[3]
    yaw_freq     = p[4]
    yaw_wave_n   = p[5]

    # Constrain by gait mode
    if mode == 'peristaltic':
        yaw_amp = 0.0
    elif mode == 'serpentine':
        slide_amp = 0.0
    slide_bias   = p[6:12]
    step_dur     = p[12]
    coupling     = p[13]

    n_slides = len(slide_ids)
    n_yaws   = len(yaw_ids)

    # Reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Settle
    settle_steps = int(SETTLE_TIME / PHYSICS_DT)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    if np.any(np.isnan(data.qpos)):
        return 1e6  # penalty for NaN

    # Record initial position
    x0 = data.xpos[head_id, 0]

    # Simulate with gait
    sim_steps = int(SIM_TIME / PHYSICS_DT)
    for step in range(sim_steps):
        t = step * PHYSICS_DT

        # Peristaltic slide wave
        for j in range(n_slides):
            phase = (2.0 * math.pi
                     * (t * slide_freq - slide_wave_n * j / n_slides)
                     + slide_bias[j])
            data.ctrl[slide_ids[j]] = -slide_amp * (1.0 + math.sin(phase))

        # Serpentine yaw wave (with optional coupling to slide phase)
        for j in range(n_yaws):
            phase = (2.0 * math.pi * yaw_freq * t
                     + 2.0 * math.pi * yaw_wave_n * j / n_yaws
                     + coupling * 2.0 * math.pi * t * slide_freq)
            data.ctrl[yaw_ids[j]] = yaw_amp * math.sin(phase)

        mujoco.mj_step(model, data)

        if np.any(np.isnan(data.qpos)):
            return 1e6

        # Mid-sim stability check every 0.5s
        if step % int(0.5 / PHYSICS_DT) == 0 and step > 0:
            z_mid = data.xpos[head_id, 2]
            if z_mid < 0.02 or z_mid > 0.25:
                return 1e6

    # Forward displacement (-X direction)
    xf = data.xpos[head_id, 0]
    yf = data.xpos[head_id, 1]
    forward = -(xf - x0)  # positive = moved forward in -X
    lateral = abs(yf)      # drift from center line (started at y=0)

    # Check stability: penalize if robot flipped or launched
    z = data.xpos[head_id, 2]
    if z < 0.02 or z > 0.25:
        return 1e6

    # Sanity check: reject implausible speeds (>500 mm/s = physics glitch)
    if abs(forward) > SIM_TIME * 0.5:
        return 1e6

    # Must go forward, not backward
    if forward < 0:
        return 1e6

    # Straightness: effective speed = forward minus heavy drift penalty
    # Robot must keep lateral drift < 5% of forward distance
    effective = forward - 15.0 * lateral
    return -effective  # negate: CMA-ES minimizes


def optimize(popsize=16, max_gen=200, test=False, mode='full'):
    """Run CMA-ES optimization.
    mode: 'full' (all params), 'peristaltic' (yaw=0), 'serpentine' (slide=0)
    """
    try:
        import cma
    except ImportError:
        print("ERROR: pip install cma")
        return

    if test:
        max_gen = 20
        popsize = 8

    print(f"CMA-ES Speed Optimization — Worm V6 [{mode.upper()}]")
    print(f"  params:    {N_PARAMS}")
    print(f"  popsize:   {popsize}")
    print(f"  max_gen:   {max_gen}")
    print(f"  sim_time:  {SIM_TIME}s per eval")
    print(f"  total evals: ~{popsize * max_gen}")

    # Build model
    print("  Building model...")
    model, data, slide_ids, yaw_ids, head_id = build_model()
    print(f"  bodies={model.nbody}, DOF={model.nv}, actuators={model.nu}")

    # Normalize search space to [0, 1]
    range_width = BOUNDS_HI - BOUNDS_LO
    x0_norm = (X0 - BOUNDS_LO) / range_width

    # Output directory
    run_dir = os.path.join(PROJECT_ROOT, "runs", f"cmaes_speed_{mode}")
    os.makedirs(run_dir, exist_ok=True)

    # CMA-ES options
    opts = cma.CMAOptions()
    opts['popsize'] = popsize
    opts['maxiter'] = max_gen
    opts['bounds'] = [0, 1]  # normalized
    opts['tolfun'] = 1e-6
    opts['verb_disp'] = 1
    opts['verb_log'] = 0

    es = cma.CMAEvolutionStrategy(x0_norm, SIGMA0, opts)

    best_fitness = 1e6
    best_params = None
    best_speed = 0.0
    history = []
    t0 = time.time()

    print(f"\n  Optimizing...")
    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for sol in solutions:
            # Denormalize
            params = BOUNDS_LO + sol * range_width
            f = evaluate(params, model, data, slide_ids, yaw_ids, head_id, mode=mode)
            fitnesses.append(f)

        es.tell(solutions, fitnesses)

        # Track best
        gen_best_idx = np.argmin(fitnesses)
        gen_best_f = fitnesses[gen_best_idx]
        gen_best_speed = -gen_best_f / SIM_TIME * 1000  # mm/s

        if gen_best_f < best_fitness:
            best_fitness = gen_best_f
            best_params = BOUNDS_LO + solutions[gen_best_idx] * range_width
            best_speed = -best_fitness / SIM_TIME * 1000

        gen_mean_speed = -np.mean(fitnesses) / SIM_TIME * 1000
        elapsed = time.time() - t0

        history.append({
            'gen': gen,
            'best_speed_mm_s': round(best_speed, 2),
            'gen_best_speed_mm_s': round(gen_best_speed, 2),
            'gen_mean_speed_mm_s': round(gen_mean_speed, 2),
            'elapsed_s': round(elapsed, 1),
        })

        if gen % 5 == 0 or gen_best_f < best_fitness + 0.001:
            print(f"  gen {gen:>4d}  best={best_speed:>7.2f} mm/s  "
                  f"gen_best={gen_best_speed:>7.2f}  "
                  f"gen_mean={gen_mean_speed:>7.2f}  "
                  f"[{elapsed:.0f}s]")

        gen += 1

    # ── Results ──
    total_time = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  CMA-ES Optimization Complete")
    print(f"  Best speed:      {best_speed:.2f} mm/s")
    print(f"  Best displacement: {-best_fitness*1000:.1f} mm in {SIM_TIME}s")
    print(f"  Generations:     {gen}")
    print(f"  Total evals:     {gen * popsize}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'═'*60}")

    # Decode best params
    print(f"\n  Best parameters:")
    print(f"    slide_amp:      {best_params[0]*1000:.2f} mm")
    print(f"    slide_freq:     {best_params[1]:.3f} Hz")
    print(f"    slide_wave_n:   {best_params[2]:.3f}")
    print(f"    yaw_amp:        {best_params[3]:.3f} rad")
    print(f"    yaw_freq:       {best_params[4]:.3f} Hz")
    print(f"    yaw_wave_n:     {best_params[5]:.3f}")
    print(f"    slide_bias:     {np.round(best_params[6:12], 3).tolist()}")
    print(f"    step_duration:  {best_params[12]:.3f} s")
    print(f"    coupling:       {best_params[13]:.3f}")

    # Save results
    results = {
        'best_speed_mm_s': round(best_speed, 2),
        'best_displacement_mm': round(-best_fitness * 1000, 2),
        'sim_time_s': SIM_TIME,
        'best_params': best_params.tolist(),
        'param_names': [
            'slide_amp', 'slide_freq', 'slide_wave_n',
            'yaw_amp', 'yaw_freq', 'yaw_wave_n',
            'slide_bias_0', 'slide_bias_1', 'slide_bias_2',
            'slide_bias_3', 'slide_bias_4', 'slide_bias_5',
            'step_duration', 'yaw_slide_coupling',
        ],
        'generations': gen,
        'total_evals': gen * popsize,
        'total_time_s': round(total_time, 1),
        'history': history,
    }

    results_path = os.path.join(run_dir, "best_gait.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CMA-ES worm speed optimization")
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--max-gen", type=int, default=200)
    ap.add_argument("--test", action="store_true", help="Quick test (20 gen)")
    ap.add_argument("--mode", choices=["full", "peristaltic", "serpentine"],
                    default="full",
                    help="Gait type: full (slide+yaw), peristaltic (slide only), serpentine (yaw only)")
    ap.add_argument("--all", action="store_true",
                    help="Run all 3 modes sequentially")
    args = ap.parse_args()

    if args.all:
        for m in ["peristaltic", "serpentine", "full"]:
            optimize(popsize=args.popsize, max_gen=args.max_gen,
                     test=args.test, mode=m)
    else:
        optimize(popsize=args.popsize, max_gen=args.max_gen,
                 test=args.test, mode=args.mode)

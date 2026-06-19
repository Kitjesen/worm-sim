"""
Search an in-place yaw primitive for Worm V6.

The searched primitive is constrained to the deployable 1 s phase clock and the
same 11-D motor target used by the residual policy adapter. A candidate must
turn both left and right while minimizing planar drift.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from action_adapter_v6 import INPLACE_YAW_PRIOR_PARAMS  # noqa: E402
from eval_v6 import wrap_angle_rad  # noqa: E402
from motor_contract_v6 import normalized_action_to_ctrl  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CTRL_DT,
    N_FRAMES,
    NUM_ACTUATORS,
    NUM_SLIDES,
    NUM_YAWS,
    PHASE_FREQ,
    WormEnvV6,
)

TWO_PI = 2.0 * math.pi

PARAM_NAMES = (
    "slide_bias",
    "slide_amp",
    "slide_freq",
    "slide_wave_n",
    "yaw_amp",
    "yaw_freq",
    "yaw_wave_n",
    "yaw_phase_rad",
)

YAW_RATE_PASS_RAD_S = 0.30
PLANAR_DRIFT_PASS_M_S = 0.05

BOUNDS_LO = np.array([
    0.05,   # slide_bias
    0.00,   # slide_amp
    0.10,   # slide_freq
    -3.00,  # slide_wave_n
    0.10,   # yaw_amp
    0.05,   # yaw_freq
    -3.00,  # yaw_wave_n
    -math.pi,
], dtype=np.float64)

BOUNDS_HI = np.array([
    0.95,
    0.90,
    1.50,
    3.00,
    1.00,
    1.50,
    3.00,
    math.pi,
], dtype=np.float64)

DEFAULT_X0 = np.array(INPLACE_YAW_PRIOR_PARAMS, dtype=np.float64)


def clip_params(params):
    return np.clip(np.asarray(params, dtype=np.float64), BOUNDS_LO, BOUNDS_HI)


def normalized_to_unit(params):
    params = clip_params(params)
    return (params - BOUNDS_LO) / (BOUNDS_HI - BOUNDS_LO)


def unit_to_params(unit_params):
    unit_params = np.clip(np.asarray(unit_params, dtype=np.float64), 0.0, 1.0)
    return BOUNDS_LO + unit_params * (BOUNDS_HI - BOUNDS_LO)


def primitive_action(params, phase, direction_sign):
    p = clip_params(params)
    slide_bias, slide_amp, slide_freq, slide_wave_n = p[:4]
    yaw_amp, yaw_freq, yaw_wave_n, yaw_phase_rad = p[4:]
    t = (float(phase) % TWO_PI) / TWO_PI
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)

    for j in range(NUM_SLIDES):
        joint_phase = (
            TWO_PI * (slide_freq * t - slide_wave_n * j / NUM_SLIDES)
        )
        action[j] = -(
            slide_bias
            + slide_amp * (0.5 + 0.5 * math.sin(joint_phase)))

    for j in range(NUM_YAWS):
        joint_phase = (
            TWO_PI * (yaw_freq * t + yaw_wave_n * j / NUM_YAWS)
            + yaw_phase_rad
        )
        action[NUM_SLIDES + j] = (
            direction_sign * yaw_amp * math.sin(joint_phase))

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def evaluate_side(params, side, seconds, seed, terrain):
    direction_sign = 1.0 if side == "left" else -1.0
    env = WormEnvV6(
        terrain=terrain,
        gait_mode="random",
        fixed_cmd_vx=0.0,
        fixed_cmd_vy=0.0,
        fixed_cmd_yaw=0.5 * direction_sign,
        command_resample_prob=0.0,
    )
    obs, _ = env.reset(seed=seed)
    del obs
    env.set_command(vx=0.0, vy=0.0, yaw_rate=0.5 * direction_sign)

    start_pos = env.data.xpos[env._root_body_id].copy()
    start_yaw = env._root_yaw_rad()
    start_xmat = env.data.xmat[env._root_body_id].reshape(3, 3)
    forward_axis = -start_xmat[:2, 0].copy()
    lateral_axis = start_xmat[:2, 1].copy()

    steps = int(seconds / CTRL_DT)
    terminated = False
    for step in range(steps):
        phase = TWO_PI * PHASE_FREQ * step * CTRL_DT
        action = primitive_action(params, phase, direction_sign)
        env.data.ctrl[:] = normalized_action_to_ctrl(action)
        for _ in range(N_FRAMES):
            mujoco.mj_step(env.model, env.data)
        if env._check_termination():
            terminated = True
            break

    elapsed = max((step + 1) * CTRL_DT, CTRL_DT)
    delta_world = env.data.xpos[env._root_body_id, :2].copy() - start_pos[:2]
    body_vx = float(np.dot(delta_world, forward_axis) / elapsed)
    body_vy = float(np.dot(delta_world, lateral_axis) / elapsed)
    yaw_rate = float(wrap_angle_rad(env._root_yaw_rad() - start_yaw) / elapsed)
    env.close()

    planar_speed = float(np.linalg.norm([body_vx, body_vy]))
    signed_yaw_rate = direction_sign * yaw_rate
    return {
        "side": side,
        "body_vx_m_s": body_vx,
        "body_vy_m_s": body_vy,
        "planar_speed_m_s": planar_speed,
        "yaw_rate_rad_s": yaw_rate,
        "signed_yaw_rate_rad_s": float(signed_yaw_rate),
        "elapsed_s": float(elapsed),
        "terminated": bool(terminated),
    }


def evaluate_params(params, seconds=6.0, seed=42, terrain="flat"):
    left = evaluate_side(params, "left", seconds, seed, terrain)
    right = evaluate_side(params, "right", seconds, seed + 1000, terrain)
    sides = (left, right)
    mean_signed_yaw = float(np.mean([s["signed_yaw_rate_rad_s"] for s in sides]))
    min_signed_yaw = float(min(s["signed_yaw_rate_rad_s"] for s in sides))
    mean_planar = float(np.mean([s["planar_speed_m_s"] for s in sides]))
    max_planar = float(max(s["planar_speed_m_s"] for s in sides))
    action_energy = float(np.mean(np.square([
        primitive_action(params, TWO_PI * i / 20.0, 1.0)
        for i in range(20)
    ])))
    yaw_term = mean_signed_yaw / 0.5
    planar_penalty = max_planar / PLANAR_DRIFT_PASS_M_S
    yaw_deficit = max(0.0, YAW_RATE_PASS_RAD_S - min_signed_yaw) / (
        YAW_RATE_PASS_RAD_S)
    score = (
        yaw_term
        - 0.90 * planar_penalty
        - 1.20 * yaw_deficit
        - 0.02 * action_energy
        - (1.0 if any(s["terminated"] for s in sides) else 0.0)
    )
    return {
        "score": float(score),
        "mean_signed_yaw_rate_rad_s": mean_signed_yaw,
        "min_signed_yaw_rate_rad_s": min_signed_yaw,
        "mean_planar_speed_m_s": mean_planar,
        "max_planar_speed_m_s": max_planar,
        "action_energy": action_energy,
        "left": left,
        "right": right,
    }


def flatten_metrics(metrics):
    return {
        "score": metrics["score"],
        "mean_signed_yaw_rate_rad_s": metrics["mean_signed_yaw_rate_rad_s"],
        "min_signed_yaw_rate_rad_s": metrics["min_signed_yaw_rate_rad_s"],
        "mean_planar_speed_m_s": metrics["mean_planar_speed_m_s"],
        "max_planar_speed_m_s": metrics["max_planar_speed_m_s"],
        "action_energy": metrics["action_energy"],
        "left_body_vx_m_s": metrics["left"]["body_vx_m_s"],
        "left_body_vy_m_s": metrics["left"]["body_vy_m_s"],
        "left_planar_speed_m_s": metrics["left"]["planar_speed_m_s"],
        "left_yaw_rate_rad_s": metrics["left"]["yaw_rate_rad_s"],
        "right_body_vx_m_s": metrics["right"]["body_vx_m_s"],
        "right_body_vy_m_s": metrics["right"]["body_vy_m_s"],
        "right_planar_speed_m_s": metrics["right"]["planar_speed_m_s"],
        "right_yaw_rate_rad_s": metrics["right"]["yaw_rate_rad_s"],
    }


def random_search(trials, seconds, seed, terrain):
    rng = np.random.default_rng(seed)
    rows = []
    best = None
    for trial in range(trials):
        if trial == 0:
            params = DEFAULT_X0.copy()
        else:
            params = unit_to_params(rng.uniform(0.0, 1.0, size=len(PARAM_NAMES)))
        metrics = evaluate_params(params, seconds=seconds,
                                  seed=seed + trial, terrain=terrain)
        row = {
            "trial": trial,
            **flatten_metrics(metrics),
            **{name: float(value) for name, value in zip(PARAM_NAMES, params)},
        }
        rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
    return best, rows


def cma_search(generations, popsize, seconds, seed, terrain):
    import cma

    rows = []
    best = None
    es = cma.CMAEvolutionStrategy(
        normalized_to_unit(DEFAULT_X0),
        0.20,
        {
            "bounds": [0.0, 1.0],
            "popsize": popsize,
            "seed": seed,
            "verbose": -9,
        },
    )
    trial = 0
    for generation in range(generations):
        candidates = es.ask()
        losses = []
        for candidate in candidates:
            params = unit_to_params(candidate)
            metrics = evaluate_params(params, seconds=seconds,
                                      seed=seed + trial, terrain=terrain)
            loss = -metrics["score"]
            losses.append(loss)
            row = {
                "trial": trial,
                "generation": generation,
                **flatten_metrics(metrics),
                **{
                    name: float(value)
                    for name, value in zip(PARAM_NAMES, params)
                },
            }
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
            trial += 1
        es.tell(candidates, losses)
    return best, rows


def params_from_row(row):
    return np.array([float(row[name]) for name in PARAM_NAMES],
                    dtype=np.float64)


def yaw_acceptance(metrics):
    return (
        float(metrics["min_signed_yaw_rate_rad_s"]) >= YAW_RATE_PASS_RAD_S
        and float(metrics["max_planar_speed_m_s"]) <= PLANAR_DRIFT_PASS_M_S
        and not bool(metrics["left"]["terminated"])
        and not bool(metrics["right"]["terminated"])
    )


def write_outputs(out_dir, summary, rows):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "yaw_prior_search_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if rows:
        csv_path = os.path.join(out_dir, "yaw_trials.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Search deployable in-place yaw primitive.")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "record", "current", "flat_omni_v45_yaw_prior_search"))
    ap.add_argument("--method", choices=("random", "cma"), default="cma")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--validation-seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=145)
    return ap.parse_args()


def main():
    args = parse_args()
    started = time.time()
    if args.method == "random":
        best, rows = random_search(args.trials, args.seconds,
                                   args.seed, args.terrain)
    else:
        best, rows = cma_search(args.generations, args.popsize,
                                args.seconds, args.seed, args.terrain)

    print(
        f"best: score={best['score']:.4f}, "
        f"yaw={best['mean_signed_yaw_rate_rad_s']:.4f}, "
        f"planar={best['max_planar_speed_m_s']:.4f}")

    validation = {}
    if args.validation_seconds > 0.0:
        validation = evaluate_params(
            params_from_row(best),
            seconds=args.validation_seconds,
            seed=args.seed + 100000,
            terrain=args.terrain)
        validation["accepted"] = yaw_acceptance(validation)
        print(
            f"validation {args.validation_seconds:.1f}s: "
            f"accepted={validation['accepted']}, "
            f"yaw={validation['mean_signed_yaw_rate_rad_s']:.4f}, "
            f"planar={validation['max_planar_speed_m_s']:.4f}")

    summary = {
        "format_version": 1,
        "created_unix_time": time.time(),
        "elapsed_wall_s": time.time() - started,
        "terrain": args.terrain,
        "method": args.method,
        "seconds_per_eval": args.seconds,
        "parameter_names": list(PARAM_NAMES),
        "parameter_bounds": {
            name: [float(lo), float(hi)]
            for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
        },
        "objective": (
            "score = mean_signed_yaw/0.5 - 0.90*max_planar/0.05 "
            "- 1.20*max(0,0.30-min_signed_yaw)/0.30 "
            "- 0.02*action_energy - termination"),
        "yaw_acceptance": {
            "min_signed_yaw_rate_rad_s_min": YAW_RATE_PASS_RAD_S,
            "max_planar_speed_m_s_max": PLANAR_DRIFT_PASS_M_S,
            "validation_seconds": float(args.validation_seconds),
        },
        "best": best,
        "validation": validation,
    }
    summary_path = write_outputs(args.out_dir, summary, rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

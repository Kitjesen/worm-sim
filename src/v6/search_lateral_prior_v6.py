"""
Search open-loop lateral primitives for Worm V6.

This tool searches a deployable, 1 s phase-clock action primitive for left and
right lateral translation. It does not change the policy ABI: the output is an
11-D normalized motor target that can later become a deterministic prior under
the existing 12-D residual-policy action.
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
    "slide_phase",
    "yaw_amp",
    "yaw_freq",
    "yaw_wave_n",
    "yaw_phase",
    "yaw_bias",
    "yaw_trim_gradient",
)

LATERAL_SPEED_PASS_M_S = 0.03
OFF_AXIS_PASS_M_S = 0.08
YAW_RATE_PASS_RAD_S = 0.20

BOUNDS_LO = np.array([
    -1.00,  # slide_bias, normalized; negative means contraction
    0.00,   # slide_amp
    0.50,   # slide_freq, Hz
    -3.00,  # slide_wave_n
    -math.pi,
    0.00,   # yaw_amp
    0.10,   # yaw_freq, Hz
    -3.00,  # yaw_wave_n
    -math.pi,
    -0.50,  # yaw_bias
    -0.40,  # yaw_trim_gradient along body
], dtype=np.float64)

BOUNDS_HI = np.array([
    0.00,
    1.00,
    1.50,
    3.00,
    math.pi,
    1.00,
    1.50,
    3.00,
    math.pi,
    0.50,
    0.40,
], dtype=np.float64)

DEFAULT_X0 = np.array([
    -0.55,
    0.45,
    1.00,
    1.00,
    -0.5 * math.pi,
    0.45,
    1.00,
    1.00,
    0.0,
    0.0,
    0.0,
], dtype=np.float64)


def clip_params(params):
    return np.clip(np.asarray(params, dtype=np.float64), BOUNDS_LO, BOUNDS_HI)


def normalized_to_unit(params):
    params = clip_params(params)
    return (params - BOUNDS_LO) / (BOUNDS_HI - BOUNDS_LO)


def unit_to_params(unit_params):
    unit_params = np.clip(np.asarray(unit_params, dtype=np.float64), 0.0, 1.0)
    return BOUNDS_LO + unit_params * (BOUNDS_HI - BOUNDS_LO)


def primitive_action(params, phase, direction_sign):
    """Return 11-D normalized motor targets for one lateral primitive."""
    p = clip_params(params)
    slide_bias, slide_amp, slide_freq, slide_wave_n, slide_phase = p[:5]
    yaw_amp, yaw_freq, yaw_wave_n, yaw_phase, yaw_bias, yaw_trim_gradient = p[5:]
    t = (float(phase) % TWO_PI) / TWO_PI
    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)

    for j in range(NUM_SLIDES):
        joint_phase = (
            TWO_PI * (t * slide_freq - slide_wave_n * j / NUM_SLIDES)
            + slide_phase
        )
        contraction = slide_bias - slide_amp * (0.5 + 0.5 * math.sin(joint_phase))
        action[j] = contraction

    center = 0.5 * (NUM_YAWS - 1)
    for j in range(NUM_YAWS):
        joint_phase = (
            TWO_PI * (t * yaw_freq + yaw_wave_n * j / NUM_YAWS)
            + yaw_phase
        )
        body_gradient = 0.0 if center <= 0.0 else (j - center) / center
        action[NUM_SLIDES + j] = direction_sign * (
            yaw_bias
            + yaw_trim_gradient * body_gradient
            + yaw_amp * math.sin(joint_phase)
        )

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def evaluate_params(params, direction, seconds=6.0, seed=42, terrain="flat"):
    direction_sign = 1.0 if direction == "left" else -1.0
    env = WormEnvV6(
        terrain=terrain,
        gait_mode="random",
        fixed_cmd_vx=0.0,
        fixed_cmd_vy=0.15 * direction_sign,
        fixed_cmd_yaw=0.0,
        command_resample_prob=0.0,
    )
    obs, _ = env.reset(seed=seed)
    del obs
    env.set_command(vx=0.0, vy=0.15 * direction_sign, yaw_rate=0.0)

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

    signed_lateral = direction_sign * body_vy
    off_axis = abs(body_vx)
    yaw_abs = abs(yaw_rate)
    action_energy = float(np.mean(np.square([
        primitive_action(params, TWO_PI * i / 20.0, direction_sign)
        for i in range(20)
    ])))
    lateral_term = signed_lateral / 0.15
    forward_penalty = off_axis / 0.08
    yaw_penalty = yaw_abs / 0.20
    lateral_deficit_penalty = max(0.0, 0.03 - signed_lateral) / 0.03
    score = (
        lateral_term
        - 0.70 * forward_penalty
        - 0.50 * yaw_penalty
        - 1.00 * lateral_deficit_penalty
        - 0.02 * action_energy
        - (1.0 if terminated else 0.0)
    )

    return {
        "direction": direction,
        "score": float(score),
        "lateral_term": float(lateral_term),
        "forward_penalty": float(forward_penalty),
        "yaw_penalty": float(yaw_penalty),
        "lateral_deficit_penalty": float(lateral_deficit_penalty),
        "body_vx_m_s": body_vx,
        "body_vy_m_s": body_vy,
        "signed_lateral_m_s": float(signed_lateral),
        "off_axis_forward_m_s": float(off_axis),
        "yaw_rate_rad_s": yaw_rate,
        "elapsed_s": float(elapsed),
        "terminated": bool(terminated),
        "action_energy": action_energy,
    }


def random_search(direction, trials, seconds, seed, terrain):
    rng = np.random.default_rng(seed)
    rows = []
    best = None
    for trial in range(trials):
        if trial == 0:
            params = DEFAULT_X0.copy()
        else:
            params = unit_to_params(rng.uniform(0.0, 1.0, size=len(PARAM_NAMES)))
        metrics = evaluate_params(
            params, direction=direction, seconds=seconds,
            seed=seed + trial, terrain=terrain)
        row = {
            "trial": trial,
            **metrics,
            **{name: float(value) for name, value in zip(PARAM_NAMES, params)},
        }
        rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
    return best, rows


def cma_search(direction, generations, popsize, seconds, seed, terrain):
    import cma

    rows = []
    best = None
    es = cma.CMAEvolutionStrategy(
        normalized_to_unit(DEFAULT_X0),
        0.22,
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
            metrics = evaluate_params(
                params, direction=direction, seconds=seconds,
                seed=seed + trial, terrain=terrain)
            loss = -metrics["score"]
            losses.append(loss)
            row = {
                "trial": trial,
                "generation": generation,
                **metrics,
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


def write_outputs(out_dir, summary, rows_by_direction):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "lateral_prior_search_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for direction, rows in rows_by_direction.items():
        if not rows:
            continue
        csv_path = os.path.join(out_dir, f"{direction}_trials.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def params_from_row(row):
    return np.array([float(row[name]) for name in PARAM_NAMES],
                    dtype=np.float64)


def lateral_acceptance(metrics):
    return (
        float(metrics["signed_lateral_m_s"]) >= LATERAL_SPEED_PASS_M_S
        and abs(float(metrics["body_vx_m_s"])) <= OFF_AXIS_PASS_M_S
        and abs(float(metrics["yaw_rate_rad_s"])) <= YAW_RATE_PASS_RAD_S
        and not bool(metrics["terminated"])
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Search deployable open-loop lateral primitives.")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "record", "current", "flat_omni_v41_lateral_prior_search"))
    ap.add_argument("--method", choices=("random", "cma"), default="cma")
    ap.add_argument("--directions", default="left,right")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--validation-seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=41)
    return ap.parse_args()


def main():
    args = parse_args()
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    rows_by_direction = {}
    best_by_direction = {}
    started = time.time()

    for idx, direction in enumerate(directions):
        if direction not in ("left", "right"):
            raise ValueError(f"unknown direction {direction!r}")
        if args.method == "random":
            best, rows = random_search(
                direction, args.trials, args.seconds,
                args.seed + 1000 * idx, args.terrain)
        else:
            best, rows = cma_search(
                direction, args.generations, args.popsize, args.seconds,
                args.seed + 1000 * idx, args.terrain)
        rows_by_direction[direction] = rows
        best_by_direction[direction] = best
        print(
            f"{direction}: score={best['score']:.4f}, "
            f"vy={best['body_vy_m_s']:.4f}, vx={best['body_vx_m_s']:.4f}, "
            f"yaw={best['yaw_rate_rad_s']:.4f}")

    validation_by_direction = {}
    if args.validation_seconds > 0.0:
        for idx, direction in enumerate(directions):
            best = best_by_direction[direction]
            metrics = evaluate_params(
                params_from_row(best),
                direction=direction,
                seconds=args.validation_seconds,
                seed=args.seed + 100000 + idx,
                terrain=args.terrain,
            )
            metrics["accepted"] = lateral_acceptance(metrics)
            validation_by_direction[direction] = metrics
            print(
                f"{direction} validation {args.validation_seconds:.1f}s: "
                f"accepted={metrics['accepted']}, "
                f"vy={metrics['body_vy_m_s']:.4f}, "
                f"vx={metrics['body_vx_m_s']:.4f}, "
                f"yaw={metrics['yaw_rate_rad_s']:.4f}")

    summary = {
        "format_version": 1,
        "created_unix_time": time.time(),
        "elapsed_wall_s": time.time() - started,
        "terrain": args.terrain,
        "method": args.method,
        "directions": directions,
        "seconds_per_eval": args.seconds,
        "parameter_names": list(PARAM_NAMES),
        "parameter_bounds": {
            name: [float(lo), float(hi)]
            for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
        },
        "objective": (
            "score = signed_vy/0.15 - 0.70*abs(body_vx)/0.08 "
            "- 0.50*abs(yaw_rate)/0.20 - max(0,0.03-signed_vy)/0.03 "
            "- 0.02*action_energy - termination"),
        "lateral_acceptance": {
            "signed_lateral_m_s_min": LATERAL_SPEED_PASS_M_S,
            "abs_body_vx_m_s_max": OFF_AXIS_PASS_M_S,
            "abs_yaw_rate_rad_s_max": YAW_RATE_PASS_RAD_S,
            "validation_seconds": float(args.validation_seconds),
        },
        "best_by_direction": best_by_direction,
        "validation_by_direction": validation_by_direction,
    }
    summary_path = write_outputs(args.out_dir, summary, rows_by_direction)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

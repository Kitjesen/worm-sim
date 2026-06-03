"""
Search open-loop mixed-planar primitives for Worm V6.

This diagnostic searches deployable 1 s phase-clock 11-D motor primitives for
diagonal body-frame commands. It does not change the 80-D observation or 12-D
policy action ABI. The result answers whether mixed vx/vy failure is purely a
policy-learning issue or whether the current primitive family lacks a usable
diagonal gait seed.
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

COMMANDS = {
    "forward_left": (0.25, 0.15),
    "forward_right": (0.25, -0.15),
    "forward_left_scan": (0.10, 0.075),
    "forward_right_scan": (0.10, -0.075),
    "reverse_left": (-0.25, 0.15),
    "reverse_right": (-0.25, -0.15),
}

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

BOUNDS_LO = np.array([
    -1.00,
    0.00,
    0.05,
    -3.00,
    -math.pi,
    0.00,
    0.05,
    -3.00,
    -math.pi,
    -0.70,
    -0.70,
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
    0.70,
    0.70,
], dtype=np.float64)

DEFAULT_X0 = np.array([
    -0.50,
    0.45,
    1.00,
    1.00,
    math.pi,
    0.45,
    1.00,
    1.00,
    0.0,
    0.0,
    0.0,
], dtype=np.float64)

DEFAULT_SCORE_WEIGHTS = {
    "planar_error": 1.25,
    "vx_deficit": 0.70,
    "vy_deficit": 0.70,
    "yaw_penalty": 0.35,
    "action_energy": 0.02,
    "termination": 1.00,
}


def clip_params(params):
    return np.clip(np.asarray(params, dtype=np.float64), BOUNDS_LO, BOUNDS_HI)


def normalized_to_unit(params):
    params = clip_params(params)
    return (params - BOUNDS_LO) / (BOUNDS_HI - BOUNDS_LO)


def unit_to_params(unit_params):
    unit_params = np.clip(np.asarray(unit_params, dtype=np.float64), 0.0, 1.0)
    return BOUNDS_LO + unit_params * (BOUNDS_HI - BOUNDS_LO)


def primitive_action(params, phase):
    """Return 11-D normalized motor targets for one mixed-planar primitive."""
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
        action[j] = (
            slide_bias
            - slide_amp * (0.5 + 0.5 * math.sin(joint_phase)))

    center = 0.5 * (NUM_YAWS - 1)
    for j in range(NUM_YAWS):
        joint_phase = (
            TWO_PI * (t * yaw_freq + yaw_wave_n * j / NUM_YAWS)
            + yaw_phase
        )
        body_gradient = 0.0 if center <= 0.0 else (j - center) / center
        action[NUM_SLIDES + j] = (
            yaw_bias
            + yaw_trim_gradient * body_gradient
            + yaw_amp * math.sin(joint_phase)
        )

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def score_metrics(metrics, cmd_vx, cmd_vy, score_weights=None):
    weights = DEFAULT_SCORE_WEIGHTS if score_weights is None else score_weights
    cmd_vec = np.array([cmd_vx, cmd_vy], dtype=np.float64)
    body_vec = np.array([
        metrics["body_vx_m_s"],
        metrics["body_vy_m_s"],
    ], dtype=np.float64)
    cmd_norm = max(float(np.linalg.norm(cmd_vec)), 1e-6)
    planar_error = float(np.linalg.norm(body_vec - cmd_vec))
    projected_speed = float(np.dot(body_vec, cmd_vec) / cmd_norm)
    vx_progress = float(np.sign(cmd_vx) * metrics["body_vx_m_s"])
    vy_progress = float(np.sign(cmd_vy) * metrics["body_vy_m_s"])
    vx_deficit = max(0.0, 0.08 - vx_progress) / 0.08
    vy_deficit = max(0.0, 0.03 - vy_progress) / 0.03
    yaw_penalty = abs(float(metrics["yaw_rate_rad_s"])) / 0.20
    error_norm = planar_error / cmd_norm
    progress_term = projected_speed / cmd_norm
    score = (
        progress_term
        - float(weights["planar_error"]) * error_norm
        - float(weights["vx_deficit"]) * vx_deficit
        - float(weights["vy_deficit"]) * vy_deficit
        - float(weights["yaw_penalty"]) * yaw_penalty
        - float(weights["action_energy"]) * float(metrics["action_energy"])
        - (float(weights["termination"]) if metrics["terminated"] else 0.0)
    )
    return {
        "score": float(score),
        "planar_error_m_s": planar_error,
        "projected_speed_m_s": projected_speed,
        "vx_progress_m_s": vx_progress,
        "vy_progress_m_s": vy_progress,
        "vx_deficit": float(vx_deficit),
        "vy_deficit": float(vy_deficit),
        "yaw_penalty": float(yaw_penalty),
    }


def evaluate_params(params, command_name, seconds=4.0, seed=42,
                    terrain="flat", score_weights=None):
    cmd_vx, cmd_vy = COMMANDS[command_name]
    env = WormEnvV6(
        terrain=terrain,
        gait_mode="random",
        fixed_cmd_vx=cmd_vx,
        fixed_cmd_vy=cmd_vy,
        fixed_cmd_yaw=0.0,
        command_resample_prob=0.0,
    )
    obs, _ = env.reset(seed=seed)
    del obs
    env.set_command(vx=cmd_vx, vy=cmd_vy, yaw_rate=0.0)

    start_pos = env.data.xpos[env._root_body_id].copy()
    start_yaw = env._root_yaw_rad()
    start_xmat = env.data.xmat[env._root_body_id].reshape(3, 3)
    forward_axis = -start_xmat[:2, 0].copy()
    lateral_axis = start_xmat[:2, 1].copy()

    steps = int(seconds / CTRL_DT)
    terminated = False
    for step in range(steps):
        phase = TWO_PI * PHASE_FREQ * step * CTRL_DT
        action = primitive_action(params, phase)
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

    action_energy = float(np.mean(np.square([
        primitive_action(params, TWO_PI * i / 20.0)
        for i in range(20)
    ])))
    metrics = {
        "command_name": command_name,
        "cmd_vx_m_s": float(cmd_vx),
        "cmd_vy_m_s": float(cmd_vy),
        "body_vx_m_s": body_vx,
        "body_vy_m_s": body_vy,
        "yaw_rate_rad_s": yaw_rate,
        "elapsed_s": float(elapsed),
        "terminated": bool(terminated),
        "action_energy": action_energy,
    }
    metrics.update(score_metrics(metrics, cmd_vx, cmd_vy, score_weights))
    return metrics


def random_search(command_name, trials, seconds, seed, terrain,
                  score_weights=None):
    rng = np.random.default_rng(seed)
    rows = []
    best = None
    for trial in range(trials):
        if trial == 0:
            params = DEFAULT_X0.copy()
        else:
            params = unit_to_params(rng.uniform(0.0, 1.0, size=len(PARAM_NAMES)))
        metrics = evaluate_params(
            params, command_name=command_name, seconds=seconds,
            seed=seed + trial, terrain=terrain,
            score_weights=score_weights)
        row = {
            "trial": trial,
            **metrics,
            **{name: float(value) for name, value in zip(PARAM_NAMES, params)},
        }
        rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
    return best, rows


def cma_search(command_name, generations, popsize, seconds, seed, terrain,
               score_weights=None):
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
                params, command_name=command_name, seconds=seconds,
                seed=seed + trial, terrain=terrain,
                score_weights=score_weights)
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


def params_from_row(row):
    return np.array([float(row[name]) for name in PARAM_NAMES],
                    dtype=np.float64)


def mixed_planar_acceptance(metrics):
    return (
        metrics["vx_progress_m_s"] >= 0.08
        and metrics["vy_progress_m_s"] >= 0.03
        and metrics["planar_error_m_s"] <= 0.16
        and abs(float(metrics["yaw_rate_rad_s"])) <= 0.20
        and not bool(metrics["terminated"])
    )


def write_outputs(out_dir, summary, rows_by_command):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "mixed_planar_prior_search_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for command_name, rows in rows_by_command.items():
        if not rows:
            continue
        csv_path = os.path.join(out_dir, f"{command_name}_trials.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Search deployable open-loop mixed-planar primitives.")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "record", "current",
        "flat_omni_v58_mixed_planar_prior_search"))
    ap.add_argument("--method", choices=("random", "cma"), default="random")
    ap.add_argument("--commands", default=",".join(COMMANDS.keys()))
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--validation-seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=58)
    ap.add_argument("--planar-error-weight", type=float,
                    default=DEFAULT_SCORE_WEIGHTS["planar_error"])
    ap.add_argument("--vx-deficit-weight", type=float,
                    default=DEFAULT_SCORE_WEIGHTS["vx_deficit"])
    ap.add_argument("--vy-deficit-weight", type=float,
                    default=DEFAULT_SCORE_WEIGHTS["vy_deficit"])
    ap.add_argument("--yaw-penalty-weight", type=float,
                    default=DEFAULT_SCORE_WEIGHTS["yaw_penalty"])
    ap.add_argument("--action-energy-weight", type=float,
                    default=DEFAULT_SCORE_WEIGHTS["action_energy"])
    ap.add_argument("--termination-weight", type=float,
                    default=DEFAULT_SCORE_WEIGHTS["termination"])
    return ap.parse_args()


def main():
    args = parse_args()
    command_names = [c.strip() for c in args.commands.split(",") if c.strip()]
    score_weights = {
        "planar_error": float(args.planar_error_weight),
        "vx_deficit": float(args.vx_deficit_weight),
        "vy_deficit": float(args.vy_deficit_weight),
        "yaw_penalty": float(args.yaw_penalty_weight),
        "action_energy": float(args.action_energy_weight),
        "termination": float(args.termination_weight),
    }
    rows_by_command = {}
    best_by_command = {}
    started = time.time()

    for idx, command_name in enumerate(command_names):
        if command_name not in COMMANDS:
            raise ValueError(f"unknown command {command_name!r}")
        if args.method == "random":
            best, rows = random_search(
                command_name, args.trials, args.seconds,
                args.seed + 1000 * idx, args.terrain,
                score_weights=score_weights)
        else:
            best, rows = cma_search(
                command_name, args.generations, args.popsize, args.seconds,
                args.seed + 1000 * idx, args.terrain,
                score_weights=score_weights)
        rows_by_command[command_name] = rows
        best_by_command[command_name] = best
        print(
            f"{command_name}: score={best['score']:.4f}, "
            f"body=({best['body_vx_m_s']:.4f},{best['body_vy_m_s']:.4f}), "
            f"err={best['planar_error_m_s']:.4f}, "
            f"yaw={best['yaw_rate_rad_s']:.4f}")

    validation_by_command = {}
    if args.validation_seconds > 0.0:
        for idx, command_name in enumerate(command_names):
            best = best_by_command[command_name]
            metrics = evaluate_params(
                params_from_row(best),
                command_name=command_name,
                seconds=args.validation_seconds,
                seed=args.seed + 100000 + idx,
                terrain=args.terrain,
                score_weights=score_weights,
            )
            metrics["accepted"] = mixed_planar_acceptance(metrics)
            validation_by_command[command_name] = metrics
            print(
                f"{command_name} validation {args.validation_seconds:.1f}s: "
                f"accepted={metrics['accepted']}, "
                f"body=({metrics['body_vx_m_s']:.4f},"
                f"{metrics['body_vy_m_s']:.4f}), "
                f"err={metrics['planar_error_m_s']:.4f}, "
                f"yaw={metrics['yaw_rate_rad_s']:.4f}")

    summary = {
        "format_version": 1,
        "created_unix_time": time.time(),
        "elapsed_wall_s": time.time() - started,
        "terrain": args.terrain,
        "method": args.method,
        "commands": command_names,
        "seconds_per_eval": args.seconds,
        "parameter_names": list(PARAM_NAMES),
        "parameter_bounds": {
            name: [float(lo), float(hi)]
            for name, lo, hi in zip(PARAM_NAMES, BOUNDS_LO, BOUNDS_HI)
        },
        "score_weights": score_weights,
        "objective": (
            "score = projected_speed/|cmd| "
            "- planar_error_weight*planar_error/|cmd| "
            "- vx_deficit_weight*vx_deficit "
            "- vy_deficit_weight*vy_deficit "
            "- yaw_penalty_weight*abs(yaw_rate)/0.20 "
            "- action_energy_weight*action_energy - termination_weight"),
        "acceptance": {
            "vx_progress_m_s_min": 0.08,
            "vy_progress_m_s_min": 0.03,
            "planar_error_m_s_max": 0.16,
            "abs_yaw_rate_rad_s_max": 0.20,
            "validation_seconds": float(args.validation_seconds),
        },
        "best_by_command": best_by_command,
        "validation_by_command": validation_by_command,
    }
    summary_path = write_outputs(args.out_dir, summary, rows_by_command)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

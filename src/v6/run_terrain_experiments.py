"""
Paper terrain experiment runner for the deployable V6 worm/snake policy.

Main matrix:
    terrains: flat, sand, slope
    modes:    worm, snake, mixed

The default method trains deployable RL policies through train_v6.py. CMA-ES is
kept as an optional open-loop baseline and uses optimize_speed.py with the
equivalent peristaltic/serpentine/full modes.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

PAPER_TERRAINS = ["flat", "sand", "slope"]
PAPER_MODES = ["worm", "snake", "mixed"]
RL_MODES = ["worm", "snake", "mixed", "random"]

MODE_TO_CMAES = {
    "worm": "peristaltic",
    "snake": "serpentine",
    "mixed": "full",
}

TERRAIN_LABELS = {
    "flat": "Flat",
    "sand": "Sand",
    "slope": "Slope",
}

MODE_LABELS = {
    "worm": "Worm",
    "snake": "Snake",
    "mixed": "Mixed",
}


def rl_run_dir(terrain, mode):
    return os.path.join(PROJECT_ROOT, "runs", f"worm_v6_ppo_{terrain}_{mode}")


def cmaes_result_path(terrain, mode):
    mapped = MODE_TO_CMAES[mode]
    return os.path.join(PROJECT_ROOT, "runs", f"cmaes_{terrain}_{mapped}",
                        "best_gait.json")


def robust_sensor_kwargs(enabled):
    if enabled:
        return {
            "encoder_pos_noise_std": 0.01,
            "encoder_vel_noise_std": 0.02,
            "imu_gravity_noise_std": 0.01,
            "imu_gyro_noise_std": 0.01,
            "action_delay_steps": 1,
        }
    return {
        "encoder_pos_noise_std": 0.0,
        "encoder_vel_noise_std": 0.0,
        "imu_gravity_noise_std": 0.0,
        "imu_gyro_noise_std": 0.0,
        "action_delay_steps": 0,
    }


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def close_float(value, expected, tol=1e-6):
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def control_timing_kwargs():
    from worm_env_v6 import (
        CTRL_DT,
        PERISTALTIC_ACTUATION_PERIOD_S,
        PHASE_FREQ,
    )

    return {
        "control_dt_s": CTRL_DT,
        "control_rate_hz": 1.0 / CTRL_DT,
        "peristaltic_actuation_period_s": PERISTALTIC_ACTUATION_PERIOD_S,
        "phase_freq_hz": PHASE_FREQ,
    }


def control_timing_compatible(config):
    actual = (config or {}).get("control_timing")
    if not isinstance(actual, dict):
        return False
    return all(
        close_float(actual.get(key), expected)
        for key, expected in control_timing_kwargs().items()
    )


def actuator_contract_compatible(config):
    from motor_contract_v6 import motor_contract

    return (
        isinstance(config, dict)
        and config.get("actuator_contract_fingerprint")
        == motor_contract()["contract_fingerprint"])


def resume_config_compatible(run_dir, robust):
    config = read_json(os.path.join(run_dir, "training_config.json"))
    if not config:
        return False
    actual = config.get("sensor_robustness", {})
    expected = robust_sensor_kwargs(robust)
    sensors_ok = all(actual.get(key) == value for key, value in expected.items())
    actuator_ok = actuator_contract_compatible(config)
    return sensors_ok and control_timing_compatible(config) and actuator_ok


def checkpoint_step(path):
    match = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
    return int(match.group(1)) if match else None


def paired_vecnormalize_path(model_path):
    stem, ext = os.path.splitext(model_path)
    if ext.lower() != ".zip":
        return None

    candidates = [f"{stem}_vecnormalize.pkl"]
    base = os.path.basename(model_path)
    directory = os.path.dirname(model_path)
    marker = "_steps.zip"
    if base.endswith(marker):
        step_part = base[:-len(marker)].split("_")[-1]
        prefix = base[:-len(step_part + marker)]
        candidates.append(os.path.join(
            directory, f"{prefix}vecnormalize_{step_part}_steps.pkl"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def latest_resume_model(run_dir, robust):
    if not resume_config_compatible(run_dir, robust):
        return None

    candidates = []
    result = read_json(os.path.join(run_dir, "training_result.json")) or {}
    final_model = os.path.join(run_dir, "final_model.zip")
    if os.path.exists(final_model) and paired_vecnormalize_path(final_model):
        candidates.append((
            int(result.get("completed_timesteps", 0)),
            final_model,
        ))

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            path = os.path.join(ckpt_dir, name)
            step = checkpoint_step(path)
            if step is not None and paired_vecnormalize_path(path):
                candidates.append((step, path))

    best_model = os.path.join(run_dir, "best_model.zip")
    best_norm = os.path.join(run_dir, "best_model_vecnormalize.pkl")
    if os.path.exists(best_model) and os.path.exists(best_norm):
        candidates.append((-1, best_model))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def rl_status(terrain, mode):
    run_dir = rl_run_dir(terrain, mode)
    best_model = os.path.join(run_dir, "best_model.zip")
    final_model = os.path.join(run_dir, "final_model.zip")
    eval_json = os.path.join(run_dir, "eval_metrics.json")
    if os.path.exists(eval_json):
        with open(eval_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if (not control_timing_compatible(metrics)
                or not actuator_contract_compatible(metrics)):
            return "stale eval"
        speed = metrics.get("mean_speed_mm_s")
        if speed is not None:
            return f"{speed:.1f} mm/s"
    if not resume_config_compatible(run_dir, robust=True):
        if os.path.exists(best_model) or os.path.exists(final_model):
            return "stale model"
    if os.path.exists(best_model):
        return "best model"
    if os.path.exists(final_model):
        return "final model"
    return "-"


def cmaes_status(terrain, mode):
    path = cmaes_result_path(terrain, mode)
    if not os.path.exists(path):
        return "-"
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    speed = result.get("best_speed_mm_s")
    return f"{speed:.1f} mm/s" if speed is not None else "done"


def print_results_table(method):
    status_fn = rl_status if method == "rl" else cmaes_status
    title = "Deployable RL" if method == "rl" else "CMA-ES baseline"
    col_w = 16
    print()
    print("=" * (18 + col_w * len(PAPER_MODES)))
    print(f"  {title} terrain x mode matrix")
    print("=" * (18 + col_w * len(PAPER_MODES)))
    header = f"{'Terrain':<18}" + "".join(
        f"{MODE_LABELS[m]:>{col_w}}" for m in PAPER_MODES)
    print(header)
    print("-" * (18 + col_w * len(PAPER_MODES)))
    for terrain in PAPER_TERRAINS:
        row = f"{TERRAIN_LABELS[terrain]:<18}"
        for mode in PAPER_MODES:
            row += f"{status_fn(terrain, mode):>{col_w}}"
        print(row)
    print("=" * (18 + col_w * len(PAPER_MODES)))


def run_rl(terrains, modes, timesteps, n_envs, test, dry_run, robust,
           resume_partial=False, train_chunk_timesteps=None):
    total = len(terrains) * len(modes)
    done = 0
    for terrain in terrains:
        for mode in modes:
            done += 1
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "train_v6.py"),
                "--terrain", terrain,
                "--gait-mode", mode,
                "--timesteps", str(10_000 if test else timesteps),
                "--n-envs", str(1 if test else n_envs),
            ]
            if train_chunk_timesteps is not None and not test:
                cmd.extend([
                    "--train-chunk-timesteps",
                    str(train_chunk_timesteps),
                ])
            if test:
                cmd.append("--test")
            if robust:
                cmd.extend([
                    "--encoder-pos-noise", "0.01",
                    "--encoder-vel-noise", "0.02",
                    "--imu-gravity-noise", "0.01",
                    "--imu-gyro-noise", "0.01",
                    "--action-delay-steps", "1",
                ])
            resume_model = None
            if resume_partial and not test:
                resume_model = latest_resume_model(rl_run_dir(terrain, mode),
                                                   robust)
                if resume_model:
                    cmd.extend(["--resume", resume_model])
            print(f"\n[{done}/{total}] RL terrain={terrain} mode={mode}")
            print("  " + " ".join(cmd))
            if resume_partial and not resume_model and not test:
                print("  resume-partial: no compatible checkpoint found")
            if not dry_run:
                subprocess.run(cmd, check=True)


def run_cmaes(terrains, modes, popsize, max_gen, test, dry_run):
    total = len(terrains) * len(modes)
    done = 0
    for terrain in terrains:
        for mode in modes:
            if mode not in MODE_TO_CMAES:
                print(f"\nSkipping CMA-ES baseline for mode={mode}")
                continue
            done += 1
            mapped = MODE_TO_CMAES[mode]
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "optimize_speed.py"),
                "--terrain", terrain,
                "--mode", mapped,
                "--popsize", str(popsize),
                "--max-gen", str(20 if test else max_gen),
            ]
            if test:
                cmd.append("--test")
            print(f"\n[{done}/{total}] CMA-ES terrain={terrain} mode={mode} ({mapped})")
            print("  " + " ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=True)


def robust_eval_args():
    return [
        "--eval-condition", "robust",
        "--encoder-pos-noise", "0.01",
        "--encoder-vel-noise", "0.02",
        "--imu-gravity-noise", "0.01",
        "--imu-gyro-noise", "0.01",
        "--action-delay-steps", "1",
        "--action-saturation", "0.8",
    ]


def run_eval(terrains, modes, episodes, eval_time, video, dry_run,
             robust_eval=False):
    total = len(terrains) * len(modes)
    done = 0
    for terrain in terrains:
        for mode in modes:
            done += 1
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "eval_v6.py"),
                "--terrain", terrain,
                "--gait-mode", mode,
                "--episodes", str(episodes),
                "--time", str(eval_time),
            ]
            if robust_eval:
                json_out = os.path.join(
                    rl_run_dir(terrain, mode), "eval_metrics_robust.json")
                cmd.extend(["--json-out", json_out])
                cmd.extend(robust_eval_args())
            if video:
                cmd.append("--video")
            label = "Robust eval" if robust_eval else "Eval"
            print(f"\n[{done}/{total}] {label} terrain={terrain} mode={mode}")
            print("  " + " ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=True)


def run_blend_scan(terrains, policy_mode, blends, episodes, eval_time, dry_run):
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "scan_gait_blend_v6.py"),
        "--terrain", *terrains,
        "--policy-mode", policy_mode,
        "--blends", blends,
        "--episodes", str(episodes),
        "--time", str(eval_time),
    ]
    if dry_run:
        cmd.append("--dry-run")
    print("\nBlend scan")
    print("  " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_summary(policy_mode, dry_run):
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "summarize_paper_results_v6.py"),
        "--policy-mode", policy_mode,
    ]
    print("\nSummarize paper results")
    print("  " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_hardware_template(dry_run):
    out_path = os.path.join(PROJECT_ROOT, "record", "v6", "hardware",
                            "hardware_log_template.csv")
    raw_path = os.path.join(PROJECT_ROOT, "record", "v6", "hardware",
                            "raw_hardware_log_template.csv")
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "validate_hardware_log_v6.py"),
        "--write-template", out_path,
    ]
    print("\nHardware log template")
    print("  " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)
    raw_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "build_hardware_obs_v6.py"),
        "--write-raw-template", raw_path,
    ]
    print("\nRaw hardware sensor template")
    print("  " + " ".join(raw_cmd))
    if not dry_run:
        subprocess.run(raw_cmd, check=True)


def run_deploy_export(terrains, modes, dry_run):
    total = len(terrains) * len(modes)
    done = 0
    for terrain in terrains:
        for mode in modes:
            done += 1
            out_dir = os.path.join(
                PROJECT_ROOT, "record", "v6", "deploy_bundles",
                f"{terrain}_{mode}")
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "deploy_policy_v6.py"),
                "export",
                "--terrain", terrain,
                "--gait-mode", mode,
                "--out-dir", out_dir,
            ]
            print(f"\n[{done}/{total}] Deploy export terrain={terrain} mode={mode}")
            print("  " + " ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Run paper terrain x gait experiments for Worm V6")
    ap.add_argument("--terrain", nargs="+", choices=PAPER_TERRAINS,
                    default=PAPER_TERRAINS)
    ap.add_argument("--mode", nargs="+", choices=RL_MODES,
                    default=PAPER_MODES)
    ap.add_argument("--method",
                    choices=["rl", "rl-eval", "robust-eval",
                             "blend-scan", "summary",
                             "hardware-template", "deploy-export",
                             "cmaes", "both"],
                    default="rl")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--train-chunk-timesteps", type=int, default=None,
                    help="Train at most this many additional timesteps "
                         "per RL run while preserving --timesteps as target")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--max-gen", type=int, default=200)
    ap.add_argument("--eval", action="store_true",
                    help="Run eval_v6.py after RL training")
    ap.add_argument("--robust", action="store_true",
                    help="Train with deployable sensor noise and 1-step action delay")
    ap.add_argument("--resume-partial", action="store_true",
                    help="Resume RL training from the latest compatible "
                         "final/best/checkpoint artifact when present")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--eval-time", type=float, default=20.0)
    ap.add_argument("--scan-policy-mode", default="random",
                    choices=["worm", "snake", "mixed", "random"])
    ap.add_argument("--blends", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results", action="store_true")
    args = ap.parse_args()

    if args.results:
        if args.method in ("rl", "rl-eval", "both"):
            print_results_table("rl")
        if args.method in ("cmaes", "both"):
            print_results_table("cmaes")
        return

    t0 = time.time()
    if args.method in ("rl", "both"):
        run_rl(args.terrain, args.mode, args.timesteps, args.n_envs,
               args.test, args.dry_run, args.robust, args.resume_partial,
               args.train_chunk_timesteps)
    if args.method == "rl-eval" or args.eval:
        run_eval(args.terrain, args.mode, args.episodes, args.eval_time,
                 args.video, args.dry_run)
    if args.method == "robust-eval":
        run_eval(args.terrain, args.mode, args.episodes, args.eval_time,
                 args.video, args.dry_run, robust_eval=True)
    if args.method == "blend-scan":
        run_blend_scan(args.terrain, args.scan_policy_mode, args.blends,
                       args.episodes, args.eval_time, args.dry_run)
    if args.method == "summary":
        run_summary(args.scan_policy_mode, args.dry_run)
    if args.method == "hardware-template":
        run_hardware_template(args.dry_run)
    if args.method == "deploy-export":
        run_deploy_export(args.terrain, args.mode, args.dry_run)
    if args.method in ("cmaes", "both"):
        run_cmaes(args.terrain, args.mode, args.popsize, args.max_gen,
                  args.test, args.dry_run)

    elapsed = (time.time() - t0) / 60.0
    print(f"\nExperiment launcher finished in {elapsed:.1f} min")
    if args.method in ("rl", "rl-eval", "both"):
        print_results_table("rl")
    if args.method in ("cmaes", "both"):
        print_results_table("cmaes")


if __name__ == "__main__":
    main()

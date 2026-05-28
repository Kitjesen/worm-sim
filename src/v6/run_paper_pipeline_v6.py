"""
Reproducible pipeline for the deployable multi-modal Worm V6 paper.

Presets:
  smoke   - fast checks, dry-run launchers, template/summary generation
  formal  - full train/eval/blend-scan/summary sequence
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

PAPER_TERRAINS = ("flat", "sand", "slope")
FIXED_MODES = ("worm", "snake", "mixed")
TRAIN_MODES = ("worm", "snake", "mixed", "random")
FORMAL_STAGES = (
    "hardware", "train", "baseline", "deploy", "eval", "scan", "summary",
    "audit")
DEFAULT_MANIFEST = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "paper_pipeline_manifest.json")


def run(cmd, dry_run=False):
    print("\n" + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def py(*args):
    return [sys.executable, *args]


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def cmd_text(cmd):
    return " ".join(str(part) for part in cmd)


def write_manifest(path, args, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    manifest = {
        "format_version": 1,
        "created_unix_time": time.time(),
        "preset": args.preset,
        "dry_run": bool(args.dry_run),
        "resume": bool(getattr(args, "resume", False)),
        "resume_partial": bool(getattr(args, "resume_partial", False)),
        "terrain": list(args.terrain),
        "train_modes": list(args.train_modes),
        "timesteps": args.timesteps,
        "train_chunk_timesteps": getattr(args, "train_chunk_timesteps", None),
        "n_envs": args.n_envs,
        "device": getattr(args, "device", "auto"),
        "episodes": args.episodes,
        "eval_time_s": args.eval_time,
        "blends": args.blends,
        "stages": list(args.stage),
        "max_records": getattr(args, "max_records", None),
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved pipeline manifest: {path}")


def run_record(record, dry_run=False, resume=False):
    status = "planned" if dry_run else "ran"
    if resume and record.get("skip_ok", False):
        status = "skipped"
        print(f"\nSKIP {record['stage']} {record['key']}: {record['skip_reason']}")
    else:
        print(f"\n[{record['stage']}] {record['key']}")
        print("  " + cmd_text(record["cmd"]))
        if not dry_run:
            subprocess.run(record["cmd"], check=True)
    record["status"] = status
    return record


def run_records(records, args):
    executed = []
    for record in records:
        executed.append(run_record(
            record, dry_run=args.dry_run, resume=args.resume))
    write_manifest(args.manifest_out, args, executed)


def filter_records(records, stages):
    selected = set(stages)
    if "all" in selected:
        return records
    return [record for record in records if record["stage"] in selected]


def limit_records(records, max_records):
    if max_records is None or max_records <= 0:
        return records
    return records[:max_records]


def run_dir(terrain, mode):
    return os.path.join(PROJECT_ROOT, "runs", f"worm_v6_ppo_{terrain}_{mode}")


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_blends(text):
    return {round(float(value), 2) for value in text.split(",") if value.strip()}


def valid_training_done(terrain, mode):
    from audit_paper_goal_v6 import valid_training_run

    ok, _ = valid_training_run(run_dir(terrain, mode), terrain, mode)
    return ok


def valid_cmaes_done(terrain, mode):
    from audit_paper_goal_v6 import cmaes_path, valid_cmaes_baseline

    ok, _ = valid_cmaes_baseline(cmaes_path(terrain, mode))
    return ok


def valid_eval_done(terrain, mode, robust=False):
    from audit_paper_goal_v6 import valid_eval_metrics

    filename = "eval_metrics_robust.json" if robust else "eval_metrics.json"
    condition = "robust" if robust else "nominal"
    path = os.path.join(run_dir(terrain, mode), filename)
    ok, _ = valid_eval_metrics(read_json(path), terrain, mode, condition)
    return ok


def valid_scan_done(terrain, blends):
    from audit_paper_goal_v6 import valid_eval_metrics

    path = os.path.join(
        PROJECT_ROOT, "runs", f"worm_v6_blend_scan_{terrain}_random",
        "scan_results.json")
    rows = read_json(path)
    if not isinstance(rows, list):
        return False
    by_blend = {
        round(float(row.get("gait_blend")), 2): row
        for row in rows
        if row.get("gait_blend") is not None
    }
    if not split_blends(blends).issubset(by_blend.keys()):
        return False
    for blend in split_blends(blends):
        row = by_blend.get(blend)
        ok, _ = valid_eval_metrics(row, terrain, "random", "nominal")
        if not (ok and row.get("policy_mode") == "random"):
            return False
    return True


def valid_deploy_bundle_done(terrain, mode="random"):
    from audit_paper_goal_v6 import OBS_DIM
    from motor_contract_v6 import action_mapping_matches, motor_contract

    bundle_dir = os.path.join(
        PROJECT_ROOT, "record", "v6", "deploy_bundles",
        f"{terrain}_{mode}")
    actor = os.path.join(bundle_dir, "policy_actor.pt")
    config = read_json(os.path.join(bundle_dir, "deploy_config.json"))
    return (
        os.path.exists(actor)
        and config is not None
        and config.get("obs_dim") == OBS_DIM
        and config.get("action_dim") == 11
        and action_mapping_matches(config.get("action_mapping"))
        and config.get("actuator_contract_fingerprint")
        == motor_contract()["contract_fingerprint"]
    )


def valid_preflight_done():
    path = os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        "hardware_deploy_preflight.json")
    data = read_json(path)
    return isinstance(data, dict) and data.get("complete") is True


def valid_sim_bridge_done():
    raw_path = os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware", "sim_flat_mixed_raw.csv")
    policy_path = os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        "sim_flat_mixed_raw_policy.csv")
    return os.path.exists(raw_path) and os.path.exists(policy_path)


def stage_record(stage, key, cmd, skip_ok=False, skip_reason="already valid"):
    return {
        "stage": stage,
        "key": key,
        "cmd": cmd,
        "cmd_text": cmd_text(cmd),
        "skip_ok": bool(skip_ok),
        "skip_reason": skip_reason,
    }


def smoke(args):
    files = [
        "worm_v6.py",
        "worm_env_v6.py",
        "train_v6.py",
        "eval_v6.py",
        "run_terrain_experiments.py",
        "scan_gait_blend_v6.py",
        "summarize_paper_results_v6.py",
        "summarize_hardware_validation_v6.py",
        "observation_contract_v6.py",
        "audit_observation_sources_v6.py",
        "validate_hardware_log_v6.py",
        "build_hardware_obs_v6.py",
        "check_controller_stream_v6.py",
        "collect_sim_hardware_log_v6.py",
        "capture_hardware_stream_v6.py",
        "deploy_policy_v6.py",
        "hardware_policy_runtime_v6.py",
        "import_hardware_trial_v6.py",
        "hardware_trial_status_v6.py",
        "preflight_hardware_deploy_v6.py",
        "audit_paper_goal_v6.py",
        "paper_status_v6.py",
        "test_deployable_obs_v6.py",
        "test_observation_contract_v6.py",
        "test_deploy_policy_v6.py",
        "test_hardware_obs_builder_v6.py",
        "test_hardware_log_validation_v6.py",
        "test_sim_hardware_log_v6.py",
        "test_controller_stream_check_v6.py",
        "test_capture_hardware_stream_v6.py",
        "test_hardware_policy_runtime_v6.py",
        "test_import_hardware_trial_v6.py",
        "test_hardware_trial_status_v6.py",
        "test_hardware_validation_summary_v6.py",
        "test_hardware_preflight_v6.py",
        "test_observation_source_audit_v6.py",
        "test_goal_audit_v6.py",
        "test_paper_pipeline_plan_v6.py",
        "test_resume_partial_v6.py",
    ]
    run(py("-m", "py_compile", *[
        os.path.join(SCRIPT_DIR, name) for name in files
    ]), dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_deployable_obs_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_observation_contract_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_deploy_policy_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_hardware_obs_builder_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_hardware_log_validation_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_sim_hardware_log_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_controller_stream_check_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_capture_hardware_stream_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_hardware_policy_runtime_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_import_hardware_trial_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_hardware_trial_status_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_hardware_validation_summary_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_hardware_preflight_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_observation_source_audit_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_goal_audit_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_paper_pipeline_plan_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "test_resume_partial_v6.py")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "rl", "--terrain", "flat", "--mode", "random",
           "--timesteps", "10000", "--n-envs", "1", "--robust",
           "--dry-run"), dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "blend-scan", "--terrain", "flat",
           "--scan-policy-mode", "random", "--blends", "0.0,0.5,1.0",
           "--episodes", "1", "--eval-time", "0.2", "--dry-run"),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "robust-eval", "--terrain", "flat",
           "--mode", "mixed", "--episodes", "1", "--eval-time", "0.2",
           "--dry-run"), dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "hardware-template"), dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "collect_sim_hardware_log_v6.py"),
           "--terrain", "flat", "--gait-mode", "mixed",
           "--gait-blend", "0.5", "--time", "0.2",
           "--output-raw", os.path.join(
               PROJECT_ROOT, "record", "v6", "hardware",
               "sim_flat_mixed_raw.csv"),
           "--output-policy", os.path.join(
               PROJECT_ROOT, "record", "v6", "hardware",
               "sim_flat_mixed_raw_policy.csv")),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "summary", "--scan-policy-mode", "random"),
        dry_run=args.dry_run)
    run(py(os.path.join(SCRIPT_DIR, "audit_paper_goal_v6.py"),
           "--allow-incomplete"), dry_run=args.dry_run)


def formal(args):
    records = filter_records(build_formal_records(args), args.stage)
    records = limit_records(records, args.max_records)
    run_records(records, args)


def build_formal_records(args):
    records = []
    records.append(stage_record(
        "hardware", "templates",
        py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "hardware-template")))
    records.append(stage_record(
        "hardware", "sim_flat_mixed_bridge",
        py(os.path.join(SCRIPT_DIR, "collect_sim_hardware_log_v6.py"),
           "--terrain", "flat", "--gait-mode", "mixed",
           "--gait-blend", "0.5", "--time", "0.2",
           "--output-raw", os.path.join(
               PROJECT_ROOT, "record", "v6", "hardware",
               "sim_flat_mixed_raw.csv"),
           "--output-policy", os.path.join(
               PROJECT_ROOT, "record", "v6", "hardware",
               "sim_flat_mixed_raw_policy.csv")),
        skip_ok=valid_sim_bridge_done()))

    for terrain in args.terrain:
        for mode in args.train_modes:
            train_cmd = py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
                           "--method", "rl", "--terrain", terrain, "--mode", mode,
                           "--timesteps", str(args.timesteps),
                           "--n-envs", str(args.n_envs),
                           "--device", args.device, "--robust")
            if getattr(args, "train_chunk_timesteps", None) is not None:
                train_cmd.extend([
                    "--train-chunk-timesteps",
                    str(args.train_chunk_timesteps),
                ])
            if getattr(args, "resume_partial", False):
                train_cmd.append("--resume-partial")
            records.append(stage_record(
                "train", f"{terrain}_{mode}",
                train_cmd,
                skip_ok=valid_training_done(terrain, mode)))

    for terrain in args.terrain:
        for mode in FIXED_MODES:
            records.append(stage_record(
                "baseline", f"{terrain}_{mode}_cmaes",
                py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
                   "--method", "cmaes", "--terrain", terrain,
                   "--mode", mode, "--popsize", "16", "--max-gen", "200"),
                skip_ok=valid_cmaes_done(terrain, mode)))

    if "random" in args.train_modes:
        for terrain in args.terrain:
            records.append(stage_record(
                "deploy", f"{terrain}_random",
                py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
                   "--method", "deploy-export", "--terrain", terrain,
                   "--mode", "random"),
                skip_ok=valid_deploy_bundle_done(terrain, "random")))

    for terrain in args.terrain:
        for mode in FIXED_MODES:
            eval_cmd = py(
                os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
                "--method", "rl-eval", "--terrain", terrain,
                "--mode", mode, "--episodes", str(args.episodes),
                "--eval-time", str(args.eval_time))
            if args.video:
                eval_cmd.append("--video")
            records.append(stage_record(
                "eval", f"{terrain}_{mode}_nominal", eval_cmd,
                skip_ok=valid_eval_done(terrain, mode, robust=False)))

    for terrain in args.terrain:
        for mode in FIXED_MODES:
            records.append(stage_record(
                "eval", f"{terrain}_{mode}_robust",
                py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
                   "--method", "robust-eval", "--terrain", terrain,
                   "--mode", mode, "--episodes", str(args.episodes),
                   "--eval-time", str(args.eval_time)),
                skip_ok=valid_eval_done(terrain, mode, robust=True)))

    for terrain in args.terrain:
        records.append(stage_record(
            "scan", f"{terrain}_random",
            py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
               "--method", "blend-scan", "--terrain", terrain,
               "--scan-policy-mode", "random", "--blends", args.blends,
               "--episodes", str(args.episodes),
               "--eval-time", str(args.eval_time)),
            skip_ok=valid_scan_done(terrain, args.blends)))

    records.append(stage_record(
        "summary", "paper_results",
        py(os.path.join(SCRIPT_DIR, "run_terrain_experiments.py"),
           "--method", "summary", "--scan-policy-mode", "random")))
    records.append(stage_record(
        "audit", "hardware_deploy_preflight",
        py(os.path.join(SCRIPT_DIR, "preflight_hardware_deploy_v6.py")),
        skip_ok=valid_preflight_done()))
    records.append(stage_record(
        "audit", "completion",
        py(os.path.join(SCRIPT_DIR, "audit_paper_goal_v6.py"),
           "--allow-incomplete")))
    return records


def main():
    ap = argparse.ArgumentParser(
        description="Run reproducible Worm V6 paper experiment pipeline")
    ap.add_argument("--preset", choices=["smoke", "formal"], default="smoke")
    ap.add_argument("--terrain", nargs="+", choices=PAPER_TERRAINS,
                    default=list(PAPER_TERRAINS))
    ap.add_argument("--train-modes", nargs="+",
                    choices=TRAIN_MODES, default=list(TRAIN_MODES))
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--train-chunk-timesteps", type=int, default=None,
                    help="For formal train stages, train at most this many "
                         "additional timesteps per selected record")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="PPO network device for train stages")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--eval-time", type=float, default=20.0)
    ap.add_argument("--blends", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--video", action="store_true",
                    help="Record representative fixed-mode evaluation videos")
    ap.add_argument("--stage", nargs="+",
                    choices=("all", *FORMAL_STAGES),
                    default=["all"],
                    help="Formal stages to run, e.g. train deploy eval scan")
    ap.add_argument("--resume", action="store_true",
                    help="Skip formal stages whose artifacts already pass audit")
    ap.add_argument("--resume-partial", action="store_true",
                    help="For formal train stages, continue from latest "
                         "compatible checkpoint instead of restarting")
    ap.add_argument("--manifest-out", default=DEFAULT_MANIFEST,
                    help="Write formal pipeline command/status manifest")
    ap.add_argument("--max-records", type=int, default=None,
                    help="Run only the first N selected formal records")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    if args.preset == "smoke":
        smoke(args)
    else:
        formal(args)


if __name__ == "__main__":
    main()

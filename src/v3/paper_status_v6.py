"""
Print the current completion status for the deployable Worm V6 paper pipeline.
"""

import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
AUDIT_JSON = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results", "completion_audit.json")
PIPELINE_MANIFEST = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "paper_pipeline_manifest.json")
PAPER_TERRAINS = ("flat", "sand", "slope")
TRAIN_MODES = ("worm", "snake", "mixed", "random")
CMAES_MODE_TO_FIXED = {
    "peristaltic": "worm",
    "serpentine": "snake",
    "full": "mixed",
}
REQUIRED_TRAIN_TIMESTEPS = 1_000_000


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def refresh_audit():
    subprocess.run([
        sys.executable,
        os.path.join(SCRIPT_DIR, "audit_paper_goal_v6.py"),
        "--allow-incomplete",
    ], check=True, stdout=subprocess.DEVNULL)


def missing_check_names(audit):
    return [
        check["name"]
        for check in audit.get("checks", [])
        if check.get("status") != "ok"
    ]


def audit_check(audit, name):
    for check in audit.get("checks", []):
        if check.get("name") == name:
            return check
    return {}


def next_missing_training_target(audit):
    check = audit_check(audit, "12 PPO training model artifacts")
    pattern = re.compile(
        r"runs/worm_v6_ppo_(flat|sand|slope)_(worm|snake|mixed|random)")
    for value in check.get("missing", []):
        match = pattern.search(value)
        if match:
            return match.group(1), match.group(2)
    return None


def next_missing_cmaes_target(audit):
    check = audit_check(audit, "9 CMA-ES open-loop baseline JSON files")
    pattern = re.compile(
        r"runs/cmaes_(flat|sand|slope)_(peristaltic|serpentine|full)")
    for value in check.get("missing", []):
        match = pattern.search(value)
        if match:
            return match.group(1), CMAES_MODE_TO_FIXED[match.group(2)]
    return None


def model_vecnormalize_pair(run_dir):
    for model_name, norm_name in (
        ("best_model.zip", "best_model_vecnormalize.pkl"),
        ("final_model.zip", "final_model_vecnormalize.pkl"),
    ):
        model_path = os.path.join(run_dir, model_name)
        norm_path = os.path.join(run_dir, norm_name)
        if os.path.exists(model_path) and os.path.exists(norm_path):
            return rel(model_path), rel(norm_path)
    return None, None


def training_matrix_progress():
    from audit_paper_goal_v6 import valid_training_run

    rows = []
    for terrain in PAPER_TERRAINS:
        for mode in TRAIN_MODES:
            run_dir = os.path.join(
                PROJECT_ROOT, "runs", f"worm_v6_ppo_{terrain}_{mode}")
            config = load_json(os.path.join(run_dir, "training_config.json"))
            result = load_json(os.path.join(run_dir, "training_result.json"))
            model_path, norm_path = model_vecnormalize_pair(run_dir)
            audit_ok, audit_info = valid_training_run(run_dir, terrain, mode)
            completed = int((result or {}).get("completed_timesteps", 0) or 0)
            planned = int(
                ((config or {}).get("training", {}) or {}).get("timesteps", 0)
                or 0)
            artifact_complete = (
                bool(model_path and norm_path)
                and planned >= REQUIRED_TRAIN_TIMESTEPS
                and completed >= REQUIRED_TRAIN_TIMESTEPS)
            rows.append({
                "terrain": terrain,
                "mode": mode,
                "run_dir": rel(run_dir),
                "exists": os.path.exists(run_dir),
                "has_model_pair": bool(model_path and norm_path),
                "model": model_path,
                "vecnormalize": norm_path,
                "planned_timesteps": planned,
                "completed_timesteps": completed,
                "artifact_complete": artifact_complete,
                "complete": audit_ok,
                "audit_reasons": audit_info.get("reasons", []),
            })
    return rows


def recommended_commands(audit, missing):
    commands = []
    if "12 PPO training model artifacts" in missing:
        target = next_missing_training_target(audit)
        if target:
            terrain, mode = target
            commands.append((
                "train_next",
                "python src/v3/run_paper_pipeline_v6.py --preset formal "
                f"--stage train --terrain {terrain} --train-modes {mode} "
                "--timesteps 1000000 --n-envs 4 --resume --resume-partial "
                "--max-records 1",
            ))
            commands.append((
                "train_next_chunk",
                "python src/v3/run_paper_pipeline_v6.py --preset formal "
                f"--stage train --terrain {terrain} --train-modes {mode} "
                "--timesteps 1000000 --train-chunk-timesteps 100000 "
                "--n-envs 4 --resume --resume-partial --max-records 1",
            ))
        commands.append((
            "train",
            "python src/v3/run_paper_pipeline_v6.py --preset formal "
            "--stage train --timesteps 1000000 --n-envs 4 "
            "--resume --resume-partial",
        ))
    if "9 CMA-ES open-loop baseline JSON files" in missing:
        target = next_missing_cmaes_target(audit)
        if target:
            terrain, mode = target
            commands.append((
                "baseline_next",
                "python src/v3/run_paper_pipeline_v6.py --preset formal "
                f"--stage baseline --terrain {terrain} --train-modes {mode} "
                "--resume --max-records 1",
            ))
            commands.append((
                "baseline_next_direct",
                "python src/v3/run_terrain_experiments.py --method cmaes "
                f"--terrain {terrain} --mode {mode} --popsize 16 "
                "--max-gen 200",
            ))
        commands.append((
            "baseline",
            "python src/v3/run_paper_pipeline_v6.py --preset formal "
            "--stage baseline --resume",
        ))
    post_train_missing = {
        "9 fixed-mode eval JSON files",
        "9 robust fixed-mode eval JSON files",
        "3 continuous gait_blend scan result files",
        "3 deployable random-policy bundles",
        "paper summary tables and figures",
    }
    if any(name in missing for name in post_train_missing):
        commands.append((
            "post_train",
            "python src/v3/run_paper_pipeline_v6.py --preset formal "
            "--stage deploy eval scan summary audit --timesteps 1000000 "
            "--n-envs 4 --resume",
        ))
    if "hardware deploy preflight report" in missing:
        commands.append((
            "hardware_preflight",
            "python src/v3/preflight_hardware_deploy_v6.py --strict",
        ))
    if "flat/sand/slope hardware logs with video references" in missing:
        commands.append((
            "hardware_status",
            "python src/v3/hardware_trial_status_v6.py --write-report",
        ))
        commands.append((
            "hardware_stream_check_flat",
            "python src/v3/check_controller_stream_v6.py --input-jsonl "
            "controller_stream.jsonl --terrain flat --mode random "
            "--video-file record/v6/videos/flat_random_hardware_demo.mp4 "
            "--gait-blend 0.5 --cmd-vel 0.025 --cmd-yaw 0.0 "
            "--bundle-dir record/v6/deploy_bundles/flat_random --strict",
        ))
        commands.append((
            "hardware_capture_flat",
            "python src/v3/capture_hardware_stream_v6.py --input-jsonl "
            "controller_stream.jsonl --output-csv "
            "record/v6/hardware/field_trials/current/flat/"
            "flat_random_raw.csv --terrain flat --mode random "
            "--video-file record/v6/videos/flat_random_hardware_demo.mp4 "
            "--gait-blend 0.5 --cmd-vel 0.025 --cmd-yaw 0.0",
        ))
        commands.append((
            "hardware_process_flat",
            "python src/v3/process_hardware_trial_v6.py --terrain flat "
            "--mode random --input-jsonl controller_stream.jsonl "
            "--raw-csv record/v6/hardware/field_trials/current/flat/"
            "flat_random_raw.csv "
            "--video-file record/v6/videos/flat_random_hardware_demo.mp4 "
            "--gait-blend 0.5 --cmd-vel 0.025 --cmd-yaw 0.0 "
            "--date YYYYMMDD",
        ))
        commands.append((
            "hardware_stream_check_sand",
            "python src/v3/check_controller_stream_v6.py --input-jsonl "
            "controller_stream.jsonl --terrain sand --mode random "
            "--video-file record/v6/videos/sand_random_hardware_demo.mp4 "
            "--gait-blend 1.0 --cmd-vel 0.025 --cmd-yaw 0.0 "
            "--bundle-dir record/v6/deploy_bundles/sand_random --strict",
        ))
        commands.append((
            "hardware_capture_sand",
            "python src/v3/capture_hardware_stream_v6.py --input-jsonl "
            "controller_stream.jsonl --output-csv "
            "record/v6/hardware/field_trials/current/sand/"
            "sand_random_raw.csv --terrain sand --mode random "
            "--video-file record/v6/videos/sand_random_hardware_demo.mp4 "
            "--gait-blend 1.0 --cmd-vel 0.025 --cmd-yaw 0.0",
        ))
        commands.append((
            "hardware_process_sand",
            "python src/v3/process_hardware_trial_v6.py --terrain sand "
            "--mode random --input-jsonl controller_stream.jsonl "
            "--raw-csv record/v6/hardware/field_trials/current/sand/"
            "sand_random_raw.csv "
            "--video-file record/v6/videos/sand_random_hardware_demo.mp4 "
            "--gait-blend 1.0 --cmd-vel 0.025 --cmd-yaw 0.0 "
            "--date YYYYMMDD",
        ))
        commands.append((
            "hardware_stream_check_slope",
            "python src/v3/check_controller_stream_v6.py --input-jsonl "
            "controller_stream.jsonl --terrain slope --mode random "
            "--video-file record/v6/videos/slope_random_hardware_demo.mp4 "
            "--gait-blend 0.0 --cmd-vel 0.025 --cmd-yaw 0.0 "
            "--bundle-dir record/v6/deploy_bundles/slope_random --strict",
        ))
        commands.append((
            "hardware_capture_slope",
            "python src/v3/capture_hardware_stream_v6.py --input-jsonl "
            "controller_stream.jsonl --output-csv "
            "record/v6/hardware/field_trials/current/slope/"
            "slope_random_raw.csv --terrain slope --mode random "
            "--video-file record/v6/videos/slope_random_hardware_demo.mp4 "
            "--gait-blend 0.0 --cmd-vel 0.025 --cmd-yaw 0.0",
        ))
        commands.append((
            "hardware_process_slope",
            "python src/v3/process_hardware_trial_v6.py --terrain slope "
            "--mode random --input-jsonl controller_stream.jsonl "
            "--raw-csv record/v6/hardware/field_trials/current/slope/"
            "slope_random_raw.csv "
            "--video-file record/v6/videos/slope_random_hardware_demo.mp4 "
            "--gait-blend 0.0 --cmd-vel 0.025 --cmd-yaw 0.0 "
            "--date YYYYMMDD",
        ))
        commands.append((
            "hardware_import_flat",
            "python src/v3/import_hardware_trial_v6.py --terrain flat "
            "--mode random --raw-csv "
            "record/v6/hardware/field_trials/current/flat/"
            "flat_random_raw.csv --video-file "
            "record/v6/videos/flat_random_hardware_demo.mp4 "
            "--gait-blend 0.5 --date YYYYMMDD",
        ))
        commands.append((
            "hardware_import_sand",
            "python src/v3/import_hardware_trial_v6.py --terrain sand "
            "--mode random --raw-csv "
            "record/v6/hardware/field_trials/current/sand/"
            "sand_random_raw.csv --video-file "
            "record/v6/videos/sand_random_hardware_demo.mp4 "
            "--gait-blend 1.0 --date YYYYMMDD",
        ))
        commands.append((
            "hardware_import_slope",
            "python src/v3/import_hardware_trial_v6.py --terrain slope "
            "--mode random --raw-csv "
            "record/v6/hardware/field_trials/current/slope/"
            "slope_random_raw.csv --video-file "
            "record/v6/videos/slope_random_hardware_demo.mp4 "
            "--gait-blend 0.0 --date YYYYMMDD",
        ))
        commands.append((
            "hardware",
            "collect flat/sand/slope real runs, then validate with "
            "python src/v3/validate_hardware_log_v6.py --input <log.csv> "
            "--expected-terrain <flat|sand|slope> --require-video "
            "--min-rows 5 --min-duration 0.1",
        ))
    if not commands:
        commands.append((
            "audit",
            "python src/v3/audit_paper_goal_v6.py",
        ))
    return commands


def status_payload(refresh=False):
    if refresh or not os.path.exists(AUDIT_JSON):
        refresh_audit()
    audit = load_json(AUDIT_JSON)
    if not audit:
        raise FileNotFoundError(rel(AUDIT_JSON))
    manifest = load_json(PIPELINE_MANIFEST)
    missing = missing_check_names(audit)
    return {
        "complete": bool(audit.get("complete", False)),
        "audit_json": rel(AUDIT_JSON),
        "pipeline_manifest": rel(PIPELINE_MANIFEST),
        "latest_manifest": manifest,
        "missing": missing,
        "next_training_target": next_missing_training_target(audit),
        "training_progress": training_matrix_progress(),
        "recommended_commands": recommended_commands(audit, missing),
    }


def print_text(payload):
    print("Worm V6 paper pipeline status")
    print(f"  complete: {payload['complete']}")
    print(f"  audit:    {payload['audit_json']}")
    print(f"  manifest: {payload['pipeline_manifest']}")

    manifest = payload.get("latest_manifest")
    if manifest:
        print("  last_pipeline:")
        print(f"    preset: {manifest.get('preset')}")
        print(f"    dry_run: {manifest.get('dry_run')}")
        print(f"    resume: {manifest.get('resume')}")
        print(f"    resume_partial: {manifest.get('resume_partial')}")
        print(f"    train_chunk_timesteps: {manifest.get('train_chunk_timesteps')}")
        print(f"    max_records: {manifest.get('max_records')}")
        print(f"    stages: {manifest.get('stages')}")
        print(f"    records: {len(manifest.get('records', []))}")

    if payload.get("next_training_target"):
        terrain, mode = payload["next_training_target"]
        print(f"  next_training_target: {terrain}/{mode}")

    progress = payload.get("training_progress", [])
    completed = sum(1 for row in progress if row["complete"])
    artifact_completed = sum(
        1 for row in progress if row.get("artifact_complete"))
    partial = [
        row for row in progress
        if row["exists"] and not row["complete"]
    ]
    print(f"  training_artifacts_present: {artifact_completed}/{len(progress)}")
    print(f"  training_current_timing: {completed}/{len(progress)}")
    if partial:
        print("  timing_or_artifact_mismatch:")
        for row in partial:
            model_flag = "model" if row["has_model_pair"] else "no-model"
            reason_text = ", ".join(row.get("audit_reasons", [])[:3])
            print(
                f"    {row['terrain']}/{row['mode']}: "
                f"{row['completed_timesteps']}/"
                f"{REQUIRED_TRAIN_TIMESTEPS} steps, {model_flag}"
                f"{' (' + reason_text + ')' if reason_text else ''}")

    if payload["missing"]:
        print("\nMissing:")
        for name in payload["missing"]:
            print(f"  - {name}")
    else:
        print("\nMissing: none")

    print("\nNext commands:")
    for label, command in payload["recommended_commands"]:
        print(f"  [{label}] {command}")


def main():
    ap = argparse.ArgumentParser(
        description="Show Worm V6 paper pipeline completion status")
    ap.add_argument("--refresh-audit", action="store_true",
                    help="Run audit_paper_goal_v6.py before reading status")
    ap.add_argument("--json", action="store_true",
                    help="Print machine-readable JSON")
    ap.add_argument("--strict", action="store_true",
                    help="Exit nonzero when the audit is incomplete")
    args = ap.parse_args()

    payload = status_payload(refresh=args.refresh_audit)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text(payload)
    if args.strict and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

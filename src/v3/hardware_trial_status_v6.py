"""
Report flat/sand/slope real-hardware trial readiness for the Worm V6 paper.

This is a diagnostic helper. It does not create or modify hardware data. It
shows whether each terrain has a captured raw CSV, a resolvable video, and a
validated formal policy CSV accepted by the paper audit.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import raw_columns  # noqa: E402
from prepare_hardware_trials_v6 import (  # noqa: E402
    DEFAULT_BEST_BLEND_CSV,
    TERRAINS,
    read_recommended_blends,
)
from validate_hardware_log_v6 import (  # noqa: E402
    VALID_MODES,
    validate_csv,
    video_reference_resolves,
)


DEFAULT_MODE = "random"
DEFAULT_MIN_ROWS = 5
DEFAULT_MIN_DURATION_S = 0.1
DEFAULT_STATUS_JSON = os.path.join(
    PROJECT_ROOT, "record", "v6", "hardware",
    "hardware_trial_status.json")
DEFAULT_STATUS_MD = os.path.join(
    PROJECT_ROOT, "record", "v6", "hardware",
    "hardware_trial_status.md")


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def resolve(path, project_root=PROJECT_ROOT):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def hardware_dir(project_root=PROJECT_ROOT):
    return os.path.join(project_root, "record", "v6", "hardware")


def field_dir(project_root=PROJECT_ROOT):
    return os.path.join(
        project_root, "record", "v6", "hardware", "field_trials",
        "current")


def video_path(terrain, mode=DEFAULT_MODE, project_root=PROJECT_ROOT):
    return os.path.join(
        project_root, "record", "v6", "videos",
        f"{terrain}_{mode}_hardware_demo.mp4")


def raw_path(terrain, mode=DEFAULT_MODE, project_root=PROJECT_ROOT):
    return os.path.join(
        field_dir(project_root), terrain, f"{terrain}_{mode}_raw.csv")


def action_path(terrain, mode=DEFAULT_MODE, project_root=PROJECT_ROOT):
    return os.path.join(
        field_dir(project_root), terrain, f"{terrain}_{mode}_actions.csv")


def policy_candidates(terrain, project_root=PROJECT_ROOT):
    root = hardware_dir(project_root)
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        if not name.endswith(".csv"):
            continue
        if "template" in name or name.endswith("_actions.csv"):
            continue
        if name.startswith(f"{terrain}_"):
            out.append(os.path.join(root, name))
    return sorted(out)


def csv_rows_duration(path, required=None):
    if not os.path.exists(path):
        return {
            "exists": False,
            "rows": 0,
            "duration_s": 0.0,
            "missing_columns": [],
        }
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [
            col for col in (required or [])
            if col not in fieldnames
        ]
        times = []
        rows = 0
        for rows, row in enumerate(reader, start=1):
            try:
                times.append(float(row.get("time_s", "")))
            except (TypeError, ValueError):
                pass
    duration = (max(times) - min(times)) if len(times) >= 2 else 0.0
    return {
        "exists": True,
        "rows": rows,
        "duration_s": float(duration),
        "missing_columns": missing,
    }


def validated_policy_candidate(terrain, candidates, min_rows, min_duration_s):
    invalid = []
    for path in candidates:
        try:
            metrics = validate_csv(
                path,
                expected_terrain=terrain,
                require_video=True,
                min_rows=min_rows,
                min_duration_s=min_duration_s,
                verbose=False,
            )
            return path, metrics, invalid
        except Exception as exc:
            invalid.append({
                "path": path,
                "reason": str(exc),
            })
    return None, None, invalid


def classify(raw, video_exists, policy_valid, min_rows, min_duration_s):
    if policy_valid:
        return "complete"
    if not raw["exists"]:
        return "needs_raw_csv"
    if raw["missing_columns"]:
        return "raw_csv_schema_error"
    if raw["rows"] < min_rows:
        return "raw_csv_too_short"
    if raw["duration_s"] < min_duration_s:
        return "raw_csv_duration_too_short"
    if not video_exists:
        return "needs_video"
    return "ready_to_import"


def import_command(terrain, mode, gait_blend, date_stamp=None):
    date_text = date_stamp or "YYYYMMDD"
    return (
        "python src\\v3\\import_hardware_trial_v6.py "
        f"--terrain {terrain} --mode {mode} "
        f"--raw-csv record\\v6\\hardware\\field_trials\\current\\{terrain}\\"
        f"{terrain}_{mode}_raw.csv "
        f"--video-file record/v6/videos/{terrain}_{mode}_hardware_demo.mp4 "
        f"--gait-blend {gait_blend:.3f} --date {date_text}"
    )


def capture_command(terrain, mode, gait_blend):
    return (
        "python src\\v3\\capture_hardware_stream_v6.py "
        "--input-jsonl controller_stream.jsonl "
        f"--output-csv record\\v6\\hardware\\field_trials\\current\\{terrain}\\"
        f"{terrain}_{mode}_raw.csv "
        f"--terrain {terrain} --mode {mode} "
        f"--video-file record/v6/videos/{terrain}_{mode}_hardware_demo.mp4 "
        f"--gait-blend {gait_blend:.3f} --cmd-vel 0.025 --cmd-yaw 0.0"
    )


def process_command(terrain, mode, gait_blend, date_stamp=None):
    date_text = date_stamp or "YYYYMMDD"
    return (
        "python src\\v3\\process_hardware_trial_v6.py "
        f"--terrain {terrain} --mode {mode} "
        "--input-jsonl controller_stream.jsonl "
        f"--raw-csv record\\v6\\hardware\\field_trials\\current\\{terrain}\\"
        f"{terrain}_{mode}_raw.csv "
        f"--video-file record/v6/videos/{terrain}_{mode}_hardware_demo.mp4 "
        f"--gait-blend {gait_blend:.3f} --cmd-vel 0.025 --cmd-yaw 0.0 "
        f"--date {date_text}"
    )


def stream_check_command(terrain, mode, gait_blend):
    return (
        "python src\\v3\\check_controller_stream_v6.py "
        "--input-jsonl controller_stream.jsonl "
        f"--terrain {terrain} --mode {mode} "
        f"--video-file record/v6/videos/{terrain}_{mode}_hardware_demo.mp4 "
        f"--gait-blend {gait_blend:.3f} --cmd-vel 0.025 --cmd-yaw 0.0 "
        f"--bundle-dir record\\v6\\deploy_bundles\\{terrain}_{mode} "
        "--strict"
    )


def video_command(terrain, mode):
    return (
        "copy or record the demo video to "
        f"record/v6/videos/{terrain}_{mode}_hardware_demo.mp4"
    )


def validate_command(terrain, policy_log):
    return (
        "python src\\v3\\validate_hardware_log_v6.py "
        f"--input {policy_log.replace('/', os.sep)} "
        f"--expected-terrain {terrain} --require-video "
        "--min-rows 5 --min-duration 0.1"
    )


def next_command_for(status, terrain, mode, gait_blend, policy_log):
    if status in (
            "needs_raw_csv", "raw_csv_schema_error",
            "raw_csv_too_short", "raw_csv_duration_too_short"):
        return capture_command(terrain, mode, gait_blend)
    if status == "needs_video":
        return video_command(terrain, mode)
    if status == "ready_to_import":
        return import_command(terrain, mode, gait_blend)
    if status == "complete":
        return validate_command(terrain, policy_log)
    return import_command(terrain, mode, gait_blend)


def terrain_status(terrain, mode=DEFAULT_MODE, project_root=PROJECT_ROOT,
                   best_blend_csv=DEFAULT_BEST_BLEND_CSV,
                   min_rows=DEFAULT_MIN_ROWS,
                   min_duration_s=DEFAULT_MIN_DURATION_S):
    best_blend_csv = resolve(best_blend_csv, project_root)
    blends = read_recommended_blends(best_blend_csv)
    gait_blend = float(blends.get(terrain, 0.5))
    raw = csv_rows_duration(
        raw_path(terrain, mode, project_root),
        required=raw_columns(),
    )
    video = video_path(terrain, mode, project_root)
    video_exists = video_reference_resolves(video, raw_path(
        terrain, mode, project_root))
    candidates = policy_candidates(terrain, project_root)
    valid_path, metrics, invalid = validated_policy_candidate(
        terrain, candidates, min_rows, min_duration_s)
    status = classify(raw, video_exists, bool(valid_path), min_rows,
                      min_duration_s)
    policy_log_rel = (
        rel(valid_path, project_root) if valid_path
        else f"record/v6/hardware/{terrain}_{mode}_YYYYMMDD.csv")
    next_command = next_command_for(
        status, terrain, mode, gait_blend, policy_log_rel)
    return {
        "terrain": terrain,
        "mode": mode,
        "recommended_gait_blend": gait_blend,
        "status": status,
        "raw_csv": rel(raw_path(terrain, mode, project_root), project_root),
        "raw_exists": raw["exists"],
        "raw_rows": raw["rows"],
        "raw_duration_s": raw["duration_s"],
        "raw_missing_columns": raw["missing_columns"],
        "video_file": rel(video, project_root),
        "video_exists": bool(video_exists),
        "policy_log_csv": policy_log_rel,
        "policy_valid": bool(valid_path),
        "policy_metrics": metrics,
        "invalid_policy_candidates": [
            {
                "path": rel(item["path"], project_root),
                "reason": item["reason"],
            }
            for item in invalid
        ],
        "import_command": import_command(
            terrain, mode, gait_blend),
        "capture_command": capture_command(terrain, mode, gait_blend),
        "process_command": process_command(terrain, mode, gait_blend),
        "stream_check_command": stream_check_command(
            terrain, mode, gait_blend),
        "validate_command": validate_command(terrain, policy_log_rel),
        "next_command": next_command,
    }


def status_payload(mode=DEFAULT_MODE, project_root=PROJECT_ROOT,
                   best_blend_csv=DEFAULT_BEST_BLEND_CSV,
                   min_rows=DEFAULT_MIN_ROWS,
                   min_duration_s=DEFAULT_MIN_DURATION_S):
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    terrains = [
        terrain_status(
            terrain,
            mode=mode,
            project_root=project_root,
            best_blend_csv=best_blend_csv,
            min_rows=min_rows,
            min_duration_s=min_duration_s,
        )
        for terrain in TERRAINS
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "min_rows": min_rows,
        "min_duration_s": min_duration_s,
        "complete": all(row["status"] == "complete" for row in terrains),
        "terrains": terrains,
    }


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Hardware Trial Status",
        "",
        f"Complete: `{str(payload['complete']).lower()}`",
        "",
        "| Terrain | Status | Raw rows | Raw duration s | Video | Policy CSV |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["terrains"]:
        lines.append(
            f"| {row['terrain']} | `{row['status']}` | "
            f"{row['raw_rows']} | {row['raw_duration_s']:.3f} | "
            f"{'yes' if row['video_exists'] else 'no'} | "
            f"{'valid' if row['policy_valid'] else row['policy_log_csv']} |")
    lines.extend(["", "## Next Commands", ""])
    for row in payload["terrains"]:
        if row["status"] != "complete":
            lines.extend([
                f"### {row['terrain']}",
                "",
                "Controller stream self-check before importing the run:",
                "",
                "```powershell",
                row["stream_check_command"],
                "```",
                "",
                "One-command processing after the controller JSONL stream and "
                "demo video are available:",
                "",
                "```powershell",
                row["process_command"],
                "```",
                "",
                "Status-specific fallback:",
                "",
                "```powershell",
                row["next_command"],
                "```",
                "",
            ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_text(payload):
    print("Worm V6 hardware trial status")
    print(f"  complete: {payload['complete']}")
    for row in payload["terrains"]:
        print(
            f"  {row['terrain']}: {row['status']} "
            f"(raw_rows={row['raw_rows']}, "
            f"duration={row['raw_duration_s']:.3f}s, "
            f"video={row['video_exists']}, "
            f"policy_valid={row['policy_valid']})")
        if row["status"] != "complete":
            print(f"    process: {row['process_command']}")
            print(f"    next: {row['next_command']}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Show flat/sand/slope Worm V6 hardware trial status")
    parser.add_argument("--mode", choices=VALID_MODES, default=DEFAULT_MODE)
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-duration", type=float,
                        default=DEFAULT_MIN_DURATION_S)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-report", action="store_true",
                        help="Write status JSON and Markdown reports")
    parser.add_argument("--json-out", default=DEFAULT_STATUS_JSON)
    parser.add_argument("--md-out", default=DEFAULT_STATUS_MD)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero unless all terrains are complete")
    return parser


def main():
    args = build_parser().parse_args()
    payload = status_payload(
        mode=args.mode,
        best_blend_csv=args.best_blend_csv,
        min_rows=args.min_rows,
        min_duration_s=args.min_duration,
    )
    if args.write_report:
        write_json(args.json_out, payload)
        write_markdown(args.md_out, payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text(payload)
        if args.write_report:
            print(f"  json: {rel(args.json_out)}")
            print(f"  md:   {rel(args.md_out)}")
    if args.strict and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

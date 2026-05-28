"""
Import one real Worm V6 hardware trial into the paper audit layout.

This script does not generate hardware data. It takes a captured raw sensor
CSV plus a video reference, runs the deployable TorchScript policy, writes the
formal 80-D hardware audit CSV, validates it, and records the import manifest.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from hardware_policy_runtime_v6 import (  # noqa: E402
    HardwarePolicyRuntime,
    run_raw_csv,
)
from prepare_hardware_trials_v6 import (  # noqa: E402
    DEFAULT_BEST_BLEND_CSV,
    read_recommended_blends,
)
from validate_hardware_log_v6 import (  # noqa: E402
    VALID_MODES,
    VALID_TERRAINS,
    video_reference_resolves,
)


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def is_url(reference):
    return urlparse(str(reference or "")).scheme in ("http", "https")


def resolve_input_path(path, project_root=PROJECT_ROOT):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def default_bundle_dir(terrain, mode, project_root=PROJECT_ROOT):
    return os.path.join(
        project_root, "record", "v6", "deploy_bundles",
        f"{terrain}_{mode}")


def default_hardware_dir(project_root=PROJECT_ROOT):
    return os.path.join(project_root, "record", "v6", "hardware")


def default_field_dir(project_root=PROJECT_ROOT):
    return os.path.join(
        project_root, "record", "v6", "hardware", "field_trials",
        "current")


def default_videos_dir(project_root=PROJECT_ROOT):
    return os.path.join(project_root, "record", "v6", "videos")


def default_policy_log_path(terrain, mode, date_stamp, hardware_dir):
    return os.path.join(hardware_dir, f"{terrain}_{mode}_{date_stamp}.csv")


def default_action_log_path(terrain, mode, field_dir):
    return os.path.join(field_dir, terrain, f"{terrain}_{mode}_actions.csv")


def default_manifest_path(hardware_dir):
    return os.path.join(hardware_dir, "hardware_trial_import_manifest.json")


def default_video_copy_path(terrain, mode, video_file, videos_dir):
    ext = os.path.splitext(urlparse(video_file).path)[1] or ".mp4"
    return os.path.join(videos_dir, f"{terrain}_{mode}_hardware_demo{ext}")


def copied_video_reference(destination, project_root):
    if os.path.normcase(os.path.abspath(project_root)) == os.path.normcase(
            os.path.abspath(PROJECT_ROOT)):
        return rel(destination, project_root)
    return os.path.abspath(destination)


def recommended_gait_blend(terrain, gait_blend, best_blend_csv):
    if gait_blend is not None:
        return max(0.0, min(1.0, float(gait_blend)))
    blends = read_recommended_blends(best_blend_csv)
    return float(blends.get(terrain, 0.5))


def video_reference_for_import(terrain, mode, video_file, project_root,
                               videos_dir, copy_video=False,
                               copy_video_to=None):
    if not video_file:
        raise ValueError("video_file is required for a hardware trial import")
    if is_url(video_file):
        if copy_video or copy_video_to:
            raise ValueError("Cannot copy an HTTP(S) video reference")
        return video_file

    source = resolve_input_path(video_file, project_root)
    if copy_video or copy_video_to:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video source not found: {source}")
        destination = copy_video_to or default_video_copy_path(
            terrain, mode, video_file, videos_dir)
        destination = resolve_input_path(destination, project_root)
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        shutil.copy2(source, destination)
        return copied_video_reference(destination, project_root)

    return video_file.replace("\\", "/")


def read_manifest(path):
    if not os.path.exists(path):
        return {
            "format_version": 1,
            "updated_at_utc": None,
            "trials": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    data.setdefault("format_version", 1)
    data.setdefault("trials", {})
    return data


def update_import_manifest(path, entry):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    manifest = read_manifest(path)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["trials"][entry["terrain"]] = entry
    manifest["complete_terrains"] = sorted(manifest["trials"].keys())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def import_trial(terrain, raw_csv, video_file, mode="random",
                 gait_blend=None, date_stamp=None, bundle_dir=None,
                 policy_log_csv=None, action_csv=None, hardware_dir=None,
                 field_dir=None, videos_dir=None, manifest_path=None,
                 best_blend_csv=DEFAULT_BEST_BLEND_CSV, copy_video=False,
                 copy_video_to=None, max_action_delta=0.2, min_rows=5,
                 min_duration_s=0.1, update_manifest=True,
                 project_root=PROJECT_ROOT):
    if terrain not in VALID_TERRAINS:
        raise ValueError(f"Unknown terrain: {terrain}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    raw_csv = resolve_input_path(raw_csv, project_root)
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    date_stamp = date_stamp or datetime.now().strftime("%Y%m%d")
    if best_blend_csv:
        best_blend_csv = resolve_input_path(best_blend_csv, project_root)
    hardware_dir = resolve_input_path(
        hardware_dir or default_hardware_dir(project_root), project_root)
    field_dir = resolve_input_path(
        field_dir or default_field_dir(project_root), project_root)
    videos_dir = resolve_input_path(
        videos_dir or default_videos_dir(project_root), project_root)
    bundle_dir = resolve_input_path(
        bundle_dir or default_bundle_dir(terrain, mode, project_root),
        project_root)
    policy_log_csv = resolve_input_path(
        policy_log_csv or default_policy_log_path(
            terrain, mode, date_stamp, hardware_dir),
        project_root)
    action_csv = resolve_input_path(
        action_csv or default_action_log_path(terrain, mode, field_dir),
        project_root)
    manifest_path = resolve_input_path(
        manifest_path or default_manifest_path(hardware_dir), project_root)
    gait_blend = recommended_gait_blend(terrain, gait_blend, best_blend_csv)

    video_ref = video_reference_for_import(
        terrain, mode, video_file,
        project_root=project_root,
        videos_dir=videos_dir,
        copy_video=copy_video,
        copy_video_to=copy_video_to,
    )
    if not video_reference_resolves(video_ref, policy_log_csv):
        raise ValueError(
            "video_file does not resolve from the final policy log: "
            f"{video_ref}")

    runtime = HardwarePolicyRuntime(bundle_dir=bundle_dir)
    result = run_raw_csv(
        runtime,
        raw_csv,
        action_csv,
        policy_log_csv=policy_log_csv,
        metadata_overrides={
            "terrain": terrain,
            "mode": mode,
            "video_file": video_ref,
            "gait_blend": gait_blend,
        },
        validate_policy_log=True,
        expected_terrain=terrain,
        require_video=True,
        min_rows=min_rows,
        min_duration_s=min_duration_s,
        max_action_delta=max_action_delta,
    )

    entry = {
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "terrain": terrain,
        "mode": mode,
        "gait_blend": gait_blend,
        "raw_csv": rel(raw_csv, project_root),
        "bundle_dir": rel(bundle_dir, project_root),
        "policy_log_csv": rel(policy_log_csv, project_root),
        "action_csv": rel(action_csv, project_root),
        "video_file": video_ref,
        "max_action_delta": max_action_delta,
        "metrics": result.get("policy_log_metrics"),
    }
    if update_manifest:
        update_import_manifest(manifest_path, entry)
        entry["manifest"] = rel(manifest_path, project_root)
    return entry


def build_parser():
    parser = argparse.ArgumentParser(
        description="Import one Worm V6 real hardware trial into audit files")
    parser.add_argument("--terrain", choices=VALID_TERRAINS, required=True)
    parser.add_argument("--raw-csv", required=True,
                        help="Captured raw hardware sensor CSV")
    parser.add_argument("--video-file", required=True,
                        help="Resolvable video reference or local source file")
    parser.add_argument("--mode", choices=VALID_MODES, default="random")
    parser.add_argument("--gait-blend", type=float, default=None,
                        help="Override gait_blend; default uses scan result")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD suffix for default output CSV")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--policy-log-csv", default=None)
    parser.add_argument("--output-csv", default=None,
                        help="Actions and physical targets CSV")
    parser.add_argument("--hardware-dir", default=None)
    parser.add_argument("--field-dir", default=None)
    parser.add_argument("--videos-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--copy-video", action="store_true",
                        help="Copy --video-file into record/v6/videos")
    parser.add_argument("--copy-video-to", default=None,
                        help="Copy --video-file to this destination")
    parser.add_argument("--max-action-delta", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--no-update-manifest", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    entry = import_trial(
        terrain=args.terrain,
        raw_csv=args.raw_csv,
        video_file=args.video_file,
        mode=args.mode,
        gait_blend=args.gait_blend,
        date_stamp=args.date,
        bundle_dir=args.bundle_dir,
        policy_log_csv=args.policy_log_csv,
        action_csv=args.output_csv,
        hardware_dir=args.hardware_dir,
        field_dir=args.field_dir,
        videos_dir=args.videos_dir,
        manifest_path=args.manifest,
        best_blend_csv=args.best_blend_csv,
        copy_video=args.copy_video,
        copy_video_to=args.copy_video_to,
        max_action_delta=args.max_action_delta,
        min_rows=args.min_rows,
        min_duration_s=args.min_duration,
        update_manifest=not args.no_update_manifest,
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()

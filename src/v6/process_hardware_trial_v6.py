"""
Run the post-capture hardware trial pipeline for one Worm V6 terrain.

This helper does not create real hardware evidence. It takes a controller JSONL
stream or an already captured raw CSV, imports it through the deployable policy,
validates the resulting paper-audit CSV, and refreshes hardware status reports.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from capture_hardware_stream_v6 import (  # noqa: E402
    capture_jsonl,
    metadata_overrides,
)
from hardware_trial_status_v6 import (  # noqa: E402
    DEFAULT_MIN_DURATION_S,
    DEFAULT_MIN_ROWS,
    DEFAULT_MODE,
    DEFAULT_STATUS_JSON,
    DEFAULT_STATUS_MD,
    action_path,
    raw_path,
    status_payload,
    video_path,
    write_json,
    write_markdown,
)
from import_hardware_trial_v6 import (  # noqa: E402
    import_trial,
    recommended_gait_blend,
)
from prepare_hardware_trials_v6 import DEFAULT_BEST_BLEND_CSV  # noqa: E402
from validate_hardware_log_v6 import VALID_MODES, VALID_TERRAINS  # noqa: E402


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def resolve(path, project_root=PROJECT_ROOT):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def terrain_row(payload, terrain):
    for row in payload.get("terrains", []):
        if row.get("terrain") == terrain:
            return row
    return {}


def process_trial(terrain, video_file, mode=DEFAULT_MODE, input_jsonl=None,
                  raw_csv=None, date_stamp=None, gait_blend=None,
                  cmd_vel=0.025, cmd_yaw=0.0, cmd_vx=None, cmd_vy=0.0,
                  bundle_dir=None,
                  policy_log_csv=None, action_csv=None, hardware_dir=None,
                  field_dir=None, videos_dir=None, manifest_path=None,
                  best_blend_csv=DEFAULT_BEST_BLEND_CSV, copy_video=False,
                  copy_video_to=None, max_action_delta=0.2,
                  min_rows=DEFAULT_MIN_ROWS,
                  min_duration_s=DEFAULT_MIN_DURATION_S, max_rows=None,
                  strict_time=True, status_json=DEFAULT_STATUS_JSON,
                  status_md=DEFAULT_STATUS_MD, write_reports=True,
                  project_root=PROJECT_ROOT):
    if terrain not in VALID_TERRAINS:
        raise ValueError(f"Unknown terrain: {terrain}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if not input_jsonl and not raw_csv:
        raise ValueError("Provide either input_jsonl or raw_csv")
    if not video_file:
        video_file = rel(video_path(terrain, mode, project_root), project_root)

    raw_csv = resolve(
        raw_csv or raw_path(terrain, mode, project_root), project_root)
    best_blend_csv = resolve(best_blend_csv, project_root)
    effective_gait_blend = recommended_gait_blend(
        terrain, gait_blend, best_blend_csv)

    capture_result = None
    if input_jsonl:
        capture_result = capture_jsonl(
            resolve(input_jsonl, project_root),
            raw_csv,
            overrides=metadata_overrides(
                terrain=terrain,
                mode=mode,
                video_file=video_file,
                gait_blend=effective_gait_blend,
                cmd_vel=cmd_vel,
                cmd_vx=cmd_vx,
                cmd_vy=cmd_vy,
                cmd_yaw=cmd_yaw,
            ),
            strict_time=strict_time,
            max_rows=max_rows,
        )

    action_csv = resolve(
        action_csv or action_path(terrain, mode, project_root), project_root)
    entry = import_trial(
        terrain=terrain,
        raw_csv=raw_csv,
        video_file=video_file,
        mode=mode,
        gait_blend=effective_gait_blend,
        date_stamp=date_stamp,
        bundle_dir=bundle_dir,
        policy_log_csv=policy_log_csv,
        action_csv=action_csv,
        hardware_dir=hardware_dir,
        field_dir=field_dir,
        videos_dir=videos_dir,
        manifest_path=manifest_path,
        best_blend_csv=best_blend_csv,
        copy_video=copy_video,
        copy_video_to=copy_video_to,
        max_action_delta=max_action_delta,
        min_rows=min_rows,
        min_duration_s=min_duration_s,
        project_root=project_root,
    )

    payload = status_payload(
        mode=mode,
        project_root=project_root,
        best_blend_csv=best_blend_csv,
        min_rows=min_rows,
        min_duration_s=min_duration_s,
    )
    status_json = resolve(status_json, project_root)
    status_md = resolve(status_md, project_root)
    if write_reports:
        write_json(status_json, payload)
        write_markdown(status_md, payload)

    return {
        "terrain": terrain,
        "mode": mode,
        "capture": capture_result,
        "import": entry,
        "terrain_status": terrain_row(payload, terrain),
        "status_complete": payload.get("complete", False),
        "status_json": rel(status_json, project_root),
        "status_md": rel(status_md, project_root),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Process one Worm V6 real hardware trial through capture, import, "
            "validation, and status reporting"))
    parser.add_argument("--terrain", choices=VALID_TERRAINS, required=True)
    parser.add_argument("--mode", choices=VALID_MODES, default=DEFAULT_MODE)
    parser.add_argument("--input-jsonl", default=None,
                        help="Controller JSONL stream captured from robot")
    parser.add_argument("--raw-csv", default=None,
                        help="Existing raw CSV; also output path when JSONL is provided")
    parser.add_argument("--video-file", default=None,
                        help="Resolvable demo video reference or local source")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD suffix for default policy log CSV")
    parser.add_argument("--gait-blend", type=float, default=None)
    parser.add_argument("--cmd-vel", type=float, default=0.025,
                        help="Legacy alias for --cmd-vx")
    parser.add_argument("--cmd-vx", type=float, default=None)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--policy-log-csv", default=None)
    parser.add_argument("--output-csv", default=None,
                        help="Actions and physical targets CSV")
    parser.add_argument("--hardware-dir", default=None)
    parser.add_argument("--field-dir", default=None)
    parser.add_argument("--videos-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--copy-video", action="store_true")
    parser.add_argument("--copy-video-to", default=None)
    parser.add_argument("--max-action-delta", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-duration", type=float,
                        default=DEFAULT_MIN_DURATION_S)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--allow-nonmonotonic-time", action="store_true")
    parser.add_argument("--status-json", default=DEFAULT_STATUS_JSON)
    parser.add_argument("--status-md", default=DEFAULT_STATUS_MD)
    parser.add_argument("--no-write-status", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    result = process_trial(
        terrain=args.terrain,
        video_file=args.video_file,
        mode=args.mode,
        input_jsonl=args.input_jsonl,
        raw_csv=args.raw_csv,
        date_stamp=args.date,
        gait_blend=args.gait_blend,
        cmd_vel=args.cmd_vel,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_yaw=args.cmd_yaw,
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
        max_rows=args.max_rows,
        strict_time=not args.allow_nonmonotonic_time,
        status_json=args.status_json,
        status_md=args.status_md,
        write_reports=not args.no_write_status,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = result["terrain_status"]
        print("Worm V6 hardware trial processed")
        print(f"  terrain: {result['terrain']}")
        print(f"  mode: {result['mode']}")
        if result["capture"]:
            print(f"  captured rows: {result['capture']['rows']}")
            print(f"  raw csv: {rel(result['capture']['output_csv'])}")
        print(f"  policy log: {result['import']['policy_log_csv']}")
        print(f"  action csv: {result['import']['action_csv']}")
        print(f"  status: {status.get('status')}")
        print(f"  status json: {result['status_json']}")
        print(f"  status md: {result['status_md']}")


if __name__ == "__main__":
    main()

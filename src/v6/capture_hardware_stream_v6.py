"""
Capture raw Worm V6 hardware sensor JSONL into the standard raw CSV format.

The robot controller can stream one JSON object per line with encoder and IMU
readings in physical units. This helper records that stream as the raw CSV
expected by import_hardware_trial_v6.py and the hardware audit pipeline.
"""

import argparse
import csv
import json
import math
import os
import sys
from contextlib import nullcontext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import neutral_raw_row, raw_columns  # noqa: E402
from validate_hardware_log_v6 import VALID_MODES, VALID_TERRAINS  # noqa: E402
from worm_env_v6 import CMD_VX_RANGE  # noqa: E402


TEXT_COLUMNS = {"terrain", "mode", "video_file"}
OPTIONAL_NUMERIC_DEFAULTS = {
    "velocity_estimate_m_s": 0.0,
    "yaw_rate_estimate_rad_s": 0.0,
}
for _i in range(11):
    OPTIONAL_NUMERIC_DEFAULTS[f"action_{_i:02d}"] = 0.0

OVERRIDABLE_COLUMNS = {
    "terrain",
    "mode",
    "video_file",
    "gait_blend",
    "cmd_vx_m_s",
    "cmd_vy_m_s",
    "cmd_yaw_rad_s",
}


def open_input_stream(path):
    if path == "-":
        return nullcontext(sys.stdin)
    return open(path, "r", encoding="utf-8")


def metadata_overrides(terrain=None, mode=None, video_file=None,
                       gait_blend=None, cmd_vel=None, cmd_yaw=None,
                       cmd_vx=None, cmd_vy=None):
    if cmd_vx is None:
        cmd_vx = cmd_vel
    return {
        "terrain": terrain,
        "mode": mode,
        "video_file": video_file,
        "gait_blend": gait_blend,
        "cmd_vx_m_s": cmd_vx,
        "cmd_vy_m_s": cmd_vy,
        "cmd_yaw_rad_s": cmd_yaw,
    }


def apply_overrides(row, overrides):
    out = dict(row)
    for key, value in (overrides or {}).items():
        if value is not None:
            out[key] = value
    return out


def required_stream_columns(use_cli_overrides=True):
    optional = set(OPTIONAL_NUMERIC_DEFAULTS)
    if use_cli_overrides:
        optional.update(OVERRIDABLE_COLUMNS)
    return [col for col in raw_columns() if col not in optional]


def example_stream_row(time_s=0.0, terrain="flat", mode="random",
                       video_file="record/v6/videos/flat_random_hardware_demo.mp4",
                       gait_blend=0.5, cmd_vel=0.025, cmd_yaw=0.0,
                       cmd_vx=None, cmd_vy=0.0,
                       include_metadata=False):
    if cmd_vx is None:
        cmd_vx = cmd_vel
    row = neutral_raw_row()
    row.update({
        "time_s": float(time_s),
        "terrain": terrain,
        "mode": mode,
        "video_file": video_file,
        "gait_blend": float(gait_blend),
        "cmd_vx_m_s": float(cmd_vx),
        "cmd_vy_m_s": float(cmd_vy),
        "cmd_yaw_rad_s": float(cmd_yaw),
    })
    for key in OPTIONAL_NUMERIC_DEFAULTS:
        row.pop(key, None)
    if not include_metadata:
        for key in OVERRIDABLE_COLUMNS:
            row.pop(key, None)
    return row


def write_jsonl_example(path, rows=5, terrain="flat", mode="random",
                        video_file=None, gait_blend=0.5, cmd_vel=0.025,
                        cmd_yaw=0.0, cmd_vx=None, cmd_vy=0.0,
                        include_metadata=False):
    video_file = video_file or (
        f"record/v6/videos/{terrain}_{mode}_hardware_demo.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(rows):
            row = example_stream_row(
                time_s=i * 0.05,
                terrain=terrain,
                mode=mode,
                video_file=video_file,
                gait_blend=gait_blend,
                cmd_vel=cmd_vel,
                cmd_vx=cmd_vx,
                cmd_vy=cmd_vy,
                cmd_yaw=cmd_yaw,
                include_metadata=include_metadata,
            )
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")
    return {
        "path": path,
        "rows": rows,
        "required_with_cli_overrides": required_stream_columns(
            use_cli_overrides=True),
    }


def write_json_schema(path):
    schema = {
        "format_version": 1,
        "description": (
            "One JSON object per controller frame. Use CLI overrides for "
            "terrain/mode/video/command metadata during field trials."),
        "required_with_cli_overrides": required_stream_columns(
            use_cli_overrides=True),
        "required_without_cli_overrides": required_stream_columns(
            use_cli_overrides=False),
        "optional_defaults": dict(OPTIONAL_NUMERIC_DEFAULTS),
        "cli_overridable_columns": sorted(OVERRIDABLE_COLUMNS),
        "raw_csv_columns": raw_columns(),
        "units": {
            "time_s": "seconds, monotonic",
            "slide_pos_m_*": "meters",
            "yaw_pos_rad_*": "radians",
            "slide_vel_m_s_*": "meters/second",
            "yaw_vel_rad_s_*": "radians/second",
            "segment_gravity_*": "unitless local gravity direction",
            "segment_gyro_rad_s_*": "radians/second in segment frame",
            "action_*": "normalized previous or commanded action in [-1, 1]",
            "velocity_estimate_m_s": (
                "external estimate for reporting only, not policy input"),
            "yaw_rate_estimate_rad_s": (
                "external estimate for reporting only, not policy input"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return schema


def _finite_float(row, key, line_no):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"JSONL line {line_no}: missing or invalid numeric field {key}")
    if not math.isfinite(value):
        raise ValueError(f"JSONL line {line_no}: non-finite field {key}")
    return value


def normalize_raw_row(row, line_no, overrides=None):
    row = apply_overrides(row, overrides)
    for key, value in OPTIONAL_NUMERIC_DEFAULTS.items():
        if key not in row or row[key] == "":
            row[key] = value

    out = {}
    missing = []
    for col in raw_columns():
        if col not in row or row[col] == "":
            missing.append(col)
            continue
        if col in TEXT_COLUMNS:
            out[col] = str(row[col])
        else:
            out[col] = _finite_float(row, col, line_no)
    if missing:
        raise ValueError(
            f"JSONL line {line_no}: missing required fields {missing}")

    if out["terrain"] not in VALID_TERRAINS:
        raise ValueError(
            f"JSONL line {line_no}: invalid terrain {out['terrain']!r}")
    if out["mode"] not in VALID_MODES:
        raise ValueError(f"JSONL line {line_no}: invalid mode {out['mode']!r}")
    if not 0.0 <= float(out["gait_blend"]) <= 1.0:
        raise ValueError(f"JSONL line {line_no}: gait_blend outside [0, 1]")
    return out


def capture_jsonl(input_jsonl, output_csv, overrides=None, strict_time=True,
                  max_rows=None):
    rows = []
    previous_time = None
    with open_input_stream(input_jsonl) as src:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            row = normalize_raw_row(raw, line_no, overrides=overrides)
            time_s = float(row["time_s"])
            if strict_time and previous_time is not None and time_s < previous_time:
                raise ValueError(
                    f"JSONL line {line_no}: time_s must be monotonic")
            previous_time = time_s
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break

    if not rows:
        raise ValueError("JSONL stream contains no sensor frames")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerows(rows)

    times = [float(row["time_s"]) for row in rows]
    return {
        "input_jsonl": input_jsonl,
        "output_csv": output_csv,
        "rows": len(rows),
        "duration_s": float(max(times) - min(times)) if len(times) >= 2 else 0.0,
        "terrain": rows[0]["terrain"],
        "mode": rows[0]["mode"],
        "video_file": rows[0]["video_file"],
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Capture Worm V6 raw hardware JSONL into raw CSV")
    parser.add_argument("--input-jsonl", default="-",
                        help="Input JSONL path or '-' for stdin")
    parser.add_argument("--output-csv", default=None,
                        help="Output raw hardware CSV path")
    parser.add_argument("--terrain", choices=VALID_TERRAINS, default=None)
    parser.add_argument("--mode", choices=VALID_MODES, default=None)
    parser.add_argument("--video-file", default=None)
    parser.add_argument("--gait-blend", type=float, default=None)
    parser.add_argument("--cmd-vel", type=float, default=None,
                        help="Legacy alias for --cmd-vx")
    parser.add_argument("--cmd-vx", type=float, default=None)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--allow-nonmonotonic-time", action="store_true")
    parser.add_argument("--write-jsonl-example", default=None,
                        help="Write a controller JSONL example")
    parser.add_argument("--write-json-schema", default=None,
                        help="Write the controller JSONL schema")
    parser.add_argument("--example-rows", type=int, default=5)
    parser.add_argument("--include-metadata-in-example", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    overrides = metadata_overrides(
        terrain=args.terrain,
        mode=args.mode,
        video_file=args.video_file,
        gait_blend=args.gait_blend,
        cmd_vx=(
            CMD_VX_RANGE[1]
            if args.cmd_vx is None and args.cmd_vel is None and args.terrain
            else (args.cmd_vx if args.cmd_vx is not None else args.cmd_vel)),
        cmd_vy=args.cmd_vy,
        cmd_yaw=0.0 if args.cmd_yaw is None and args.terrain else args.cmd_yaw,
    )
    results = {}
    if args.write_json_schema:
        results["json_schema"] = write_json_schema(args.write_json_schema)
    if args.write_jsonl_example:
        results["jsonl_example"] = write_jsonl_example(
            args.write_jsonl_example,
            rows=args.example_rows,
            terrain=args.terrain or "flat",
            mode=args.mode or "random",
            video_file=args.video_file,
            gait_blend=args.gait_blend if args.gait_blend is not None else 0.5,
            cmd_vel=(
                CMD_VX_RANGE[1] if args.cmd_vel is None else args.cmd_vel),
            cmd_vx=args.cmd_vx,
            cmd_vy=args.cmd_vy,
            cmd_yaw=0.0 if args.cmd_yaw is None else args.cmd_yaw,
            include_metadata=args.include_metadata_in_example,
        )
    if not args.output_csv:
        if not results:
            parser = build_parser()
            parser.error("Provide --output-csv or a --write-* option")
        print(json.dumps(results, indent=2))
        return
    result = capture_jsonl(
        args.input_jsonl,
        args.output_csv,
        overrides=overrides,
        strict_time=not args.allow_nonmonotonic_time,
        max_rows=args.max_rows,
    )
    results["capture"] = result
    print(json.dumps(results if len(results) > 1 else result, indent=2))


if __name__ == "__main__":
    main()

"""
Check a Worm V6 controller JSONL stream before importing it as hardware data.

This tool is for field use. It does not create paper evidence; it catches
sensor-schema, timing, IMU normalization, joint-range, and deploy-policy output
problems before a run is imported into the hardware validation archive.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from capture_hardware_stream_v6 import (  # noqa: E402
    metadata_overrides,
    normalize_raw_row,
    open_input_stream,
)
from hardware_policy_runtime_v6 import HardwarePolicyRuntime  # noqa: E402
from motor_contract_v6 import (  # noqa: E402
    SLIDE_TARGET_MAX_M,
    SLIDE_TARGET_MIN_M,
    YAW_TARGET_MAX_RAD,
    YAW_TARGET_MIN_RAD,
)
from validate_hardware_log_v6 import AXES, VALID_MODES, VALID_TERRAINS  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    IMU_GYRO_SCALE,
    NUM_ACTUATORS,
    NUM_IMUS,
    NUM_SLIDES,
    OBS_DIM,
    SLIDE_RANGE_VAL,
    SLIDE_VEL_SCALE,
    YAW_RANGE_VAL,
    YAW_VEL_SCALE,
)


DEFAULT_MIN_ROWS = 5
DEFAULT_MIN_DURATION_S = 0.1
DEFAULT_MIN_RATE_HZ = 1.0
DEFAULT_MAX_RATE_HZ = 500.0


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def resolve(path, project_root=PROJECT_ROOT):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def default_bundle_dir(terrain, mode, project_root=PROJECT_ROOT):
    return os.path.join(
        project_root, "record", "v6", "deploy_bundles",
        f"{terrain}_{mode}")


def finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def append_limited(items, message, limit):
    if len(items) < limit:
        items.append(message)


def sensor_health(row, line_no, errors, warnings, max_messages,
                  gravity_norm_min, gravity_norm_max, position_margin):
    slide_pad = SLIDE_RANGE_VAL * max(position_margin - 1.0, 0.0)
    yaw_pad = YAW_RANGE_VAL * max(position_margin - 1.0, 0.0)
    slide_min = SLIDE_TARGET_MIN_M - slide_pad
    slide_max = SLIDE_TARGET_MAX_M + slide_pad
    yaw_min = YAW_TARGET_MIN_RAD - yaw_pad
    yaw_max = YAW_TARGET_MAX_RAD + yaw_pad
    slide_vel_limit = SLIDE_VEL_SCALE * 10.0
    yaw_vel_limit = YAW_VEL_SCALE * 10.0
    gyro_limit = IMU_GYRO_SCALE * 10.0
    gravity_norms = []

    for i in range(NUM_SLIDES):
        pos = finite_float(row.get(f"slide_pos_m_{i:02d}"))
        vel = finite_float(row.get(f"slide_vel_m_s_{i:02d}"))
        if pos < slide_min or pos > slide_max:
            append_limited(
                errors,
                f"line {line_no}: slide_pos_m_{i:02d}={pos:.6g} "
                f"outside [{slide_min:.6g}, {slide_max:.6g}] m",
                max_messages,
            )
        if abs(vel) > slide_vel_limit:
            append_limited(
                warnings,
                f"line {line_no}: slide_vel_m_s_{i:02d}={vel:.6g} is large",
                max_messages,
            )

    for i in range(NUM_ACTUATORS - NUM_SLIDES):
        pos = finite_float(row.get(f"yaw_pos_rad_{i:02d}"))
        vel = finite_float(row.get(f"yaw_vel_rad_s_{i:02d}"))
        if pos < yaw_min or pos > yaw_max:
            append_limited(
                errors,
                f"line {line_no}: yaw_pos_rad_{i:02d}={pos:.6g} "
                f"outside [{yaw_min:.6g}, {yaw_max:.6g}] rad",
                max_messages,
            )
        if abs(vel) > yaw_vel_limit:
            append_limited(
                warnings,
                f"line {line_no}: yaw_vel_rad_s_{i:02d}={vel:.6g} is large",
                max_messages,
            )

    for seg in range(NUM_IMUS):
        gravity = np.array([
            finite_float(row.get(f"segment_gravity_{seg:02d}_{axis}"))
            for axis in AXES
        ], dtype=np.float32)
        norm = float(np.linalg.norm(gravity))
        gravity_norms.append(norm)
        if not gravity_norm_min <= norm <= gravity_norm_max:
            append_limited(
                errors,
                f"line {line_no}: segment {seg:02d} gravity norm "
                f"{norm:.4f} outside [{gravity_norm_min}, "
                f"{gravity_norm_max}]",
                max_messages,
            )
        for axis in AXES:
            gyro = finite_float(row.get(f"segment_gyro_rad_s_{seg:02d}_{axis}"))
            if abs(gyro) > gyro_limit:
                append_limited(
                    warnings,
                    f"line {line_no}: segment_gyro_rad_s_{seg:02d}_{axis}="
                    f"{gyro:.6g} is large",
                    max_messages,
                )
    return gravity_norms


def make_runtime(bundle_dir, terrain, mode, project_root):
    if bundle_dir is None and terrain and mode:
        bundle_dir = default_bundle_dir(terrain, mode, project_root)
    if bundle_dir is None:
        return None, None
    bundle_dir = resolve(bundle_dir, project_root)
    return HardwarePolicyRuntime(bundle_dir=bundle_dir), bundle_dir


def action_health(runtime, row, max_action_delta, line_no, errors,
                  max_messages, stats):
    prediction = runtime.predict(
        row,
        update_state=True,
        max_action_delta=max_action_delta,
    )
    obs = prediction["observation"]
    action = prediction["action"]
    slide = prediction["slide_targets_m"]
    yaw = prediction["yaw_targets_rad"]
    if obs.shape != (OBS_DIM,) or not np.all(np.isfinite(obs)):
        append_limited(
            errors,
            f"line {line_no}: invalid observation shape or finite check",
            max_messages,
        )
    if action.shape != (NUM_ACTUATORS,) or not np.all(np.isfinite(action)):
        append_limited(
            errors,
            f"line {line_no}: invalid action shape or finite check",
            max_messages,
        )
    if np.any(action < -1.0) or np.any(action > 1.0):
        append_limited(
            errors,
            f"line {line_no}: normalized action outside [-1, 1]",
            max_messages,
        )
    stats["max_abs_action"] = max(
        stats["max_abs_action"], float(np.max(np.abs(action))))
    stats["max_abs_slide_target_m"] = max(
        stats["max_abs_slide_target_m"], float(np.max(np.abs(slide))))
    stats["max_abs_yaw_target_rad"] = max(
        stats["max_abs_yaw_target_rad"], float(np.max(np.abs(yaw))))


def check_stream(input_jsonl, terrain=None, mode=None, video_file=None,
                 gait_blend=None, cmd_vel=None, cmd_yaw=None,
                 bundle_dir=None, project_root=PROJECT_ROOT,
                 min_rows=DEFAULT_MIN_ROWS,
                 min_duration_s=DEFAULT_MIN_DURATION_S,
                 min_rate_hz=DEFAULT_MIN_RATE_HZ,
                 max_rate_hz=DEFAULT_MAX_RATE_HZ,
                 max_action_delta=0.2,
                 gravity_norm_min=0.5,
                 gravity_norm_max=1.5,
                 position_margin=1.25,
                 strict_time=True,
                 max_messages=20):
    overrides = metadata_overrides(
        terrain=terrain,
        mode=mode,
        video_file=video_file,
        gait_blend=gait_blend,
        cmd_vel=cmd_vel,
        cmd_yaw=cmd_yaw,
    )
    runtime = None
    resolved_bundle = None
    if bundle_dir is not None:
        runtime, resolved_bundle = make_runtime(
            bundle_dir, terrain, mode, project_root)

    errors = []
    warnings = []
    rows = 0
    valid_rows = 0
    times = []
    gravity_norms = []
    action_stats = {
        "max_abs_action": 0.0,
        "max_abs_slide_target_m": 0.0,
        "max_abs_yaw_target_rad": 0.0,
    }

    with open_input_stream(input_jsonl) as src:
        previous_time = None
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                append_limited(
                    errors,
                    f"line {line_no}: invalid JSON: {exc}",
                    max_messages,
                )
                continue
            if not isinstance(raw, dict):
                append_limited(
                    errors,
                    f"line {line_no}: JSONL row is not an object",
                    max_messages,
                )
                continue
            try:
                row = normalize_raw_row(raw, line_no, overrides=overrides)
            except Exception as exc:
                append_limited(errors, str(exc), max_messages)
                continue

            time_s = finite_float(row.get("time_s"))
            if previous_time is not None:
                dt = time_s - previous_time
                if strict_time and dt < 0.0:
                    append_limited(
                        errors,
                        f"line {line_no}: time_s decreased",
                        max_messages,
                    )
                if dt == 0.0:
                    append_limited(
                        warnings,
                        f"line {line_no}: repeated time_s={time_s:.6g}",
                        max_messages,
                    )
            previous_time = time_s
            times.append(time_s)
            valid_rows += 1
            gravity_norms.extend(sensor_health(
                row, line_no, errors, warnings, max_messages,
                gravity_norm_min, gravity_norm_max, position_margin))
            if runtime is not None:
                try:
                    action_health(
                        runtime, row, max_action_delta, line_no,
                        errors, max_messages, action_stats)
                except Exception as exc:
                    append_limited(
                        errors,
                        f"line {line_no}: deploy policy failed: {exc}",
                        max_messages,
                    )

    duration = (max(times) - min(times)) if len(times) >= 2 else 0.0
    positive_dts = [
        b - a for a, b in zip(times, times[1:])
        if b - a > 0.0
    ]
    sample_rate = (
        1.0 / float(np.mean(positive_dts)) if positive_dts else None)
    if valid_rows < min_rows:
        append_limited(
            errors,
            f"valid rows {valid_rows} below required {min_rows}",
            max_messages,
        )
    if duration < min_duration_s:
        append_limited(
            errors,
            f"duration {duration:.3f}s below required {min_duration_s:.3f}s",
            max_messages,
        )
    if sample_rate is not None and sample_rate < min_rate_hz:
        append_limited(
            errors,
            f"sample rate {sample_rate:.3f}Hz below {min_rate_hz:.3f}Hz",
            max_messages,
        )
    if sample_rate is not None and sample_rate > max_rate_hz:
        append_limited(
            warnings,
            f"sample rate {sample_rate:.3f}Hz above {max_rate_hz:.3f}Hz",
            max_messages,
        )

    complete = not errors
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "input_jsonl": input_jsonl,
        "terrain": terrain,
        "mode": mode,
        "bundle_dir": rel(resolved_bundle, project_root)
        if resolved_bundle else None,
        "rows": rows,
        "valid_rows": valid_rows,
        "duration_s": float(duration),
        "sample_rate_hz": float(sample_rate) if sample_rate is not None else None,
        "min_rows": min_rows,
        "min_duration_s": min_duration_s,
        "gravity_norm": {
            "min": float(min(gravity_norms)) if gravity_norms else None,
            "max": float(max(gravity_norms)) if gravity_norms else None,
            "mean": float(np.mean(gravity_norms)) if gravity_norms else None,
            "accepted_range": [gravity_norm_min, gravity_norm_max],
        },
        "policy_runtime": {
            "enabled": runtime is not None,
            "obs_dim": OBS_DIM,
            "action_dim": NUM_ACTUATORS,
            **action_stats,
        },
        "errors": errors,
        "warnings": warnings,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Controller Stream Check",
        "",
        f"Complete: `{str(payload['complete']).lower()}`",
        f"Input: `{payload['input_jsonl']}`",
        f"Rows: `{payload['valid_rows']}/{payload['rows']}`",
        f"Duration: `{payload['duration_s']:.3f} s`",
        f"Sample rate: `{payload['sample_rate_hz']}` Hz",
        "",
        "This is a field stream health check. It is not a substitute for the "
        "final hardware CSV plus video evidence.",
        "",
        "## Policy Runtime",
        "",
        f"- Enabled: `{str(payload['policy_runtime']['enabled']).lower()}`",
        f"- Observation dimension: `{payload['policy_runtime']['obs_dim']}`",
        f"- Action dimension: `{payload['policy_runtime']['action_dim']}`",
        f"- Max abs action: `{payload['policy_runtime']['max_abs_action']}`",
        "",
        "## Gravity Norm",
        "",
        f"- Min: `{payload['gravity_norm']['min']}`",
        f"- Max: `{payload['gravity_norm']['max']}`",
        f"- Mean: `{payload['gravity_norm']['mean']}`",
        "",
    ]
    if payload["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend([f"- {item}" for item in payload["errors"]])
        lines.append("")
    if payload["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend([f"- {item}" for item in payload["warnings"]])
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check a Worm V6 controller JSONL stream before import")
    parser.add_argument("--input-jsonl", default="-",
                        help="Input controller JSONL path or '-'")
    parser.add_argument("--terrain", choices=VALID_TERRAINS, default=None)
    parser.add_argument("--mode", choices=VALID_MODES, default=None)
    parser.add_argument("--video-file", default=None)
    parser.add_argument("--gait-blend", type=float, default=None)
    parser.add_argument("--cmd-vel", type=float, default=None)
    parser.add_argument("--cmd-yaw", type=float, default=None)
    parser.add_argument("--bundle-dir", default=None,
                        help="Optional deploy bundle for live policy output "
                             "shape/range checks.")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-duration", type=float,
                        default=DEFAULT_MIN_DURATION_S)
    parser.add_argument("--min-rate-hz", type=float,
                        default=DEFAULT_MIN_RATE_HZ)
    parser.add_argument("--max-rate-hz", type=float,
                        default=DEFAULT_MAX_RATE_HZ)
    parser.add_argument("--max-action-delta", type=float, default=0.2)
    parser.add_argument("--gravity-norm-min", type=float, default=0.5)
    parser.add_argument("--gravity-norm-max", type=float, default=1.5)
    parser.add_argument("--position-margin", type=float, default=1.25)
    parser.add_argument("--allow-nonmonotonic-time", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero unless the stream check is clean")
    return parser


def main():
    args = build_parser().parse_args()
    payload = check_stream(
        input_jsonl=args.input_jsonl,
        terrain=args.terrain,
        mode=args.mode,
        video_file=args.video_file,
        gait_blend=args.gait_blend,
        cmd_vel=args.cmd_vel,
        cmd_yaw=args.cmd_yaw,
        bundle_dir=args.bundle_dir,
        min_rows=args.min_rows,
        min_duration_s=args.min_duration,
        min_rate_hz=args.min_rate_hz,
        max_rate_hz=args.max_rate_hz,
        max_action_delta=args.max_action_delta,
        gravity_norm_min=args.gravity_norm_min,
        gravity_norm_max=args.gravity_norm_max,
        position_margin=args.position_margin,
        strict_time=not args.allow_nonmonotonic_time,
    )
    if args.json_out:
        write_json(args.json_out, payload)
    if args.md_out:
        write_markdown(args.md_out, payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Worm V6 controller stream check")
        print(f"  complete: {payload['complete']}")
        print(f"  rows: {payload['valid_rows']}/{payload['rows']}")
        print(f"  duration_s: {payload['duration_s']:.3f}")
        print(f"  sample_rate_hz: {payload['sample_rate_hz']}")
        print(f"  policy_runtime: {payload['policy_runtime']['enabled']}")
        if payload["errors"]:
            print("  errors:")
            for error in payload["errors"]:
                print(f"    - {error}")
        if payload["warnings"]:
            print("  warnings:")
            for warning in payload["warnings"]:
                print(f"    - {warning}")
    if args.strict and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

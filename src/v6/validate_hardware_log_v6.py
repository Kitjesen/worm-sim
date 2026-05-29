"""
Validate or create a Worm V6 hardware log template.

The hardware log must contain the same 80 policy-observation fields used by
WormEnvV6, plus commanded actions and external evaluation estimates. This keeps
sim and real deployment inputs isomorphic.
"""

import argparse
import csv
import json
import math
import os
import sys
from urllib.parse import urlparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from motor_contract_v6 import action_mapping_config, motor_contract  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    GAIT_BLENDS,
    NUM_ACTUATORS,
    NUM_IMUS,
    NUM_SLIDES,
    OBS_DIM,
    OBS_LAYOUT,
    SLIDE_RANGE_VAL,
    YAW_RANGE_VAL,
)

AXES = ("x", "y", "z")
VALID_TERRAINS = ("flat", "sand", "slope")
VALID_MODES = tuple(GAIT_BLENDS.keys()) + ("random",)


def observation_columns():
    cols = ["cmd_vx_norm", "cmd_vy_norm", "cmd_yaw_norm"]
    cols += [f"joint_pos_{i:02d}" for i in range(NUM_ACTUATORS)]
    cols += [f"joint_vel_{i:02d}" for i in range(NUM_ACTUATORS)]
    cols += [f"previous_action_{i:02d}" for i in range(NUM_ACTUATORS)]
    cols += [
        f"segment_gravity_{seg:02d}_{axis}"
        for seg in range(NUM_IMUS)
        for axis in AXES
    ]
    cols += [
        f"segment_gyro_{seg:02d}_{axis}"
        for seg in range(NUM_IMUS)
        for axis in AXES
    ]
    cols += ["phase_sin", "phase_cos"]
    assert len(cols) == OBS_DIM
    return cols


def required_columns():
    cols = ["time_s", "terrain", "mode", "video_file"]
    cols += observation_columns()
    cols += [f"action_{i:02d}" for i in range(NUM_ACTUATORS)]
    cols += ["velocity_estimate_m_s", "yaw_rate_estimate_rad_s"]
    return cols


def neutral_example_row():
    row = {
        "time_s": 0.0,
        "terrain": "flat",
        "mode": "worm",
        "video_file": "example.mp4",
        "cmd_vx_norm": 0.5,
        "cmd_vy_norm": 0.0,
        "cmd_yaw_norm": 0.0,
        "phase_sin": 0.0,
        "phase_cos": 1.0,
        "velocity_estimate_m_s": 0.0,
        "yaw_rate_estimate_rad_s": 0.0,
    }
    for i in range(NUM_ACTUATORS):
        row[f"joint_pos_{i:02d}"] = 0.0
        row[f"joint_vel_{i:02d}"] = 0.0
        row[f"previous_action_{i:02d}"] = 0.0
        row[f"action_{i:02d}"] = 0.0
    for seg in range(NUM_IMUS):
        row[f"segment_gravity_{seg:02d}_x"] = 0.0
        row[f"segment_gravity_{seg:02d}_y"] = 0.0
        row[f"segment_gravity_{seg:02d}_z"] = -1.0
        row[f"segment_gyro_{seg:02d}_x"] = 0.0
        row[f"segment_gyro_{seg:02d}_y"] = 0.0
        row[f"segment_gyro_{seg:02d}_z"] = 0.0
    return row


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=required_columns())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_template(path):
    write_csv(path, rows=[])
    example_path = path.replace(".csv", "_example.csv")
    write_csv(example_path, rows=[neutral_example_row()])
    meta_path = path.replace(".csv", "_schema.json")
    from observation_contract_v6 import observation_contract

    contract = observation_contract()
    schema = {
        "obs_dim": OBS_DIM,
        "observation_contract_fingerprint": contract["abi_fingerprint"],
        "obs_layout": {
            key: [value.start, value.stop]
            for key, value in OBS_LAYOUT.items()
        },
        "num_actuators": NUM_ACTUATORS,
        "num_imus": NUM_IMUS,
        "required_columns": required_columns(),
        "example_log": example_path,
        "action_mapping": action_mapping_config(),
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "formal_validation": {
            "terrains": list(VALID_TERRAINS),
            "modes": list(VALID_MODES),
            "required_video_file": True,
            "min_rows": 5,
            "min_duration_s": 0.1,
            "time_s": "monotonic nondecreasing",
            "action_range": [-1.05, 1.05],
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote hardware CSV template: {path}")
    print(f"Wrote hardware CSV example: {example_path}")
    print(f"Wrote hardware schema JSON: {meta_path}")


def video_reference_resolves(reference, csv_path):
    reference = str(reference or "").strip()
    if not reference:
        return False

    parsed = urlparse(reference)
    if parsed.scheme in ("http", "https"):
        return True
    if parsed.scheme == "file":
        return os.path.exists(parsed.path)

    candidates = []
    if os.path.isabs(reference):
        candidates.append(reference)
    else:
        candidates.extend([
            os.path.join(os.path.dirname(os.path.abspath(csv_path)), reference),
            os.path.join(PROJECT_ROOT, reference),
            os.path.join(PROJECT_ROOT, "record", "v6", "videos", reference),
            os.path.join(
                PROJECT_ROOT, "record", "v6", "videos",
                os.path.basename(reference)),
        ])
    return any(os.path.exists(candidate) for candidate in candidates)


def _float(row, col):
    return float(row[col])


def validate_csv(path, expected_terrain=None, require_video=False,
                 min_rows=1, min_duration_s=0.0, strict_time=True,
                 verbose=True):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing = [c for c in required_columns() if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        obs_cols = observation_columns()
        speeds = []
        times = []
        terrains = set()
        modes = set()
        videos = set()
        unresolved_videos = set()
        rows = 0
        bad_rows = set()
        for rows, row in enumerate(reader, start=1):
            try:
                time_s = _float(row, "time_s")
                obs = np.array([_float(row, c) for c in obs_cols],
                               dtype=np.float32)
                actions = np.array(
                    [_float(row, f"action_{i:02d}")
                     for i in range(NUM_ACTUATORS)],
                    dtype=np.float32)
                speed = _float(row, "velocity_estimate_m_s")
                yaw_rate = _float(row, "yaw_rate_estimate_rad_s")
            except (TypeError, ValueError):
                bad_rows.add(rows)
                continue

            if not math.isfinite(time_s):
                bad_rows.add(rows)
            else:
                times.append(time_s)

            terrain = str(row.get("terrain", "")).strip()
            mode = str(row.get("mode", "")).strip()
            video = str(row.get("video_file", "")).strip()
            if terrain:
                terrains.add(terrain)
            if mode:
                modes.add(mode)
            if video:
                videos.add(video)
                if not video_reference_resolves(video, path):
                    unresolved_videos.add(video)

            if expected_terrain is not None and terrain != expected_terrain:
                bad_rows.add(rows)
            if terrain and terrain not in VALID_TERRAINS:
                bad_rows.add(rows)
            if mode and mode not in VALID_MODES:
                bad_rows.add(rows)

            if obs.shape != (OBS_DIM,) or not np.all(np.isfinite(obs)):
                bad_rows.add(rows)
            if (actions.shape != (NUM_ACTUATORS,)
                    or not np.all(np.isfinite(actions))
                    or np.any(actions < -1.05)
                    or np.any(actions > 1.05)):
                bad_rows.add(rows)
            command = obs[OBS_LAYOUT["command"]]
            if np.any(command < -1.05) or np.any(command > 1.05):
                bad_rows.add(rows)
            gravity = obs[OBS_LAYOUT["segment_gravity"]].reshape(NUM_IMUS, 3)
            gravity_norm = np.linalg.norm(gravity, axis=1)
            if (not np.all(np.isfinite(gravity_norm))
                    or np.any(gravity_norm < 0.5)
                    or np.any(gravity_norm > 1.5)):
                bad_rows.add(rows)
            phase = obs[OBS_LAYOUT["phase_clock"]]
            phase_norm = float(np.linalg.norm(phase))
            if not 0.5 <= phase_norm <= 1.5:
                bad_rows.add(rows)
            if not math.isfinite(yaw_rate):
                bad_rows.add(rows)
            if math.isfinite(speed):
                speeds.append(speed)
            else:
                bad_rows.add(rows)

        if rows == 0:
            raise ValueError("CSV contains no data rows")
        if rows < min_rows:
            raise ValueError(f"CSV has {rows} rows, expected at least {min_rows}")
        duration_s = (max(times) - min(times)) if len(times) >= 2 else 0.0
        if min_duration_s and duration_s < min_duration_s:
            raise ValueError(
                f"CSV duration {duration_s:.3f}s below {min_duration_s:.3f}s")
        if strict_time and any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("time_s must be monotonic nondecreasing")
        if require_video and not videos:
            raise ValueError("CSV has no nonempty video_file references")
        if require_video and unresolved_videos:
            raise ValueError(
                "Unresolved video_file references: "
                f"{sorted(unresolved_videos)[:10]}")
        if bad_rows:
            raise ValueError(
                "Invalid hardware log rows: "
                f"{sorted(bad_rows)[:10]}")

    metrics = {
        "path": path,
        "rows": rows,
        "duration_s": float(duration_s),
        "obs_dim": OBS_DIM,
        "terrains": sorted(terrains),
        "modes": sorted(modes),
        "video_files": sorted(videos),
        "mean_velocity_estimate_m_s": float(np.mean(speeds)) if speeds else None,
        "mean_velocity_estimate_mm_s": (
            float(np.mean(speeds) * 1000.0) if speeds else None),
    }
    if verbose:
        print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Validate Worm V6 hardware log is policy-input isomorphic")
    ap.add_argument("--input", default=None, help="Hardware CSV log to validate")
    ap.add_argument("--expected-terrain", choices=VALID_TERRAINS, default=None)
    ap.add_argument("--require-video", action="store_true",
                    help="Require at least one resolvable video reference")
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="Minimum log duration in seconds")
    ap.add_argument("--allow-nonmonotonic-time", action="store_true")
    ap.add_argument("--write-template", default=None,
                    help="Write an empty hardware CSV template")
    args = ap.parse_args()

    if args.write_template:
        write_template(args.write_template)
    if args.input:
        validate_csv(
            args.input,
            expected_terrain=args.expected_terrain,
            require_video=args.require_video,
            min_rows=args.min_rows,
            min_duration_s=args.min_duration,
            strict_time=not args.allow_nonmonotonic_time)
    if not args.write_template and not args.input:
        ap.error("Provide --input or --write-template")


if __name__ == "__main__":
    main()

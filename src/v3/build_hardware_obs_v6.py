"""
Build deployable Worm V6 policy-observation CSV from raw hardware sensors.

Raw logs should keep physical units from the controller:
  - slide positions in meters, slide velocities in m/s
  - yaw positions in radians, yaw velocities in rad/s
  - IMU projected gravity in each segment frame
  - IMU angular velocity in each segment frame, rad/s

The output CSV matches validate_hardware_log_v6.py: 80 policy-observation
columns, 11 normalized actions, and evaluation metadata.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from validate_hardware_log_v6 import (  # noqa: E402
    AXES,
    required_columns,
    validate_csv,
    write_csv,
)
from worm_env_v6 import (  # noqa: E402
    CMD_VEL_RANGE,
    CMD_YAW_RANGE,
    IMU_GYRO_SCALE,
    NUM_ACTUATORS,
    NUM_IMUS,
    NUM_SLIDES,
    PHASE_FREQ,
    SLIDE_RANGE_VAL,
    SLIDE_VEL_SCALE,
    YAW_RANGE_VAL,
    YAW_VEL_SCALE,
)


def raw_columns():
    cols = [
        "time_s",
        "terrain",
        "mode",
        "video_file",
        "cmd_vel_m_s",
        "cmd_yaw_rad_s",
        "gait_blend",
    ]
    cols += [f"slide_pos_m_{i:02d}" for i in range(NUM_SLIDES)]
    cols += [f"yaw_pos_rad_{i:02d}" for i in range(NUM_ACTUATORS - NUM_SLIDES)]
    cols += [f"slide_vel_m_s_{i:02d}" for i in range(NUM_SLIDES)]
    cols += [f"yaw_vel_rad_s_{i:02d}" for i in range(NUM_ACTUATORS - NUM_SLIDES)]
    cols += [
        f"segment_gravity_{seg:02d}_{axis}"
        for seg in range(NUM_IMUS)
        for axis in AXES
    ]
    cols += [
        f"segment_gyro_rad_s_{seg:02d}_{axis}"
        for seg in range(NUM_IMUS)
        for axis in AXES
    ]
    cols += [f"action_{i:02d}" for i in range(NUM_ACTUATORS)]
    cols += ["velocity_estimate_m_s", "yaw_rate_estimate_rad_s"]
    return cols


def neutral_raw_row():
    row = {
        "time_s": 0.0,
        "terrain": "flat",
        "mode": "worm",
        "video_file": "example.mp4",
        "cmd_vel_m_s": CMD_VEL_RANGE[1] * 0.5,
        "cmd_yaw_rad_s": 0.0,
        "gait_blend": 0.0,
        "velocity_estimate_m_s": 0.0,
        "yaw_rate_estimate_rad_s": 0.0,
    }
    for i in range(NUM_SLIDES):
        row[f"slide_pos_m_{i:02d}"] = 0.0
        row[f"slide_vel_m_s_{i:02d}"] = 0.0
    for i in range(NUM_ACTUATORS - NUM_SLIDES):
        row[f"yaw_pos_rad_{i:02d}"] = 0.0
        row[f"yaw_vel_rad_s_{i:02d}"] = 0.0
    for seg in range(NUM_IMUS):
        row[f"segment_gravity_{seg:02d}_x"] = 0.0
        row[f"segment_gravity_{seg:02d}_y"] = 0.0
        row[f"segment_gravity_{seg:02d}_z"] = -1.0
        row[f"segment_gyro_rad_s_{seg:02d}_x"] = 0.0
        row[f"segment_gyro_rad_s_{seg:02d}_y"] = 0.0
        row[f"segment_gyro_rad_s_{seg:02d}_z"] = 0.0
    for i in range(NUM_ACTUATORS):
        row[f"action_{i:02d}"] = 0.0
    return row


def write_raw_template(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
    example_path = path.replace(".csv", "_example.csv")
    with open(example_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns())
        writer.writeheader()
        writer.writerow(neutral_raw_row())
    print(f"Wrote raw hardware template: {path}")
    print(f"Wrote raw hardware example: {example_path}")


def read_float(row, key, default=None):
    value = row.get(key, default)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"Missing required numeric column: {key}")
        value = default
    return float(value)


def action_vector(row):
    return np.array([
        np.clip(read_float(row, f"action_{i:02d}", 0.0), -1.0, 1.0)
        for i in range(NUM_ACTUATORS)
    ], dtype=np.float32)


def build_output_row(raw, previous_action):
    time_s = read_float(raw, "time_s")
    out = {
        "time_s": time_s,
        "terrain": raw.get("terrain", ""),
        "mode": raw.get("mode", ""),
        "video_file": raw.get("video_file", ""),
        "cmd_vel_norm": np.clip(
            read_float(raw, "cmd_vel_m_s") / max(CMD_VEL_RANGE[1], 1e-6),
            0.0, 1.0),
        "cmd_yaw_norm": np.clip(
            read_float(raw, "cmd_yaw_rad_s") /
            max(abs(CMD_YAW_RANGE[1]), 1e-6), -1.0, 1.0),
        "gait_blend": np.clip(read_float(raw, "gait_blend"), 0.0, 1.0),
        "velocity_estimate_m_s": read_float(raw, "velocity_estimate_m_s", 0.0),
        "yaw_rate_estimate_rad_s": read_float(
            raw, "yaw_rate_estimate_rad_s", 0.0),
    }

    for i in range(NUM_SLIDES):
        out[f"joint_pos_{i:02d}"] = np.clip(
            read_float(raw, f"slide_pos_m_{i:02d}") /
            max(SLIDE_RANGE_VAL, 1e-6), -1.0, 1.0)
        out[f"joint_vel_{i:02d}"] = np.clip(
            read_float(raw, f"slide_vel_m_s_{i:02d}", 0.0) /
            max(SLIDE_VEL_SCALE, 1e-6), -10.0, 10.0)

    for i in range(NUM_ACTUATORS - NUM_SLIDES):
        dst = NUM_SLIDES + i
        out[f"joint_pos_{dst:02d}"] = np.clip(
            read_float(raw, f"yaw_pos_rad_{i:02d}") /
            max(YAW_RANGE_VAL, 1e-6), -1.0, 1.0)
        out[f"joint_vel_{dst:02d}"] = np.clip(
            read_float(raw, f"yaw_vel_rad_s_{i:02d}", 0.0) /
            max(YAW_VEL_SCALE, 1e-6), -10.0, 10.0)

    for i, value in enumerate(previous_action):
        out[f"previous_action_{i:02d}"] = float(value)

    for seg in range(NUM_IMUS):
        gravity = np.array([
            read_float(raw, f"segment_gravity_{seg:02d}_{axis}")
            for axis in AXES
        ], dtype=np.float32)
        norm = float(np.linalg.norm(gravity))
        if norm > 1e-6:
            gravity = gravity / norm
        for axis, value in zip(AXES, gravity):
            out[f"segment_gravity_{seg:02d}_{axis}"] = float(value)
        for axis in AXES:
            out[f"segment_gyro_{seg:02d}_{axis}"] = np.clip(
                read_float(raw, f"segment_gyro_rad_s_{seg:02d}_{axis}", 0.0) /
                max(IMU_GYRO_SCALE, 1e-6), -10.0, 10.0)

    phase = 2.0 * math.pi * PHASE_FREQ * time_s
    out["phase_sin"] = math.sin(phase)
    out["phase_cos"] = math.cos(phase)

    current_action = action_vector(raw)
    for i, value in enumerate(current_action):
        out[f"action_{i:02d}"] = float(value)
    return out, current_action


def convert_raw_csv(input_path, output_path, validate=True):
    out_rows = []
    previous_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Raw CSV has no header")
        missing = [col for col in raw_columns() if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Raw CSV missing columns: {missing}")
        for raw in reader:
            out, current_action = build_output_row(raw, previous_action)
            out_rows.append(out)
            previous_action = current_action

    if not out_rows:
        raise ValueError("Raw CSV contains no data rows")
    write_csv(output_path, out_rows)
    if validate:
        validate_csv(output_path)
    return {"input": input_path, "output": output_path, "rows": len(out_rows)}


def main():
    ap = argparse.ArgumentParser(
        description="Convert raw Worm V6 hardware sensors into policy CSV")
    ap.add_argument("--input-raw", default=None, help="Raw hardware CSV")
    ap.add_argument("--output", default=None,
                    help="Output deployable policy-input CSV")
    ap.add_argument("--write-raw-template", default=None,
                    help="Write raw sensor CSV template and example")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    if args.write_raw_template:
        write_raw_template(args.write_raw_template)
    if args.input_raw:
        if not args.output:
            root, ext = os.path.splitext(args.input_raw)
            args.output = f"{root}_policy_input{ext or '.csv'}"
        result = convert_raw_csv(
            args.input_raw, args.output, validate=not args.no_validate)
        print(result)
    if not args.write_raw_template and not args.input_raw:
        ap.error("Provide --input-raw or --write-raw-template")


if __name__ == "__main__":
    main()

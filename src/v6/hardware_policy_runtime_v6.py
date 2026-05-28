"""
Online hardware runtime wrapper for a deployable Worm V6 policy.

The controller-facing API consumes one raw sensor frame in physical units
matching build_hardware_obs_v6.py and returns normalized policy actions plus
physical slide/yaw targets. It keeps previous_action internally, so a real
control loop does not need to construct the 80-D observation by hand.
"""

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import build_output_row, raw_columns  # noqa: E402
from deploy_policy_v6 import (  # noqa: E402
    action_mapping_fields,
    action_targets,
    read_config,
)
from validate_hardware_log_v6 import (  # noqa: E402
    VALID_MODES,
    VALID_TERRAINS,
    required_columns,
    validate_csv,
)
from worm_env_v6 import NUM_ACTUATORS  # noqa: E402


def default_bundle_dir():
    return os.path.join(
        PROJECT_ROOT, "record", "v6", "deploy_bundles", "flat_random")


def default_output_path(input_raw_csv):
    root, ext = os.path.splitext(input_raw_csv)
    return f"{root}_runtime_actions{ext or '.csv'}"


class HardwarePolicyRuntime:
    def __init__(self, bundle_dir=None, config_path=None, actor_path=None,
                 device="cpu"):
        bundle_dir = bundle_dir or default_bundle_dir()
        config_path = config_path or os.path.join(bundle_dir, "deploy_config.json")
        actor_path = actor_path or os.path.join(bundle_dir, "policy_actor.pt")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"TorchScript actor not found: {actor_path}")

        self.config = read_config(config_path)
        self.actor = torch.jit.load(actor_path, map_location=device)
        self.actor.eval()
        self.device = device
        self.obs_cols = self.config["observation_columns"]
        self.action_cols = self.config["action_columns"]
        self.obs_dim = int(self.config["obs_dim"])
        self.action_dim = int(self.config["action_dim"])
        self.mapping = action_mapping_fields(self.config)
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self, previous_action=None):
        if previous_action is None:
            self.previous_action[:] = 0.0
            return
        previous_action = np.asarray(previous_action, dtype=np.float32)
        if previous_action.shape != (self.action_dim,):
            raise ValueError(
                f"previous_action shape {previous_action.shape}, "
                f"expected {(self.action_dim,)}")
        self.previous_action[:] = np.clip(previous_action, -1.0, 1.0)

    def _raw_with_action_defaults(self, raw_sensor):
        raw = dict(raw_sensor)
        for i, value in enumerate(self.previous_action):
            raw.setdefault(f"action_{i:02d}", float(value))
        raw.setdefault("velocity_estimate_m_s", 0.0)
        raw.setdefault("yaw_rate_estimate_rad_s", 0.0)
        raw.setdefault("terrain", self.config.get("terrain", ""))
        raw.setdefault("mode", self.config.get("gait_mode", ""))
        raw.setdefault("video_file", "")
        return raw

    def build_observation(self, raw_sensor):
        raw = self._raw_with_action_defaults(raw_sensor)
        policy_row, _ = build_output_row(raw, self.previous_action)
        obs = np.array(
            [float(policy_row[col]) for col in self.obs_cols],
            dtype=np.float32)
        if obs.shape != (self.obs_dim,) or not np.all(np.isfinite(obs)):
            raise ValueError("Raw sensor frame produced an invalid observation")
        return obs, policy_row

    def _limited_action(self, action, max_action_delta=None):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if max_action_delta is None:
            return action
        max_action_delta = float(max_action_delta)
        if max_action_delta < 0.0:
            raise ValueError("max_action_delta must be >= 0")
        delta = np.clip(
            action - self.previous_action,
            -max_action_delta,
            max_action_delta,
        )
        return np.clip(self.previous_action + delta, -1.0, 1.0).astype(np.float32)

    def predict(self, raw_sensor, update_state=True, max_action_delta=None):
        obs, policy_row = self.build_observation(raw_sensor)
        with torch.no_grad():
            raw_action = self.actor(
                torch.from_numpy(obs).to(self.device).unsqueeze(0)
            ).cpu().numpy()[0].astype(np.float32)
        if (raw_action.shape != (self.action_dim,)
                or not np.all(np.isfinite(raw_action))):
            raise RuntimeError(
                f"Actor returned invalid action shape {raw_action.shape}")
        action = self._limited_action(raw_action, max_action_delta)
        targets = action_targets(self.config, action)
        if update_state:
            self.previous_action[:] = action
        return {
            "observation": obs,
            "policy_row": policy_row,
            "raw_action": np.clip(raw_action, -1.0, 1.0).astype(np.float32),
            "action": action,
            "slide_targets_m": targets["slide_targets_m"],
            "yaw_targets_rad": targets["yaw_targets_rad"],
        }

    def output_fieldnames(self, raw_row=None):
        metadata_cols = [
            col for col in ("time_s", "terrain", "mode", "video_file")
            if raw_row is None or col in raw_row
        ]
        return [
            "row_index",
            *metadata_cols,
            *self.action_cols,
            *self.mapping["slide_target_cols"],
            *self.mapping["yaw_target_cols"],
        ]

    def format_output_row(self, row_index, raw_row, prediction):
        out = {"row_index": row_index}
        for col in ("time_s", "terrain", "mode", "video_file"):
            if col in raw_row:
                out[col] = raw_row[col]
        for col, value in zip(self.action_cols, prediction["action"]):
            out[col] = f"{float(value):.8f}"
        for col, value in zip(
                self.mapping["slide_target_cols"],
                prediction["slide_targets_m"]):
            out[col] = f"{float(value):.8f}"
        for col, value in zip(
                self.mapping["yaw_target_cols"],
                prediction["yaw_targets_rad"]):
            out[col] = f"{float(value):.8f}"
        return out

    def format_policy_log_row(self, raw_row, prediction):
        """Return the formal audit CSV row for one online policy step."""
        policy_row = dict(prediction["policy_row"])
        out = {}
        for col in ("time_s", "terrain", "mode", "video_file"):
            out[col] = policy_row.get(col, raw_row.get(col, ""))
        for col in self.obs_cols:
            out[col] = policy_row[col]
        for col, value in zip(self.action_cols, prediction["action"]):
            out[col] = f"{float(value):.8f}"
        out["velocity_estimate_m_s"] = policy_row.get(
            "velocity_estimate_m_s", raw_row.get("velocity_estimate_m_s", 0.0))
        out["yaw_rate_estimate_rad_s"] = policy_row.get(
            "yaw_rate_estimate_rad_s",
            raw_row.get("yaw_rate_estimate_rad_s", 0.0))
        return out


def required_sensor_columns():
    action_cols = {f"action_{i:02d}" for i in range(NUM_ACTUATORS)}
    optional = action_cols | {
        "velocity_estimate_m_s", "yaw_rate_estimate_rad_s",
        "terrain", "mode", "video_file",
    }
    return [col for col in raw_columns() if col not in optional]


def apply_metadata_overrides(row, overrides=None):
    if not overrides:
        return row
    out = dict(row)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def run_raw_csv(runtime, input_raw_csv, output_csv, policy_log_csv=None,
                metadata_overrides=None, validate_policy_log=False,
                expected_terrain=None, require_video=False, min_rows=1,
                min_duration_s=0.0, max_action_delta=None):
    rows = []
    policy_rows = []
    with open(input_raw_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Raw CSV has no header")
        missing = [
            col for col in required_sensor_columns()
            if col not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"Raw CSV missing sensor columns: {missing}")
        for row_index, row in enumerate(reader):
            row = apply_metadata_overrides(row, metadata_overrides)
            prediction = runtime.predict(
                row, update_state=True, max_action_delta=max_action_delta)
            rows.append(runtime.format_output_row(row_index, row, prediction))
            if policy_log_csv:
                policy_rows.append(runtime.format_policy_log_row(
                    row, prediction))

    if not rows:
        raise ValueError("Raw CSV contains no data rows")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=runtime.output_fieldnames(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "input_raw_csv": input_raw_csv,
        "output_csv": output_csv,
        "rows": len(rows),
    }
    if policy_log_csv:
        os.makedirs(os.path.dirname(os.path.abspath(policy_log_csv)),
                    exist_ok=True)
        with open(policy_log_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=required_columns())
            writer.writeheader()
            writer.writerows(policy_rows)
        result["policy_log_csv"] = policy_log_csv
        if validate_policy_log:
            result["policy_log_metrics"] = validate_csv(
                policy_log_csv,
                expected_terrain=expected_terrain,
                require_video=require_video,
                min_rows=min_rows,
                min_duration_s=min_duration_s,
                verbose=False,
            )
    return result


def prediction_json_row(row_index, raw_row, prediction):
    return {
        "row_index": row_index,
        "time_s": raw_row.get("time_s", prediction["policy_row"].get("time_s")),
        "terrain": raw_row.get("terrain", prediction["policy_row"].get("terrain", "")),
        "mode": raw_row.get("mode", prediction["policy_row"].get("mode", "")),
        "action": [float(v) for v in prediction["action"]],
        "raw_action": [float(v) for v in prediction["raw_action"]],
        "slide_targets_m": [float(v) for v in prediction["slide_targets_m"]],
        "yaw_targets_rad": [float(v) for v in prediction["yaw_targets_rad"]],
    }


def open_input_stream(path):
    if path == "-":
        return nullcontext(sys.stdin)
    return open(path, "r", encoding="utf-8")


def open_output_stream(path):
    if path == "-":
        return nullcontext(sys.stdout)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "w", encoding="utf-8", newline="\n")


def run_jsonl_stream(runtime, input_jsonl, output_jsonl="-",
                     policy_log_csv=None, metadata_overrides=None,
                     validate_policy_log=False, expected_terrain=None,
                     require_video=False, min_rows=1, min_duration_s=0.0,
                     max_action_delta=None):
    rows = 0
    policy_rows = []
    with open_input_stream(input_jsonl) as src, open_output_stream(output_jsonl) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            raw = apply_metadata_overrides(raw, metadata_overrides)
            prediction = runtime.predict(
                raw, update_state=True, max_action_delta=max_action_delta)
            dst.write(json.dumps(
                prediction_json_row(rows, raw, prediction),
                separators=(",", ":")))
            dst.write("\n")
            dst.flush()
            if policy_log_csv:
                policy_rows.append(runtime.format_policy_log_row(
                    raw, prediction))
            rows += 1

    if rows == 0:
        raise ValueError("JSONL stream contains no sensor frames")

    result = {
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "rows": rows,
    }
    if policy_log_csv:
        os.makedirs(os.path.dirname(os.path.abspath(policy_log_csv)),
                    exist_ok=True)
        with open(policy_log_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=required_columns())
            writer.writeheader()
            writer.writerows(policy_rows)
        result["policy_log_csv"] = policy_log_csv
        if validate_policy_log:
            result["policy_log_metrics"] = validate_csv(
                policy_log_csv,
                expected_terrain=expected_terrain,
                require_video=require_video,
                min_rows=min_rows,
                min_duration_s=min_duration_s,
                verbose=False,
            )
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Run a deployable Worm V6 policy on raw hardware sensors")
    ap.add_argument("--bundle-dir", default=default_bundle_dir())
    ap.add_argument("--config", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--input-raw-csv", default=None,
                    help="Raw hardware sensor CSV in physical units")
    ap.add_argument("--input-jsonl", default=None,
                    help="Raw sensor JSONL stream or '-' for stdin")
    ap.add_argument("--output-jsonl", default="-",
                    help="Action JSONL stream path or '-' for stdout")
    ap.add_argument("--output-csv", default=None,
                    help="Output actions and physical targets CSV")
    ap.add_argument("--policy-log-csv", default=None,
                    help="Write formal hardware audit CSV with observations "
                         "and predicted actions")
    ap.add_argument("--terrain", choices=VALID_TERRAINS, default=None,
                    help="Override terrain metadata for every raw row")
    ap.add_argument("--mode", choices=VALID_MODES, default=None,
                    help="Override mode metadata for every raw row")
    ap.add_argument("--video-file", default=None,
                    help="Override video_file metadata for every raw row")
    ap.add_argument("--gait-blend", type=float, default=None,
                    help="Override gait_blend command for every raw row")
    ap.add_argument("--validate-policy-log", action="store_true",
                    help="Validate --policy-log-csv after writing it")
    ap.add_argument("--expected-terrain", choices=VALID_TERRAINS, default=None)
    ap.add_argument("--require-video", action="store_true")
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--max-action-delta", type=float, default=None,
                    help="Optional normalized action slew-rate limit per frame")
    args = ap.parse_args()

    runtime = HardwarePolicyRuntime(
        bundle_dir=args.bundle_dir,
        config_path=args.config,
        actor_path=args.actor)
    overrides = {
        "terrain": args.terrain,
        "mode": args.mode,
        "video_file": args.video_file,
        "gait_blend": args.gait_blend,
    }
    if bool(args.input_raw_csv) == bool(args.input_jsonl):
        ap.error("Provide exactly one of --input-raw-csv or --input-jsonl")

    common = {
        "policy_log_csv": args.policy_log_csv,
        "metadata_overrides": overrides,
        "validate_policy_log": args.validate_policy_log,
        "expected_terrain": args.expected_terrain or args.terrain,
        "require_video": args.require_video,
        "min_rows": args.min_rows,
        "min_duration_s": args.min_duration,
        "max_action_delta": args.max_action_delta,
    }
    if args.input_jsonl:
        result = run_jsonl_stream(
            runtime,
            args.input_jsonl,
            output_jsonl=args.output_jsonl,
            **common,
        )
        summary_stream = sys.stderr if args.output_jsonl == "-" else sys.stdout
        print(json.dumps(result, indent=2), file=summary_stream)
    else:
        output_csv = args.output_csv or default_output_path(args.input_raw_csv)
        result = run_raw_csv(
            runtime,
            args.input_raw_csv,
            output_csv,
            **common,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
Formal observation ABI for deployable Worm V6 policies.

The contract documents the exact 80-D policy input vector, its hardware sensor
sources, and the privileged simulation quantities that must never enter the
policy observation. It is used by deploy configs, paper artifacts, and tests.
"""

import argparse
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from build_hardware_obs_v6 import raw_columns  # noqa: E402
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import (  # noqa: E402
    CMD_VEL_RANGE,
    CMD_YAW_RANGE,
    CTRL_DT,
    IMU_GYRO_SCALE,
    NUM_ACTUATORS,
    NUM_IMUS,
    NUM_SLIDES,
    OBS_DIM,
    OBS_LAYOUT,
    PERISTALTIC_ACTUATION_PERIOD_S,
    PHASE_FREQ,
    SLIDE_RANGE_VAL,
    SLIDE_VEL_SCALE,
    YAW_RANGE_VAL,
    YAW_VEL_SCALE,
)

DEFAULT_CONTRACT_JSON = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "observation_contract.json")
DEFAULT_CONTRACT_MD = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "observation_contract.md")


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def layout_json():
    return {
        key: [value.start, value.stop]
        for key, value in OBS_LAYOUT.items()
    }


def segment_columns(columns, name):
    start, stop = layout_json()[name]
    return columns[start:stop]


def stable_hash(payload):
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_contract():
    columns = observation_columns()
    layout = layout_json()
    groups = [
        {
            "name": "command",
            "range": layout["command"],
            "columns": segment_columns(columns, "command"),
            "hardware_source": (
                "high-level command interface: cmd_vel_m_s, cmd_yaw_rad_s, "
                "and gait_blend"),
            "normalization": {
                "cmd_vel_norm": f"cmd_vel_m_s / {CMD_VEL_RANGE[1]}",
                "cmd_yaw_norm": f"cmd_yaw_rad_s / {CMD_YAW_RANGE[1]}",
                "gait_blend": "clipped to [0, 1]",
            },
        },
        {
            "name": "joint_pos",
            "range": layout["joint_pos"],
            "columns": segment_columns(columns, "joint_pos"),
            "hardware_source": (
                "11 joint encoders: 6 slide positions and 5 yaw positions"),
            "normalization": {
                "slide": f"slide_pos_m / {SLIDE_RANGE_VAL}",
                "yaw": f"yaw_pos_rad / {YAW_RANGE_VAL}",
            },
        },
        {
            "name": "joint_vel",
            "range": layout["joint_vel"],
            "columns": segment_columns(columns, "joint_vel"),
            "hardware_source": (
                "11 joint encoder velocity estimates from the actuator "
                "controller"),
            "normalization": {
                "slide": f"slide_vel_m_s / {SLIDE_VEL_SCALE}",
                "yaw": f"yaw_vel_rad_s / {YAW_VEL_SCALE}",
            },
        },
        {
            "name": "previous_action",
            "range": layout["previous_action"],
            "columns": segment_columns(columns, "previous_action"),
            "hardware_source": (
                "controller memory of the previous normalized policy action"),
            "normalization": "already normalized in [-1, 1]",
        },
        {
            "name": "segment_gravity",
            "range": layout["segment_gravity"],
            "columns": segment_columns(columns, "segment_gravity"),
            "hardware_source": (
                "one IMU per body segment: local gravity direction from "
                "attitude/accelerometer fusion"),
            "normalization": "unit vector per segment",
        },
        {
            "name": "segment_gyro",
            "range": layout["segment_gyro"],
            "columns": segment_columns(columns, "segment_gyro"),
            "hardware_source": (
                "one IMU per body segment: local angular velocity"),
            "normalization": f"gyro_rad_s / {IMU_GYRO_SCALE}",
        },
        {
            "name": "phase_clock",
            "range": layout["phase_clock"],
            "columns": segment_columns(columns, "phase_clock"),
            "hardware_source": (
                "controller clock, not a simulator state estimate"),
            "normalization": f"sin/cos(2*pi*{PHASE_FREQ}*time_s)",
        },
    ]
    return {
        "format_version": 1,
        "name": "worm_v6_deployable_observation_abi",
        "obs_dim": OBS_DIM,
        "obs_layout": layout,
        "observation_columns": columns,
        "raw_hardware_columns": raw_columns(),
        "sensor_counts": {
            "actuated_joints": NUM_ACTUATORS,
            "slide_joints": NUM_SLIDES,
            "yaw_joints": NUM_ACTUATORS - NUM_SLIDES,
            "segment_imus": NUM_IMUS,
        },
        "control_timing": {
            "control_dt_s": CTRL_DT,
            "control_rate_hz": 1.0 / CTRL_DT,
            "peristaltic_actuation_period_s": (
                PERISTALTIC_ACTUATION_PERIOD_S),
            "phase_freq_hz": PHASE_FREQ,
        },
        "groups": groups,
        "allowed_policy_inputs": [
            "joint encoder positions",
            "joint encoder velocities",
            "per-segment IMU gravity direction",
            "per-segment IMU angular velocity",
            "commanded speed, commanded yaw rate, gait_blend",
            "previous normalized action",
            "controller phase clock",
        ],
        "forbidden_policy_inputs": [
            "MuJoCo freejoint global position",
            "MuJoCo freejoint global orientation quaternion",
            "MuJoCo root/base linear velocity",
            "MuJoCo root/base angular velocity as a privileged state",
            "global pose from external tracking",
            "motion-capture state",
            "ground-truth terrain contact or slip labels",
            "reward-only forward/lateral/yaw velocity measurements",
        ],
        "reward_only_quantities": [
            "forward speed",
            "yaw rate",
            "lateral drift",
            "energy/action cost",
            "termination and stability diagnostics",
        ],
        "deployability_claim": (
            "The policy input is reconstructable from onboard joint encoders, "
            "distributed IMUs, command metadata, controller memory, and a local "
            "clock. Training rewards may use simulator truth, but the policy "
            "observation ABI may not."),
    }


def observation_contract():
    payload = base_contract()
    payload["abi_fingerprint"] = stable_hash(payload)
    return payload


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Deployable Observation Contract",
        "",
        f"- ABI fingerprint: `{payload['abi_fingerprint']}`",
        f"- Observation dimension: `{payload['obs_dim']}`",
        f"- Actuated joints: `{payload['sensor_counts']['actuated_joints']}`",
        f"- Segment IMUs: `{payload['sensor_counts']['segment_imus']}`",
        f"- Control rate: `{payload['control_timing']['control_rate_hz']:.1f} Hz`",
        f"- Peristaltic actuation period: `{payload['control_timing']['peristaltic_actuation_period_s']:.3f} s`",
        f"- Phase clock frequency: `{payload['control_timing']['phase_freq_hz']:.3f} Hz`",
        "",
        "## Policy Input Groups",
        "",
        "| Group | Range | Count | Hardware source |",
        "| --- | --- | ---: | --- |",
    ]
    for group in payload["groups"]:
        start, stop = group["range"]
        lines.append(
            f"| {group['name']} | [{start}, {stop}) | "
            f"{stop - start} | {group['hardware_source']} |")
    lines.extend([
        "",
        "## Forbidden Policy Inputs",
        "",
    ])
    lines.extend([
        f"- {item}" for item in payload["forbidden_policy_inputs"]
    ])
    lines.extend([
        "",
        "## Reward-Only Quantities",
        "",
    ])
    lines.extend([
        f"- {item}" for item in payload["reward_only_quantities"]
    ])
    lines.extend([
        "",
        "## Deployability Claim",
        "",
        payload["deployability_claim"],
        "",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_contract(json_path=DEFAULT_CONTRACT_JSON,
                   md_path=DEFAULT_CONTRACT_MD):
    payload = observation_contract()
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    return payload


def attach_contract_to_config(config):
    payload = observation_contract()
    config = dict(config)
    config["observation_contract_fingerprint"] = payload["abi_fingerprint"]
    config["observation_contract"] = payload
    return config


def refresh_bundle_contract(bundle_dir):
    config_path = os.path.join(bundle_dir, "deploy_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"deploy_config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = attach_contract_to_config(config)
    write_json(config_path, config)
    return config_path


def default_bundle_dirs(project_root=PROJECT_ROOT):
    root = os.path.join(project_root, "record", "v6", "deploy_bundles")
    return [
        os.path.join(root, "flat_random"),
        os.path.join(root, "sand_random"),
        os.path.join(root, "slope_random"),
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Write or validate the Worm V6 observation contract")
    parser.add_argument("--json-out", default=DEFAULT_CONTRACT_JSON)
    parser.add_argument("--md-out", default=DEFAULT_CONTRACT_MD)
    parser.add_argument("--refresh-bundles", action="store_true",
                        help="Attach the current contract to deploy configs")
    return parser


def main():
    args = build_parser().parse_args()
    payload = write_contract(args.json_out, args.md_out)
    refreshed = []
    if args.refresh_bundles:
        for bundle_dir in default_bundle_dirs():
            refreshed.append(rel(refresh_bundle_contract(bundle_dir)))
    print(json.dumps({
        "contract_json": rel(args.json_out),
        "contract_md": rel(args.md_out),
        "abi_fingerprint": payload["abi_fingerprint"],
        "refreshed_bundles": refreshed,
    }, indent=2))


if __name__ == "__main__":
    main()

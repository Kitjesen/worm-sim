"""
Audit Worm V6 policy observation sources for deployability.

The policy observation may use simulated encoders and IMUs during training, but
the quantities must be reconstructable on the real robot. This audit documents
the exact source groups and checks that privileged MuJoCo root/freejoint truth
does not enter WormEnvV6._get_obs().
"""

import argparse
import inspect
import json
import os
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from observation_contract_v6 import observation_contract  # noqa: E402
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import OBS_DIM, OBS_LAYOUT, WormEnvV6  # noqa: E402

DEFAULT_JSON = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "observation_source_audit.json")
DEFAULT_MD = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "observation_source_audit.md")


FORBIDDEN_OBS_PATTERNS = {
    "root_linear_velocity_x": r"self\.data\.qvel\s*\[\s*0\s*\]",
    "root_linear_velocity_y": r"self\.data\.qvel\s*\[\s*1\s*\]",
    "root_linear_velocity_z": r"self\.data\.qvel\s*\[\s*2\s*\]",
    "root_angular_velocity_roll": r"self\.data\.qvel\s*\[\s*3\s*\]",
    "root_angular_velocity_pitch": r"self\.data\.qvel\s*\[\s*4\s*\]",
    "root_angular_velocity_yaw": r"self\.data\.qvel\s*\[\s*5\s*\]",
    "global_position": r"self\.data\.xpos",
    "mocap_state": r"mocap",
    "base_linvel_alias": r"base_linvel",
    "freejoint_name": r"freejoint|free_joint",
}

REWARD_ONLY_PATTERNS = {
    "forward_speed": r"self\.data\.qvel\s*\[\s*0\s*\]",
    "lateral_speed": r"self\.data\.qvel\s*\[\s*1\s*\]",
    "yaw_rate": r"self\.data\.qvel\s*\[\s*5\s*\]",
    "termination_height": r"self\.data\.xpos",
    "termination_nan_qpos": r"self\.data\.qpos",
}


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


def regex_hits(source, patterns):
    hits = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, source, flags=re.IGNORECASE)
        if matches:
            hits[name] = len(matches)
    return hits


def group_columns(columns, group):
    start, stop = layout_json()[group]
    return columns[start:stop]


def expected_groups(columns):
    return [
        {
            "group": "command",
            "range": layout_json()["command"],
            "columns": group_columns(columns, "command"),
            "sim_source": ["self._cmd_vel", "self._cmd_yaw", "self._gait_blend"],
            "real_source": [
                "controller command velocity",
                "controller command yaw rate",
                "controller-selected gait_blend",
            ],
            "policy_allowed": True,
        },
        {
            "group": "joint_pos",
            "range": layout_json()["joint_pos"],
            "columns": group_columns(columns, "joint_pos"),
            "sim_source": ["self.data.qpos[self._act_qpos_idx[i]]"],
            "real_source": ["11 joint encoder positions"],
            "policy_allowed": True,
        },
        {
            "group": "joint_vel",
            "range": layout_json()["joint_vel"],
            "columns": group_columns(columns, "joint_vel"),
            "sim_source": ["self.data.qvel[self._act_qvel_idx[i]]"],
            "real_source": ["11 joint encoder velocity estimates"],
            "policy_allowed": True,
        },
        {
            "group": "previous_action",
            "range": layout_json()["previous_action"],
            "columns": group_columns(columns, "previous_action"),
            "sim_source": ["self._last_action"],
            "real_source": ["controller memory of previous normalized action"],
            "policy_allowed": True,
        },
        {
            "group": "segment_gravity",
            "range": layout_json()["segment_gravity"],
            "columns": group_columns(columns, "segment_gravity"),
            "sim_source": [
                "self.data.xmat[body_id] projected gravity for each IMU body"
            ],
            "real_source": [
                "one IMU per segment: gravity direction from attitude fusion"
            ],
            "policy_allowed": True,
        },
        {
            "group": "segment_gyro",
            "range": layout_json()["segment_gyro"],
            "columns": group_columns(columns, "segment_gyro"),
            "sim_source": [
                "mujoco.mj_objectVelocity(..., body_id, local_vel, 1)"
            ],
            "real_source": ["one IMU per segment: local angular velocity"],
            "policy_allowed": True,
        },
        {
            "group": "phase_clock",
            "range": layout_json()["phase_clock"],
            "columns": group_columns(columns, "phase_clock"),
            "sim_source": ["self._step_count * CTRL_DT"],
            "real_source": ["controller clock"],
            "policy_allowed": True,
        },
    ]


def source_contains_all(source, tokens):
    return {
        token: (token in source)
        for token in tokens
    }


def build_audit():
    obs_source = inspect.getsource(WormEnvV6._get_obs)
    reward_source = inspect.getsource(WormEnvV6._compute_reward)
    termination_source = inspect.getsource(WormEnvV6._check_termination)
    columns = observation_columns()
    contract = observation_contract()
    layout = layout_json()

    group_tokens = {
        "command": ["self._cmd_vel", "self._cmd_yaw", "self._gait_blend"],
        "joint_pos": ["self.data.qpos", "self._act_qpos_idx"],
        "joint_vel": ["self.data.qvel", "self._act_qvel_idx"],
        "previous_action": ["self._last_action"],
        "segment_gravity": ["self.data.xmat", "gravity_world"],
        "segment_gyro": ["mujoco.mj_objectVelocity", "local_vel[:3]"],
        "phase_clock": ["self._step_count", "PHASE_FREQ"],
    }
    group_checks = {
        name: source_contains_all(obs_source, tokens)
        for name, tokens in group_tokens.items()
    }
    missing_group_tokens = {
        name: [token for token, present in checks.items() if not present]
        for name, checks in group_checks.items()
    }
    missing_group_tokens = {
        name: missing
        for name, missing in missing_group_tokens.items()
        if missing
    }
    forbidden_hits = regex_hits(obs_source, FORBIDDEN_OBS_PATTERNS)
    reward_only_hits = regex_hits(
        reward_source + "\n" + termination_source,
        REWARD_ONLY_PATTERNS,
    )
    layout_matches_columns = len(columns) == OBS_DIM and all(
        (stop - start) == len(group_columns(columns, name))
        for name, (start, stop) in layout.items()
    )
    complete = (
        not forbidden_hits
        and not missing_group_tokens
        and layout_matches_columns
        and contract.get("obs_dim") == OBS_DIM
        and contract.get("observation_columns") == columns
    )
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "obs_dim": OBS_DIM,
        "observation_contract_fingerprint": contract["abi_fingerprint"],
        "layout": layout,
        "layout_matches_columns": layout_matches_columns,
        "groups": expected_groups(columns),
        "group_source_token_checks": group_checks,
        "missing_group_source_tokens": missing_group_tokens,
        "forbidden_policy_observation_hits": forbidden_hits,
        "reward_only_privileged_hits": reward_only_hits,
        "source_files": {
            "environment": "src/v6/worm_env_v6.py",
            "hardware_builder": "src/v6/build_hardware_obs_v6.py",
            "contract": "src/v6/observation_contract_v6.py",
        },
        "interpretation": (
            "MuJoCo qpos/qvel/xmat/objectVelocity are acceptable here only as "
            "simulated encoder/IMU readouts matching the hardware ABI. "
            "Privileged root velocity/pose terms may appear in reward or "
            "termination code, but must not appear in WormEnvV6._get_obs()."),
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Observation Source Audit",
        "",
        f"Complete: `{str(payload['complete']).lower()}`",
        f"Observation dimension: `{payload['obs_dim']}`",
        f"ABI fingerprint: `{payload['observation_contract_fingerprint']}`",
        "",
        payload["interpretation"],
        "",
        "## Policy Observation Groups",
        "",
        "| Group | Range | Count | Sim source | Real source |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for group in payload["groups"]:
        start, stop = group["range"]
        lines.append(
            f"| {group['group']} | [{start}, {stop}) | {stop - start} | "
            f"{'; '.join(group['sim_source'])} | "
            f"{'; '.join(group['real_source'])} |")

    lines.extend([
        "",
        "## Forbidden Policy Observation Hits",
        "",
    ])
    if payload["forbidden_policy_observation_hits"]:
        for name, count in payload["forbidden_policy_observation_hits"].items():
            lines.append(f"- `{name}`: `{count}`")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Reward-Only Privileged Hits",
        "",
    ])
    if payload["reward_only_privileged_hits"]:
        for name, count in payload["reward_only_privileged_hits"].items():
            lines.append(f"- `{name}`: `{count}`")
    else:
        lines.append("- none")

    if payload["missing_group_source_tokens"]:
        lines.extend(["", "## Missing Group Tokens", ""])
        for name, tokens in payload["missing_group_source_tokens"].items():
            lines.append(f"- `{name}`: {', '.join(tokens)}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_audit(json_path=DEFAULT_JSON, md_path=DEFAULT_MD):
    payload = build_audit()
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit Worm V6 deployable observation sources")
    parser.add_argument("--json-out", default=DEFAULT_JSON)
    parser.add_argument("--md-out", default=DEFAULT_MD)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    payload = write_audit(args.json_out, args.md_out)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Worm V6 observation source audit")
        print(f"  complete: {payload['complete']}")
        print(f"  json: {rel(args.json_out)}")
        print(f"  md:   {rel(args.md_out)}")
    if args.strict and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

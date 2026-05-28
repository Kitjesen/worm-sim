"""
Preflight deployable Worm V6 hardware policy bundles.

This check does not create hardware evidence. It verifies that each terrain's
deploy bundle can consume the raw encoder/IMU sensor schema, construct the
formal 80-D observation ABI, run the TorchScript actor, and emit finite
normalized actions plus physical joint targets.
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

from build_hardware_obs_v6 import neutral_raw_row, raw_columns  # noqa: E402
from deploy_policy_v6 import action_mapping_fields, read_config  # noqa: E402
from hardware_policy_runtime_v6 import HardwarePolicyRuntime  # noqa: E402
from motor_contract_v6 import (  # noqa: E402
    action_mapping_matches,
    motor_contract,
)
from observation_contract_v6 import observation_contract  # noqa: E402
from prepare_hardware_trials_v6 import (  # noqa: E402
    DEFAULT_BEST_BLEND_CSV,
    TERRAINS,
    read_recommended_blends,
)
from validate_hardware_log_v6 import observation_columns  # noqa: E402
from worm_env_v6 import NUM_ACTUATORS, NUM_IMUS, NUM_SLIDES, OBS_DIM  # noqa: E402


DEFAULT_PREFLIGHT_JSON = os.path.join(
    PROJECT_ROOT, "record", "v6", "hardware",
    "hardware_deploy_preflight.json")
DEFAULT_PREFLIGHT_MD = os.path.join(
    PROJECT_ROOT, "record", "v6", "hardware",
    "hardware_deploy_preflight.md")
DEFAULT_MODE = "random"


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


def bundle_dir(terrain, mode=DEFAULT_MODE, project_root=PROJECT_ROOT):
    return os.path.join(
        project_root, "record", "v6", "deploy_bundles",
        f"{terrain}_{mode}")


def status(ok):
    return "ok" if ok else "failed"


def check(name, ok, detail=None):
    item = {"name": name, "status": status(bool(ok))}
    if detail is not None:
        item["detail"] = detail
    return item


def all_ok(checks):
    return all(item.get("status") == "ok" for item in checks)


def json_float(value):
    return float(np.asarray(value, dtype=np.float64))


def global_checks():
    contract = observation_contract()
    actuator_contract = motor_contract()
    return [
        check("obs_dim", contract.get("obs_dim") == OBS_DIM, OBS_DIM),
        check(
            "observation_columns",
            contract.get("observation_columns") == observation_columns(),
            len(observation_columns()),
        ),
        check(
            "raw_hardware_columns",
            contract.get("raw_hardware_columns") == raw_columns(),
            len(raw_columns()),
        ),
        check(
            "sensor_counts",
            contract.get("sensor_counts") == {
                "actuated_joints": NUM_ACTUATORS,
                "slide_joints": NUM_SLIDES,
                "yaw_joints": NUM_ACTUATORS - NUM_SLIDES,
                "segment_imus": NUM_IMUS,
            },
            contract.get("sensor_counts"),
        ),
        check(
            "actuator_contract",
            actuator_contract.get("actuator_count") == NUM_ACTUATORS
            and action_mapping_matches(actuator_contract.get("action_mapping")),
            actuator_contract.get("contract_fingerprint"),
        ),
    ]


def sample_raw_sensor(terrain, mode, gait_blend):
    row = neutral_raw_row()
    row.update({
        "terrain": terrain,
        "mode": mode,
        "gait_blend": float(np.clip(gait_blend, 0.0, 1.0)),
        "video_file": f"record/v6/videos/{terrain}_{mode}_hardware_demo.mp4",
    })
    return row


def preflight_terrain(terrain, mode=DEFAULT_MODE, gait_blend=0.5,
                      project_root=PROJECT_ROOT, bundle=None,
                      max_action_delta=0.2):
    bundle = resolve(bundle, project_root) or bundle_dir(
        terrain, mode, project_root)
    actor_path = os.path.join(bundle, "policy_actor.pt")
    config_path = os.path.join(bundle, "deploy_config.json")
    checks = [
        check("bundle_dir_exists", os.path.isdir(bundle), rel(bundle, project_root)),
        check("torchscript_actor_exists", os.path.exists(actor_path),
              rel(actor_path, project_root)),
        check("deploy_config_exists", os.path.exists(config_path),
              rel(config_path, project_root)),
    ]
    errors = []
    prediction_summary = {}

    config = None
    if os.path.exists(config_path):
        try:
            config = read_config(config_path)
        except Exception as exc:
            errors.append(f"read_config: {exc}")

    contract = observation_contract()
    if isinstance(config, dict):
        mapping = config.get("action_mapping", {})
        checks.extend([
            check("config_terrain", config.get("terrain") == terrain,
                  config.get("terrain")),
            check("config_gait_mode", config.get("gait_mode") == mode,
                  config.get("gait_mode")),
            check("config_obs_dim", config.get("obs_dim") == OBS_DIM,
                  config.get("obs_dim")),
            check(
                "config_observation_columns",
                config.get("observation_columns") == observation_columns(),
                len(config.get("observation_columns", [])),
            ),
            check("config_action_dim", config.get("action_dim") == NUM_ACTUATORS,
                  config.get("action_dim")),
            check(
                "config_action_mapping",
                action_mapping_matches(mapping),
                mapping,
            ),
            check(
                "actuator_contract_fingerprint",
                config.get("actuator_contract_fingerprint")
                == motor_contract()["contract_fingerprint"]
                and config.get("actuator_contract", {}).get(
                    "contract_fingerprint")
                == motor_contract()["contract_fingerprint"],
                config.get("actuator_contract_fingerprint"),
            ),
            check(
                "observation_contract_fingerprint",
                config.get("observation_contract_fingerprint")
                == contract["abi_fingerprint"],
                config.get("observation_contract_fingerprint"),
            ),
        ])

    if os.path.exists(actor_path) and isinstance(config, dict):
        try:
            runtime = HardwarePolicyRuntime(bundle_dir=bundle)
            raw = sample_raw_sensor(terrain, mode, gait_blend)
            prediction = runtime.predict(
                raw,
                update_state=False,
                max_action_delta=max_action_delta,
            )
            obs = prediction["observation"]
            action = prediction["action"]
            slide_targets = prediction["slide_targets_m"]
            yaw_targets = prediction["yaw_targets_rad"]
            mapping_fields = action_mapping_fields(config)
            checks.extend([
                check("observation_shape", obs.shape == (OBS_DIM,),
                      list(obs.shape)),
                check("observation_finite", np.all(np.isfinite(obs))),
                check("action_shape", action.shape == (NUM_ACTUATORS,),
                      list(action.shape)),
                check("action_finite", np.all(np.isfinite(action))),
                check(
                    "action_range",
                    np.all(action >= -1.0) and np.all(action <= 1.0),
                    [json_float(np.min(action)), json_float(np.max(action))],
                ),
                check("slide_target_shape", slide_targets.shape == (NUM_SLIDES,),
                      list(slide_targets.shape)),
                check(
                    "slide_target_range",
                    np.all(slide_targets >= mapping_fields["slide_min_m"] - 1e-9)
                    and np.all(
                        slide_targets <= mapping_fields["slide_max_m"] + 1e-9),
                    [
                        json_float(np.min(slide_targets)),
                        json_float(np.max(slide_targets)),
                    ],
                ),
                check(
                    "yaw_target_shape",
                    yaw_targets.shape == (NUM_ACTUATORS - NUM_SLIDES,),
                    list(yaw_targets.shape),
                ),
                check(
                    "yaw_target_range",
                    np.all(yaw_targets >= mapping_fields["yaw_min_rad"] - 1e-9)
                    and np.all(
                        yaw_targets <= mapping_fields["yaw_max_rad"] + 1e-9),
                    [
                        json_float(np.min(yaw_targets)),
                        json_float(np.max(yaw_targets)),
                    ],
                ),
            ])
            prediction_summary = {
                "obs_dim": int(obs.shape[0]),
                "action_dim": int(action.shape[0]),
                "max_abs_action": json_float(np.max(np.abs(action))),
                "max_abs_slide_target_m": json_float(
                    np.max(np.abs(slide_targets))),
                "max_abs_yaw_target_rad": json_float(
                    np.max(np.abs(yaw_targets))),
            }
        except Exception as exc:
            errors.append(f"runtime_predict: {exc}")

    ok = all_ok(checks) and not errors
    return {
        "terrain": terrain,
        "mode": mode,
        "recommended_gait_blend": float(gait_blend),
        "bundle_dir": rel(bundle, project_root),
        "actor": rel(actor_path, project_root),
        "config": rel(config_path, project_root),
        "status": status(ok),
        "checks": checks,
        "errors": errors,
        "prediction": prediction_summary,
    }


def preflight_payload(mode=DEFAULT_MODE, project_root=PROJECT_ROOT,
                      best_blend_csv=DEFAULT_BEST_BLEND_CSV,
                      max_action_delta=0.2):
    best_blend_csv = resolve(best_blend_csv, project_root)
    blends = read_recommended_blends(best_blend_csv)
    contract = observation_contract()
    checks = global_checks()
    terrains = [
        preflight_terrain(
            terrain,
            mode=mode,
            gait_blend=blends.get(terrain, 0.5),
            project_root=project_root,
            max_action_delta=max_action_delta,
        )
        for terrain in TERRAINS
    ]
    complete = all_ok(checks) and all(
        row.get("status") == "ok" for row in terrains)
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "mode": mode,
        "obs_dim": OBS_DIM,
        "observation_contract_fingerprint": contract["abi_fingerprint"],
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "best_blend_csv": rel(best_blend_csv, project_root),
        "max_action_delta": max_action_delta,
        "global_checks": checks,
        "terrains": terrains,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Hardware Deploy Preflight",
        "",
        f"Complete: `{str(payload['complete']).lower()}`",
        f"Observation ABI: `{payload['observation_contract_fingerprint']}`",
        f"Actuator contract: `{payload['actuator_contract_fingerprint']}`",
        "",
        "This is a deploy-chain readiness check. It is not real hardware "
        "validation evidence.",
        "",
        "## Terrain Bundles",
        "",
        "| Terrain | Status | gait_blend | Bundle | Max abs action |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in payload["terrains"]:
        prediction = row.get("prediction", {})
        lines.append(
            f"| {row['terrain']} | `{row['status']}` | "
            f"{row['recommended_gait_blend']:.3f} | "
            f"`{row['bundle_dir']}` | "
            f"{prediction.get('max_abs_action', '')} |")

    lines.extend([
        "",
        "## Global ABI Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ])
    for item in payload["global_checks"]:
        lines.append(
            f"| {item['name']} | `{item['status']}` | "
            f"{item.get('detail', '')} |")

    errors = [
        f"{row['terrain']}: {error}"
        for row in payload["terrains"]
        for error in row.get("errors", [])
    ]
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend([f"- {error}" for error in errors])

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python src\\v3\\preflight_hardware_deploy_v6.py --strict",
        "```",
        "",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_reports(json_path=DEFAULT_PREFLIGHT_JSON,
                  md_path=DEFAULT_PREFLIGHT_MD, **kwargs):
    payload = preflight_payload(**kwargs)
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preflight Worm V6 deployable hardware policy bundles")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["random"])
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--max-action-delta", type=float, default=0.2)
    parser.add_argument("--json-out", default=DEFAULT_PREFLIGHT_JSON)
    parser.add_argument("--md-out", default=DEFAULT_PREFLIGHT_MD)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero unless preflight is complete")
    return parser


def main():
    args = build_parser().parse_args()
    payload = write_reports(
        json_path=args.json_out,
        md_path=args.md_out,
        mode=args.mode,
        best_blend_csv=args.best_blend_csv,
        max_action_delta=args.max_action_delta,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Worm V6 hardware deploy preflight")
        print(f"  complete: {payload['complete']}")
        print(f"  json: {rel(args.json_out)}")
        print(f"  md:   {rel(args.md_out)}")
        for row in payload["terrains"]:
            print(
                f"  {row['terrain']}: {row['status']} "
                f"(gait_blend={row['recommended_gait_blend']:.3f})")
    if args.strict and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

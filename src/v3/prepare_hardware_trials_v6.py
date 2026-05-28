"""
Prepare field-trial folders for Worm V6 hardware validation.

The paper audit still requires real flat/sand/slope logs with video references.
This helper does not fake hardware data. It creates per-terrain collection
templates, recommended gait_blend values from the completed blend scan, and the
exact conversion / validation / replay commands needed after a robot run.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from build_hardware_obs_v6 import (  # noqa: E402
    neutral_raw_row,
    raw_columns,
)
from capture_hardware_stream_v6 import (  # noqa: E402
    write_json_schema,
    write_jsonl_example,
)
from validate_hardware_log_v6 import (  # noqa: E402
    neutral_example_row,
    required_columns,
)

TERRAINS = ("flat", "sand", "slope")
DEFAULT_MODE = "random"
DEFAULT_BEST_BLEND_CSV = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "best_blend_by_terrain.csv")


def rel(path):
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def ps_path(path):
    return rel(path).replace("/", "\\")


def write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_recommended_blends(path):
    blends = {}
    if not path or not os.path.exists(path):
        return blends
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            terrain = row.get("terrain")
            try:
                blend = float(row.get("gait_blend", ""))
            except ValueError:
                continue
            if terrain in TERRAINS:
                blends[terrain] = max(0.0, min(1.0, blend))
    return blends


def terrain_raw_example(terrain, mode, gait_blend, video_file):
    row = neutral_raw_row()
    row["terrain"] = terrain
    row["mode"] = mode
    row["gait_blend"] = gait_blend
    row["video_file"] = video_file
    return row


def terrain_policy_example(terrain, mode, gait_blend, video_file):
    row = neutral_example_row()
    row["terrain"] = terrain
    row["mode"] = mode
    row["gait_blend"] = gait_blend
    row["video_file"] = video_file
    return row


def command_list(terrain, mode, trial_dir, gait_blend=None):
    stem = f"{terrain}_{mode}"
    raw_log = os.path.join(trial_dir, terrain, f"{stem}_raw.csv")
    policy_log = os.path.join(PROJECT_ROOT, "record", "v6", "hardware",
                              f"{stem}_YYYYMMDD.csv")
    action_log = os.path.join(trial_dir, terrain, f"{stem}_actions.csv")
    video_file = f"record/v6/videos/{stem}_hardware_demo.mp4"
    bundle_dir = os.path.join(PROJECT_ROOT, "record", "v6", "deploy_bundles",
                              f"{terrain}_{mode}")
    gait_blend_arg = (
        f"--gait-blend {gait_blend:.3f} " if gait_blend is not None else "")
    return [
        (
            "preflight deploy bundle and observation ABI",
            "python src\\v3\\preflight_hardware_deploy_v6.py --strict",
        ),
        (
            "check controller JSONL stream before import",
            "python src\\v3\\check_controller_stream_v6.py "
            "--input-jsonl controller_stream.jsonl "
            f"--terrain {terrain} --mode {mode} "
            f"--video-file {video_file} "
            f"{gait_blend_arg}"
            "--cmd-vel 0.025 --cmd-yaw 0.0 "
            f"--bundle-dir {ps_path(bundle_dir)} --strict",
        ),
        (
            "optional: live JSONL controller bridge",
            "python src\\v3\\hardware_policy_runtime_v6.py "
            f"--bundle-dir {ps_path(bundle_dir)} "
            "--input-jsonl - --output-jsonl - "
            f"--policy-log-csv {ps_path(policy_log)} "
            f"--terrain {terrain} --mode {mode} {gait_blend_arg}"
            f"--video-file {video_file} --max-action-delta 0.2",
        ),
        (
            "capture controller JSONL stream into raw CSV",
            "python src\\v3\\capture_hardware_stream_v6.py "
            "--input-jsonl controller_stream.jsonl "
            f"--output-csv {ps_path(raw_log)} "
            f"--terrain {terrain} --mode {mode} "
            f"--video-file {video_file} "
            f"{gait_blend_arg}"
            "--cmd-vel 0.025 --cmd-yaw 0.0",
        ),
        (
            "one-command post-capture processing",
            "python src\\v3\\process_hardware_trial_v6.py "
            f"--terrain {terrain} --mode {mode} "
            "--input-jsonl controller_stream.jsonl "
            f"--raw-csv {ps_path(raw_log)} "
            f"--video-file {video_file} "
            f"{gait_blend_arg}"
            "--cmd-vel 0.025 --cmd-yaw 0.0 "
            "--date YYYYMMDD",
        ),
        (
            "import captured raw trial into audit files",
            "python src\\v3\\import_hardware_trial_v6.py "
            f"--terrain {terrain} --mode {mode} "
            f"--raw-csv {ps_path(raw_log)} "
            f"--video-file {video_file} "
            f"{gait_blend_arg}"
            "--date YYYYMMDD",
        ),
        (
            "run deploy policy on raw sensors and write formal audit CSV",
            "python src\\v3\\hardware_policy_runtime_v6.py "
            f"--bundle-dir {ps_path(bundle_dir)} "
            f"--input-raw-csv {ps_path(raw_log)} "
            f"--output-csv {ps_path(action_log)} "
            f"--policy-log-csv {ps_path(policy_log)} "
            f"--terrain {terrain} --mode {mode} {gait_blend_arg}"
            f"--video-file {video_file} "
            "--max-action-delta 0.2 "
            "--validate-policy-log --require-video "
            f"--expected-terrain {terrain} --min-rows 5 --min-duration 0.1",
        ),
        (
            "validate policy CSV for the paper audit",
            "python src\\v3\\validate_hardware_log_v6.py "
            f"--input {ps_path(policy_log)} --expected-terrain {terrain} "
            "--require-video --min-rows 5 --min-duration 0.1",
        ),
        (
            "optional: replay deploy bundle on logged observations",
            "python src\\v3\\deploy_policy_v6.py replay "
            f"--bundle-dir {ps_path(bundle_dir)} --input-csv {ps_path(policy_log)} "
            f"--output-csv {ps_path(action_log)}",
        ),
        (
            "show hardware trial status",
            "python src\\v3\\hardware_trial_status_v6.py --write-report",
        ),
        (
            "refresh completion audit",
            "python src\\v3\\paper_status_v6.py --refresh-audit",
        ),
    ]


def terrain_readme(terrain, mode, gait_blend, trial_dir):
    stem = f"{terrain}_{mode}"
    video_file = f"record/v6/videos/{stem}_hardware_demo.mp4"
    commands = command_list(terrain, mode, trial_dir, gait_blend=gait_blend)
    lines = [
        f"# Worm V6 Hardware Trial: {terrain}",
        "",
        f"- Terrain: `{terrain}`",
        f"- Policy bundle: `record/v6/deploy_bundles/{stem}`",
        f"- Mode label: `{mode}`",
        f"- Recommended gait_blend from scan: `{gait_blend:.3f}`",
        f"- Required video reference in CSV: `{video_file}`",
        f"- Controller JSONL example: "
        f"`record/v6/hardware/field_trials/current/{terrain}/{stem}_controller_stream_example.jsonl`",
        f"- Controller JSONL schema: "
        f"`record/v6/hardware/field_trials/current/{terrain}/{stem}_controller_stream_schema.json`",
        "",
        "## Capture Requirements",
        "",
        "- Record raw encoder positions and velocities for all 11 actuators.",
        "- Record one IMU per segment: local gravity direction and angular velocity.",
        "- Record normalized policy actions and physical command timing.",
        "- Do not use external pose, MuJoCo state, or ground-truth base velocity as policy input.",
        "- Keep at least 5 rows and at least 0.1 s duration for audit validation.",
        "",
        "## Commands After Capture",
        "",
    ]
    for title, command in commands:
        lines.extend([f"### {title}", "", "```powershell", command, "```", ""])
    lines.extend([
        "## Final Files Expected By Audit",
        "",
        f"- `record/v6/hardware/{stem}_YYYYMMDD.csv`",
        f"- `record/v6/videos/{stem}_hardware_demo.mp4`",
        "",
    ])
    return "\n".join(lines)


def write_top_readme(path, manifest):
    lines = [
        "# Worm V6 Hardware Field Trial Package",
        "",
        "This package prepares the real-robot validation step for the paper.",
        "It contains templates and commands only; it is not proof of a hardware run.",
        "",
        "The current audit is complete only after one validated CSV and one video",
        "reference exist for each terrain: flat, sand, and slope.",
        "",
        "## Terrain Plan",
        "",
        "| Terrain | Bundle | Recommended gait_blend | Trial README |",
        "| --- | --- | ---: | --- |",
    ]
    for entry in manifest["terrains"]:
        lines.append(
            f"| {entry['terrain']} | `{entry['bundle_dir']}` | "
            f"{entry['recommended_gait_blend']:.3f} | "
            f"`{entry['readme']}` |")
    lines.extend([
        "",
        "## Final Audit Command",
        "",
        "```powershell",
        "python src\\v3\\paper_status_v6.py --refresh-audit",
        "```",
        "",
        "## Hardware Status Command",
        "",
        "```powershell",
        "python src\\v3\\hardware_trial_status_v6.py --write-report",
        "```",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def prepare_trials(out_dir, mode=DEFAULT_MODE, best_blend_csv=None,
                   force=False):
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Use --force.")

    recommended = read_recommended_blends(best_blend_csv)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": rel(out_dir),
        "mode": mode,
        "terrains": [],
    }
    for terrain in TERRAINS:
        gait_blend = recommended.get(terrain, 0.5)
        terrain_dir = os.path.join(out_dir, terrain)
        os.makedirs(terrain_dir, exist_ok=True)
        stem = f"{terrain}_{mode}"
        video_file = f"record/v6/videos/{stem}_hardware_demo.mp4"

        raw_template = os.path.join(terrain_dir, f"{stem}_raw_template.csv")
        raw_example = os.path.join(terrain_dir, f"{stem}_raw_example.csv")
        stream_example = os.path.join(
            terrain_dir, f"{stem}_controller_stream_example.jsonl")
        stream_schema = os.path.join(
            terrain_dir, f"{stem}_controller_stream_schema.json")
        policy_template = os.path.join(terrain_dir, f"{stem}_policy_template.csv")
        policy_example = os.path.join(terrain_dir, f"{stem}_policy_example.csv")
        readme_path = os.path.join(terrain_dir, "README.md")

        write_rows(raw_template, raw_columns(), [])
        write_rows(raw_example, raw_columns(), [
            terrain_raw_example(terrain, mode, gait_blend, video_file)
        ])
        write_jsonl_example(
            stream_example,
            rows=5,
            terrain=terrain,
            mode=mode,
            video_file=video_file,
            gait_blend=gait_blend,
            include_metadata=False,
        )
        write_json_schema(stream_schema)
        write_rows(policy_template, required_columns(), [])
        write_rows(policy_example, required_columns(), [
            terrain_policy_example(terrain, mode, gait_blend, video_file)
        ])
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(terrain_readme(terrain, mode, gait_blend, out_dir))

        manifest["terrains"].append({
            "terrain": terrain,
            "bundle_dir": f"record/v6/deploy_bundles/{stem}",
            "recommended_gait_blend": gait_blend,
            "raw_template": rel(raw_template),
            "raw_example": rel(raw_example),
            "controller_stream_example": rel(stream_example),
            "controller_stream_schema": rel(stream_schema),
            "policy_template": rel(policy_template),
            "policy_example": rel(policy_example),
            "readme": rel(readme_path),
            "final_policy_log": f"record/v6/hardware/{stem}_YYYYMMDD.csv",
            "final_video": video_file,
            "commands": [
                {"name": name, "command": command}
                for name, command in command_list(
                    terrain, mode, out_dir, gait_blend=gait_blend)
            ],
        })

    manifest_path = os.path.join(out_dir, "hardware_trial_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_top_readme(os.path.join(out_dir, "README.md"), manifest)
    return manifest_path, manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare Worm V6 flat/sand/slope hardware trial package")
    parser.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware", "field_trials",
        "current"))
    parser.add_argument("--mode", default=DEFAULT_MODE,
                        choices=["random"])
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    manifest_path, manifest = prepare_trials(
        args.out_dir,
        mode=args.mode,
        best_blend_csv=args.best_blend_csv,
        force=args.force,
    )
    print(json.dumps({
        "manifest": rel(manifest_path),
        "out_dir": manifest["out_dir"],
        "terrains": [
            {
                "terrain": entry["terrain"],
                "recommended_gait_blend": entry["recommended_gait_blend"],
                "readme": entry["readme"],
            }
            for entry in manifest["terrains"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

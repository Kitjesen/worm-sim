"""
Build a compact, human-readable index for Worm V6 training artifacts.

The index is intentionally derived from current files under record/v6 and runs/.
It is a viewing aid for the paper workflow, not a substitute for the completion
audit or real hardware logs.
"""

import argparse
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

PAPER_TERRAINS = ("flat", "sand", "slope")
TRAIN_MODES = ("worm", "snake", "mixed", "random")
FIXED_MODES = ("worm", "snake", "mixed")
VIDEO_CANDIDATES = (
    "training_arena_1280x720.mp4",
    "gait_comparison_1280x720.mp4",
    "worm_v6_combined.mp4",
    "worm_v6_worm.mp4",
    "eval_straight_fast.mp4",
    "eval_turn_left.mp4",
    "eval_turn_right.mp4",
)


def posix_rel(path, start):
    try:
        return os.path.relpath(path, start).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def repo_rel(path):
    return posix_rel(path, PROJECT_ROOT)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value, digits=3):
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def file_status(path):
    return {
        "path": path,
        "exists": os.path.exists(path),
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
    }


def training_artifacts():
    rows = []
    for terrain in PAPER_TERRAINS:
        for mode in TRAIN_MODES:
            run_dir = os.path.join(
                PROJECT_ROOT, "runs", f"worm_v6_ppo_{terrain}_{mode}")
            model = os.path.join(run_dir, "best_model.zip")
            vecnorm = os.path.join(run_dir, "best_model_vecnormalize.pkl")
            result = load_json(os.path.join(run_dir, "training_result.json"),
                               default={})
            rows.append({
                "terrain": terrain,
                "mode": mode,
                "run_dir": run_dir,
                "model": model,
                "vecnormalize": vecnorm,
                "model_ok": os.path.exists(model),
                "vecnormalize_ok": os.path.exists(vecnorm),
                "completed_timesteps": result.get("completed_timesteps", ""),
            })
    return rows


def deploy_bundles():
    rows = []
    for terrain in PAPER_TERRAINS:
        bundle_dir = os.path.join(
            PROJECT_ROOT, "record", "v6", "deploy_bundles",
            f"{terrain}_random")
        actor = os.path.join(bundle_dir, "policy_actor.pt")
        config = os.path.join(bundle_dir, "deploy_config.json")
        rows.append({
            "terrain": terrain,
            "bundle_dir": bundle_dir,
            "actor": actor,
            "config": config,
            "actor_ok": os.path.exists(actor),
            "config_ok": os.path.exists(config),
        })
    return rows


def available_videos():
    video_dir = os.path.join(PROJECT_ROOT, "record", "v6", "videos")
    rows = []
    for name in VIDEO_CANDIDATES:
        path = os.path.join(video_dir, name)
        if os.path.exists(path):
            rows.append(file_status(path))
    return rows


def hardware_status_rows():
    from hardware_trial_status_v6 import (
        DEFAULT_MODE,
        process_command,
        stream_check_command,
    )

    status_path = os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        "hardware_trial_status.json")
    data = load_json(status_path, default={})
    rows = []
    for item in data.get("terrains", []):
        terrain = item.get("terrain", "")
        mode = item.get("mode", DEFAULT_MODE)
        gait_blend = item.get("recommended_gait_blend", 0.5)
        generated_process = (
            process_command(terrain, mode, float(gait_blend))
            if terrain else "")
        generated_stream_check = (
            stream_check_command(terrain, mode, float(gait_blend))
            if terrain else "")
        rows.append({
            "terrain": terrain,
            "status": item.get("status", ""),
            "raw_rows": item.get("raw_rows", 0),
            "duration_s": item.get("duration_s", 0.0),
            "video": item.get("video_exists", False),
            "policy_valid": item.get("policy_valid", False),
            "stream_check_command": item.get(
                "stream_check_command", generated_stream_check),
            "process_command": item.get(
                "process_command", generated_process),
            "next_command": item.get("next_command", ""),
        })
    return rows


def link(path, out_dir, label=None):
    label = label or os.path.basename(path)
    return f"[{label}]({posix_rel(path, out_dir)})"


def status_text(ok):
    return "ok" if ok else "missing"


def build_index_text(out_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    paper_dir = os.path.join(PROJECT_ROOT, "record", "v6", "paper_results")
    hardware_dir = os.path.join(PROJECT_ROOT, "record", "v6", "hardware")
    fixed_rows = load_csv(os.path.join(paper_dir, "fixed_mode_summary.csv"))
    best_rows = load_csv(os.path.join(paper_dir, "best_blend_by_terrain.csv"))
    claims = load_json(os.path.join(paper_dir, "paper_claims.json"),
                       default={})
    audit = load_json(os.path.join(paper_dir, "completion_audit.json"),
                      default={})
    preflight = load_json(
        os.path.join(hardware_dir, "hardware_deploy_preflight.json"),
        default={},
    )
    hardware_summary = load_json(
        os.path.join(paper_dir, "hardware_validation_summary.json"),
        default={},
    )
    observation_audit = load_json(
        os.path.join(paper_dir, "observation_source_audit.json"),
        default={},
    )

    lines = [
        "# Worm V6 Training Result Index",
        "",
        "This page indexes the current deployable multimodal snake/worm "
        "training results. It only reports artifacts already present on disk.",
        "",
        "## Open First",
        "",
        f"- {link(os.path.join(paper_dir, 'summary.md'), out_dir, 'paper result summary')}",
        f"- {link(os.path.join(paper_dir, 'fixed_mode_speed.svg'), out_dir, 'fixed-mode speed figure')}",
        f"- {link(os.path.join(paper_dir, 'blend_scan_speed.svg'), out_dir, 'gait_blend scan figure')}",
        f"- {link(os.path.join(paper_dir, 'paper_claims.md'), out_dir, 'paper claim analysis')}",
        f"- {link(os.path.join(paper_dir, 'observation_contract.md'), out_dir, 'deployable observation contract')}",
        f"- {link(os.path.join(paper_dir, 'observation_source_audit.md'), out_dir, 'observation source audit')}",
        f"- {link(os.path.join(paper_dir, 'paper_video_manifest.md'), out_dir, 'representative simulation videos')}",
        f"- {link(os.path.join(paper_dir, 'hardware_validation_summary.md'), out_dir, 'hardware validation summary')}",
        f"- {link(os.path.join(hardware_dir, 'hardware_deploy_preflight.md'), out_dir, 'hardware deploy preflight')}",
        f"- {link(os.path.join(paper_dir, 'completion_audit.md'), out_dir, 'completion audit')}",
        "",
        "## Fixed-Mode RL Results",
        "",
        "| Terrain | Mode | Speed mm/s | Success | Slip proxy | Robust speed mm/s | Robust success |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fixed_rows:
        lines.append(
            f"| {row.get('terrain', '')} | {row.get('mode', '')} | "
            f"{fmt(row.get('rl_speed_mm_s'))} | "
            f"{fmt(row.get('rl_success_rate'))} | "
            f"{fmt(row.get('rl_slip_proxy'))} | "
            f"{fmt(row.get('robust_speed_mm_s'))} | "
            f"{fmt(row.get('robust_success_rate'))} |")

    lines.extend([
        "",
        "## Best Continuous gait_blend Scan",
        "",
        "| Terrain | Best gait_blend | Label | Speed mm/s | Success |",
        "| --- | ---: | --- | ---: | ---: |",
    ])
    terrain_selection = {
        row.get("terrain"): row
        for row in claims.get("terrain_mode_selection", [])
    }
    for row in best_rows:
        terrain = row.get("terrain", "")
        claim_row = terrain_selection.get(terrain, {})
        lines.append(
            f"| {terrain} | {fmt(row.get('gait_blend'))} | "
            f"{claim_row.get('scan_best_label', '')} | "
            f"{fmt(row.get('mean_speed_mm_s'))} | "
            f"{fmt(row.get('success_rate'))} |")

    adaptive = claims.get("adaptive_best_blend_average", {})
    lines.extend([
        "",
        "## Cross-Terrain Claim Snapshot",
        "",
        f"- Adaptive best-blend average speed: {fmt(adaptive.get('avg_speed_mm_s'))} mm/s.",
        f"- Adaptive best-blend average success: {fmt(adaptive.get('avg_success_rate'))}.",
        f"- Completion audit: {'complete' if audit.get('complete') else 'incomplete'}.",
    ])
    for item in claims.get("claim_assessments", []):
        lines.append(f"- {item.get('status', '')}: {item.get('claim', '')}")
    cautions = claims.get("cautions", [])
    if cautions:
        lines.extend(["", "## Current Cautions", ""])
        lines.extend([f"- {caution}" for caution in cautions])

    lines.extend([
        "",
        "## Training Runs",
        "",
        "| Terrain | Mode | Model | VecNormalize | Completed steps | Run dir |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    for row in training_artifacts():
        lines.append(
            f"| {row['terrain']} | {row['mode']} | "
            f"{status_text(row['model_ok'])} | "
            f"{status_text(row['vecnormalize_ok'])} | "
            f"{row['completed_timesteps']} | "
            f"{link(row['run_dir'], out_dir, repo_rel(row['run_dir']))} |")

    lines.extend([
        "",
        "## Deployable Policy Bundles",
        "",
        "| Terrain | TorchScript actor | Deploy config | Bundle |",
        "| --- | --- | --- | --- |",
    ])
    for row in deploy_bundles():
        lines.append(
            f"| {row['terrain']} | {status_text(row['actor_ok'])} | "
            f"{status_text(row['config_ok'])} | "
            f"{link(row['bundle_dir'], out_dir, repo_rel(row['bundle_dir']))} |")

    forbidden_hits = observation_audit.get(
        "forbidden_policy_observation_hits", {})
    reward_only_hits = observation_audit.get(
        "reward_only_privileged_hits", {})
    reward_only_text = ", ".join(
        sorted(reward_only_hits.keys())) if reward_only_hits else "none"
    lines.extend([
        "",
        "## Observation Source Audit",
        "",
        f"- Complete: {str(observation_audit.get('complete', False)).lower()}.",
        f"- Observation dimension: {observation_audit.get('obs_dim', '')}.",
        f"- Forbidden policy observation hits: {len(forbidden_hits)}.",
        f"- Reward-only privileged hits: {reward_only_text}.",
    ])

    lines.extend([
        "",
        "## Hardware Deploy Preflight",
        "",
        f"- Complete: {str(preflight.get('complete', False)).lower()}.",
        f"- Observation ABI: {preflight.get('observation_contract_fingerprint', '')}.",
        "",
        "| Terrain | Status | gait_blend | Max abs action |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in preflight.get("terrains", []):
        prediction = row.get("prediction", {})
        lines.append(
            f"| {row.get('terrain', '')} | {row.get('status', '')} | "
            f"{fmt(row.get('recommended_gait_blend'))} | "
            f"{fmt(prediction.get('max_abs_action'))} |")
    if not preflight.get("terrains"):
        lines.append("| missing | missing |  |  |")

    lines.extend([
        "",
        "## Hardware Validation Summary",
        "",
        f"- Complete: {str(hardware_summary.get('complete', False)).lower()}.",
        f"- Validated terrains: {hardware_summary.get('aggregate', {}).get('validated_terrains', 0)}/{hardware_summary.get('aggregate', {}).get('required_terrains', 0)}.",
        "",
        "| Terrain | Status | Evidence | gait_blend | Velocity mm/s | Video |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for row in hardware_summary.get("rows", []):
        lines.append(
            f"| {row.get('terrain', '')} | {row.get('status', '')} | "
            f"{row.get('evidence_level', '')} | "
            f"{fmt(row.get('recommended_gait_blend'))} | "
            f"{fmt(row.get('mean_velocity_estimate_mm_s'))} | "
            f"{row.get('video_exists', False)} |")
    if not hardware_summary.get("rows"):
        lines.append("| missing | missing | missing |  |  | False |")

    videos = available_videos()
    video_manifest = load_json(os.path.join(
        paper_dir, "paper_video_manifest.json"), default={})
    lines.extend([
        "",
        "## Viewable Videos",
        "",
    ])
    for row in video_manifest.get("records", []):
        video = row.get("video", {})
        if video.get("exists"):
            lines.append(
                f"- {link(os.path.join(PROJECT_ROOT, video['path']), out_dir, video['path'])} "
                f"({video.get('size_bytes', 0) / (1024 * 1024):.1f} MiB, "
                f"{row.get('terrain')} best-blend preview)")
    if videos:
        for row in videos:
            mib = row["size"] / (1024 * 1024)
            lines.append(
                f"- {link(row['path'], out_dir, os.path.basename(row['path']))} "
                f"({mib:.1f} MiB)")
    else:
        lines.append("- No selected video artifacts found.")

    hardware_rows = hardware_status_rows()
    lines.extend([
        "",
        "## Hardware Trial Status",
        "",
        "| Terrain | Status | Raw rows | Duration s | Video | Policy valid |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ])
    if hardware_rows:
        for row in hardware_rows:
            lines.append(
                f"| {row['terrain']} | {row['status']} | "
                f"{row['raw_rows']} | {fmt(row['duration_s'])} | "
                f"{row['video']} | {row['policy_valid']} |")
    else:
        lines.append("| missing | missing | 0 | 0.000 | False | False |")

    stream_check_commands = [
        row["stream_check_command"] for row in hardware_rows
        if row.get("stream_check_command") and row.get("status") != "complete"
    ]
    if stream_check_commands:
        lines.extend(["", "Controller stream self-check:", ""])
        for command in stream_check_commands:
            lines.extend(["```powershell", command, "```", ""])

    process_commands = [
        row["process_command"] for row in hardware_rows
        if row.get("process_command") and row.get("status") != "complete"
    ]
    if process_commands:
        lines.extend(["", "One-command hardware processing:", ""])
        for command in process_commands:
            lines.extend(["```powershell", command, "```", ""])

    next_commands = [
        row["next_command"] for row in hardware_rows
        if row.get("next_command")
    ]
    if next_commands:
        lines.extend(["", "Status-specific fallback commands:", ""])
        for command in next_commands:
            lines.extend(["```powershell", command, "```", ""])

    return "\n".join(lines).rstrip() + "\n"


def write_index(out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    text = build_index_text(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Build a Markdown index for Worm V6 training results")
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "results_index.md"))
    args = ap.parse_args()
    out = write_index(args.out)
    print(f"Saved results index: {repo_rel(out)}")


if __name__ == "__main__":
    main()

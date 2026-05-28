"""
Render representative Worm V6 policy videos for paper inspection.

The script uses the trained random gait-conditioned policy on each paper
terrain, with gait_blend set to the current best blend scan result. It writes a
small manifest so the paper summary can link concrete video artifacts.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

PAPER_TERRAINS = ("flat", "sand", "slope")
DEFAULT_BEST_BLEND_CSV = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "best_blend_by_terrain.csv")
DEFAULT_OUT_JSON = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "paper_video_manifest.json")
DEFAULT_OUT_MD = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results",
    "paper_video_manifest.md")


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def read_best_blends(path):
    blends = {terrain: 0.5 for terrain in PAPER_TERRAINS}
    if not os.path.exists(path):
        return blends
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            terrain = row.get("terrain")
            if terrain not in blends:
                continue
            try:
                blends[terrain] = max(0.0, min(1.0, float(row["gait_blend"])))
            except (KeyError, TypeError, ValueError):
                continue
    return blends


def video_path(terrain):
    return os.path.join(
        PROJECT_ROOT, "record", "v6", "videos",
        f"eval_{terrain}_random.mp4")


def metrics_path(terrain):
    return os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        f"video_eval_{terrain}_random.json")


def eval_command(terrain, gait_blend, episodes, time_s, seed):
    return [
        sys.executable,
        os.path.join(SCRIPT_DIR, "eval_v6.py"),
        "--terrain", terrain,
        "--gait-mode", "random",
        "--gait-blend", f"{gait_blend:.3f}",
        "--episodes", str(episodes),
        "--time", str(time_s),
        "--seed", str(seed),
        "--json-out", metrics_path(terrain),
        "--video",
    ]


def file_info(path):
    return {
        "path": rel(path),
        "exists": os.path.exists(path),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
    }


def build_record(terrain, gait_blend, episodes, time_s, seed):
    vpath = video_path(terrain)
    mpath = metrics_path(terrain)
    return {
        "terrain": terrain,
        "policy_mode": "random",
        "gait_blend": gait_blend,
        "command": " ".join(eval_command(
            terrain, gait_blend, episodes, time_s, seed)),
        "video": file_info(vpath),
        "metrics": file_info(mpath),
    }


def render_videos(episodes=1, time_s=6.0, seed=2200,
                  best_blend_csv=DEFAULT_BEST_BLEND_CSV,
                  skip_existing=True, dry_run=False):
    blends = read_best_blends(best_blend_csv)
    records = []
    for terrain in PAPER_TERRAINS:
        gait_blend = blends[terrain]
        vpath = video_path(terrain)
        mpath = metrics_path(terrain)
        should_run = not dry_run and not (
            skip_existing
            and os.path.exists(vpath)
            and os.path.getsize(vpath) > 0
            and os.path.exists(mpath)
        )
        if should_run:
            os.makedirs(os.path.dirname(vpath), exist_ok=True)
            os.makedirs(os.path.dirname(mpath), exist_ok=True)
            subprocess.run(
                eval_command(terrain, gait_blend, episodes, time_s, seed),
                check=True,
            )
        record = build_record(
            terrain, gait_blend, episodes, time_s, seed)
        record["status"] = (
            "planned" if dry_run
            else "ok" if (
                record["video"]["exists"]
                and record["video"]["size_bytes"] > 0
                and record["metrics"]["exists"])
            else "missing")
        records.append(record)
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "time_s": time_s,
        "seed": seed,
        "best_blend_csv": rel(best_blend_csv),
        "complete": all(row["status"] == "ok" for row in records),
        "records": records,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Representative Simulation Videos",
        "",
        f"Complete: `{str(payload['complete']).lower()}`",
        "",
        "| Terrain | Policy | gait_blend | Status | Video | Metrics |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in payload["records"]:
        video = row["video"]["path"] if row["video"]["exists"] else "missing"
        metrics = (
            row["metrics"]["path"] if row["metrics"]["exists"]
            else "missing")
        lines.append(
            f"| {row['terrain']} | {row['policy_mode']} | "
            f"{row['gait_blend']:.3f} | {row['status']} | "
            f"{video} | {metrics} |")
    lines.extend(["", "## Reproduce", ""])
    for row in payload["records"]:
        lines.extend([
            f"### {row['terrain']}",
            "",
            "```powershell",
            row["command"],
            "```",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_manifest(payload, json_out=DEFAULT_OUT_JSON, md_out=DEFAULT_OUT_MD):
    write_json(json_out, payload)
    write_markdown(md_out, payload)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render representative Worm V6 paper videos")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--time", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=2200)
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--json-out", default=DEFAULT_OUT_JSON)
    parser.add_argument("--md-out", default=DEFAULT_OUT_MD)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    payload = render_videos(
        episodes=args.episodes,
        time_s=args.time,
        seed=args.seed,
        best_blend_csv=args.best_blend_csv,
        skip_existing=not args.force,
        dry_run=args.dry_run,
    )
    write_manifest(payload, args.json_out, args.md_out)
    print(json.dumps({
        "manifest": rel(args.json_out),
        "markdown": rel(args.md_out),
        "complete": payload["complete"],
        "records": [
            {
                "terrain": row["terrain"],
                "status": row["status"],
                "video": row["video"]["path"],
                "size_bytes": row["video"]["size_bytes"],
            }
            for row in payload["records"]
        ],
    }, indent=2))
    if not args.allow_missing and not args.dry_run and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

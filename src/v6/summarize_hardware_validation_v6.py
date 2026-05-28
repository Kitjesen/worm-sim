"""
Summarize Worm V6 real-hardware validation evidence for the paper.

The report is deliberately evidence-driven. It marks terrains as pending until
validated hardware CSV logs with resolvable video references exist; it never
substitutes templates, controller examples, or simulation logs for real runs.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from hardware_trial_status_v6 import (  # noqa: E402
    DEFAULT_MIN_DURATION_S,
    DEFAULT_MIN_ROWS,
    DEFAULT_MODE,
    status_payload,
)
from prepare_hardware_trials_v6 import DEFAULT_BEST_BLEND_CSV  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results")
DEFAULT_JSON = os.path.join(
    DEFAULT_OUT_DIR, "hardware_validation_summary.json")
DEFAULT_MD = os.path.join(
    DEFAULT_OUT_DIR, "hardware_validation_summary.md")
DEFAULT_CSV = os.path.join(
    DEFAULT_OUT_DIR, "hardware_validation_summary.csv")


def rel(path, project_root=PROJECT_ROOT):
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def fmt(value, digits=3):
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def metric_value(metrics, key):
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value in ("", None):
        return None
    return value


def evidence_level(row):
    if row.get("policy_valid"):
        return "validated_hardware"
    if row.get("raw_exists") and row.get("video_exists"):
        return "ready_to_import"
    if row.get("raw_exists"):
        return "captured_raw_pending_video_or_import"
    return "pending_real_run"


def summary_row(row):
    metrics = row.get("policy_metrics") or {}
    return {
        "terrain": row.get("terrain"),
        "mode": row.get("mode"),
        "recommended_gait_blend": row.get("recommended_gait_blend"),
        "status": row.get("status"),
        "evidence_level": evidence_level(row),
        "raw_csv": row.get("raw_csv"),
        "raw_rows": row.get("raw_rows"),
        "raw_duration_s": row.get("raw_duration_s"),
        "video_file": row.get("video_file"),
        "video_exists": bool(row.get("video_exists")),
        "policy_log_csv": row.get("policy_log_csv"),
        "policy_valid": bool(row.get("policy_valid")),
        "policy_rows": metric_value(metrics, "rows"),
        "policy_duration_s": metric_value(metrics, "duration_s"),
        "mean_velocity_estimate_mm_s": metric_value(
            metrics, "mean_velocity_estimate_mm_s"),
        "mean_velocity_estimate_m_s": metric_value(
            metrics, "mean_velocity_estimate_m_s"),
        "video_references": len(metrics.get("video_files", []))
        if isinstance(metrics, dict) else 0,
        "stream_check_command": row.get("stream_check_command"),
        "process_command": row.get("process_command"),
        "validate_command": row.get("validate_command"),
        "next_command": row.get("next_command"),
    }


def build_summary(mode=DEFAULT_MODE, project_root=PROJECT_ROOT,
                  best_blend_csv=DEFAULT_BEST_BLEND_CSV,
                  min_rows=DEFAULT_MIN_ROWS,
                  min_duration_s=DEFAULT_MIN_DURATION_S):
    payload = status_payload(
        mode=mode,
        project_root=project_root,
        best_blend_csv=best_blend_csv,
        min_rows=min_rows,
        min_duration_s=min_duration_s,
    )
    rows = [summary_row(row) for row in payload.get("terrains", [])]
    complete_rows = [row for row in rows if row["policy_valid"]]
    blockers = [
        {
            "terrain": row["terrain"],
            "status": row["status"],
            "next_command": row["next_command"],
        }
        for row in rows
        if not row["policy_valid"]
    ]
    velocities = [
        row.get("mean_velocity_estimate_mm_s")
        for row in complete_rows
        if row.get("mean_velocity_estimate_mm_s") is not None
    ]
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": bool(payload.get("complete")),
        "mode": mode,
        "min_rows": min_rows,
        "min_duration_s": min_duration_s,
        "requirement": (
            "One validated real-hardware CSV with a resolvable video "
            "reference for each terrain: flat, sand, and slope."),
        "non_evidence": [
            "hardware templates",
            "controller stream examples",
            "simulation-to-hardware bridge logs",
            "deploy preflight reports",
        ],
        "aggregate": {
            "validated_terrains": len(complete_rows),
            "required_terrains": len(rows),
            "mean_velocity_estimate_mm_s": mean(velocities),
        },
        "rows": rows,
        "blockers": blockers,
    }


def write_csv(path, rows):
    fields = [
        "terrain", "mode", "recommended_gait_blend", "status",
        "evidence_level", "raw_csv", "raw_rows", "raw_duration_s",
        "video_file", "video_exists", "policy_log_csv", "policy_valid",
        "policy_rows", "policy_duration_s", "mean_velocity_estimate_mm_s",
        "mean_velocity_estimate_m_s", "video_references",
        "stream_check_command", "process_command", "validate_command",
        "next_command",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, payload):
    lines = [
        "# Worm V6 Hardware Validation Summary",
        "",
        f"Complete: `{str(payload['complete']).lower()}`",
        "",
        payload["requirement"],
        "",
        "## Terrain Evidence",
        "",
        "| Terrain | Status | Evidence | gait_blend | Rows | Duration s | Velocity mm/s | Video |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['terrain']} | `{row['status']}` | "
            f"{row['evidence_level']} | "
            f"{fmt(row['recommended_gait_blend'])} | "
            f"{fmt(row['policy_rows'] or row['raw_rows'])} | "
            f"{fmt(row['policy_duration_s'] or row['raw_duration_s'])} | "
            f"{fmt(row['mean_velocity_estimate_mm_s'])} | "
            f"{'yes' if row['video_exists'] else 'no'} |")

    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Validated terrains: `{payload['aggregate']['validated_terrains']}/{payload['aggregate']['required_terrains']}`",
        f"- Mean velocity estimate: `{fmt(payload['aggregate']['mean_velocity_estimate_mm_s'])}` mm/s",
        "",
        "## Not Counted As Hardware Evidence",
        "",
    ])
    lines.extend([f"- {item}" for item in payload["non_evidence"]])

    if payload["blockers"]:
        lines.extend(["", "## Remaining Actions", ""])
        for item in payload["blockers"]:
            lines.extend([
                f"### {item['terrain']}",
                "",
                f"Status: `{item['status']}`",
                "",
                "```powershell",
                item["next_command"],
                "```",
                "",
            ])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_summary(json_path=DEFAULT_JSON, md_path=DEFAULT_MD,
                  csv_path=DEFAULT_CSV, **kwargs):
    payload = build_summary(**kwargs)
    write_json(json_path, payload)
    write_csv(csv_path, payload["rows"])
    write_markdown(md_path, payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize Worm V6 real-hardware validation evidence")
    parser.add_argument("--mode", choices=["random"], default=DEFAULT_MODE)
    parser.add_argument("--best-blend-csv", default=DEFAULT_BEST_BLEND_CSV)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-duration", type=float,
                        default=DEFAULT_MIN_DURATION_S)
    parser.add_argument("--json-out", default=DEFAULT_JSON)
    parser.add_argument("--md-out", default=DEFAULT_MD)
    parser.add_argument("--csv-out", default=DEFAULT_CSV)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero unless all terrain evidence exists")
    return parser


def main():
    args = build_parser().parse_args()
    payload = write_summary(
        json_path=args.json_out,
        md_path=args.md_out,
        csv_path=args.csv_out,
        mode=args.mode,
        best_blend_csv=args.best_blend_csv,
        min_rows=args.min_rows,
        min_duration_s=args.min_duration,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Worm V6 hardware validation summary")
        print(f"  complete: {payload['complete']}")
        print(
            "  validated: "
            f"{payload['aggregate']['validated_terrains']}/"
            f"{payload['aggregate']['required_terrains']}")
        print(f"  json: {rel(args.json_out)}")
        print(f"  md:   {rel(args.md_out)}")
        print(f"  csv:  {rel(args.csv_out)}")
    if args.strict and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

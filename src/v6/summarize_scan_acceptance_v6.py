"""Summarize Worm V6 strict command-scan acceptance artifacts.

This helper combines a scan JSON with its strict analysis JSON. It is intended
for paper-facing result tables where the acceptance verdict and failure group
matter as much as RMSE.
"""

import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _find_command(commands, target, tol=1e-8):
    tvx, tvy, tyaw = target
    for row in commands:
        vx = float(row.get("cmd_vx_m_s", 0.0))
        vy = float(row.get("cmd_vy_m_s", 0.0))
        yaw = float(row.get("cmd_yaw_rad_s", 0.0))
        if abs(vx - tvx) <= tol and abs(vy - tvy) <= tol and abs(yaw - tyaw) <= tol:
            return row
    return None


def _command_key(row):
    return (
        float(row.get("cmd_vx_m_s", 0.0)),
        float(row.get("cmd_vy_m_s", 0.0)),
        float(row.get("cmd_yaw_rad_s", 0.0)),
    )


def _component_sign_ok(cmd, measured):
    cmd = float(cmd or 0.0)
    measured = float(measured or 0.0)
    if abs(cmd) <= 1e-9:
        return True
    return cmd * measured > 0.0


def _mixed_component_sign_ok(row):
    cmd_vx = float(row.get("cmd_vx_m_s", 0.0))
    cmd_vy = float(row.get("cmd_vy_m_s", 0.0))
    cmd_yaw = float(row.get("cmd_yaw_rad_s", 0.0))
    if abs(cmd_vx) <= 1e-9 or abs(cmd_vy) <= 1e-9:
        return True
    if abs(cmd_yaw) > 1e-9:
        return True
    return (
        _component_sign_ok(cmd_vx, row.get("body_vx_m_s", 0.0))
        and _component_sign_ok(cmd_vy, row.get("body_vy_m_s", 0.0))
    )


def _selected_command(scan, analysis):
    commands = list(scan.get("commands", []))
    planar_sign_failures = [
        row for row in commands
        if (
            abs(float(row.get("cmd_vx_m_s", 0.0))) > 1e-9
            or abs(float(row.get("cmd_vy_m_s", 0.0))) > 1e-9
        )
        and not bool(row.get("planar_sign_ok", True))
    ]
    if planar_sign_failures:
        return max(planar_sign_failures, key=lambda r: float(
            r.get("planar_error_m_s", 0.0)) + float(
            r.get("yaw_error_rad_s", 0.0)))

    mixed_component_failures = [
        row for row in commands
        if not _mixed_component_sign_ok(row)
    ]
    if mixed_component_failures:
        return max(mixed_component_failures, key=lambda r: float(
            r.get("planar_error_m_s", 0.0)) + float(
            r.get("yaw_error_rad_s", 0.0)))

    yaw_sign_failures = [
        row for row in commands
        if abs(float(row.get("cmd_yaw_rad_s", 0.0))) > 1e-9
        and not bool(row.get("yaw_sign_ok", True))
    ]
    if yaw_sign_failures:
        return max(yaw_sign_failures, key=lambda r: float(
            r.get("planar_error_m_s", 0.0)) + float(
            r.get("yaw_error_rad_s", 0.0)))

    worst_commands = analysis.get("worst_commands", [])
    if worst_commands:
        target = _command_key(worst_commands[0])
        return _find_command(commands, target) or worst_commands[0]

    if commands:
        return max(commands, key=lambda r: float(
            r.get("planar_error_m_s", 0.0)) + float(
            r.get("yaw_error_rad_s", 0.0)))
    return {}


def summarize_item(label, scan_path, analysis_path):
    scan = load_json(scan_path)
    analysis = load_json(analysis_path)
    summary = scan.get("summary", {})
    acceptance = analysis.get("acceptance", {})
    measured = acceptance.get("measured", {})
    adapter = scan.get("action_adapter", {})
    counter = _selected_command(scan, analysis)

    return {
        "label": label,
        "scan_path": os.path.normpath(scan_path),
        "analysis_path": os.path.normpath(analysis_path),
        "accepted": acceptance.get("accepted"),
        "failed_conditions": ",".join(acceptance.get("failed_conditions", [])),
        "dominant_failure_group": analysis.get("dominant_failure_group"),
        "adapter_version": adapter.get("version"),
        "gait_prior_scale": scan.get("gait_prior_scale"),
        "policy_residual_scale": scan.get("policy_residual_scale"),
        "gait_blend": scan.get("gait_blend"),
        "num_commands": summary.get("num_commands"),
        "planar_rmse_m_s": measured.get(
            "planar_rmse_m_s", summary.get("planar_rmse_m_s")),
        "yaw_rmse_rad_s": measured.get(
            "yaw_rmse_rad_s", summary.get("yaw_rmse_rad_s")),
        "wrong_planar_sign_count": measured.get(
            "wrong_planar_sign_count", summary.get("wrong_planar_sign_count")),
        "wrong_yaw_sign_count": measured.get(
            "wrong_yaw_sign_count", summary.get("wrong_yaw_sign_count")),
        "wrong_mixed_component_sign_count": measured.get(
            "wrong_mixed_component_sign_count"),
        "planar_error_exceed_count": summary.get("planar_error_exceed_count"),
        "off_axis_exceed_count": summary.get("off_axis_exceed_count"),
        "fixed_lateral_strict_gate_passed": summary.get(
            "fixed_lateral_strict_gate_passed"),
        "counter_cmd_vx_m_s": counter.get("cmd_vx_m_s"),
        "counter_cmd_vy_m_s": counter.get("cmd_vy_m_s"),
        "counter_cmd_yaw_rad_s": counter.get("cmd_yaw_rad_s"),
        "counter_body_vx_m_s": counter.get("body_vx_m_s"),
        "counter_body_vy_m_s": counter.get("body_vy_m_s"),
        "counter_yaw_rate_rad_s": counter.get("yaw_rate_rad_s"),
        "counter_planar_sign_ok": counter.get("planar_sign_ok"),
        "counter_mixed_component_sign_ok": _mixed_component_sign_ok(counter),
        "counter_mean_prior_component_l2": counter.get(
            "mean_prior_component_l2"),
        "counter_mean_residual_component_l2": counter.get(
            "mean_residual_component_l2"),
    }


def render_markdown(rows, title):
    lines = [
        f"# {title}",
        "",
        "Selected row: sign-failure command when available, otherwise worst "
        "strict-analysis command.",
        "",
        "| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | "
        "Wrong planar | Wrong yaw | Wrong mixed comp | Planar exceed | Off-axis exceed | "
        "Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | "
        "Projected sign | Mixed comp sign |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "--- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cmd_tuple = (
            f"({_fmt(row.get('counter_cmd_vx_m_s'))}, "
            f"{_fmt(row.get('counter_cmd_vy_m_s'))}, "
            f"{_fmt(row.get('counter_cmd_yaw_rad_s'))})"
        )
        counter_tuple = (
            f"({_fmt(row.get('counter_body_vx_m_s'))}, "
            f"{_fmt(row.get('counter_body_vy_m_s'))}, "
            f"{_fmt(row.get('counter_yaw_rate_rad_s'))})"
        )
        sign = row.get("counter_planar_sign_ok")
        sign_text = "n/a" if sign is None else ("pass" if sign else "fail")
        comp_sign = row.get("counter_mixed_component_sign_ok")
        comp_sign_text = (
            "n/a" if comp_sign is None else ("pass" if comp_sign else "fail"))
        failed = row.get("failed_conditions") or "-"
        lines.append(
            f"| `{row['label']}` | {_fmt(row.get('accepted'))} | `{failed}` | "
            f"`{row.get('adapter_version')}` | "
            f"{_fmt(row.get('planar_rmse_m_s'))} | "
            f"{_fmt(row.get('yaw_rmse_rad_s'))} | "
            f"{row.get('wrong_planar_sign_count')} | "
            f"{row.get('wrong_yaw_sign_count')} | "
            f"{row.get('wrong_mixed_component_sign_count')} | "
            f"{row.get('planar_error_exceed_count')} | "
            f"{row.get('off_axis_exceed_count')} | "
            f"{_fmt(row.get('fixed_lateral_strict_gate_passed'))} | "
            f"`{cmd_tuple}` | "
            f"`{counter_tuple}` | {sign_text} | {comp_sign_text} |")
    lines.append("")
    return "\n".join(lines)


def write_json(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format_version": 1, "items": rows}, f, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize strict scan acceptance results")
    ap.add_argument(
        "--item",
        action="append",
        nargs=3,
        metavar=("LABEL", "SCAN_JSON", "ANALYSIS_JSON"),
        required=True,
        help="Add one labeled scan plus analysis pair.",
    )
    ap.add_argument("--title", default="Worm V6 Strict Scan Acceptance Summary")
    ap.add_argument("--out-json")
    ap.add_argument("--out-csv")
    ap.add_argument("--out-md")
    args = ap.parse_args()

    rows = [summarize_item(label, scan, analysis)
            for label, scan, analysis in args.item]
    markdown = render_markdown(rows, args.title)

    if args.out_json:
        write_json(args.out_json, rows)
    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)),
                    exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(markdown)
    print(markdown)


if __name__ == "__main__":
    main()

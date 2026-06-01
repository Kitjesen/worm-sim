"""Analyze Worm V6 command-tracking scans by command class.

The 35-command scan is the acceptance surface for continuous vx/vy/yaw
tracking. This helper makes the proof stricter: primitive-direction success is
reported separately from mixed-command composition failures.
"""

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional


EPS = 1e-9
PLANAR_RMSE_TARGET_M_S = 0.10
YAW_RMSE_TARGET_RAD_S = 0.20
VX_ERROR_TARGET_M_S = 0.10
VY_ERROR_TARGET_M_S = 0.10
OFF_AXIS_TARGET_M_S = 0.08
ZERO_SPEED_TARGET_M_S = 0.02
YAW_ONLY_PLANAR_TARGET_M_S = 0.08

COMMAND_CLASS_ORDER = [
    "stop",
    "pure_vx",
    "pure_vy",
    "pure_yaw",
    "mixed_vx_vy",
    "mixed_vx_yaw",
    "mixed_vy_yaw",
    "full_mixed",
]
SCAN_ROW_TELEMETRY_KEYS = (
    "mean_gait_blend",
    "mean_learned_gait_blend",
    "mean_raw_gait_gate_action",
    "mean_prior_component_l2",
    "mean_residual_component_l2",
    "mean_applied_action_l2",
    "mean_desired_gait_blend",
    "mean_gait_gate_error",
    "mean_reward_component_tracking_cost",
    "mean_reward_planar_component_deficit_penalty",
    "mean_mixed_planar_fullscale_gate",
    "mean_reward_mixed_planar_fullscale_deficit_penalty",
    "mean_reward_axial_prior_preserve_penalty",
    "mean_axial_prior_residual_cancellation",
    "mean_axial_slide_activity_deficit",
)


def _f(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def classify_command(row: Dict) -> str:
    has_vx = abs(_f(row, "cmd_vx_m_s")) > EPS
    has_vy = abs(_f(row, "cmd_vy_m_s")) > EPS
    has_yaw = abs(_f(row, "cmd_yaw_rad_s")) > EPS
    if not has_vx and not has_vy and not has_yaw:
        return "stop"
    if has_vx and not has_vy and not has_yaw:
        return "pure_vx"
    if not has_vx and has_vy and not has_yaw:
        return "pure_vy"
    if not has_vx and not has_vy and has_yaw:
        return "pure_yaw"
    if has_vx and has_vy and not has_yaw:
        return "mixed_vx_vy"
    if has_vx and not has_vy and has_yaw:
        return "mixed_vx_yaw"
    if not has_vx and has_vy and has_yaw:
        return "mixed_vy_yaw"
    return "full_mixed"


def _rmse(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _planar_speed(row: Dict) -> float:
    return math.hypot(_f(row, "body_vx_m_s"), _f(row, "body_vy_m_s"))


def _tracking_score(row: Dict) -> float:
    planar = _f(row, "planar_error_m_s") / PLANAR_RMSE_TARGET_M_S
    yaw = _f(row, "yaw_error_rad_s") / YAW_RMSE_TARGET_RAD_S
    return planar * planar + yaw * yaw


def _compact_command(row: Dict) -> Dict:
    return {
        "class": row["class"],
        "cmd_vx_m_s": _f(row, "cmd_vx_m_s"),
        "cmd_vy_m_s": _f(row, "cmd_vy_m_s"),
        "cmd_yaw_rad_s": _f(row, "cmd_yaw_rad_s"),
        "body_vx_m_s": _f(row, "body_vx_m_s"),
        "body_vy_m_s": _f(row, "body_vy_m_s"),
        "yaw_rate_rad_s": _f(row, "yaw_rate_rad_s"),
        "vx_error_m_s": _f(row, "vx_error_m_s"),
        "vy_error_m_s": _f(row, "vy_error_m_s"),
        "planar_error_m_s": _f(row, "planar_error_m_s"),
        "yaw_error_rad_s": _f(row, "yaw_error_rad_s"),
        "off_axis_speed_m_s": _f(row, "off_axis_speed_m_s"),
        "tracking_score": _tracking_score(row),
        "planar_sign_ok": bool(row.get("planar_sign_ok", True)),
        "yaw_sign_ok": bool(row.get("yaw_sign_ok", True)),
    }


def _group_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "count": 0,
            "vx_error_rmse_m_s": None,
            "vy_error_rmse_m_s": None,
            "planar_rmse_m_s": None,
            "yaw_rmse_rad_s": None,
            "mean_off_axis_speed_m_s": None,
            "max_planar_error_m_s": None,
            "max_yaw_error_rad_s": None,
            "vx_error_exceed_count": 0,
            "vy_error_exceed_count": 0,
            "planar_error_exceed_count": 0,
            "yaw_error_exceed_count": 0,
            "off_axis_exceed_count": 0,
            "wrong_planar_sign_count": 0,
            "wrong_yaw_sign_count": 0,
            "dominant_error_score": 0.0,
            "telemetry_means": {},
            "worst_command": None,
        }
    worst = max(rows, key=_tracking_score)
    telemetry_means = {}
    for key in SCAN_ROW_TELEMETRY_KEYS:
        values = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if values:
            telemetry_means[key] = float(round(sum(values) / len(values), 10))
    return {
        "count": len(rows),
        "vx_error_rmse_m_s": _rmse(_f(r, "vx_error_m_s") for r in rows),
        "vy_error_rmse_m_s": _rmse(_f(r, "vy_error_m_s") for r in rows),
        "planar_rmse_m_s": _rmse(_f(r, "planar_error_m_s") for r in rows),
        "yaw_rmse_rad_s": _rmse(_f(r, "yaw_error_rad_s") for r in rows),
        "mean_off_axis_speed_m_s": _mean(
            _f(r, "off_axis_speed_m_s") for r in rows),
        "max_planar_error_m_s": max(_f(r, "planar_error_m_s") for r in rows),
        "max_yaw_error_rad_s": max(_f(r, "yaw_error_rad_s") for r in rows),
        "vx_error_exceed_count": sum(
            1 for r in rows if abs(_f(r, "vx_error_m_s")) > VX_ERROR_TARGET_M_S),
        "vy_error_exceed_count": sum(
            1 for r in rows if abs(_f(r, "vy_error_m_s")) > VY_ERROR_TARGET_M_S),
        "planar_error_exceed_count": sum(
            1 for r in rows
            if _f(r, "planar_error_m_s") > PLANAR_RMSE_TARGET_M_S),
        "yaw_error_exceed_count": sum(
            1 for r in rows
            if _f(r, "yaw_error_rad_s") > YAW_RMSE_TARGET_RAD_S),
        "off_axis_exceed_count": sum(
            1 for r in rows
            if _f(r, "off_axis_speed_m_s") > OFF_AXIS_TARGET_M_S),
        "wrong_planar_sign_count": sum(
            1 for r in rows if not bool(r.get("planar_sign_ok", True))),
        "wrong_yaw_sign_count": sum(
            1 for r in rows if not bool(r.get("yaw_sign_ok", True))),
        "dominant_error_score": sum(_tracking_score(r) for r in rows),
        "telemetry_means": telemetry_means,
        "worst_command": _compact_command(worst),
    }


def _acceptance(summary: Dict, rows: List[Dict]) -> Dict:
    failed = []
    planar_rmse = summary.get("planar_rmse_m_s")
    yaw_rmse = summary.get("yaw_rmse_rad_s")
    wrong_planar = int(summary.get("wrong_planar_sign_count", 0) or 0)
    wrong_yaw = int(summary.get("wrong_yaw_sign_count", 0) or 0)
    zero_speed = summary.get("zero_command_mean_speed_m_s")
    yaw_only_speed = summary.get("yaw_only_mean_planar_speed_m_s")

    planar_command_rows = [
        r for r in rows
        if abs(_f(r, "cmd_vx_m_s")) > EPS or abs(_f(r, "cmd_vy_m_s")) > EPS
    ]
    mean_off_axis = _mean(
        _f(r, "off_axis_speed_m_s") for r in planar_command_rows)

    if planar_rmse is None or float(planar_rmse) > PLANAR_RMSE_TARGET_M_S:
        failed.append("planar_rmse_m_s")
    if yaw_rmse is None or float(yaw_rmse) > YAW_RMSE_TARGET_RAD_S:
        failed.append("yaw_rmse_rad_s")
    if wrong_planar != 0:
        failed.append("wrong_planar_sign_count")
    if wrong_yaw != 0:
        failed.append("wrong_yaw_sign_count")
    if mean_off_axis is None or mean_off_axis > OFF_AXIS_TARGET_M_S:
        failed.append("mean_off_axis_speed_m_s")
    if zero_speed is None or float(zero_speed) > ZERO_SPEED_TARGET_M_S:
        failed.append("zero_command_mean_speed_m_s")
    if yaw_only_speed is None or float(yaw_only_speed) > YAW_ONLY_PLANAR_TARGET_M_S:
        failed.append("yaw_only_mean_planar_speed_m_s")

    return {
        "accepted": len(failed) == 0,
        "failed_conditions": failed,
        "targets": {
            "planar_rmse_m_s": PLANAR_RMSE_TARGET_M_S,
            "yaw_rmse_rad_s": YAW_RMSE_TARGET_RAD_S,
            "mean_off_axis_speed_m_s": OFF_AXIS_TARGET_M_S,
            "zero_command_mean_speed_m_s": ZERO_SPEED_TARGET_M_S,
            "yaw_only_mean_planar_speed_m_s": YAW_ONLY_PLANAR_TARGET_M_S,
        },
        "measured": {
            "planar_rmse_m_s": planar_rmse,
            "yaw_rmse_rad_s": yaw_rmse,
            "wrong_planar_sign_count": wrong_planar,
            "wrong_yaw_sign_count": wrong_yaw,
            "mean_off_axis_speed_m_s": mean_off_axis,
            "zero_command_mean_speed_m_s": zero_speed,
            "yaw_only_mean_planar_speed_m_s": yaw_only_speed,
        },
    }


def analyze_scan_payload(payload: Dict, top_k: int = 8) -> Dict:
    rows = []
    for row in payload.get("commands", []):
        row_with_class = dict(row)
        row_with_class["class"] = classify_command(row)
        rows.append(row_with_class)

    grouped_rows = {
        name: [r for r in rows if r["class"] == name]
        for name in COMMAND_CLASS_ORDER
    }
    groups = {
        name: _group_summary(grouped_rows[name])
        for name in COMMAND_CLASS_ORDER
    }
    present_groups = [
        (name, info) for name, info in groups.items() if info["count"] > 0
    ]
    dominant_group = None
    if present_groups:
        dominant_group = max(
            present_groups,
            key=lambda item: item[1]["dominant_error_score"],
        )[0]
    worst_commands = [
        _compact_command(r)
        for r in sorted(rows, key=_tracking_score, reverse=True)[:top_k]
    ]
    return {
        "format_version": 1,
        "source_scan": {
            "terrain": payload.get("terrain"),
            "gait_mode": payload.get("gait_mode"),
            "time_s": payload.get("time_s"),
            "model_path": payload.get("model_path"),
            "eval_condition": payload.get("eval_condition"),
            "sensor_noise": payload.get("sensor_noise"),
            "action_delay_steps": payload.get("action_delay_steps"),
            "action_saturation": payload.get("action_saturation"),
            "reward_contract": payload.get("reward_contract"),
            "action_adapter_version": (
                payload.get("action_adapter", {}).get("version")
                if isinstance(payload.get("action_adapter"), dict)
                else None
            ),
        },
        "acceptance": _acceptance(payload.get("summary", {}), rows),
        "dominant_failure_group": dominant_group,
        "groups": groups,
        "worst_commands": worst_commands,
        "original_summary": payload.get("summary", {}),
    }


def render_markdown(analysis: Dict) -> str:
    acceptance = analysis["acceptance"]
    verdict = "PASS" if acceptance["accepted"] else "FAIL"
    lines = [
        "# Worm V6 command scan strict analysis",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Proof rule",
        "",
        "Six primitive directions are a necessary condition, not a sufficient "
        "condition, for continuous `vx/vy/yaw` tracking. A policy is accepted "
        "only if the scan satisfies component RMSE, direction-sign, off-axis, "
        "stop, and yaw-only drift gates. One failed command class is a "
        "constructive counterexample to the continuous-tracking claim.",
        "",
    ]
    source = analysis.get("source_scan", {})
    if source.get("eval_condition"):
        lines += [
            "## Evaluation condition",
            "",
            f"- condition: `{source.get('eval_condition')}`",
            f"- sensor_noise: `{source.get('sensor_noise')}`",
            f"- action_delay_steps: `{source.get('action_delay_steps')}`",
            f"- action_saturation: `{source.get('action_saturation')}`",
            "",
        ]
    lines += [
        "## Acceptance gate",
        "",
        "| Condition | Measured | Target | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    measured = acceptance["measured"]
    targets = acceptance["targets"]
    for key, target in targets.items():
        value = measured.get(key)
        failed = key in acceptance["failed_conditions"]
        value_text = "n/a" if value is None else f"`{float(value):.4f}`"
        lines.append(
            f"| `{key}` | {value_text} | `{target:.4f}` | "
            f"{'fail' if failed else 'pass'} |")
    for key in ("wrong_planar_sign_count", "wrong_yaw_sign_count"):
        value = measured.get(key)
        failed = key in acceptance["failed_conditions"]
        lines.append(
            f"| `{key}` | `{int(value)}` | `0` | "
            f"{'fail' if failed else 'pass'} |")

    lines += [
        "",
        f"Dominant failure group: `{analysis['dominant_failure_group']}`",
        "",
        "## Command-class decomposition",
        "",
        "| Class | N | Planar RMSE | Yaw RMSE | vx exceed | vy exceed | "
        "planar exceed | yaw exceed | off-axis exceed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in COMMAND_CLASS_ORDER:
        info = analysis["groups"][name]
        if info["count"] == 0:
            continue
        planar = info["planar_rmse_m_s"]
        yaw = info["yaw_rmse_rad_s"]
        lines.append(
            f"| `{name}` | {info['count']} | "
            f"`{planar:.4f}` | `{yaw:.4f}` | "
            f"{info['vx_error_exceed_count']} | "
            f"{info['vy_error_exceed_count']} | "
            f"{info['planar_error_exceed_count']} | "
            f"{info['yaw_error_exceed_count']} | "
            f"{info['off_axis_exceed_count']} |")

    telemetry_rows = [
        (name, analysis["groups"][name])
        for name in COMMAND_CLASS_ORDER
        if analysis["groups"][name]["count"] > 0
        and analysis["groups"][name]["telemetry_means"]
    ]
    if telemetry_rows:
        lines += [
            "",
            "## Telemetry by command class",
            "",
            "| Class | Mean gait blend | Prior L2 | Residual L2 | "
            "Action L2 | Tracking cost | Fullscale gate | Fullscale deficit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, info in telemetry_rows:
            telemetry = info["telemetry_means"]

            def fmt(key):
                value = telemetry.get(key)
                return "n/a" if value is None else f"`{value:.4f}`"

            lines.append(
                f"| `{name}` | "
                f"{fmt('mean_gait_blend')} | "
                f"{fmt('mean_prior_component_l2')} | "
                f"{fmt('mean_residual_component_l2')} | "
                f"{fmt('mean_applied_action_l2')} | "
                f"{fmt('mean_reward_component_tracking_cost')} | "
                f"{fmt('mean_mixed_planar_fullscale_gate')} | "
                f"{fmt('mean_reward_mixed_planar_fullscale_deficit_penalty')} |")

    lines += [
        "",
        "## Worst commands",
        "",
        "| Class | Command `(vx, vy, yaw)` | Measured `(vx, vy, yaw)` | "
        "Planar err | Yaw err | Score |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in analysis["worst_commands"]:
        lines.append(
            f"| `{row['class']}` | "
            f"(`{row['cmd_vx_m_s']:.3f}`, `{row['cmd_vy_m_s']:.3f}`, "
            f"`{row['cmd_yaw_rad_s']:.3f}`) | "
            f"(`{row['body_vx_m_s']:.3f}`, `{row['body_vy_m_s']:.3f}`, "
            f"`{row['yaw_rate_rad_s']:.3f}`) | "
            f"`{row['planar_error_m_s']:.4f}` | "
            f"`{row['yaw_error_rad_s']:.4f}` | "
            f"`{row['tracking_score']:.2f}` |")
    lines.append("")
    return "\n".join(lines)


def load_scan(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze a Worm V6 command-tracking scan by command class")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    analysis = analyze_scan_payload(load_scan(args.scan), top_k=args.top_k)
    if args.out_json:
        write_json(args.out_json, analysis)
    if args.out_md:
        write_text(args.out_md, render_markdown(analysis))
    print(json.dumps({
        "accepted": analysis["acceptance"]["accepted"],
        "failed_conditions": analysis["acceptance"]["failed_conditions"],
        "dominant_failure_group": analysis["dominant_failure_group"],
    }, indent=2))


if __name__ == "__main__":
    main()

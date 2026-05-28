"""
Generate body-segment trajectory plots for Worm V6 gait inspection.

This is a visualization/diagnostic script, not a training script. It simulates
the open-loop snake, worm, and combined gait controllers from worm_v6.py, logs
the head and each body segment position over time, and writes plots plus CSVs.
"""

import argparse
import csv
import math
import os
import sys

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from worm_v6 import (  # noqa: E402
    BODY_Z,
    NUM_SLIDES,
    NUM_YAWS,
    SLIDE_RANGE_VAL,
    SNAKE_AMP,
    SNAKE_FREQ,
    SNAKE_WAVES,
    STEP_DURATION,
    build_xml,
    setup_terrain,
)


SEGMENT_NAMES = ["base_link"] + [f"back{i}_Link" for i in range(1, 7)]
SEGMENT_LABELS = ["head"] + [f"seg{i}" for i in range(1, 7)]
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "record", "v6", "trajectory")


def ensure_urdf():
    mesh_dir = os.path.join(PROJECT_ROOT, "meshes")
    urdf_path = os.path.join(mesh_dir, "longworm2", "longworm2.SLDASM.urdf")
    if os.path.exists(urdf_path):
        return mesh_dir, urdf_path

    src_urdf = os.path.join(
        "D:/inovxio/3d/longworm2/longworm2.SLDASM/urdf",
        "longworm2.SLDASM.urdf")
    if not os.path.exists(src_urdf):
        raise FileNotFoundError(
            f"URDF not found at {urdf_path} or {src_urdf}")
    os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
    import shutil
    shutil.copy2(src_urdf, urdf_path)
    return mesh_dir, urdf_path


def build_model(terrain):
    mesh_dir, urdf_path = ensure_urdf()
    del urdf_path
    xml_str = build_xml(mesh_dir, os.path.join(
        mesh_dir, "longworm2", "longworm2.SLDASM.urdf"), terrain=terrain)
    model = mujoco.MjModel.from_xml_string(xml_str)
    setup_terrain(model, terrain)
    data = mujoco.MjData(model)
    return model, data


def actuator_ids(model):
    slide_ids = []
    yaw_ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("act_back"):
            slide_ids.append(i)
        elif name and name.startswith("act_front"):
            yaw_ids.append(i)
    if len(slide_ids) != NUM_SLIDES or len(yaw_ids) != NUM_YAWS:
        raise RuntimeError(
            f"Expected {NUM_SLIDES} slides/{NUM_YAWS} yaws, got "
            f"{len(slide_ids)}/{len(yaw_ids)}")
    return slide_ids, yaw_ids


def body_ids(model):
    ids = []
    for name in SEGMENT_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Body not found: {name}")
        ids.append(body_id)
    return ids


def apply_open_loop_control(data, mode, t, slide_ids, yaw_ids):
    data.ctrl[:] = 0.0

    if mode in ("worm", "combined"):
        wave_len = len(slide_ids)
        for j, act_id in enumerate(slide_ids):
            phase = 2.0 * math.pi * (t / STEP_DURATION + j / wave_len)
            data.ctrl[act_id] = -SLIDE_RANGE_VAL * 0.5 * (
                1.0 + math.sin(phase))

    if mode in ("snake", "combined"):
        for j, act_id in enumerate(yaw_ids):
            phase = (
                2.0 * math.pi * SNAKE_FREQ * t
                + 2.0 * math.pi * SNAKE_WAVES * j / len(yaw_ids)
            )
            data.ctrl[act_id] = SNAKE_AMP * math.sin(phase)


def simulate_mode(mode, terrain, duration_s, sample_hz):
    model, data = build_model(terrain)
    slide_ids, yaw_ids = actuator_ids(model)
    seg_ids = body_ids(model)

    settle_steps = int(2.0 / model.opt.timestep)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    sample_dt = 1.0 / sample_hz
    next_sample_t = 0.0
    records = []
    total_steps = int(duration_s / model.opt.timestep)
    for step in range(total_steps + 1):
        t = step * model.opt.timestep
        apply_open_loop_control(data, mode, t, slide_ids, yaw_ids)
        mujoco.mj_step(model, data)

        if t + 1e-12 < next_sample_t:
            continue
        positions = data.xpos[seg_ids].copy()
        records.append((t, positions))
        next_sample_t += sample_dt

    times = np.array([row[0] for row in records], dtype=float)
    pos = np.stack([row[1] for row in records], axis=0)
    return times, pos


def direction_summary(mode, times, pos):
    head = pos[:, 0, :]
    start = head[0]
    end = head[-1]
    delta = end - start
    xy_norm = float(np.linalg.norm(delta[:2]))
    unit_xy = delta[:2] / xy_norm if xy_norm > 1e-9 else np.zeros(2)
    forward_mm = float(-delta[0] * 1000.0)
    lateral_mm = float(delta[1] * 1000.0)
    displacement_mm = float(np.linalg.norm(delta) * 1000.0)
    speed_mm_s = forward_mm / max(float(times[-1] - times[0]), 1e-9)
    angle_deg = math.degrees(math.atan2(delta[1], delta[0]))
    if delta[0] < 0:
        direction_label = "forward (-X)"
    elif delta[0] > 0:
        direction_label = "backward (+X)"
    else:
        direction_label = "lateral/no X motion"
    return {
        "mode": mode,
        "direction_label": direction_label,
        "head_delta_x_mm": float(delta[0] * 1000.0),
        "head_delta_y_mm": lateral_mm,
        "head_delta_z_mm": float(delta[2] * 1000.0),
        "head_unit_x": float(unit_xy[0]),
        "head_unit_y": float(unit_xy[1]),
        "world_xy_angle_deg": float(angle_deg),
        "forward_minus_x_mm": forward_mm,
        "lateral_y_mm": lateral_mm,
        "head_displacement_mm": displacement_mm,
        "forward_speed_mm_s": speed_mm_s,
    }


def write_trajectory_csv(path, mode, times, pos):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "time_s", "segment_index", "segment_label",
            "body_name", "x_m", "y_m", "z_m",
            "forward_minus_x_mm", "lateral_y_mm", "delta_z_mm",
            "height_z_mm",
        ])
        start = pos[0]
        for ti, t in enumerate(times):
            for si, label in enumerate(SEGMENT_LABELS):
                xyz = pos[ti, si]
                writer.writerow([
                    mode,
                    f"{t:.6f}",
                    si,
                    label,
                    SEGMENT_NAMES[si],
                    f"{xyz[0]:.9f}",
                    f"{xyz[1]:.9f}",
                    f"{xyz[2]:.9f}",
                    f"{-(xyz[0] - start[si, 0]) * 1000.0:.6f}",
                    f"{(xyz[1] - start[si, 1]) * 1000.0:.6f}",
                    f"{(xyz[2] - start[si, 2]) * 1000.0:.6f}",
                    f"{xyz[2] * 1000.0:.6f}",
                ])


def write_summary_csv(path, summaries):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = [
        "mode", "direction_label", "head_delta_x_mm", "head_delta_y_mm",
        "head_delta_z_mm", "head_unit_x", "head_unit_y",
        "world_xy_angle_deg", "forward_minus_x_mm", "lateral_y_mm",
        "head_displacement_mm", "forward_speed_mm_s",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def write_markdown_report(path, terrain, duration_s, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "# Worm V6 Body-Segment Trajectory Report",
        "",
        f"- Terrain: `{terrain}`",
        f"- Duration: `{duration_s:.2f} s`",
        "- Forward convention: `-X` in the MuJoCo world frame",
        "- Segment order: `head/base_link`, then `back1_Link` ... `back6_Link`",
        "",
        "| Mode | Head direction | Head delta XY (mm) | Unit XY | Forward speed (mm/s) | Absolute position plot | Relative motion plot | CSV |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in records:
        lines.append(
            f"| {row['mode']} | {row['direction_label']} | "
            f"({row['head_delta_x_mm']:.1f}, {row['head_delta_y_mm']:.1f}) | "
            f"({row['head_unit_x']:.3f}, {row['head_unit_y']:.3f}) | "
            f"{row['forward_speed_mm_s']:.2f} | "
            f"{row['absolute_plot']} | {row['relative_plot']} | "
            f"{row['csv']} |")
    lines.extend([
        "",
        "`absolute_positions` plots show each body segment's actual world "
        "position time history: world `X(t)`, `Y(t)`, and `Z(t)`. Segment "
        "initial positions are different because the robot has physical length.",
        "",
        "`relative_motion` plots show each body segment's motion relative to "
        "its own initial position, useful for comparing displacement phases.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_absolute_mode(path, mode, terrain, duration_s, times, pos, summary):
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(SEGMENT_LABELS)))
    pos_mm = pos * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Worm V6 {mode} absolute segment positions on {terrain} "
        f"({duration_s:.1f}s, forward = -X)",
        fontsize=14,
    )

    ax = axes[0, 0]
    for si, label in enumerate(SEGMENT_LABELS):
        ax.plot(pos_mm[:, si, 0], pos_mm[:, si, 1],
                color=colors[si], linewidth=1.6, label=label)
        ax.scatter(pos_mm[0, si, 0], pos_mm[0, si, 1],
                   color=colors[si], marker="o", s=16)
        ax.scatter(pos_mm[-1, si, 0], pos_mm[-1, si, 1],
                   color=colors[si], marker="x", s=24)
    head_start = pos_mm[0, 0, :2]
    head_delta = pos_mm[-1, 0, :2] - head_start
    ax.arrow(
        head_start[0], head_start[1],
        head_delta[0], head_delta[1],
        length_includes_head=True,
        head_width=8,
        head_length=12,
        linewidth=2.0,
        color="black",
        alpha=0.85,
    )
    ref_x = float(np.max(pos_mm[:, :, 0])) - 20.0
    ref_y = float(np.min(pos_mm[:, :, 1])) + 20.0
    ax.arrow(ref_x, ref_y, -60, 0, length_includes_head=True,
             head_width=7, head_length=10, color="tab:red")
    ax.text(ref_x - 70, ref_y + 10, "forward -X", color="tab:red")
    ax.set_title(
        "World XY paths; circles=start, x=end; black arrow=head motion")
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    for si, label in enumerate(SEGMENT_LABELS):
        ax.plot(times, pos_mm[:, si, 0],
                color=colors[si], linewidth=1.5, label=label)
        ax.scatter(times[0], pos_mm[0, si, 0],
                   color=colors[si], marker="o", s=12)
    ax.set_title("World X position of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("world X (mm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for si, label in enumerate(SEGMENT_LABELS):
        ax.plot(times, pos_mm[:, si, 1],
                color=colors[si], linewidth=1.5, label=label)
    ax.set_title("World Y position of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("world Y (mm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for si, label in enumerate(SEGMENT_LABELS):
        ax.plot(times, pos_mm[:, si, 2],
                color=colors[si], linewidth=1.5, label=label)
    ax.axhline(BODY_Z * 1000.0, color="0.4", linestyle="--",
               linewidth=1.0, label="nominal body z")
    ax.set_title("World Z position of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("world Z (mm)")
    ax.grid(True, alpha=0.3)

    text = (
        f"Head direction: {summary['direction_label']}\n"
        f"Head delta XY: ({summary['head_delta_x_mm']:.1f}, "
        f"{summary['head_delta_y_mm']:.1f}) mm\n"
        f"Unit XY: ({summary['head_unit_x']:.3f}, "
        f"{summary['head_unit_y']:.3f})\n"
        f"Forward speed: {summary['forward_speed_mm_s']:.2f} mm/s"
    )
    fig.text(0.012, 0.012, text, fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_relative_mode(path, mode, terrain, duration_s, times, pos, summary):
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(SEGMENT_LABELS)))
    pos_mm = pos * 1000.0
    start_mm = pos_mm[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Worm V6 {mode} relative segment motion on {terrain} "
        f"({duration_s:.1f}s, forward = -X)",
        fontsize=14,
    )

    ax = axes[0, 0]
    for si, label in enumerate(SEGMENT_LABELS):
        forward = -(pos_mm[:, si, 0] - start_mm[si, 0])
        lateral = pos_mm[:, si, 1] - start_mm[si, 1]
        ax.plot(forward, lateral,
                color=colors[si], linewidth=1.6, label=label)
        ax.scatter(forward[0], lateral[0],
                   color=colors[si], marker="o", s=16)
        ax.scatter(forward[-1], lateral[-1],
                   color=colors[si], marker="x", s=24)
    head_forward = -(pos_mm[-1, 0, 0] - start_mm[0, 0])
    head_lateral = pos_mm[-1, 0, 1] - start_mm[0, 1]
    ax.arrow(
        0.0, 0.0,
        head_forward, head_lateral,
        length_includes_head=True,
        head_width=8,
        head_length=12,
        linewidth=2.0,
        color="black",
        alpha=0.85,
    )
    all_forward = -(pos_mm[:, :, 0] - start_mm[None, :, 0])
    all_lateral = pos_mm[:, :, 1] - start_mm[None, :, 1]
    ref_x = float(np.min(all_forward)) + 20.0
    ref_y = float(np.min(all_lateral)) + 20.0
    ax.arrow(ref_x, ref_y, 60, 0, length_includes_head=True,
             head_width=7, head_length=10, color="tab:red")
    ax.text(ref_x + 5, ref_y + 10, "forward (-world X)", color="tab:red")
    ax.set_title("Relative XY motion; all segment starts are at origin")
    ax.set_xlabel("forward displacement -Delta X (mm)")
    ax.set_ylabel("lateral displacement Delta Y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    for si, label in enumerate(SEGMENT_LABELS):
        forward = -(pos_mm[:, si, 0] - start_mm[si, 0])
        ax.plot(times, forward, color=colors[si], linewidth=1.5, label=label)
    ax.set_title("Forward displacement of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("-Delta X (mm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for si, label in enumerate(SEGMENT_LABELS):
        lateral = pos_mm[:, si, 1] - start_mm[si, 1]
        ax.plot(times, lateral, color=colors[si], linewidth=1.5, label=label)
    ax.set_title("Lateral displacement of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Delta Y (mm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for si, label in enumerate(SEGMENT_LABELS):
        dz = pos_mm[:, si, 2] - start_mm[si, 2]
        ax.plot(times, dz,
                color=colors[si], linewidth=1.5, label=label)
    ax.axhline(0.0, color="0.4", linestyle="--",
               linewidth=1.0, label="initial height")
    ax.set_title("Height displacement of each segment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Delta Z (mm)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Worm V6 body-segment trajectories")
    parser.add_argument(
        "--modes", nargs="+", default=["snake", "worm", "combined"],
        choices=["snake", "worm", "combined"])
    parser.add_argument("--terrain", default="flat",
                        choices=["flat", "sand", "slope"])
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    records = []
    for mode in args.modes:
        times, pos = simulate_mode(
            mode, args.terrain, args.duration, args.sample_hz)
        summary = direction_summary(mode, times, pos)
        summaries.append(summary)
        abs_png_path = os.path.join(
            args.out_dir, f"{args.terrain}_{mode}_absolute_positions.png")
        rel_png_path = os.path.join(
            args.out_dir, f"{args.terrain}_{mode}_relative_motion.png")
        csv_path = os.path.join(
            args.out_dir, f"{args.terrain}_{mode}_segment_trajectories.csv")
        plot_absolute_mode(
            abs_png_path, mode, args.terrain, args.duration, times, pos,
            summary)
        plot_relative_mode(
            rel_png_path, mode, args.terrain, args.duration, times, pos,
            summary)
        write_trajectory_csv(csv_path, mode, times, pos)
        records.append({
            "mode": mode,
            "absolute_plot": os.path.relpath(
                abs_png_path, PROJECT_ROOT).replace("\\", "/"),
            "relative_plot": os.path.relpath(
                rel_png_path, PROJECT_ROOT).replace("\\", "/"),
            "csv": os.path.relpath(csv_path, PROJECT_ROOT).replace("\\", "/"),
            **summary,
        })

    summary_path = os.path.join(
        args.out_dir, f"{args.terrain}_direction_summary.csv")
    report_path = os.path.join(
        args.out_dir, f"{args.terrain}_trajectory_report.md")
    write_summary_csv(summary_path, summaries)
    write_markdown_report(report_path, args.terrain, args.duration, records)
    print("Generated trajectory artifacts:")
    for row in records:
        print(
            f"  {row['mode']:>8s}: {row['direction_label']}, "
            f"head_delta=({row['head_delta_x_mm']:.1f}, "
            f"{row['head_delta_y_mm']:.1f}) mm, "
            f"speed={row['forward_speed_mm_s']:.2f} mm/s")
        print(f"            absolute: {row['absolute_plot']}")
        print(f"            relative: {row['relative_plot']}")
        print(f"            csv:  {row['csv']}")
    print(
        "  summary: "
        + os.path.relpath(summary_path, PROJECT_ROOT).replace("\\", "/"))
    print(
        "  report:  "
        + os.path.relpath(report_path, PROJECT_ROOT).replace("\\", "/"))


if __name__ == "__main__":
    main()

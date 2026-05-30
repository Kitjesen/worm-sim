"""
Render a visual-only steel-strip preview.

This script shows how curved spring-steel strips can be generated as many small
box segments. It is a renderer preview, not a physics model.
"""

import argparse
import math
import os
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402

from visual_steel_strip_geometry_v6 import (  # noqa: E402
    DEFAULT_EXTRA_COMPRESSION_RANGE_M,
    DEFAULT_NORMAL_GAP_M,
    DEFAULT_PREBEND_COMPRESSION_M,
    STRIP_CIRCLE_R,
    box_corners,
    generate_steel_strip_boxes,
)


FACE_INDICES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4),
)


def _box_faces(box):
    corners = box_corners(box)
    return [[corners[idx] for idx in face] for face in FACE_INDICES]


def _draw_plate(ax, x, radius=0.085, color=(0.72, 0.74, 0.78, 0.25)):
    theta = np.linspace(0.0, 2.0 * math.pi, 80)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    xs = np.full_like(y, x)
    ax.plot(xs, y, z, color=(0.45, 0.48, 0.52), linewidth=1.0)
    ax.scatter([x], [0.0], [0.0], color=(0.25, 0.28, 0.32), s=8)
    ax.plot(
        [x, x],
        [-radius, radius],
        [0.0, 0.0],
        color=color,
        linewidth=8,
        alpha=0.25,
    )


def draw_strip_state(ax, target_extra_compression_m,
                     normal_gap_m=DEFAULT_NORMAL_GAP_M):
    target_extra = max(0.0, float(target_extra_compression_m))
    effective_compression_m = DEFAULT_PREBEND_COMPRESSION_M + target_extra
    target_gap = normal_gap_m - target_extra
    distance = max(target_gap, 0.006)
    shown_extra = max(0.0, normal_gap_m - distance)
    parent = np.array([-distance * 0.5, 0.0, 0.0])
    child = np.array([distance * 0.5, 0.0, 0.0])
    boxes = generate_steel_strip_boxes(
        parent,
        child,
        np.eye(3),
        normal_gap_m,
        arc_segments=16,
        prebend_compression_m=DEFAULT_PREBEND_COMPRESSION_M,
        extra_compression_range_m=DEFAULT_EXTRA_COMPRESSION_RANGE_M,
    )

    faces = []
    for box in boxes:
        faces.extend(_box_faces(box))
    collection = Poly3DCollection(
        faces,
        facecolors=(0.03, 0.03, 0.035, 0.92),
        edgecolors=(0.0, 0.0, 0.0, 0.12),
        linewidths=0.1,
    )
    ax.add_collection3d(collection)

    _draw_plate(ax, parent[0])
    _draw_plate(ax, child[0])

    # Draw the centerline and outer reference radius.
    ax.plot(
        [parent[0], child[0]],
        [0.0, 0.0],
        [0.0, 0.0],
        color=(0.8, 0.1, 0.1),
        linewidth=1.0,
        alpha=0.55,
    )
    if target_extra <= 1e-9:
        title = (
            f"effective compression = {effective_compression_m * 1000:.0f} mm "
            f"(pre-bend) | gap = {distance * 1000:.0f} mm")
    else:
        title = (
            f"effective compression = {effective_compression_m * 1000:.0f} mm "
            f"| gap = {distance * 1000:.0f} mm")
    ax.set_title(title)
    if shown_extra + 1e-9 < target_extra:
        ax.text2D(
            0.05,
            0.04,
            f"shown with {distance * 1000:.0f} mm clearance",
            transform=ax.transAxes,
            fontsize=8,
            color=(0.35, 0.35, 0.35),
        )
    span = normal_gap_m * 1.3
    lim = STRIP_CIRCLE_R + 0.055
    ax.set_xlim(-span, span)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.8, 1.0, 1.0))
    ax.view_init(elev=18, azim=-58)
    ax.set_axis_off()
    return boxes


def render_static(out_path):
    fig = plt.figure(figsize=(14, 4.5), dpi=180)
    extra_compressions_m = [0.0, 0.025, 0.050]
    for idx, extra_compression_m in enumerate(extra_compressions_m, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        draw_strip_state(ax, extra_compression_m)
    fig.suptitle(
        "Worm V6 spring-steel visual: 50 mm pre-bend plus up to 50 mm additional compression",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_cycle(out_path):
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(7.2, 4.6), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    frames = 40

    def update(frame_idx):
        ax.cla()
        phase = 2.0 * math.pi * frame_idx / frames
        additional_compression = 0.025 * (1.0 - math.cos(phase))
        gap_m = max(0.050 - additional_compression, 0.006)
        draw_strip_state(ax, additional_compression)
        ax.set_title(
            "box-segment visual steel strip | effective compression "
            f"{(0.050 + additional_compression) * 1000:.1f} mm "
            f"| gap {gap_m * 1000:.1f} mm",
            fontsize=8,
        )

    anim = FuncAnimation(fig, update, frames=frames, interval=70)
    anim.save(out_path, writer=PillowWriter(fps=14))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            "record", "v6", "steel_strip_visual_preview"),
    )
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    static_path = os.path.join(args.out_dir, "steel_strip_prebend_states.png")
    render_static(static_path)
    print(f"wrote {static_path}")
    legacy_static_path = os.path.join(args.out_dir, "steel_strip_box_states.png")
    shutil.copyfile(static_path, legacy_static_path)
    print(f"wrote {legacy_static_path}")

    if not args.no_gif:
        gif_path = os.path.join(args.out_dir, "steel_strip_prebend_cycle.gif")
        render_cycle(gif_path)
        print(f"wrote {gif_path}")
        legacy_gif_path = os.path.join(args.out_dir, "steel_strip_box_cycle.gif")
        shutil.copyfile(gif_path, legacy_gif_path)
        print(f"wrote {legacy_gif_path}")


if __name__ == "__main__":
    main()

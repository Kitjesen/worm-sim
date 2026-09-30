"""Render solved ribbon geometry; no rope, contact, or time dynamics is implied."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


RIBBON_COLORS = ("#87a3b4", "#9bb2bf", "#7895a7", "#a8bdc8")
PLATE_COLOR = "#b5bac0"


def paper_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                         "svg.fonttype": "none", "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white",
                         "axes.grid": False, "axes.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "mathtext.fontset": "dejavuserif"})


def paper_axis(ax, lo, hi, *, show_axes=True):
    for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(lo[dim], hi[dim])
    ax.set_proj_type("ortho")
    ax.set_box_aspect(hi - lo, zoom=1 if show_axes else 1.22)
    ax.view_init(elev=23, azim=-58)
    ax.grid(False)
    if not show_axes:
        ax.set_axis_off()
        return
    ax.set(xlabel="$x$ (mm)", ylabel="$y$ (mm)", zlabel="$z$ (mm)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis.set_major_locator(MaxNLocator(3))
        axis.labelpad = 0
    ax.tick_params(pad=0, length=2)


def ribbon_faces(frame, width):
    nodes = np.asarray(frame["nodes_m"], dtype=float)
    directors = np.asarray(frame["width_directors"], dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2:
        raise ValueError("nodes_m must contain at least two XYZ points")
    if directors.shape != (len(nodes) - 1, 3):
        raise ValueError("width_directors must contain one XYZ vector per edge")
    norms = np.linalg.norm(directors, axis=1)
    if not np.isfinite(nodes).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("Ribbon geometry must be finite with nonzero directors")
    offset = 0.5 * width * directors / norms[:, None]
    return np.stack((nodes[:-1] - offset, nodes[1:] - offset,
                     nodes[1:] + offset, nodes[:-1] + offset), axis=1)


def plate_points(center, radius, axis):
    basis = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, basis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angle = np.linspace(0, 2 * np.pi, 65)
    return center + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)


def move_plate(points, center, frame):
    rotation = np.asarray(frame["plate_rotation"], dtype=float)
    translation = np.asarray(frame["plate_translation_m"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Plate pose needs a 3x3 rotation and XYZ translation")
    return center + translation + (points - center) @ rotation.T


def render(path):
    paper_style()
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    meta, cases = data["metadata"], data["cases"]
    model = "sano" if any(c["model"] == "sano" for c in cases) else "kirchhoff"
    displayed = sorted((c for c in cases if c["model"] == model), key=lambda c: c["strip"])
    if not 1 <= len(displayed) <= 8:
        raise ValueError("Expected one to eight ribbons")
    frames = displayed[0]["frames"]
    if not frames:
        raise ValueError("No solved frames to render")
    progress = np.array([f["progress"] for f in frames], dtype=float)
    for case in cases:
        if len(case["frames"]) != len(frames) or not np.allclose(
                [f["progress"] for f in case["frames"]], progress):
            raise ValueError("All cases must share the same saved load schedule")
    width, radius = float(meta["width_m"]), float(meta["plate_radius_m"])
    front = np.asarray(meta["front_plate_center_m"], dtype=float)
    back = np.asarray(meta["back_plate_center_world_m"], dtype=float)
    axis = np.array([1.0, 0.0, 0.0])
    if width <= 0 or radius <= 0 or np.linalg.norm(back - front) < 1e-12:
        raise ValueError("Positive dimensions and distinct plate centers are required")
    disks = [plate_points(c, radius, axis) for c in (front, back)]
    surfaces = [[ribbon_faces(f, width) * 1000 for f in c["frames"]] for c in displayed]
    all_points = [s.reshape(-1, 3) for strip in surfaces for s in strip]
    all_points += [disks[0] * 1000]
    all_points += [move_plate(disks[1], back, f) * 1000 for f in frames]
    points = np.concatenate(all_points)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite plate or ribbon geometry")
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1)
    lo, hi = lo - span * 0.1, hi + span * 0.1
    smoke = bool(meta.get("renderer_smoke"))
    description = json.dumps({key: meta[key] for key in
                              ("loading", "reference", "path_status", "unloading_converged", "renderer_smoke")
                              if key in meta}, ensure_ascii=False)

    def draw(ax, index, show_axes=True):
        ax.clear()
        f = frames[index]
        for j, case in enumerate(displayed):
            ax.add_collection3d(Poly3DCollection(surfaces[j][index], facecolors=RIBBON_COLORS[j % 4],
                                               edgecolors="#607889", linewidths=.18, alpha=.92))
        for j, disk in enumerate((disks[0], move_plate(disks[1], back, f))):
            ax.add_collection3d(Poly3DCollection([disk * 1000], facecolors=PLATE_COLOR,
                                               edgecolors="#6b747b", linewidths=.65, alpha=.18))
        paper_axis(ax, lo, hi, show_axes=show_axes)

    peak = max(range(len(frames)), key=lambda i: (abs(frames[i]["compression_m"]),
                                                  abs(frames[i]["yaw_rad"])))
    fig = plt.figure(figsize=(7.2, 6.2), layout="constrained")
    fig.get_layout_engine().set(h_pad=.04, hspace=.06, w_pad=.06, wspace=.08)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.65, 1])
    for j, index in enumerate((0, peak)):
        ax = fig.add_subplot(grid[0, j], projection="3d")
        draw(ax, index)
        ax.set_title(f"({chr(97+j)})", loc="left", pad=2)
    force_ax, energy_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for model, color in (("sano", "#157b80"), ("kirchhoff", "#c76b34")):
        group = [c for c in cases if c["model"] == model]
        if not group:
            continue
        force = np.sum([[f["back_support_force_n"] for f in c["frames"]] for c in group], axis=0) @ axis
        energy = np.sum([[f["energy_j"] for f in c["frames"]] for c in group], axis=0) * 1000
        force_ax.plot(progress, force, color=color, label=model.capitalize(), linewidth=1.3)
        energy_ax.plot(progress, energy, color=color, label=model.capitalize(), linewidth=1.3)
    force_ax.set_ylabel("$F_x$ (N)")
    energy_ax.set_ylabel("$U$ (mJ)")
    for label, ax in zip(("(c)", "(d)"), (force_ax, energy_ax)):
        ax.set_xlabel("Path parameter, $s$")
        ax.set_title(label, loc="left", pad=5)
        ax.legend()
    if smoke:
        fig.text(.985, .99, "SMOKE", ha="right", va="top", color="#777777")
    preview = path.parent / "preview.png"
    svg = path.parent / "preview.svg"
    fig.savefig(preview, dpi=300, metadata={"Description": description})
    fig.savefig(svg, metadata={"Description": description})
    plt.close(fig)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    if smoke:
        fig.text(.025, .975, "SMOKE", ha="left", va="top", color="#777777")
    def update(index):
        draw(ax, index, show_axes=False)
    movie = FuncAnimation(fig, update, frames=range(len(frames)), interval=200, repeat=True)
    gif = path.parent / "simulation.gif"
    movie.save(gif, writer=PillowWriter(fps=5), dpi=100)
    plt.close(fig)
    return {"preview": str(preview), "preview_svg": str(svg), "animation": str(gif)}


def self_check():
    frame = {"nodes_m": [[0, 0, 0], [1, 0, 0]], "width_directors": [[0, 1, 0]],
             "plate_rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
             "plate_translation_m": [0, 0, 2]}
    faces = ribbon_faces(frame, 0.2)
    assert faces.shape == (1, 4, 3)
    assert np.allclose(faces[0, 3] - faces[0, 0], [0, 0.2, 0])
    assert np.allclose(move_plate(np.array([[2, 0, 0]]), np.array([1, 0, 0]), frame), [[1, 1, 2]])
    try:
        ribbon_faces({**frame, "width_directors": [[0, 0, 0]]}, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero director accepted")
    print("render geometry checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        print(json.dumps(render(args.path), indent=2))

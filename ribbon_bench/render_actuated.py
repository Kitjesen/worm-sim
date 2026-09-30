"""Render solved cable-driven equilibria; slack lines show routes, not sag shapes."""
import argparse
import json
from pathlib import Path

from render import ribbon_faces, plate_points, move_plate, paper_style, paper_axis, RIBBON_COLORS, PLATE_COLOR
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def render(path, output=None, smoke=False):
    paper_style()
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    meta, cases, actuation = data["metadata"], data["cases"], data["actuation_frames"]
    smoke = smoke or bool(meta.get("renderer_smoke"))
    model = "sano" if any(c["model"] == "sano" for c in cases) else "kirchhoff"
    strips = sorted((c for c in cases if c["model"] == model), key=lambda c: c["strip"])
    if len(strips) != 8 or len({c["strip"] for c in strips}) != 8:
        raise ValueError("Eight independently solved strips are required")
    frames = strips[0]["frames"]
    if len(frames) < 2 or len(actuation) != len(frames):
        raise ValueError("Incomplete or unsynchronized actuation/strip frames")
    schedule = np.array([[f["progress"], f["compression_m"], f["yaw_rad"],
                          *np.asarray(f["plate_rotation"]).ravel(), *f["plate_translation_m"]]
                         for f in frames], dtype=float)
    if (schedule.shape != (len(frames), 15) or not np.isfinite(schedule).all()
            or not np.all(np.diff(schedule[:, 0]) > 0)):
        raise ValueError("A finite, increasing saved loading schedule is required")
    for case in strips:
        actual = np.array([[f["progress"], f["compression_m"], f["yaw_rad"],
                            *np.asarray(f["plate_rotation"]).ravel(), *f["plate_translation_m"]]
                           for f in case["frames"]], dtype=float)
        if actual.shape != schedule.shape or not np.allclose(actual, schedule, rtol=0, atol=1e-10):
            raise ValueError("Strip loading paths or plate poses differ")
    routes = np.asarray([f["routes_m"] for f in actuation], dtype=float)
    tensions = np.asarray([f["tensions_n"] for f in actuation], dtype=float)
    lengths = np.asarray([f["rest_lengths_m"] for f in actuation], dtype=float)
    gaps = np.asarray([f["plate_gap_m"] for f in actuation], dtype=float)
    contacts = np.asarray([f["contact_force_n"] for f in actuation], dtype=float)
    if (routes.shape != (len(frames), 4, 2, 3) or tensions.shape != (len(frames), 4)
            or lengths.shape != tensions.shape or gaps.shape != (len(frames),)
            or contacts.shape != gaps.shape or np.any(tensions < -1e-10) or np.any(lengths <= 0)
            or not all(np.isfinite(a).all() for a in (routes, tensions, lengths, gaps, contacts))):
        raise ValueError("Invalid routes, tensions, rest lengths, gaps or contact forces")
    if "progress" in actuation[0] and not np.allclose(
            [f["progress"] for f in actuation], schedule[:, 0], rtol=0, atol=1e-10):
        raise ValueError("Cable and strip progress values differ")
    width, radius = float(meta["width_m"]), float(meta["plate_radius_m"])
    front, back = np.array(meta["front_plate_center_m"]), np.array(meta["back_plate_center_world_m"])
    if width <= 0 or radius <= 0:
        raise ValueError("Positive ribbon width and plate radius are required")
    disks = [plate_points(c, radius, np.array([1., 0, 0])) for c in (front, back)]
    surfaces = [[ribbon_faces(f, width) * 1000 for f in c["frames"]] for c in strips]
    points = np.concatenate([s.reshape(-1, 3) for strip in surfaces for s in strip]
                            + [routes.reshape(-1, 3) * 1000, disks[0] * 1000]
                            + [move_plate(disks[1], back, f) * 1000 for f in frames])
    if not np.isfinite(points).all():
        raise ValueError("Non-finite rendered geometry")
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(hi - lo, 1) * .1
    lo, hi = lo - padding, hi + padding
    progress = schedule[:, 0]
    cable_colors = ["#ad5b4f", "#517f9b", "#877293", "#9b823a"]
    description = json.dumps({"loading": meta.get("loading", "Cable driven; 2 DOF guided plate; quasi-static"),
                              "cables": "Dashed lines are slack routes, not sag shapes",
                              "contact": "Frictionless plate stop", "renderer_smoke": smoke,
                              "path_status": meta.get("path_status"),
                              "unloading_converged": meta.get("unloading_converged"),
                              "target_yaw_deg": meta.get("target_yaw_deg")}, ensure_ascii=False)

    def draw(ax, index, show_axes=True):
        ax.clear()
        f = frames[index]
        for j in range(len(strips)):
            ax.add_collection3d(Poly3DCollection(surfaces[j][index], facecolors=RIBBON_COLORS[j % 4],
                edgecolors="#607889", linewidths=.18, alpha=.88))
        for j, disk in enumerate((disks[0], move_plate(disks[1], back, f))):
            ax.add_collection3d(Poly3DCollection([disk * 1000], facecolors=PLATE_COLOR,
                                                 edgecolors="#6b747b", linewidths=.65, alpha=.18))
        for j, route in enumerate(routes[index] * 1000):
            ax.plot(*route.T, color=cable_colors[j], linewidth=1.5,
                    linestyle="--" if tensions[index, j] <= 1e-8 else "-")
            ax.scatter(*route.T, color=cable_colors[j], s=8, depthshade=False)
        paper_axis(ax, lo, hi, show_axes=show_axes)

    peak = max(range(len(frames)), key=lambda i: (abs(frames[i]["yaw_rad"]),
                                                 abs(frames[i]["compression_m"]), i))
    fig = plt.figure(figsize=(7.6, 6.2), layout="constrained")
    fig.get_layout_engine().set(h_pad=.04, hspace=.06, w_pad=.06, wspace=.14)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.65, 1])
    for j, index in enumerate((0, peak)):
        ax = fig.add_subplot(grid[0, j], projection="3d")
        draw(ax, index)
        ax.set_title(f"({chr(97+j)})", loc="left", pad=2)
    pose_ax, tension_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    yaw_ax = pose_ax.twinx()
    compression_line, = pose_ax.plot(progress, schedule[:, 1] * 1000, color="#517f9b", linewidth=1.3, label=r"$\delta$")
    yaw_line, = yaw_ax.plot(progress, np.degrees(schedule[:, 2]), color="#ad5b4f", linewidth=1.3, label=r"$\psi$")
    pose_ax.set_ylabel(r"$\delta$ (mm)", color="#517f9b")
    yaw_ax.set_ylabel(r"$\psi$ (deg)", color="#ad5b4f")
    yaw_ax.spines["right"].set_visible(True)
    pose_ax.legend(handles=[compression_line, yaw_line], loc="best")
    pose_ax.set_title("(c)", loc="left", pad=5)
    for j, color in enumerate(cable_colors):
        tension_ax.plot(progress, tensions[:, j], color=color, linewidth=1.3, label=f"$T_{j+1}$")
    tension_ax.set_ylabel("$T$ (N)")
    tension_ax.set_title("(d)", loc="left", pad=5)
    tension_ax.legend(ncols=2)
    for ax in (pose_ax, tension_ax):
        ax.set_xlabel("Path parameter, $s$")
    if smoke:
        fig.text(.985, .99, "SMOKE", ha="right", va="top", color="#777777")
    output = Path(output).resolve() if output else path.parent
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "preview.png", dpi=300, metadata={"Description": description})
    fig.savefig(output / "preview.svg", metadata={"Description": description})
    plt.close(fig)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    if smoke:
        fig.text(.025, .975, "SMOKE", ha="left", va="top", color="#777777")
    def update(index):
        draw(ax, index, show_axes=False)
    movie = FuncAnimation(fig, update, frames=range(len(frames)), interval=200, repeat=True)
    movie.save(output / "simulation.gif", writer=PillowWriter(fps=5), dpi=100)
    plt.close(fig)
    return {"preview": str(output / "preview.png"), "preview_svg": str(output / "preview.svg"),
            "animation": str(output / "simulation.gif")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="label synthetic renderer test data prominently")
    args = parser.parse_args()
    try:
        print(json.dumps(render(args.path, args.output, args.smoke), indent=2))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

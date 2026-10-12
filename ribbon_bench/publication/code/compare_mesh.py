"""Compare three completed ribbon meshes using only their saved equilibrium states."""
import argparse
import json
from pathlib import Path

from render import (ribbon_faces, plate_points, move_plate, paper_style, paper_axis,
                    RIBBON_COLORS, PLATE_COLOR)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def load_results(paths, smoke=False):
    if len(paths) != 3:
        raise ValueError("Provide three results.json paths in 33, 65, 129-node order")
    results = []
    for path in map(Path, paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta, cases = data["metadata"], data["cases"]
            if not np.isfinite(meta["wall_seconds"]) or meta["wall_seconds"] <= 0:
                raise ValueError("missing or invalid completed solve time")
            keys = [(c["model"], c["strip"]) for c in cases]
            if not keys or len(set(keys)) != len(keys):
                raise ValueError("empty or duplicate strip/model cases")
            schedule = np.array([[f["progress"], f["compression_m"], f["yaw_rad"],
                                  *np.asarray(f["plate_rotation"]).ravel(), *f["plate_translation_m"]]
                                 for f in cases[0]["frames"]], dtype=float)
            if (schedule.shape[0] < 25 or schedule.shape[1:] != (15,)
                    or not np.isfinite(schedule).all() or not np.all(np.diff(schedule[:, 0]) > 0)
                    or not np.allclose(schedule[[0, -1], 0], [0, 4])
                    or not np.allclose(schedule[0, 1:], schedule[-1, 1:])):
                raise ValueError("need a complete 0-to-4 load/unload cycle with >=25 saved states")
            for case in cases:
                actual = np.array([[f["progress"], f["compression_m"], f["yaw_rad"],
                                    *np.asarray(f["plate_rotation"]).ravel(), *f["plate_translation_m"]]
                                   for f in case["frames"]], dtype=float)
                if actual.shape != schedule.shape or not np.allclose(actual, schedule, rtol=0, atol=1e-10):
                    raise ValueError("strips have different or incomplete loading paths")
                for frame in case["frames"]:
                    if len(frame["nodes_m"]) != meta["nodes"]:
                        raise ValueError("node count differs from metadata")
                    ribbon_faces(frame, meta["width_m"])
                    residuals = [frame["free_force_residual_n"], frame["free_moment_residual_nm"]]
                    if (not np.isfinite(residuals).all() or min(residuals) < 0
                            or max(residuals) > 1e-6 or not np.isfinite(frame["energy_j"])):
                        raise ValueError("saved state fails the equilibrium/finite-energy checks")
                forces = np.array([f["back_support_force_n"] for f in case["frames"]])
                if forces.shape != (len(schedule), 3) or not np.isfinite(forces).all():
                    raise ValueError("missing or non-finite support force")
            if results:
                other = results[0]
                if set(keys) != {(c["model"], c["strip"]) for c in other["cases"]}:
                    raise ValueError("strip/model sets differ between meshes")
                if schedule.shape != other["schedule"].shape or not np.allclose(
                        schedule, other["schedule"], rtol=0, atol=1e-10):
                    raise ValueError("loading paths differ between meshes")
                for key in ("width_m", "thickness_m", "youngs_modulus_pa", "poisson_ratio",
                            "plate_radius_m", "front_plate_center_m", "back_plate_center_world_m"):
                    if not np.allclose(meta[key], other["metadata"][key], rtol=0, atol=1e-12):
                        raise ValueError(f"physical input differs: {key}")
                for key in ("parameters_sha256", "urdf_sha256", "upstream_commit", "workers", "initial_guess"):
                    if meta.get(key) != other["metadata"].get(key):
                        raise ValueError(f"input provenance differs: {key}")
            data["schedule"] = schedule
            results.append(data)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Incomplete or incompatible result {path}: {exc}") from exc
    if not smoke and [d["metadata"]["nodes"] for d in results] != [33, 65, 129]:
        raise ValueError("Expected 33, 65, 129 nodes in order; --smoke is only for renderer checks")
    if not smoke:
        model = "sano" if any(c["model"] == "sano" for c in results[0]["cases"]) else "kirchhoff"
        if {c["strip"] for c in results[0]["cases"] if c["model"] == model} != set(range(8)):
            raise ValueError("Incomplete comparison: all eight strips (0 through 7) are required")
    return results


def compare(paths, output=None, smoke=False):
    results = load_results(paths, smoke)
    output = Path(output) if output else Path(paths[0]).resolve().parent.parent / ("smoke" if smoke else "")
    model = "sano" if any(c["model"] == "sano" for c in results[0]["cases"]) else "kirchhoff"
    groups = [sorted((c for c in data["cases"] if c["model"] == model), key=lambda c: c["strip"])
              for data in results]
    if not 1 <= len(groups[0]) <= 8:
        raise ValueError("Expected one to eight independent ribbons per mesh")
    meta = results[0]["metadata"]
    front, back = np.array(meta["front_plate_center_m"]), np.array(meta["back_plate_center_world_m"])
    disks = [plate_points(c, meta["plate_radius_m"], np.array([1., 0, 0])) for c in (front, back)]
    frames = groups[0][0]["frames"]
    surfaces = [[[ribbon_faces(f, meta["width_m"]) * 1000 for f in c["frames"]] for c in group]
                for group in groups]
    points = np.concatenate([face.reshape(-1, 3) for mesh in surfaces for strip in mesh for face in strip]
                            + [disks[0] * 1000]
                            + [move_plate(disks[1], back, f) * 1000 for f in frames])
    if not np.isfinite(points).all():
        raise ValueError("Non-finite geometry")
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(hi - lo, 1) * 0.09
    lo, hi = lo - padding, hi + padding
    progress = results[0]["schedule"][:, 0]
    paper_style()

    def draw(axes, index, show_axes):
        frame = frames[index]
        for j, ax in enumerate(axes):
            ax.clear()
            for k, case in enumerate(groups[j]):
                ax.add_collection3d(Poly3DCollection(surfaces[j][k][index],
                    facecolors=RIBBON_COLORS[case["strip"] % len(RIBBON_COLORS)],
                    edgecolors="#354b5c", linewidths=.2, alpha=.94))
            for disk in (disks[0], move_plate(disks[1], back, frame)):
                ax.add_collection3d(Poly3DCollection([disk * 1000], facecolors=PLATE_COLOR,
                                                      edgecolors="#737b82", linewidths=.55, alpha=.2))
            paper_axis(ax, lo, hi, show_axes=show_axes)
            ax.text2D(.025, .94, f"({chr(97+j)}) $N={results[j]['metadata']['nodes']}$",
                      transform=ax.transAxes, fontsize=10, ha="left", va="top")

    def watermark(fig):
        if smoke:
            fig.text(.5, .5, "SMOKE TEST", ha="center", va="center", rotation=20,
                     fontsize=28, weight="bold", color="#a33b32", alpha=.45)

    output.mkdir(parents=True, exist_ok=True)
    peak = max(range(len(frames)), key=lambda i: (abs(frames[i]["compression_m"]), abs(frames[i]["yaw_rad"])))
    fig = plt.figure(figsize=(9.6, 5.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.1, top=.995, hspace=.4, wspace=.04)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.65, .85])
    axes = [fig.add_subplot(grid[0, j], projection="3d") for j in range(3)]
    draw(axes, peak, True)
    force_ax = fig.add_subplot(grid[1, :])
    for j, (group, color, style) in enumerate(zip(groups, ("#344d65", "#738b9d", "#a1acb5"), ("-", "--", ":"))):
        force = np.sum([[f["back_support_force_n"][0] for f in c["frames"]] for c in group], axis=0)
        force_ax.plot(progress, force, color=color, linestyle=style, linewidth=1.25,
                      label=f"$N={results[j]['metadata']['nodes']}$")
    force_ax.set(xlabel=r"Path parameter, $s$", ylabel=r"$F_x\;(\mathrm{N})$", xlim=(0, 4))
    force_ax.spines[["top", "right"]].set_visible(False)
    force_ax.tick_params(direction="out", width=.6, length=3)
    force_ax.legend(loc="lower center", ncols=3, frameon=False, handlelength=2.5)
    watermark(fig)
    fig.savefig(output / "preview.png", dpi=300, facecolor="white")
    fig.savefig(output / "preview.svg", facecolor="white")
    plt.close(fig)

    fig = plt.figure(figsize=(9.6, 3.25))
    fig.subplots_adjust(left=.008, right=.992, bottom=.005, top=.995, wspace=.015)
    axes = [fig.add_subplot(1, 3, j+1, projection="3d") for j in range(3)]
    watermark(fig)
    movie = FuncAnimation(fig, lambda index: draw(axes, index, False),
                          frames=range(len(frames)), interval=200, repeat=True)
    movie.save(output / "comparison.gif", writer=PillowWriter(fps=5), dpi=140)
    plt.close(fig)
    return {"preview": str((output / "preview.png").resolve()),
            "vector": str((output / "preview.svg").resolve()),
            "animation": str((output / "comparison.gif").resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="label a renderer test; permit repeated resolutions")
    parser.add_argument("--check-only", action="store_true", help="validate completed inputs without rendering")
    args = parser.parse_args()
    try:
        if args.check_only:
            load_results(args.paths, args.smoke)
            print("All three completed meshes have matching load paths and strip sets")
        else:
            print(json.dumps(compare(args.paths, args.output, args.smoke), indent=2))
    except ValueError as exc:
        parser.error(str(exc))

"""Replay recorded V6 qpos: CAD navigation view and close-up, no physics edits.

The parabolic strips from worm_v6 are visualization only, not Sano elasticity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from worm_v6 import inject_strips, STRIP_RGBA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_indices(count, frames):
    if count < 2 or frames < 2:
        raise ValueError("At least two source and displayed frames are required")
    return np.unique(np.rint(np.linspace(0, count - 1, min(count, frames))).astype(int))


def window_indices(times, frames, start=None, end=None):
    times = np.asarray(times)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Source times must be finite and strictly increasing")
    start = float(times[0]) if start is None else start
    end = float(times[-1]) if end is None else end
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise ValueError("Provide a finite ordered time window")
    selected = np.flatnonzero((times >= start) & (times <= end))
    return selected[sample_indices(len(selected), frames)]


def line(scene, a, b, rgba, radius=0.012):
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Render geometry buffer exhausted")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                       np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        np.r_[a[:2], 0.008], np.r_[b[:2], 0.008])
    scene.ngeom += 1


def plots(output, arrays, summary, comparison=None, prefix="robot_tracking", route=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    path, com = arrays["path_m"], arrays["com_m"]
    ax = axes[0]
    for obstacle in summary["scene"]["obstacles"]:
        center, half = np.array(obstacle["center_m"]), np.array(obstacle["half_size_m"])
        ax.add_patch(Rectangle(center-half, *(2*half), color="#ad6146", alpha=0.8))
    if route and "point_stroke_id" in route:
        labels = np.asarray(route["point_stroke_id"])
        arcs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        actual_labels = labels[np.clip(np.searchsorted(arcs, arrays["progress_m"]), 0, len(labels)-1)]
        segments = np.stack((path[:-1], path[1:]), axis=1)
        ax.add_collection(LineCollection(segments, colors=["#c7c7c7" if label < 0 else "#2477a2" for label in labels[1:]],
                                         linewidths=.9, linestyles="dashed"))
        for label in [-1, *range(len(route["stroke_names"]))]:
            actual = com[:, :2].copy(); actual[actual_labels != label] = np.nan
            ax.plot(actual[:,0], actual[:,1], color="#d7d7d7" if label<0 else "#d17c36", lw=.7 if label<0 else 1.5)
        ax.plot([], [], "--", color="#2477a2", label="Target strokes")
        ax.plot([], [], color="#d17c36", label="Measured COM")
    else:
        ax.plot(path[:, 0], path[:, 1], "--", color="#2477a2", lw=1.7, label="A* path")
        ax.plot(com[:, 0], com[:, 1], color="#d17c36", lw=1.6, label="Measured COM")
    if comparison is not None:
        other = comparison["com_m"]
        ax.plot(other[:, 0], other[:, 1], ":", color="#914452", lw=1.5, label="Open loop")
        ax.scatter(*other[-1, :2], marker="x", s=36, color="#914452", zorder=5)
    ax.scatter(*path[0], marker="o", s=26, color="#2477a2", zorder=5)
    ax.scatter(*path[-1], marker="*", s=75, color="#2477a2", zorder=5)
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].plot(arrays["time_s"], 100*arrays["cross_track_m"], color="#d17c36", lw=1.4)
    axes[1].axhline(100*summary["cross_track_rms_m"], color="#2477a2", ls="--", lw=1,
                    label="RMS")
    axes[1].set(xlabel="Time (s)", ylabel="Distance to target path (cm)", ylim=(0, None))
    axes[1].legend(frameon=False, fontsize=9)
    for extension in ("png", "svg"):
        name = "tracking_metrics" if prefix == "robot_tracking" else f"{prefix}_metrics"
        fig.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(fig)
    if route and "point_stroke_id" in route:
        fig, ax = plt.subplots(figsize=(15, 3.6), constrained_layout=True)
        ax.add_collection(LineCollection(segments, colors=["#dedede" if label < 0 else "#2477a2" for label in labels[1:]],
                                         linewidths=.9, linestyles="dashed"))
        for label in [-1, *range(len(route["stroke_names"]))]:
            actual = com[:, :2].copy(); actual[actual_labels != label] = np.nan
            ax.plot(actual[:,0], actual[:,1], color="#d0d0d0" if label<0 else "#d17c36", lw=.7 if label<0 else 1.4)
        ax.plot([], [], "--", color="#2477a2", label="Target strokes")
        ax.plot([], [], color="#d17c36", label="Measured COM: strokes")
        ax.plot([], [], color="#d0d0d0", label="Continuous transfers")
        ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25))
        for extension in ("png", "svg"):
            fig.savefig(output/f"{prefix}_letters.{extension}", dpi=200)
        plt.close(fig)


def render(args):
    started = time.perf_counter()
    output = Path(args.run).resolve()
    source = [output/name for name in ("trajectory.npz", "summary.json", "scene.xml")]
    route_path = output / "route.json"
    if route_path.exists():
        source.append(route_path)
    before = {p.name: digest(p) for p in source}
    summary = json.loads(source[1].read_text(encoding="utf-8"))
    route = json.loads(route_path.read_text(encoding="utf-8")) if route_path.exists() else None
    with np.load(source[0]) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    comparison, compare_hash = None, None
    if args.compare:
        compare_file = Path(args.compare).resolve()/"trajectory.npz"
        compare_hash = digest(compare_file)
        with np.load(compare_file) as archive:
            comparison = {"com_m": archive["com_m"].copy()}
    indices = window_indices(arrays["time_s"], args.frames, args.start_time, args.end_time)
    selected_times = arrays["time_s"][indices]
    physical_duration = float(selected_times[-1]-selected_times[0])
    motion_playback = args.playback if args.playback is not None else physical_duration
    speed_label = f"{physical_duration/motion_playback:.2f}x"
    stroke_labels = np.asarray(route["point_stroke_id"]) if route and "point_stroke_id" in route else None
    path_arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(arrays["path_m"], axis=0), axis=1))]
    scene_xml = ET.fromstring(source[2].read_text(encoding="utf-8"))
    mesh_dir = Path(__file__).resolve().parents[2]/"meshes"
    scene_xml.find("compiler").set("meshdir", str(mesh_dir))
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene_xml, encoding="unicode"))
    data = mujoco.MjData(model)
    if arrays["qpos"].shape[1] != model.nq:
        raise ValueError("Recorded qpos is incompatible with saved scene")
    # Only change render appearance. No mj_step, coordinate offsets or ctrl edits.
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        # MuJoCo planes collide infinitely; enlarge only their visible patch for long routes.
        model.geom_size[floor, :2] = 200.
        material = model.geom_matid[floor]
        model.mat_texid[material] = -1
        model.mat_rgba[material] = [1, 1, 1, 1]
        model.mat_emission[material] = 0.4
    model.light_active[:] = False
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.45
    model.vis.headlight.specular[:] = 0
    STRIP_RGBA[:] = [0.30, 0.42, 0.49, 1]
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    close_option = mujoco.MjvOption()
    close_option.geomgroup[3:] = 0
    for i in range(len(summary["scene"]["obstacles"])):
        obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
        model.geom_group[obstacle] = 5
    option.geomgroup[5] = 1
    get_body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    slide_pairs = [(get_body("base_link"), get_body("back1_Link"))]
    slide_pairs += [(get_body(f"front{i}_Link"), get_body(f"back{i}_Link")) for i in range(2, 7)]
    spacings = [0.151] + [0.1175]*5
    all_xy = np.vstack((arrays["path_m"], arrays["com_m"][:, :2]))
    for obstacle in summary["scene"]["obstacles"]:
        center, half = np.asarray(obstacle["center_m"]), np.asarray(obstacle["half_size_m"])
        all_xy = np.vstack((all_xy, center-half, center+half))
    low, high = all_xy.min(0)-0.8, all_xy.max(0)+0.8
    center = (low+high)/2
    global_camera = mujoco.MjvCamera()
    global_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    global_camera.lookat[:] = [*center, 0]
    global_camera.azimuth, global_camera.elevation = 90, -90
    half_fovy = math.radians(float(model.vis.global_.fovy)/2)
    overview_width = 1360 if args.overview_only else 800
    global_camera.distance = max((high-low)[1], (high-low)[0]/(overview_width/500))/(2*math.tan(half_fovy))
    close_camera = mujoco.MjvCamera()
    close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    close_camera.distance = 1.6
    close_camera.azimuth, close_camera.elevation = 135, -35
    width, height = 1360, 500
    renderer = mujoco.Renderer(model, height=height, width=overview_width, max_geom=6000)
    close_renderer = None if args.overview_only else mujoco.Renderer(model, height=height, width=560, max_geom=6000)
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 16)
    path_radius = max(.012, float((high-low)[0])/1200)
    images, com_error, body_error = [], 0., 0.
    try:
        for number, index in enumerate(indices):
            data.qpos[:] = arrays["qpos"][index]
            data.qvel[:] = arrays["qvel"][index]
            data.time = float(arrays["time_s"][index])
            mujoco.mj_forward(model, data)
            com_error = max(com_error, float(np.max(np.abs(data.subtree_com[int(arrays["head_id"])]-arrays["com_m"][index]))))
            body_error = max(body_error, float(np.max(np.abs(data.xpos-arrays["body_pos_m"][index]))))
            renderer.update_scene(data, global_camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            inject_strips(renderer.scene, data, slide_pairs, spacings)
            for segment, (a, b) in enumerate(zip(arrays["path_m"][:-1], arrays["path_m"][1:])):
                distance = float(np.linalg.norm(b-a))
                colour = [0.65, 0.65, 0.65, 0.45] if stroke_labels is not None and stroke_labels[segment+1] < 0 else [0.14, 0.43, 0.64, 1]
                for s in np.arange(0, distance, 0.15):
                    line(renderer.scene, a+(b-a)*(s/distance),
                         a+(b-a)*(min(s+0.08, distance)/distance), colour, path_radius)
            trail_indices = np.unique(np.linspace(0, index, min(index+1, 1500), dtype=int))
            trail = arrays["com_m"][trail_indices]
            for k, (a, b) in enumerate(zip(trail[:-1], trail[1:])):
                stroke = int(np.clip(np.searchsorted(path_arcs, arrays["progress_m"][trail_indices[k+1]]), 0, len(path_arcs)-1))
                colour = [0.7, 0.7, 0.7, 0.45] if stroke_labels is not None and stroke_labels[stroke] < 0 else [0.81, 0.43, 0.17, 1]
                line(renderer.scene, a, b, colour, path_radius*.75)
            overview = Image.fromarray(renderer.render())
            image = Image.new("RGB", (width, height), "white")
            image.paste(overview, (0, 0))
            if close_renderer is not None:
                close_camera.lookat[:] = data.subtree_com[int(arrays["head_id"])]+[0, 0, 0.02]
                close_renderer.update_scene(data, close_camera, scene_option=close_option)
                close_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                inject_strips(close_renderer.scene, data, slide_pairs, spacings)
                image.paste(Image.fromarray(close_renderer.render()), (800, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((11, 10, 225, 57), fill="white")
            draw.line((21, 23, 51, 23), fill="#2477a2", width=3)
            draw.text((58, 13), "Target strokes" if stroke_labels is not None else "A* path", fill="#333333", font=font)
            draw.line((21, 43, 51, 43), fill="#d17c36", width=3)
            draw.text((58, 36), "Measured COM", fill="#333333", font=font)
            scale = height/(2*global_camera.distance*math.tan(half_fovy))
            draw.line((23, height-30, 23+round(scale), height-30), fill="#555555", width=2)
            draw.text((25, height-50), "1 m", fill="#333333", font=font)
            draw.text((1140, 13), f"t = {data.time:.1f} s", fill="#333333", font=font)
            draw.text((1140, 36), f"Playback {speed_label}", fill="#333333", font=font)
            if number == len(indices)-1:
                image.save(output/f"{args.prefix}.png")
            images.append(image.quantize(colors=128))
            if number % 20 == 0:
                print(f"Rendered {number+1}/{len(indices)} real qpos frames", flush=True)
    finally:
        renderer.close()
        if close_renderer is not None:
            close_renderer.close()
    if com_error > 1e-9 or body_error > 1e-9:
        raise AssertionError(f"Replay differs from saved physics: COM={com_error}, bodies={body_error}")
    frame_times_ms = np.rint((selected_times-selected_times[0])/physical_duration*motion_playback*100)*10
    durations = np.diff(frame_times_ms).astype(int).tolist()+[1200]
    if min(durations) < 10:
        raise ValueError("Time window has too many GIF frames for the requested playback")
    gif_path = output/f"{args.prefix}.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    plots(output, arrays, summary, comparison, args.prefix, route)
    after = {p.name: digest(p) for p in source}
    assert before == after, "Rendering altered the original rollout"
    if args.compare:
        assert compare_hash == digest(compare_file), "Plotting altered the comparison rollout"
    with Image.open(gif_path) as gif:
        count, playback = gif.n_frames, 0
        for i in range(count):
            gif.seek(i)
            playback += gif.info["duration"]
    assert count == len(indices) and tuple(gif.size) == (width, height)
    manifest = {"source_sha256": before, "renderer_sha256": digest(Path(__file__)),
                "source_unchanged": True, "replay": "Actual saved qpos/qvel + mj_forward; no translation of old animation",
                "strip_model": "worm_v6.inject_strips parabolic visual geometry only; not Sano",
                "visual_scope": "Global route overview only" if args.overview_only else "Global view includes obstacles; CAD close-up hides obstacle visuals for clarity",
                "source_frame_count": len(arrays["time_s"]), "source_simulation_time_s": float(arrays["time_s"][-1]),
                "selected_indices": indices.tolist(), "gif_frame_count": count,
                "selected_start_time_s": float(selected_times[0]),
                "selected_end_time_s": float(selected_times[-1]),
                "selected_physical_duration_s": physical_duration,
                "gif_size_px": [width, height], "gif_playback_s": playback/1000,
                "gif_motion_playback_s": sum(durations[:-1])/1000,
                "playback_speedup": physical_duration/(sum(durations[:-1])/1000),
                "final_frame_hold_s": 1.2, "max_replay_com_error_m": com_error,
                "max_replay_body_error_m": body_error, "render_wall_time_s": time.perf_counter()-started,
                "comparison_npz_sha256": compare_hash}
    manifest_name = "render_manifest.json" if args.prefix == "robot_tracking" else f"{args.prefix}_render_manifest.json"
    (output/manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_indices"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--playback", type=float, default=None, help="Motion playback seconds; default is true 1x time")
    parser.add_argument("--start-time", type=float, help="First recorded time to include, in seconds")
    parser.add_argument("--end-time", type=float, help="Last recorded time to include, in seconds")
    parser.add_argument("--prefix", default="robot_tracking", help="Output filename prefix")
    parser.add_argument("--overview-only", action="store_true", help="Show route progress without a speeded-up joint close-up")
    parser.add_argument("--compare", help="Optional open-loop run for the static path plot")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert np.array_equal(sample_indices(11, 4), [0, 3, 7, 10])
        assert np.array_equal(sample_indices(3, 100), [0, 1, 2])
        assert np.array_equal(window_indices(np.arange(11.), 4), sample_indices(11, 4))
        assert np.array_equal(window_indices(np.arange(11.), 100, 2, 8), np.arange(2, 9))
        assert np.array_equal(window_indices(np.arange(11.), 3, 2.1, 7.9), [3, 5, 7])
        for check in (lambda: sample_indices(1, 20),
                      lambda: window_indices(np.arange(11.), 20, 20, 30),
                      lambda: window_indices(np.arange(11.), 20, 5, 5),
                      lambda: window_indices(np.arange(11.), 20, 8, 2)):
            try:
                check()
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid frame selection accepted")
        print("Frame sampling, inclusive window endpoints, and invalid-input checks passed")
    else:
        if not args.run or args.frames < 2 or (args.playback is not None and (not math.isfinite(args.playback) or args.playback <= 0)):
            parser.error("Provide a run path, frames >= 2, and positive finite playback")
        if not args.prefix or not all(c.isalnum() or c in "-_" for c in args.prefix):
            parser.error("Prefix must contain only letters, digits, '-' and '_'")
        render(args)

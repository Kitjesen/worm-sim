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
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from worm_v6 import inject_strips, STRIP_RGBA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_indices(count, frames):
    if count < 2 or frames < 2:
        raise ValueError("At least two source and displayed frames are required")
    return np.unique(np.rint(np.linspace(0, count - 1, min(count, frames))).astype(int))


def line(scene, a, b, rgba, radius=0.012):
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Render geometry buffer exhausted")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                       np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        np.r_[a[:2], 0.008], np.r_[b[:2], 0.008])
    scene.ngeom += 1


def plots(output, arrays, summary, comparison=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    path, com = arrays["path_m"], arrays["com_m"]
    ax = axes[0]
    for obstacle in summary["scene"]["obstacles"]:
        center, half = np.array(obstacle["center_m"]), np.array(obstacle["half_size_m"])
        ax.add_patch(Rectangle(center-half, *(2*half), color="#ad6146", alpha=0.8))
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
    axes[1].set(xlabel="Time (s)", ylabel="Distance to A* path (cm)", ylim=(0, None))
    axes[1].legend(frameon=False, fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / f"tracking_metrics.{extension}", dpi=180)
    plt.close(fig)


def render(args):
    started = time.perf_counter()
    output = Path(args.run).resolve()
    source = [output/name for name in ("trajectory.npz", "summary.json", "scene.xml")]
    before = {p.name: digest(p) for p in source}
    summary = json.loads(source[1].read_text(encoding="utf-8"))
    with np.load(source[0]) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    comparison, compare_hash = None, None
    if args.compare:
        compare_file = Path(args.compare).resolve()/"trajectory.npz"
        compare_hash = digest(compare_file)
        with np.load(compare_file) as archive:
            comparison = {"com_m": archive["com_m"].copy()}
    indices = sample_indices(len(arrays["time_s"]), args.frames)
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
    low, high = all_xy.min(0)-0.8, all_xy.max(0)+0.8
    center = (low+high)/2
    global_camera = mujoco.MjvCamera()
    global_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    global_camera.lookat[:] = [*center, 0]
    global_camera.azimuth, global_camera.elevation = 90, -90
    half_fovy = math.radians(float(model.vis.global_.fovy)/2)
    global_camera.distance = max((high-low)[1], (high-low)[0]/(800/500))/(2*math.tan(half_fovy))
    close_camera = mujoco.MjvCamera()
    close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    close_camera.distance = 1.6
    close_camera.azimuth, close_camera.elevation = 135, -35
    width, height = 1360, 500
    renderer = mujoco.Renderer(model, height=height, width=800, max_geom=6000)
    close_renderer = mujoco.Renderer(model, height=height, width=560, max_geom=6000)
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
            for a, b in zip(arrays["path_m"][:-1], arrays["path_m"][1:]):
                distance = float(np.linalg.norm(b-a))
                for s in np.arange(0, distance, 0.15):
                    line(renderer.scene, a+(b-a)*(s/distance),
                         a+(b-a)*(min(s+0.08, distance)/distance), [0.14, 0.43, 0.64, 1])
            trail = arrays["com_m"][:index+1:3]
            if len(trail) and not np.array_equal(trail[-1], arrays["com_m"][index]):
                trail = np.vstack((trail, arrays["com_m"][index]))
            for a, b in zip(trail[:-1], trail[1:]):
                line(renderer.scene, a, b, [0.81, 0.43, 0.17, 1], 0.009)
            overview = Image.fromarray(renderer.render())
            close_camera.lookat[:] = data.subtree_com[int(arrays["head_id"])]+[0, 0, 0.02]
            close_renderer.update_scene(data, close_camera, scene_option=close_option)
            close_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            inject_strips(close_renderer.scene, data, slide_pairs, spacings)
            closeup = Image.fromarray(close_renderer.render())
            image = Image.new("RGB", (width, height), "white")
            image.paste(overview, (0, 0))
            image.paste(closeup, (800, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((11, 10, 225, 57), fill="white")
            draw.line((21, 23, 51, 23), fill="#2477a2", width=3)
            draw.text((58, 17), "A* path", fill="#333333")
            draw.line((21, 43, 51, 43), fill="#d17c36", width=3)
            draw.text((58, 37), "Measured COM", fill="#333333")
            scale = height/(2*global_camera.distance*math.tan(half_fovy))
            draw.line((23, height-30, 23+round(scale), height-30), fill="#555555", width=2)
            draw.text((25, height-50), "1 m", fill="#333333")
            draw.text((1210, 13), f"t = {data.time:.1f} s", fill="#333333")
            if number == len(indices)-1:
                image.save(output/"robot_tracking.png")
            images.append(image.quantize(colors=128))
            if number % 20 == 0:
                print(f"Rendered {number+1}/{len(indices)} real qpos frames", flush=True)
    finally:
        renderer.close()
        close_renderer.close()
    if com_error > 1e-9 or body_error > 1e-9:
        raise AssertionError(f"Replay differs from saved physics: COM={com_error}, bodies={body_error}")
    duration = max(10, round(args.playback*1000/len(images)/10)*10)
    durations = [duration]*len(images)
    durations[-1] += 1200
    gif_path = output/"robot_tracking.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    plots(output, arrays, summary, comparison)
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
                "visual_scope": "Global view includes obstacles; CAD close-up hides obstacle visuals for clarity",
                "source_frame_count": len(arrays["time_s"]), "source_simulation_time_s": float(arrays["time_s"][-1]),
                "selected_indices": indices.tolist(), "gif_frame_count": count,
                "gif_size_px": [width, height], "gif_playback_s": playback/1000,
                "playback_speedup": float(arrays["time_s"][-1])/(duration*len(indices)/1000),
                "final_frame_hold_s": 1.2, "max_replay_com_error_m": com_error,
                "max_replay_body_error_m": body_error, "render_wall_time_s": time.perf_counter()-started,
                "comparison_npz_sha256": compare_hash}
    (output/"render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_indices"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--playback", type=float, default=18.)
    parser.add_argument("--compare", help="Optional open-loop run for the static path plot")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert np.array_equal(sample_indices(11, 4), [0, 3, 7, 10])
        assert np.array_equal(sample_indices(3, 100), [0, 1, 2])
        try:
            sample_indices(1, 20)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid one-frame rollout accepted")
        print("Frame sampling, endpoints, and invalid-input checks passed")
    else:
        if not args.run or args.frames < 2 or not math.isfinite(args.playback) or args.playback <= 0:
            parser.error("Provide a run path, frames >= 2, and positive finite playback")
        render(args)

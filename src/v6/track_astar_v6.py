"""A* and feedback steering of the existing V6 MuJoCo robot.

Only joint targets drive the robot. This is the wheeled V6 navigation model,
not the Sano ribbon/contact solver. Run --self-check before the demonstration.
"""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from worm_v6 import build_xml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "record/v6/astar_tracking_20261002/closed_loop"
SCENE = {
    "bounds_m": [-6.8, 1.0, -2.8, 2.8],
    "goal_m": [-5.5, 0.0],
    "obstacles": [{"center_m": [-2.8, 0.0], "half_size_m": [0.30, 0.40]}],
    "grid_m": 0.10,
    "collision_envelope_m": 0.70,
    "tracking_margin_m": 0.15,
    "goal_tolerance_m": 0.15,
}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def obstacle_distance(points, obstacles):
    """Euclidean distance to closed rectangles (zero inside)."""
    points = np.asarray(points)
    distance = np.full(points.shape[:-1], np.inf)
    for obstacle in obstacles:
        delta = np.maximum(np.abs(points - obstacle["center_m"])
                           - obstacle["half_size_m"], 0)
        distance = np.minimum(distance, np.linalg.norm(delta, axis=-1))
    return distance


def segment_clear(a, b, obstacles, radius, spacing=0.025):
    count = max(2, math.ceil(np.linalg.norm(np.asarray(b) - a) / spacing) + 1)
    points = np.linspace(a, b, count)
    # Distance is 1-Lipschitz; half a sample gap certifies the unsampled span.
    return bool(np.all(obstacle_distance(points, obstacles) > radius + spacing / 2))


def astar(start, goal, scene):
    xmin, xmax, ymin, ymax = scene["bounds_m"]
    resolution = scene["grid_m"]
    origin = np.array([xmin, ymin])
    shape = np.rint((np.array([xmax, ymax]) - origin) / resolution).astype(int) + 1
    radius = scene["collision_envelope_m"] + scene["tracking_margin_m"]
    def key(point):
        return tuple(np.rint((np.asarray(point) - origin) / resolution).astype(int))
    def point(cell):
        return origin + np.asarray(cell) * resolution
    grid = origin + np.indices(tuple(shape)).transpose(1, 2, 0) * resolution
    free = obstacle_distance(grid, scene["obstacles"]) > radius + resolution / math.sqrt(2)
    source, target = key(start), key(goal)
    for cell in (source, target):
        if any(v < 0 or v >= shape[i] for i, v in enumerate(cell)) or not free[cell]:
            raise ValueError("Start or goal is outside the free grid")
    if not segment_clear(start, point(source), scene["obstacles"], radius):
        raise ValueError("Start cannot reach its grid cell")
    frontier = [(float(np.linalg.norm(point(source) - point(target))), 0.0, source)]
    costs, parents = {source: 0.0}, {}
    expanded = 0
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cost > costs[cell]:
            continue
        expanded += 1
        if cell == target:
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            other = cell[0] + dx, cell[1] + dy
            if not (0 <= other[0] < shape[0] and 0 <= other[1] < shape[1]) or not free[other]:
                continue
            if dx and dy and not (free[cell[0] + dx, cell[1]] and free[cell[0], cell[1] + dy]):
                continue
            candidate = cost + resolution * math.hypot(dx, dy)
            if candidate < costs.get(other, math.inf):
                costs[other], parents[other] = candidate, cell
                heapq.heappush(frontier, (candidate + float(np.linalg.norm(point(other) - point(target))), candidate, other))
    else:
        raise ValueError("A* found no path")
    cells, cell = [target], target
    while cell != source:
        cell = parents[cell]
        cells.append(cell)
    raw = np.vstack((start, [point(c) for c in cells[::-1]], goal))
    raw = raw[np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-9]]
    # ponytail: visibility shortcuts suffice in this open arena; use curvature
    # constrained planning when narrow passages require an articulated footprint.
    simplified, index = [raw[0]], 0
    while index < len(raw) - 1:
        last = len(raw) - 1
        while not segment_clear(raw[index], raw[last], scene["obstacles"], radius):
            last -= 1
            if last <= index:
                raise ValueError("No certified path segment")
        simplified.append(raw[last])
        index = last
    return raw, np.asarray(simplified), expanded


def projection(position, path):
    vectors = np.diff(path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    fractions = np.clip(np.sum((position - path[:-1]) * vectors, axis=1) / lengths**2, 0, 1)
    points = path[:-1] + fractions[:, None] * vectors
    distances = np.linalg.norm(position - points, axis=1)
    index = int(np.argmin(distances))
    progress = np.r_[0, np.cumsum(lengths)][index] + fractions[index] * lengths[index]
    return float(progress), float(distances[index])


def point_along(path, progress):
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    index = min(int(np.searchsorted(cumulative[1:], progress, side="right")), len(lengths) - 1)
    fraction = np.clip((progress - cumulative[index]) / lengths[index], 0, 1)
    return path[index] + fraction * (path[index + 1] - path[index])


def yaw_targets(t, bias, amplitude, frequency, phase_offsets, startup_s):
    ramp = 0.5 * (1 - math.cos(math.pi * min(1., t / startup_s)))
    return ramp * amplitude * np.sin(2 * math.pi * frequency * t + phase_offsets) + bias


def make_model(scene=SCENE):
    meshes = ROOT / "meshes"
    tree = ET.fromstring(build_xml(str(meshes), str(meshes / "longworm2/longworm2.SLDASM.urdf")))
    for i, obstacle in enumerate(scene["obstacles"]):
        x, y = obstacle["center_m"]
        hx, hy = obstacle["half_size_m"]
        ET.SubElement(tree.find("worldbody"), "geom", name=f"obstacle_{i}", type="box",
                      pos=f"{x} {y} 0.25", size=f"{hx} {hy} 0.25", rgba="0.67 0.28 0.16 1",
                      contype="1", conaffinity="3", friction="0.8 0.005 0.001")
    xml = ET.tostring(tree, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml


def self_check():
    scene = dict(SCENE, bounds_m=[-3, 3, -3, 3], goal_m=[2, 0],
                 obstacles=[{"center_m": [0, 0], "half_size_m": [0.2, 0.3]}],
                 collision_envelope_m=0.3, tracking_margin_m=0.1)
    _, path, _ = astar(np.array([-2., 0]), np.array([2., 0]), scene)
    assert len(path) > 2 and sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) > 4
    assert all(segment_clear(a, b, scene["obstacles"], 0.4) for a, b in zip(path[:-1], path[1:]))
    assert not segment_clear([-2, 0], [2, 0], scene["obstacles"], 0.4)
    assert np.allclose(point_along(path, 1e6), path[-1])
    assert projection(path[0], path) == (0.0, 0.0)
    try:
        astar([0, 0], [2, 0], scene)
    except ValueError:
        pass
    else:
        raise AssertionError("Blocked start accepted")
    phases = np.arange(5) * 0.7
    assert np.allclose(yaw_targets(0., 0., 0.2, 0.4, phases, 2.), 0.)
    assert np.allclose(yaw_targets(2., 0.01, 0.2, 0.4, phases, 2.),
                       0.2 * np.sin(2 * math.pi * 0.4 * 2 + phases) + 0.01)
    assert np.max(np.abs(yaw_targets(3., 0.02, 0.2, 0.4, phases, 2.))) <= 0.22
    print("A* clearance, endpoints, blocked-start and gentle-wave checks passed")


def run(args):
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trajectory.npz").exists():
        raise FileExistsError("Choose a new output to preserve the previous rollout")
    source_paths = (Path(__file__), ROOT / "src/v6/worm_v6.py", ROOT / "src/v6/motor_contract_v6.py",
                    ROOT / "runs/cmaes_flat_serpentine/best_gait.json",
                    ROOT / "meshes/longworm2/longworm2.SLDASM.urdf")
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    model, xml = make_model()
    (output / "scene.xml").write_text(xml, encoding="utf-8")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(round(1 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    getid = lambda kind, name: mujoco.mj_name2id(model, kind, name)
    head = getid(mujoco.mjtObj.mjOBJ_BODY, "base_link")
    tail = getid(mujoco.mjtObj.mjOBJ_BODY, "back6_Link")
    yaw_ids = [getid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_front{i}") for i in range(2, 7)]
    collision_ids = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_contype != 0))
    obstacle_ids = [getid(mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}") for i in range(len(SCENE["obstacles"]))]
    params_path = ROOT / "runs/cmaes_flat_serpentine/best_gait.json"
    params = np.array(json.loads(params_path.read_text())["best_params"])
    amplitude = math.radians(args.yaw_amplitude_deg)
    frequency = args.yaw_frequency
    phase_offsets = 2 * math.pi * params[5] * np.arange(5) / 5
    com = data.subtree_com[head].copy()
    raw_path, path, expanded = astar(com[:2], np.array(SCENE["goal_m"]), SCENE)
    delta = data.xpos[tail, :2] - data.xpos[head, :2]
    heading = math.atan2(delta[1], delta[0])
    dt = float(model.opt.timestep)
    if args.duration < dt:
        raise ValueError("Duration must allow at least one physics step")
    if args.max_bias + amplitude > 1.57:
        raise ValueError("Gait amplitude plus bias exceeds yaw joint limits")
    bias, progress, collision_steps, max_envelope = 0., 0., 0, 0.
    min_clearance, torque_peak, force_peak = math.inf, 0., 0.
    warning_start = [w.number for w in data.warning]
    records = {name: [] for name in ("time_s", "qpos", "qvel", "com_m", "body_pos_m",
                                    "ctrl", "heading_rad", "bias_rad", "cross_track_m",
                                    "target_m", "progress_m")}
    success, reason = False, "time_limit"
    steps = round(args.duration / dt)
    def record(t, target, cross_track):
        values = (t, data.qpos.copy(), data.qvel.copy(), data.subtree_com[head].copy(),
                  data.xpos.copy(), data.ctrl.copy(), heading, bias, cross_track,
                  target.copy(), progress)
        for name, value in zip(records, values):
            records[name].append(value)
    record(0., path[0], 0.)
    print(f"A*: {expanded} expanded cells, {len(path)} waypoints, {np.linalg.norm(np.diff(path, axis=0), axis=1).sum():.3f} m", flush=True)
    print(f"path = {path.tolist()}", flush=True)
    for step in range(steps):
        t = step * dt
        if step % 10 == 0:
            com = data.subtree_com[head].copy()
            measured, cross_track = projection(com[:2], path)
            progress = max(progress, measured)
            target = point_along(path, progress + args.lookahead)
            delta = data.xpos[tail, :2] - data.xpos[head, :2]
            heading += (1 - math.exp(-10 * dt / 0.4)) * wrap(math.atan2(delta[1], delta[0]) - heading)
            desired = math.atan2(target[1] - com[1], target[0] - com[0])
            # Calibration: negative yaw target bias produces positive world yaw.
            command = 0. if args.open_loop else float(np.clip(-args.gain * wrap(desired - heading), -args.max_bias, args.max_bias))
            bias += (1 - math.exp(-10 * dt / 0.35)) * (command - bias)
        data.ctrl[:] = 0
        data.ctrl[yaw_ids] = yaw_targets(t, bias, amplitude, frequency, phase_offsets, args.startup)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        torque_peak = max(torque_peak, float(np.max(np.abs(data.actuator_force[yaw_ids]))))
        force_peak = max(force_peak, float(np.max(np.abs(data.actuator_force[:6]))))
        colliding = any(c.geom1 in obstacle_ids or c.geom2 in obstacle_ids for c in data.contact)
        collision_steps += int(colliding)
        if step % 25 == 0:
            envelope = float(np.max(np.linalg.norm(data.geom_xpos[collision_ids, :2] - data.subtree_com[head, :2], axis=1) + model.geom_rbound[collision_ids]))
            max_envelope = max(max_envelope, envelope)
            for obstacle in obstacle_ids:
                for geom in collision_ids:
                    min_clearance = min(min_clearance, float(mujoco.mj_geomDistance(model, data, int(geom), obstacle, 10., None)))
            if envelope > SCENE["collision_envelope_m"]:
                reason = "collision_envelope_exceeded"
                break
        if step % 50 == 49:
            _, cross_track = projection(data.subtree_com[head, :2], path)
            record((step + 1) * dt, target, cross_track)
        if not np.all(np.isfinite(data.qpos)) or any(w.number > warning_start[i] for i, w in enumerate(data.warning)):
            reason = "invalid_physics_state"
            break
        if not 0.02 < data.xpos[head, 2] < 0.28:
            reason = "unstable_height"
            break
        if colliding:
            reason = "obstacle_contact"
            break
        if np.linalg.norm(data.subtree_com[head, :2] - path[-1]) <= SCENE["goal_tolerance_m"]:
            success, reason = True, "goal_reached"
            break
        if step % 5000 == 4999:
            print(f"t={(step+1)*dt:.1f}s COM={data.subtree_com[head,:2].round(3)} error={cross_track:.3f}m bias={bias:.3f}", flush=True)
    final_time = (step + 1) * dt
    if abs(records["time_s"][-1] - final_time) > 1e-8:
        _, cross_track = projection(data.subtree_com[head, :2], path)
        record(final_time, target, cross_track)
    arrays = {name: np.asarray(value) for name, value in records.items()}
    np.savez_compressed(output / "trajectory.npz", **arrays, path_m=path, raw_astar_m=raw_path,
                        collision_geom_ids=collision_ids, body_mass_kg=model.body_mass,
                        head_id=head, tail_id=tail)
    sources_unchanged = hashes == {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    summary = {"model": "V6 MuJoCo wheeled navigation; ribbons are visual only, not Sano",
               "success": success, "stop_reason": reason, "sim_time_s": final_time,
               "wall_time_s": time.perf_counter() - started, "mujoco_version": mujoco.__version__,
               "python_version": sys.version, "numpy_version": np.__version__, "platform": platform.platform(),
               "physics_dt_s": dt, "controller_dt_s": 10*dt, "record_dt_s": 50*dt,
               "scene": SCENE, "controller": {"lookahead_m": args.lookahead, "gain": args.gain,
                   "max_bias_rad": args.max_bias, "open_loop": args.open_loop,
                   "heading_filter_s": 0.4, "bias_filter_s": 0.35},
               "gait": {"yaw_amplitude_rad": amplitude, "effective_frequency_hz": float(frequency),
                        "yaw_wave_number": float(params[5]), "slide_target_m": 0.,
                        "startup_ramp_s": args.startup, "startup_shape": "half-cosine"},
               "planned_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
               "travelled_com_length_m": float(np.linalg.norm(np.diff(arrays["com_m"][:,:2], axis=0), axis=1).sum()),
               "distance_note": "COM arc length and cross-track errors sampled at 0.1 s; final frame may have a shorter interval",
               "net_displacement_m": float(np.linalg.norm(arrays["com_m"][-1,:2]-arrays["com_m"][0,:2])),
               "goal_error_m": float(np.linalg.norm(arrays["com_m"][-1,:2]-path[-1])),
               "cross_track_rms_m": float(np.sqrt(np.mean(arrays["cross_track_m"]**2))),
               "cross_track_max_m": float(np.max(arrays["cross_track_m"])),
               "obstacle_contact_steps": collision_steps, "collision_check_dt_s": dt,
               "collision_geom_count": len(collision_ids), "min_geom_clearance_m": min_clearance,
               "distance_and_envelope_check_dt_s": 25*dt, "max_collision_envelope_m": max_envelope,
               "peak_yaw_torque_nm": torque_peak, "peak_slide_force_n": force_peak,
               "source_sha256": hashes, "sources_unchanged_during_run": sources_unchanged,
               "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
               "qpos_drive": "Only mj_step integrates root; joint targets set in data.ctrl"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not sources_unchanged:
        raise RuntimeError("Source changed during the run; results preserved but cannot be certified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--duration", type=float, default=600.)
    parser.add_argument("--lookahead", type=float, default=0.65)
    parser.add_argument("--gain", type=float, default=0.12)
    parser.add_argument("--max-bias", type=float, default=0.05)
    parser.add_argument("--yaw-amplitude-deg", type=float, default=20.)
    parser.add_argument("--yaw-frequency", type=float, default=0.4)
    parser.add_argument("--startup", type=float, default=2.)
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    options = parser.parse_args()
    if options.self_check:
        self_check()
    else:
        if not all(math.isfinite(v) and v > 0 for v in (options.duration, options.lookahead, options.gain,
                options.max_bias, options.yaw_amplitude_deg, options.yaw_frequency, options.startup)):
            parser.error("Duration and control parameters must be finite and positive")
        run(options)

"""Project steel-strip benchmark using the authors' Discrete Elastic Ribbons solver.

Quasi-static prescribed endplate motion, not a tendon/contact/locomotion simulation.
All lengths are metres; input assumptions are retained and recorded in results.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import root
import dismech as dm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
VENDOR = HERE / "vendor/discrete-elastic-ribbon"
COMMIT = "c9d341164e2927fc24b2c43dff97fcfb492cf700"


def unit(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError("Degenerate direction in steel geometry")
    return v / length


def read_project(path: Path):
    raw = path.read_bytes()
    p = json.loads(raw)
    urdf = HERE / "publication/data/cad_reference.urdf"
    urdf_raw = urdf.read_bytes()
    joint = next(j for j in ET.fromstring(urdf_raw).findall("joint")
                 if j.find("child").get("link") == "back2_Link")
    origin = joint.find("origin")
    delta = np.fromstring(origin.get("xyz"), sep=" ")
    if not np.allclose(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0):
        raise ValueError("This adapter expects the project's unrotated CAD plate reference")
    for key in ("strip_width_m", "strip_thickness_m", "youngs_modulus_pa", "steel_density_kg_m3"):
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not -1 < p["poisson_ratio"] < .5:
        raise ValueError("Invalid elastic Poisson ratio")
    return p, delta, dict(parameters_sha256=hashlib.sha256(raw).hexdigest(),
                         urdf_sha256=hashlib.sha256(urdf_raw).hexdigest(),
                         source_parameters=str(path), source_urdf=str(urdf))


def strip_geometry(p, delta, strip, n):
    """Reuse CAD clamp pairs and parabola; resample to equal CHORD lengths.

    The upstream analytical energies use one common delta_l. Equal u or even
    equal arc length leaves unequal chords at finite resolution.
    """
    front = np.asarray(p["front_clamp_hole_pairs_m"][strip])
    back = np.asarray(p["back_clamp_hole_pairs_m"][strip])
    radial_f = front.mean(0) - p["front_plate_center_m"]
    radial_b = back.mean(0) - p["back_plate_center_m"]
    radial_f[0] = radial_b[0] = 0
    radial_f, radial_b = unit(radial_f), unit(radial_b)
    half = p["strip_thickness_m"] / 2
    a = front.mean(0) + [-half, 0, 0] + p["tab_bolt_to_fold_m"] * radial_f
    b = back.mean(0) + [half, 0, 0] + p["tab_bolt_to_fold_m"] * radial_b + delta
    def curve(u):
        u = np.asarray(u)[:, None]
        return a + u * (b-a) + 4*p["stress_free_bow_m"]*u*(1-u)*radial_f
    dense_u = np.linspace(0, 1, 4097)
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve(dense_u), axis=0), axis=1))]
    initial = np.interp(np.linspace(0, cumulative[-1], n), cumulative, dense_u)
    def residual(u_inner):
        lengths = np.linalg.norm(np.diff(curve(np.r_[0, u_inner, 1]), axis=0), axis=1)
        return np.diff(lengths)
    solved = root(residual, initial[1:-1], tol=1e-10)
    u = np.r_[0, solved.x, 1]
    points = curve(u)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not solved.success or np.any(np.diff(u) <= 0) or np.ptp(lengths)/lengths.mean() > 1e-8:
        raise RuntimeError("Equal-chord resampling failed")
    tangents = unit(np.diff(points, axis=0))
    seed = unit(front[1]-front[0])
    widths = unit(seed - np.sum(tangents*seed, axis=1)[:, None]*tangents)
    return points, widths


def refresh(robot, q):
    a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
    m1, m2 = robot.compute_material_directors(q, a1, a2)
    twist = robot.compute_reference_twist(robot.bend_twist_springs, q, a1, robot.state.ref_twist)
    return robot.update(q=q, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=twist)


def make_robot(p, points, widths, model):
    n = len(points)
    w, h, nu = p["strip_width_m"], p["strip_thickness_m"], p["poisson_ratio"]
    # Match the existing rectangular Saint-Venant torsional constant.
    major, minor = max(w, h), min(w, h)
    torsion = major*minor**3*(1/3-.21*(minor/major)*(1-minor**4/(12*major**4)))
    geom = dm.GeomParams(rod_r0=h, shell_h=0, axs=w*h, jxs=torsion,
                         ixs1=w*h**3/12, ixs2=h*w**3/12)
    material = dm.Material(p["steel_density_kg_m3"], p["youngs_modulus_pa"],
                           p["youngs_modulus_pa"], nu, nu)
    params = dm.SimParams(static_sim=True, two_d_sim=False, use_mid_edge=False,
                         use_line_search=False, log_data=False, log_step=1, show_floor=False,
                         dt=.01, max_iter=120, total_time=1, plot_step=1,
                         tol=1e-8, ftol=1e-10, dtol=1e-12)
    geometry = dm.Geometry(points, np.column_stack((np.arange(n-1), np.arange(1, n))),
                           np.empty((0, 3), dtype=int))
    robot = dm.SoftRobot(geom, material, geometry, params, dm.Environment())
    # m2 is the width direction; m1 is the thickness normal (m2 x tangent).
    desired_m1 = np.cross(widths, unit(np.diff(points, axis=0)))
    theta = np.unwrap(np.arctan2(np.sum(desired_m1*robot.state.a2, axis=1),
                                np.sum(desired_m1*robot.state.a1, axis=1)))
    q = robot.state.q.copy()
    q[3*n:] = theta
    robot = refresh(robot, q).fix_nodes(np.array([0, 1, n-2, n-1]))
    # Natural strains MUST be cached after material frames, before any loading.
    with contextlib.redirect_stdout(io.StringIO()):
        stepper = dm.ImplicitEulerTimeStepper(robot, energy_model=model,
                    sano_zeta=math.sqrt((1-nu)*w**4/(60*h*h)))
    robot, _, _ = stepper.step(robot)
    assert abs(float(stepper.compute_total_elastic_energy(robot.state))) < 1e-12
    assert np.allclose(robot.state.m2, widths, atol=1e-8)
    return robot, stepper


def pose(progress, compression, yaw):
    phase = min(int(progress), 3)
    u = progress - phase
    d, a = [(u*compression, 0), (compression, u*yaw),
            (compression, (1-u)*yaw), ((1-u)*compression, 0)][phase]
    c, s = math.cos(a), math.sin(a)
    return d, a, np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def impose(robot, rest, widths, center, progress, compression, yaw, q=None):
    d, _, rotation = pose(progress, compression, yaw)
    q = robot.state.q.copy() if q is None else np.asarray(q, dtype=float).copy()
    q[:6] = rest[:2].ravel()
    q[3*(len(rest)-2):3*len(rest)] = ((rest[-2:]-center)@rotation.T + center + [d, 0, 0]).ravel()
    a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
    for edge, desired_width in ((0, widths[0]), (len(rest)-2, rotation@widths[-1])):
        nodes = q[:3*len(rest)].reshape(-1, 3)
        tangent = unit(nodes[edge+1]-nodes[edge])
        normal = np.cross(desired_width, tangent)
        angle = math.atan2(normal@a2[edge], normal@a1[edge])
        dof = 3*len(rest)+edge
        q[dof] = angle + 2*math.pi*round((q[dof]-angle)/(2*math.pi))
    return refresh(robot, q)


def predict(robot, center, begin, end, compression, yaw):
    """Distribute the plate increment as an initial guess, never as a constraint."""
    d0, _, r0 = pose(begin, compression, yaw)
    d1, _, r1 = pose(end, compression, yaw)
    q = robot.state.q.copy()
    n = len(robot.node_dof_indices)
    points = q[:3*n].reshape(n, 3)
    transformed = (points-center-[d0, 0, 0]) @ (r1 @ r0.T).T + center + [d1, 0, 0]
    weight = np.clip((np.arange(n)-1)/(n-3), 0, 1)
    points += weight[:, None] * (transformed-points)
    return refresh(robot, q)


def elastic_gradient(stepper, robot):
    # Upstream _forces is +grad(E) when no external forces are enabled.
    stepper._compute_forces_and_jacobian(robot, robot.state.q, np.zeros(robot.n_dof))
    return stepper._forces.copy()


def solve_case(p, delta, strip, n, model, steps, compression, yaw):
    started = time.perf_counter()
    rest, widths = strip_geometry(p, delta, strip, n)
    robot, stepper = make_robot(p, rest, widths, model)
    center = delta + p["back_plate_center_m"]
    subdivisions = 0
    def advance(current, begin, end, depth=0):
        nonlocal subdivisions
        try:
            guess = predict(current, center, begin, end, compression, yaw)
            target = impose(guess, rest, widths, center, end, compression, yaw)
            candidate, _, _ = stepper.step(target)
            grad = elastic_gradient(stepper, candidate)
            if (not np.isfinite(candidate.state.q).all() or not np.isfinite(grad).all()
                    or np.max(np.abs(grad[candidate.state.free_dof])) > 1e-6):
                raise RuntimeError("Equilibrium residual above tolerance")
            return candidate
        except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
            if depth >= 8:
                raise
            subdivisions += 1
            midpoint = (begin+end)/2
            halfway = advance(current, begin, midpoint, depth+1)
            return advance(halfway, midpoint, end, depth+1)
    frames = []
    previous = 0.0
    for progress in np.linspace(0, 4, 4*steps+1):
        if progress:
            robot = advance(robot, previous, progress)
        d, a, rotation = pose(progress, compression, yaw)
        grad = elastic_gradient(stepper, robot)
        node_grad = grad[:3*n].reshape(n, 3)
        free_nodes = robot.state.free_dof[robot.state.free_dof < 3*n]
        free_twists = robot.state.free_dof[robot.state.free_dof >= 3*n]
        frames.append(dict(progress=float(progress), compression_m=float(d), yaw_rad=float(a),
                           plate_rotation=rotation.tolist(), plate_translation_m=[float(d), 0, 0],
                           nodes_m=robot.state.q[:3*n].reshape(n, 3).tolist(),
                           width_directors=robot.state.m2.tolist(),
                           energy_j=float(stepper.compute_total_elastic_energy(robot.state)),
                           back_support_force_n=node_grad[-2:].sum(0).tolist(),
                           front_support_force_n=node_grad[:2].sum(0).tolist(),
                           free_force_residual_n=float(np.max(np.abs(grad[free_nodes]))),
                           free_moment_residual_nm=float(np.max(np.abs(grad[free_twists]))),
                           force_balance_n=float(np.linalg.norm(node_grad.sum(0)))))
        print(f"    [{model}, strip {strip}] saved {len(frames)}/{4*steps+1} equilibrium states "
              f"({time.perf_counter()-started:.1f}s)", flush=True)
        previous = progress
    return dict(model=model, strip=strip, rest_nodes_m=rest.tolist(), frames=frames,
                adaptive_subdivisions=subdivisions,
                final_shape_error_m=float(np.max(np.linalg.norm(np.asarray(frames[-1]["nodes_m"])-rest, axis=1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=HERE/"output/parameters.snapshot.json")
    parser.add_argument("--output", type=Path, default=HERE/"output")
    parser.add_argument("--nodes", type=int, default=17)
    parser.add_argument("--steps", type=int, default=6, help="saved increments per loading phase")
    parser.add_argument("--workers", type=int, default=1, help="parallel independent strip solves")
    parser.add_argument("--strips", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--models", nargs="+", choices=["sano", "kirchhoff"], default=["sano"])
    parser.add_argument("--compression-mm", type=float, default=20)
    parser.add_argument("--yaw-deg", type=float, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.nodes < 7 or args.steps < 1 or not np.isfinite([args.compression_mm, args.yaw_deg]).all() or args.compression_mm < 0:
        parser.error("Need >=7 nodes, positive steps and finite nonnegative compression")
    if args.workers < 1:
        parser.error("workers must be a positive integer")
    p, delta, provenance = read_project(args.parameters)
    if any(i < 0 or i >= p["strip_count"] for i in args.strips):
        parser.error("strip index outside project geometry")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=VENDOR, text=True).strip()
    if revision != COMMIT:
        raise RuntimeError("Vendor revision differs from the reviewed implementation")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"parameters.snapshot.json").write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = [(model, strip) for model in args.models for strip in args.strips]
    workers = min(args.workers, len(jobs))
    metadata = dict(**provenance, upstream_commit=revision, nodes=args.nodes, workers=workers,
                    requested_strips=args.strips, requested_models=args.models,
                    width_m=p["strip_width_m"], thickness_m=p["strip_thickness_m"],
                    youngs_modulus_pa=p["youngs_modulus_pa"], poisson_ratio=p["poisson_ratio"],
                    plate_radius_m=p["plate_stop_radius_m"], front_plate_center_m=p["front_plate_center_m"],
                    back_plate_center_world_m=(delta+p["back_plate_center_m"]).tolist(),
                    loading="Quasi-static prescribed plate translation/yaw; no ropes, contact, gravity or locomotion",
                    reference="Project provisional parabola assumed stress-free; no measured assembly preload",
                    discretization="Equal chord lengths; two vertices and edge twist clamped at each end",
                    initial_guess="Affine plate-increment blend; interior nodes remain unconstrained",
                    force_convention="External support force applied to each strip (+gradient of elastic energy)")
    result = dict(metadata=metadata, cases=[])
    start = time.perf_counter()
    completed = {}
    def save_case(index, case):
        completed[index] = case
        result["cases"] = [completed[i] for i in sorted(completed)]
        print(f"  [{case['model']}, strip {case['strip']}] done: {case['adaptive_subdivisions']} subdivisions; "
              f"return error {case['final_shape_error_m']*1000:.6g} mm", flush=True)
        metadata["wall_seconds"] = time.perf_counter()-start
        (args.output/"checkpoint.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if workers == 1:
        for index, (model, strip) in enumerate(jobs):
            print(f"Solving {model}, strip {strip}, {args.nodes} nodes", flush=True)
            case = solve_case(p, delta, strip, args.nodes, model, args.steps,
                              args.compression_mm/1000, math.radians(args.yaw_deg))
            save_case(index, case)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for index, (model, strip) in enumerate(jobs):
                print(f"Submitting {model}, strip {strip}, {args.nodes} nodes", flush=True)
                future = pool.submit(solve_case, p, delta, strip, args.nodes, model, args.steps,
                                     args.compression_mm/1000, math.radians(args.yaw_deg))
                futures[future] = index
            for future in as_completed(futures):
                save_case(futures[future], future.result())
    metadata["wall_seconds"] = time.perf_counter()-start
    (args.output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.output/'results.json'} ({metadata['wall_seconds']:.1f}s)", flush=True)
    if args.render:
        from render import render
        render(args.output/"results.json")


if __name__ == "__main__":
    main()

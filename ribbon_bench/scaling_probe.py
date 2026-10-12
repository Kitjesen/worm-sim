"""Temporary coordinate-scaling experiment; preserves upstream RobustSolver."""
import json
import sys
import time

import numpy as np

import run as bench

n = int(sys.argv[1]) if len(sys.argv) > 1 else 33
strip = 2 if n == 33 else 0
reference_path = bench.HERE / f"mesh_demo/baseline{n}/results.json"
reference = json.loads(reference_path.read_text(encoding="utf-8"))
expected = next(case for case in reference["cases"] if case["strip"] == strip)
parameters = reference_path.parent / "parameters.snapshot.json"
if not parameters.exists():
    parameters = bench.HERE / "refinement33/parameters.snapshot.json"
p, delta, _ = bench.read_project(parameters)
old_make = bench.make_robot
stats = {}


def make_scaled(p, points, widths, model):
    robot, stepper = old_make(p, points, widths, model)
    length = float(np.linalg.norm(points[-1] - points[0]))
    scale = np.where(robot.state.free_dof < 3*len(points), length, 1.0)
    solver = stepper._solver
    original_solve = solver.solve
    stats.update(length_scale_m=length, solver=solver, calls=0)

    def solve(jacobian, force):
        stats["calls"] += 1
        return scale * original_solve(scale[:, None]*jacobian*scale[None, :], scale*force)

    solver.solve = solve
    return robot, stepper


bench.make_robot = make_scaled
started = time.perf_counter()
case = bench.solve_case(p, delta, strip, n, "sano", 6, .020, np.deg2rad(5))
elapsed = time.perf_counter()-started
frames, old = case["frames"], expected["frames"]
assert len(frames) == len(old)
summary = dict(nodes=n, strip=strip, wall_seconds=elapsed,
               subdivisions=case["adaptive_subdivisions"],
               reference_subdivisions=expected["adaptive_subdivisions"],
               length_scale_m=stats["length_scale_m"], linear_solves=stats["calls"],
               regularizations=stats["solver"]._regularization_count,
               max_condition_seen=stats["solver"]._max_condition_seen,
               max_force_difference_n=float(np.max(np.abs(np.asarray([f["back_support_force_n"] for f in frames])-np.asarray([f["back_support_force_n"] for f in old])))),
               max_energy_difference_j=float(np.max(np.abs(np.asarray([f["energy_j"] for f in frames])-np.asarray([f["energy_j"] for f in old])))),
               max_node_difference_m=float(np.max(np.linalg.norm(np.asarray([f["nodes_m"] for f in frames])-np.asarray([f["nodes_m"] for f in old]), axis=2))),
               max_force_residual_n=max(f["free_force_residual_n"] for f in frames),
               max_moment_residual_nm=max(f["free_moment_residual_nm"] for f in frames))
out = bench.HERE / f"probe_prediction/scaled{n}_strip{strip}"
out.mkdir(parents=True, exist_ok=True)
(out/"comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
metadata = {**reference["metadata"], "wall_seconds": elapsed,
            "requested_strips": [strip], "experiment": "Affine predictor plus translation/twist coordinate scaling",
            "reference_results": str(reference_path)}
(out/"results.json").write_text(json.dumps(dict(metadata=metadata, cases=[case]), indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
assert summary["max_force_difference_n"] < 1e-6, summary
assert summary["max_energy_difference_j"] < 1e-10, summary
assert summary["max_node_difference_m"] < 1e-8, summary
assert summary["max_force_residual_n"] <= 1e-6, summary
assert summary["max_moment_residual_nm"] <= 1e-6, summary

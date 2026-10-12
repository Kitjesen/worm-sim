"""Compare the integrated predictor against the saved pre-predictor 33-node solve."""
import json
import time

import numpy as np

import run as bench


def main():
    reference_path = bench.HERE / "refinement33/results.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["cases"][0]
    p, delta, provenance = bench.read_project(reference_path.parent / "parameters.snapshot.json")
    assert hasattr(bench, "predict"), "Run this after run.predict has been integrated"
    n = reference["metadata"]["nodes"]
    old_frames = expected["frames"]
    compression = max(f["compression_m"] for f in old_frames)
    yaw = max(f["yaw_rad"] for f in old_frames)
    steps = (len(old_frames)-1)//4
    started = time.perf_counter()
    case = bench.solve_case(p, delta, expected["strip"], n, expected["model"], steps, compression, yaw)
    elapsed = time.perf_counter()-started
    frames = case["frames"]
    assert len(frames) == len(old_frames)
    assert np.allclose([f["progress"] for f in frames], [f["progress"] for f in old_frames])
    force = np.asarray([f["back_support_force_n"] for f in frames])
    old_force = np.asarray([f["back_support_force_n"] for f in old_frames])
    energy = np.asarray([f["energy_j"] for f in frames])
    old_energy = np.asarray([f["energy_j"] for f in old_frames])
    nodes = np.asarray([f["nodes_m"] for f in frames])
    old_nodes = np.asarray([f["nodes_m"] for f in old_frames])
    summary = dict(nodes=n, wall_seconds=elapsed,
                   reference_wall_seconds=reference["metadata"]["wall_seconds"],
                   adaptive_subdivisions=case["adaptive_subdivisions"],
                   reference_subdivisions=expected["adaptive_subdivisions"],
                   max_force_difference_n=float(np.max(np.abs(force-old_force))),
                   max_energy_difference_j=float(np.max(np.abs(energy-old_energy))),
                   max_node_difference_m=float(np.max(np.linalg.norm(nodes-old_nodes, axis=2))),
                   max_free_force_residual_n=max(f["free_force_residual_n"] for f in frames),
                   max_free_moment_residual_nm=max(f["free_moment_residual_nm"] for f in frames))
    assert summary["max_force_difference_n"] < 1e-6, summary
    assert summary["max_energy_difference_j"] < 1e-10, summary
    assert summary["max_node_difference_m"] < 1e-8, summary
    assert summary["max_free_force_residual_n"] <= 1e-6, summary
    assert summary["max_free_moment_residual_nm"] <= 1e-6, summary
    out = bench.HERE / "probe_prediction/integration"
    out.mkdir(parents=True, exist_ok=True)
    metadata = {**reference["metadata"], **provenance, "wall_seconds": elapsed,
                "predictor": "run.predict through run.solve_case"}
    (out / "results.json").write_text(json.dumps(dict(metadata=metadata, cases=[case]), indent=2, allow_nan=False), encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

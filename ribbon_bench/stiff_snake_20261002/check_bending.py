"""Read-only polygonal steel-bend metrics; these are not a plastic-yield test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent


def bending(points):
    edges = np.diff(points, axis=-2)
    lengths = np.linalg.norm(edges, axis=-1)
    if not np.isfinite(points).all() or np.any(lengths <= 1e-12):
        raise ValueError('Nonfinite points or degenerate steel edge')
    tangent = edges/lengths[..., None]
    cosine = np.sum(tangent[..., :-1, :]*tangent[..., 1:, :], axis=-1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.))), lengths


def measure(folder):
    folder = folder.resolve()
    summary_file = folder/'summary.json'
    trajectory_file = folder/'trajectory.npz'
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    with np.load(trajectory_file, allow_pickle=False) as data:
        n = summary['metadata']['nodes']
        points = data['q'][..., :3*n].reshape(len(summary['frames']), -1, n, 3)
        rest = data['rest_nodes_m']
    assert summary['status'] == 'completed'
    times = np.array([row['time_s'] for row in summary['frames']])
    assert np.all(np.diff(times) > 0)
    angles, lengths = bending(points)
    rest_angles, rest_lengths = bending(rest)
    ratio = lengths/rest_lengths[None]

    def location(index):
        frame, strip, spring = (int(v) for v in index)
        node = spring+1
        return dict(time_s=float(times[frame]), frame=frame, steel_index_zero_based=strip,
                    module_index_one_based=strip//8+1, cad_module_index=strip//8+2,
                    steel_in_module_one_based=strip%8+1, vertex_index_zero_based=node,
                    at_clamp_transition=node in (1, n-2))

    peak = np.unravel_index(np.argmax(angles), angles.shape)
    change = np.abs(angles-rest_angles[None])
    change_peak = np.unravel_index(np.argmax(change), angles.shape)
    assert np.max(np.abs(points[0]-rest)) < 1e-12
    min_edge = np.unravel_index(np.argmin(ratio), ratio.shape)
    result = dict(
        source=str(folder.relative_to(BENCH)).replace('\\', '/'),
        summary_sha256=hashlib.sha256(summary_file.read_bytes()).hexdigest(),
        trajectory_sha256=hashlib.sha256(trajectory_file.read_bytes()).hexdigest(),
        nodes=n, saved_frames=len(times),
        initial_max_turn_angle_deg=float(angles[0].max()),
        all_frames_max_turn_angle_deg=float(angles.max()),
        max_turn_location=location(peak),
        max_absolute_turn_angle_change_from_rest_deg=float(change.max()),
        max_angle_change_location=location(change_peak),
        min_reference_edge_length_mm=float(rest_lengths.min()*1000),
        min_current_edge_length_mm=float(lengths.min()*1000),
        min_edge_length_ratio_to_rest=float(ratio.min()),
        max_edge_length_ratio_to_rest=float(ratio.max()),
        min_edge_ratio_location=dict(time_s=float(times[min_edge[0]]),
            steel_index_zero_based=int(min_edge[1]), edge_index_zero_based=int(min_edge[2])),
        near_180deg_turn_samples=int(np.count_nonzero(angles >= 165.)),
        near_180deg_reporting_threshold_deg=165.,
        modules=[dict(module_index_one_based=s+1, cad_module_index=s+2,
             max_turn_angle_deg=float(angles[:, 8*s:8*(s+1)].max()),
             max_turn_change_from_rest_deg=float(change[:, 8*s:8*(s+1)].max()))
             for s in range(points.shape[1]//8)],
        scope='Saved-frame N9 polygonal turning angle and length only. Angles include stress-free bow; angle change is geometric, not a material curvature/strain or plastic yield measure. Clamp transition vertices are 1 and N-2; idealized CAD mounting-tab folds lie outside this elastic chain. No interpolation or new solve.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='*')
    parser.add_argument('--output', type=Path, default=HERE/'bending_metrics.json')
    args = parser.parse_args()
    straight = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    right = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    np.testing.assert_allclose(bending(straight)[0], [0.], atol=1e-12)
    np.testing.assert_allclose(bending(right)[0], [90.], atol=1e-12)
    runs = args.runs or [BENCH/'snake_wave_20261002/snake_n9_cuda']
    result = dict(status='passed', algorithm_self_check='straight/right-angle passed',
                  cases=[measure(path) for path in runs])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

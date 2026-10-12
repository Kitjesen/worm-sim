"""Check one completed prescribed target against the four-cable tension cone.

This tests the saved target branch, not reachability of the whole workspace.
Positive tension pulls each back outlet toward its fixed front anchor.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from cable_loads import CableLoads


def fit_wrench(columns, required, length_scale):
    """Scale moment by a physical length so both residual rows have units N."""
    row_scale = np.array([1., 1./length_scale])
    matrix = np.asarray(columns)*row_scale[:, None]
    target = np.asarray(required)*row_scale
    positive, _ = nnls(matrix, target)
    signed, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != 2:
        raise ValueError("Cable wrench matrix does not span both plate coordinates")
    positive_residual = columns@positive-required
    signed_residual = columns@signed-required
    distance = float(np.linalg.norm(positive_residual*row_scale))
    tolerance = 1e-5  # N in the [force, moment / length_scale] coordinates.
    feasible = distance <= tolerance
    certificate = None
    if not feasible:
        normal = (target-matrix@positive)/distance
        certificate = dict(normal_in_scaled_wrench_space=normal.tolist(),
                           max_dot_with_cable_column=float(np.max(normal@matrix)),
                           dot_with_required_wrench_n=float(normal@target))
        # Projection onto a convex cone gives a separating plane for an outside target.
        assert certificate['max_dot_with_cable_column'] <= 1e-8
        assert certificate['dot_with_required_wrench_n'] > tolerance
    assert np.linalg.norm(signed_residual*row_scale) < 1e-8*max(1., np.linalg.norm(target))
    return dict(length_scale_m=float(length_scale),
                scaling='[force_N, moment_Nm / length_scale_m]; both rows are N',
                wrench_columns_per_newton=np.asarray(columns).tolist(),
                required_wrench_n_nm=np.asarray(required).tolist(),
                nonnegative_tensions_n=positive.tolist(),
                nonnegative_achieved_wrench_n_nm=(columns@positive).tolist(),
                nonnegative_residual_achieved_minus_required_n_nm=positive_residual.tolist(),
                scaled_cone_distance_n=distance, scaled_feasibility_tolerance_n=tolerance,
                nonnegative_fit_feasible=bool(feasible),
                signed_minimum_norm_tensions_n=signed.tolist(),
                signed_residual_n_nm=signed_residual.tolist(),
                scaled_matrix_condition=float(singular_values[0]/singular_values[-1]),
                separating_plane=certificate)


def check(path):
    path = Path(path).resolve()
    if path.name not in ('results.json', 'loading_results.json'):
        raise ValueError('Read a completed results.json or explicitly verified loading_results.json, not a checkpoint')
    data = json.loads(path.read_text(encoding='utf-8'))
    meta, cases, controls = data['metadata'], data['cases'], data['actuation_frames']
    if (len(cases) != 8 or {case['strip'] for case in cases} != set(range(8))
            or any(case['model'] != 'sano' for case in cases)):
        raise ValueError('Need exactly eight Sano strips with indices 0 through 7')
    if 'prescribed' not in meta['loading'].lower():
        raise ValueError('Need the prescribed-plate reference, whose residual is the required external wrench')
    frames = cases[0]['frames']
    if len(frames) < 2 or len(controls) != len(frames):
        raise ValueError('Need at least two synchronized saved frames')
    progress = np.array([frame['progress'] for frame in frames])
    if not np.isfinite(progress).all() or np.any(np.diff(progress) <= 0) or abs(progress[0]) > 1e-9:
        raise ValueError('Need finite, increasing load progress starting at zero')
    for case in cases:
        if (len(case['frames']) != len(frames) or not np.allclose(
                [frame['progress'] for frame in case['frames']], progress, rtol=0, atol=1e-9)):
            raise ValueError('All eight strips must have synchronized frames')
        for frame in case['frames']:
            if not all(np.isfinite(np.asarray(frame[key], dtype=float)).all() for key in
                       ('progress', 'compression_m', 'yaw_rad', 'nodes_m', 'width_directors',
                        'energy_j', 'free_force_residual_n', 'free_moment_residual_nm')):
                raise ValueError('Reference contains non-finite strip data')
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for frame in controls for value in frame.values()):
        raise ValueError('Reference contains non-finite actuation data')
    loading_only = path.name == 'loading_results.json'
    if loading_only:
        if (meta.get('loading_path_complete') is not True or 'unloading_converged' not in meta
                or (meta['unloading_converged'] is not None and meta['unloading_converged'] is not False)
                or meta.get('path_status') != 'loading_only' or abs(progress[-1]-2.) > 1e-9):
            raise ValueError('Loading-only reference needs explicit completion/status flags and progress 0 through 2')
    elif (abs(progress[-1]-4.) > 1e-9 or meta.get('unloading_converged') is False
          or any('final_shape_error_m' not in case for case in cases)):
        raise ValueError('results.json must contain the completed full load/unload path')
    pose = np.array([meta['target_compression_mm']/1000, np.deg2rad(meta['target_yaw_deg'])])
    matches = [i for i, frame in enumerate(frames) if np.allclose(
        [frame['compression_m'], frame['yaw_rad']], pose, rtol=0, atol=1e-9)]
    if not matches:
        raise ValueError('Requested target pose is absent from the completed reference')
    index = matches[0]
    if loading_only and index != len(frames)-1:
        raise ValueError('The completed loading segment must end at its requested target')
    control = controls[index]
    if np.max(np.abs(control['tensions_n'])) > 1e-10:
        raise ValueError('Prescribed reference cables must be slack')
    target_frames = [case['frames'][index] for case in cases]
    if not all(np.allclose([f['compression_m'], f['yaw_rad']], pose, rtol=0, atol=1e-9) for f in target_frames):
        raise ValueError('Strip frames disagree on target pose')
    force_error = max(f['free_force_residual_n'] for f in target_frames)
    moment_error = max(f['free_moment_residual_nm'] for f in target_frames)
    if force_error > 1e-6 or moment_error > 1e-7:
        raise ValueError('Target steel strips have not met the coupled equilibrium tolerances')

    p = json.loads((path.parent/'parameters.snapshot.json').read_text(encoding='utf-8'))
    delta = np.asarray(meta['back_plate_center_world_m'])-p['back_plate_center_m']
    loads = CableLoads(p, delta)
    geometry = loads.evaluate(pose, np.ones(4))
    front, back = geometry['routes_m'][:, 0], geometry['routes_m'][:, 1]
    pull = (front-back)/geometry['lengths_m'][:, None]
    arms = back-loads.center-[pose[0], 0., 0.]
    columns = np.vstack((pull[:, 0], np.cross(arms, pull)[:, 2]))
    # Independent check: physical cable wrench is minus the span-length derivative.
    finite_difference = np.zeros((2, 4))
    for j, step in enumerate((1e-7, 1e-6)):
        offset = np.eye(2)[j]*step
        plus = loads.evaluate(pose+offset, np.ones(4))['lengths_m']
        minus = loads.evaluate(pose-offset, np.ones(4))['lengths_m']
        finite_difference[j] = -(plus-minus)/(2*step)
    assert np.allclose(columns, finite_difference, rtol=1e-7, atol=1e-9)
    required = np.array([control['plate_force_residual_n'], control['plate_moment_residual_nm']])
    report = fit_wrench(columns, required, float(p['plate_stop_radius_m']))
    report.update(reference_results=str(path), frame_index=index,
                  reference_path_status='loading_only' if loading_only else 'full_cycle',
                  reference_unloading_converged=meta['unloading_converged'] if loading_only else True,
                  target_compression_mm=float(pose[0]*1000), target_yaw_deg=float(np.degrees(pose[1])),
                  cable_order='snapshot [side, upper/lower], row-major flattening',
                  reference_free_force_residual_n=float(force_error),
                  reference_free_moment_residual_nm=float(moment_error),
                  reference_contact_force_n=float(control['contact_force_n']),
                  wrench_derivative_max_abs_difference=float(np.max(np.abs(columns-finite_difference))),
                  scope='Static cable-wrench feasibility at this saved prescribed target branch only; not a full-workspace reachability conclusion',
                  signed_solution_meaning='Exact minimum-norm algebraic fit; negative tension is not available to a cable')
    output = path.parent/'wrench_feasibility.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved {output}')


def self_check():
    columns = np.array([[1., 1., 1., 1.], [.03, .03, -.03, -.03]])
    assert fit_wrench(columns, np.array([1., 0.]), .055)['nonnegative_fit_feasible']
    outside = fit_wrench(columns, np.array([1., .05]), .055)
    assert not outside['nonnegative_fit_feasible']
    assert min(outside['signed_minimum_norm_tensions_n']) < 0
    assert np.linalg.norm(outside['signed_residual_n_nm']) < 1e-12
    print('Wrench-cone self-check passed: feasible cone interior and outside target with exact signed fit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', type=Path,
                        default=Path(__file__).resolve().parent/'prescribed_30_refined/results.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        check(args.results)

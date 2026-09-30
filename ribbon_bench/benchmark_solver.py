"""Compare the reference and accelerated solver on the same complete cable path."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def compare(reference, candidate):
    for key in ('parameters_sha256', 'urdf_sha256', 'nodes', 'upstream_commit'):
        assert reference['metadata'][key] == candidate['metadata'][key], key
    limits = {'nodes_m': 1e-6, 'width_directors': 1e-4, 'compression_m': 1e-6,
              'yaw_rad': 1e-5, 'energy_j': 1e-7, 'back_support_force_n': 2e-4,
              'front_support_force_n': 2e-4, 'tensions_n': 2e-4,
              'total_energy_j': 1e-6}
    errors = dict.fromkeys(limits, 0.)
    for result in (reference, candidate):
        assert [case['strip'] for case in result['cases']] == list(range(8))
        progress = np.asarray([frame['progress'] for frame in result['actuation_frames']])
        assert len(progress) >= 5 and np.isfinite(progress).all()
        assert progress[0] == 0 and progress[-1] == 4 and np.all(np.diff(progress) > 0)
        for case in result['cases']:
            assert np.array_equal([frame['progress'] for frame in case['frames']], progress)
            for frame in case['frames']:
                for key in limits.keys() - {'tensions_n', 'total_energy_j'}:
                    assert np.isfinite(frame[key]).all(), key
                assert np.shape(frame['nodes_m']) == (result['metadata']['nodes'], 3)
                assert np.shape(frame['width_directors']) == (result['metadata']['nodes']-1, 3)
        for frame in result['actuation_frames']:
            assert len(frame['tensions_n']) == len(frame['rest_lengths_m']) == 4
            assert np.isfinite(frame['tensions_n']).all() and np.isfinite(frame['total_energy_j'])
    assert len(reference['cases']) == len(candidate['cases']) == 8
    for a, b in zip(reference['cases'], candidate['cases']):
        assert len(a['frames']) == len(b['frames'])
        for old, new in zip(a['frames'], b['frames']):
            assert old['progress'] == new['progress']
            for key in limits.keys() - {'tensions_n', 'total_energy_j'}:
                errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(old[key])-new[key]))))
            assert new['free_force_residual_n'] <= 1e-6
            assert new['free_moment_residual_nm'] <= 1e-7
    assert len(reference['actuation_frames']) == len(candidate['actuation_frames'])
    for old, new in zip(reference['actuation_frames'], candidate['actuation_frames']):
        assert old['progress'] == new['progress']
        assert np.array_equal(old['rest_lengths_m'], new['rest_lengths_m'])
        for key in ('tensions_n', 'total_energy_j'):
            errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(old[key])-new[key]))))
        assert abs(new['plate_force_residual_n']) <= 1e-5
        assert abs(new['plate_moment_residual_nm']) <= 1e-6
        assert min(new['tensions_n']) >= 0
    for key, limit in limits.items():
        assert errors[key] <= limit, (key, errors[key], limit)
    return {'status': 'passed', 'max_absolute_differences': errors, 'limits': limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=33)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or args.nodes < 7:
        parser.error('Need positive repeats/steps and at least seven nodes')
    if args.output.exists():
        parser.error('Choose a new output directory to preserve earlier evidence')
    args.output.mkdir(parents=True)
    source_hashes = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                     for name in ('actuate.py', 'fast_sano.py', 'benchmark_solver.py')}
    timings = {'reference': [], 'fast': []}
    validation = []
    for repeat in range(args.repeats):
        results = {}
        # Alternate order to reduce systematic warm-cache/thermal bias.
        for backend in (('reference', 'fast') if repeat % 2 == 0 else ('fast', 'reference')):
            out = args.output / f'{backend}_{repeat+1}'
            command = [sys.executable, str(HERE/'actuate.py'), '--solver', backend,
                       '--nodes', str(args.nodes), '--steps', str(args.steps),
                       '--compression-mm', '40', '--yaw-deg', '30', '--output', str(out)]
            print(f'Run {repeat+1}/{args.repeats}: {backend}', flush=True)
            subprocess.run(command, cwd=HERE, check=True)
            results[backend] = json.loads((out/'results.json').read_text(encoding='utf-8'))
            timings[backend].append(results[backend]['metadata']['wall_seconds'])
        validation.append(compare(results['reference'], results['fast']))
        print(json.dumps(validation[-1]), flush=True)
    medians = {key: statistics.median(values) for key, values in timings.items()}
    assert source_hashes == {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                            for name in source_hashes}, 'Source changed during benchmark'
    summary = dict(status='passed', nodes=args.nodes, strips=8, states=4*args.steps+1,
                   repeats=args.repeats, seconds=timings, median_seconds=medians,
                   median_speedup=medians['reference']/medians['fast'],
                   timing_scope='CLI loading path including checkpoint serialization, excluding imports/initialization/rendering',
                   hardware=platform.processor(), platform=platform.platform(),
                   python=sys.version, numpy=np.__version__, validation=validation,
                   thread_environment={key: os.environ.get(key) for key in
                       ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                   source_sha256=source_hashes)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

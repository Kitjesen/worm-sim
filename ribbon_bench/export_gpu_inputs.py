"""Export real normalized Sano strains and CPU64 references; never solve a path."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType

import numpy as np

from actuate import PlateSystem
from fast_sano import energy_grad_hess_batch
from run import COMMIT, HERE, VENDOR, read_project


COEFFICIENTS = ('EA', 'EI1', 'EI2', 'GJ', 'delta_l', 'inv_dl', 'scaling',
                'energy_norm', 'zeta', 'h')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=HERE/'solver_benchmark_20261001/fast_1/results.json')
    parser.add_argument('--output', type=Path, default=HERE/'gpu_probe_20261001')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; preserve the earlier probe')
    data = json.loads(args.input.read_text(encoding='utf-8'))
    metadata, cases, controls = data['metadata'], data['cases'], data['actuation_frames']
    assert metadata['nodes'] == 33
    assert [c['strip'] for c in cases] == list(range(8))
    progress = np.array([f['progress'] for f in controls])
    assert progress[0] == 0 and progress[-1] == 4 and np.all(np.diff(progress) > 0)
    assert all(np.array_equal([f['progress'] for f in c['frames']], progress) for c in cases)
    parameters = args.input.parent/'parameters.snapshot.json'
    p, delta, provenance = read_project(parameters)
    for key in ('parameters_sha256', 'urdf_sha256'):
        assert provenance[key] == metadata[key], key
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip()
    assert revision == metadata['upstream_commit'] == COMMIT

    sys.path.insert(0, str(HERE/'prescribed_30_unload_diagnostic'))
    restore = importlib.import_module('continue').restore
    first_frames = cases[0]['frames']
    peak = int(np.argmax([abs(f['yaw_rad']) for f in first_frames]))
    peak_compression = int(np.argmax([f['compression_m'] for f in first_frames]))
    roles = {'initial': 0, 'peak_yaw': peak, 'peak_compression': peak_compression,
             'yaw_unloading': int(np.argmin(abs(progress-2.5))),
             'compression_unloading': int(np.argmin(abs(progress-3.5))),
             'returned': len(progress)-1}
    selected = sorted(set(roles.values()))
    system = PlateSystem(p, delta, nodes=33, solver='fast')
    fields = {k: [] for k in ('x', 'energy', 'gradient', 'hessian', *COEFFICIENTS,
                              'frame_index', 'strip_index', 'spring_index', 'physical_energy_scale')}
    peak_robots = None
    for frame_index in selected:
        snapshot = {'cases': [{**case, 'frames': [case['frames'][frame_index]]} for case in cases]}
        robots = restore(system, snapshot)
        if frame_index == peak:
            peak_robots = robots
        for strip, (robot, stepper) in enumerate(zip(robots, system.steppers)):
            elastic, = stepper._TimeStepper__elastic_energies
            x = elastic.get_strain(robot.state)-elastic._nat_strain
            x[:, 1:] *= (elastic.h/elastic.l_eff)[:, None]
            energy, gradient, hessian = energy_grad_hess_batch(stepper.energy_model, x)
            count = len(x)
            assert count == 31, count
            fields['x'].append(x)
            fields['energy'].append(energy)
            fields['gradient'].append(gradient)
            fields['hessian'].append(hessian)
            for name in COEFFICIENTS:
                fields[name].append(np.full(count, getattr(stepper.energy_model, name), dtype=np.float64))
            for name, value in (('frame_index', frame_index), ('strip_index', strip)):
                fields[name].append(np.full(count, value, dtype=np.int64))
            fields['spring_index'].append(np.arange(count, dtype=np.int64))
            fields['physical_energy_scale'].append(.5*elastic.EA*elastic.l_eff)
    arrays = {key: np.concatenate(value) for key, value in fields.items()}
    arrays['peak_indices'] = np.flatnonzero(arrays['frame_index'] == peak)
    arrays['single_indices'] = np.flatnonzero((arrays['frame_index'] == peak) & (arrays['strip_index'] == 0))
    assert len(arrays['peak_indices']) == 248 and len(arrays['single_indices']) == 31
    assert all(np.isfinite(value).all() for value in arrays.values())

    # Isolate material-output quantization; geometry and assembly stay CPU64.
    # This is not a full float32/GPU evaluation or a new equilibrium solve.
    frame = first_frames[peak]
    plate = np.array([frame['compression_m'], frame['yaw_rad']])
    lengths = np.asarray(controls[peak]['rest_lengths_m'])
    baseline = system.evaluate(peak_robots, plate, lengths)
    assert baseline['error'] <= 1, 'Restored peak must still satisfy original equilibrium tolerances'
    originals = [s.energy_model.compute_energy_grad_hess_batch for s in system.steppers]
    def rounded(model, x):
        return tuple(a.astype(np.float32).astype(np.float64) for a in energy_grad_hess_batch(model, x))
    try:
        for stepper in system.steppers:
            stepper.energy_model.compute_energy_grad_hess_batch = MethodType(rounded, stepper.energy_model)
        quantized = system.evaluate(peak_robots, plate, lengths)
    finally:
        for stepper, original in zip(system.steppers, originals):
            stepper.energy_model.compute_energy_grad_hess_batch = original
    force_error, free_force_error, moment_error = 0., 0., 0.
    for (g0, _), (g1, _), free in zip(baseline['reactions'], quantized['reactions'], system.free):
        difference = abs(g1-g0)
        force_error = max(force_error, float(max(difference[:99])))
        free_force_error = max(free_force_error, float(max(difference[free[free < 99]])))
        moment_error = max(moment_error, float(max(difference[free[free >= 99]])))
    idx = arrays['peak_indices']
    physical_energy = float(arrays['energy'][idx] @ arrays['physical_energy_scale'][idx])
    rounded_energy = float(arrays['energy'][idx].astype(np.float32).astype(np.float64)
                           @ arrays['physical_energy_scale'][idx])
    rounding = dict(scope='Only E/g/H outputs rounded to float32 and promoted to float64; input/geometry/assembly remain float64; no Newton solve',
        max_node_force_component_error_n=force_error, max_free_node_force_component_error_n=free_force_error,
        max_free_twist_moment_error_nm=moment_error, plate_force_error_n=float(abs(quantized['gp'][0]-baseline['gp'][0])),
        plate_moment_error_nm=float(abs(quantized['gp'][1]-baseline['gp'][1])),
        baseline_free_force_residual_n=baseline['free_force'], rounded_free_force_residual_n=quantized['free_force'],
        baseline_scaled_equilibrium_error=baseline['error'], rounded_scaled_equilibrium_error=quantized['error'],
        steel_energy_reference_j=physical_energy, steel_energy_rounded_j=rounded_energy,
        steel_energy_absolute_error_j=abs(rounded_energy-physical_energy))
    sources = [args.input, parameters, Path(provenance['source_urdf']), Path(__file__),
               HERE/'fast_sano.py', HERE/'run.py', HERE/'actuate.py', HERE/'cable_loads.py',
               HERE/'prescribed_30_unload_diagnostic/continue.py']
    record = dict(status='exported_and_reference_peak_validated', upstream_commit=revision,
        source_result=str(args.input.resolve()), source_result_sha256=digest(args.input),
        sources=[dict(path=str(path.resolve()), sha256=digest(path)) for path in sources],
        selected_frame_roles=roles, selected_frame_indices=selected,
        frames=[dict(index=i, progress=float(progress[i]), compression_m=first_frames[i]['compression_m'],
                     yaw_rad=first_frames[i]['yaw_rad']) for i in selected],
        rows=len(arrays['x']), rows_per_strip_frame=31, peak_rows=248, single_rows=31,
        layout={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in arrays.items()},
        normalized_strain='x=(current-natural); columns 1:4 multiplied by h/l_eff in CPU float64 before any downcast',
        coefficients='Per-row actual model attributes; normalized material E/g/H. Multiply E by physical_energy_scale for Joules.',
        row_order='Selected frame ascending, then strip 0..7, then spring 0..30',
        batch_examples={'31': 'single_indices', '248': 'peak_indices',
                        '7936': 'tile peak_indices 32 times', '63488': 'tile peak_indices 256 times'},
        float32_output_rounding_at_peak=rounding)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'inputs.npz', **arrays)
    record['inputs_npz_sha256'] = digest(args.output/'inputs.npz')
    (args.output/'inputs.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), rows=len(arrays['x']), selected=roles,
                         peak_rows=248, single_rows=31, rounding=rounding), indent=2))


if __name__ == '__main__':
    main()

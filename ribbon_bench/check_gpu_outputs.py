"""Assemble recorded GPU material outputs at a saved CPU64 geometry; no solve."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType

import numpy as np

from actuate import PlateSystem
from export_gpu_inputs import COEFFICIENTS, digest
from fast_sano import energy_grad_hess_batch
from run import COMMIT, HERE, VENDOR, read_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, default=HERE/'gpu_probe_20261001')
    parser.add_argument('--input', type=Path,
                        default=HERE/'solver_benchmark_20261001/fast_1/results.json')
    parser.add_argument('--frame', type=int, default=12)
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--gpu-results', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.probe/'assembled_gpu_check.json'
    if output.exists():
        parser.error('Output already exists; use a new --output to preserve this check')
    inputs_path = args.probe/'inputs.npz'
    gpu_path = args.gpu_results or args.probe/'directml/gpu_outputs.npz'
    summary_path = args.summary or gpu_path.parent/'summary.json'
    metadata = json.loads((args.probe/'inputs.json').read_text(encoding='utf-8'))
    gpu_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert digest(inputs_path) == metadata['inputs_npz_sha256'] == gpu_summary['input_sha256']
    assert digest(args.input) == metadata['source_result_sha256']
    assert gpu_summary['validation'][args.precision]['status'] == 'passed_material_check'
    gpu_source = summary_path.parent/'gpu_probe.snapshot.py'
    if not gpu_source.exists():
        gpu_source = HERE/'gpu_probe.py'
    assert digest(gpu_source) == gpu_summary['source_sha256'], 'Executed GPU script SHA mismatch'
    with np.load(inputs_path, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    with np.load(gpu_path, allow_pickle=False) as archive:
        gpu = archive[args.precision]
    assert gpu.dtype in (np.float32, np.float64) and gpu.shape == (len(inputs['x']), 21)
    assert np.isfinite(gpu).all()
    # gpu_probe promotes its readback to NumPy float64 before saving.
    assert np.array_equal(gpu, gpu.astype(args.precision).astype(np.float64))
    storage_dtype = str(gpu.dtype)
    gpu = gpu.astype(np.float64)
    data = json.loads(args.input.read_text(encoding='utf-8'))
    cases = data['cases']
    assert [case['strip'] for case in cases] == list(range(8))
    assert data['metadata']['nodes'] == 33
    parameters = args.input.parent/'parameters.snapshot.json'
    p, delta, provenance = read_project(parameters)
    for key in ('parameters_sha256', 'urdf_sha256'):
        assert provenance[key] == data['metadata'][key], key
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip()
    assert revision == metadata['upstream_commit'] == COMMIT
    dependencies = [HERE/'run.py', HERE/'actuate.py', HERE/'fast_sano.py',
                    HERE/'cable_loads.py', HERE/'prescribed_30_unload_diagnostic/continue.py']
    original_hashes = {Path(item['path'].replace('\\', '/')).name: item['sha256'] for item in metadata['sources']}
    source_compatibility_exceptions = []
    published_paths = {
        'run.py': (
            '94670e07ae5aff95ac42c519b9781843a7f0740629035d7bcf163fe4e471e2c8',
            'c29e57430b9d5f55a059aede748592da354c9c6f4f4ca906031fdc2fd3e4f23e',
            ['CAD URDF: PROJECT/worm-sim/meshes/longworm2/longworm2.SLDASM.urdf -> HERE/publication/data/cad_reference.urdf',
             'CLI default parameters: PROJECT/single_segment/parameters.json -> HERE/output/parameters.snapshot.json; this check supplies an explicit snapshot path']),
        'cable_loads.py': (
            '5b4a05a0812b3e1dc0c7b35f93e54a469ce362bd14c5a6490d32ae8c0b3b9c48',
            '7237696ca0b91305a080909615526f19db2a369dfd3dd9cb10596d19fabe9d29',
            ['Unexecuted self_check CAD URDF: here.parent/worm-sim/meshes/longworm2/longworm2.SLDASM.urdf -> here/publication/data/cad_reference.urdf; CableLoads physics is byte-identical'])}
    for path in dependencies:
        current_sha256 = digest(path)
        recorded_sha256 = original_hashes[path.name]
        if current_sha256 != recorded_sha256:
            # Each complete-file hash pair differs only at the documented paths.
            allowed = published_paths.get(path.name)
            assert allowed and (recorded_sha256, current_sha256) == allowed[:2], f'Export source changed: {path}'
            source_compatibility_exceptions.append(dict(path=str(path.resolve()),
                recorded_sha256=recorded_sha256, current_sha256=current_sha256,
                changes=allowed[2],
                verification='Only this exact ordered pair of complete-file SHA-256 hashes is accepted; all other dependency hashes remain strict. Actual parameter and CAD bytes are checked against original result hashes.'))
    sys.path.insert(0, str(HERE/'prescribed_30_unload_diagnostic'))
    restore = importlib.import_module('continue').restore
    system = PlateSystem(p, delta, nodes=33, solver='fast')
    snapshot = {'cases': [{**case, 'frames': [case['frames'][args.frame]]} for case in cases]}
    robots = restore(system, snapshot)
    frame = cases[0]['frames'][args.frame]
    assert all(case['frames'][args.frame]['progress'] == frame['progress'] for case in cases)
    plate = np.asarray([frame['compression_m'], frame['yaw_rad']])
    lengths = np.asarray(data['actuation_frames'][args.frame]['rest_lengths_m'])
    baseline = system.evaluate(robots, plate, lengths)
    assert baseline['error'] <= 1, 'CPU64 reconstruction must satisfy existing tolerances'
    row_indices, calls, input_checks = [], [0]*8, []
    for strip, (robot, stepper) in enumerate(zip(robots, system.steppers)):
        indices = np.flatnonzero((inputs['frame_index'] == args.frame) & (inputs['strip_index'] == strip))
        assert len(indices) == 31 and np.array_equal(inputs['spring_index'][indices], np.arange(31))
        elastic, = stepper._TimeStepper__elastic_energies
        x = elastic.get_strain(robot.state)-elastic._nat_strain
        x[:, 1:] *= (elastic.h/elastic.l_eff)[:, None]
        assert np.array_equal(x, inputs['x'][indices]), f'Restored strain mismatch, strip {strip}'
        for name in COEFFICIENTS:
            assert np.all(inputs[name][indices] == getattr(stepper.energy_model, name)), name
        assert np.array_equal(inputs['physical_energy_scale'][indices], .5*elastic.EA*elastic.l_eff)
        for actual, name in zip(energy_grad_hess_batch(stepper.energy_model, x), ('energy', 'gradient', 'hessian')):
            assert np.array_equal(actual, inputs[name][indices]), f'CPU64 reference mismatch: {name}'
        row_indices.append(indices)

    def recorded_material(strip, indices, use_gpu):
        def evaluate(model, x):
            if hasattr(x, 'detach'):
                x = x.detach().numpy()
            x = np.asarray(x)
            expected = inputs['x'][indices]
            difference = np.abs(x-expected)
            # The upstream evaluation rebuilds directors even at unchanged q.
            # Require CPU64 roundoff only, and record GPU-cast equality separately.
            assert np.all(difference <= 64*np.finfo(np.float64).eps*np.maximum(1., np.abs(expected))), f'Assembly strain mismatch, strip {strip}'
            if use_gpu:
                input_checks.append(dict(strip=strip, max_absolute_difference=float(difference.max()),
                    cpu64_bitwise_equal=bool(np.array_equal(x, expected)),
                    gpu_precision_bitwise_equal=bool(np.array_equal(x.astype(args.precision), expected.astype(args.precision)))))
            calls[strip] += 1
            if not use_gpu:
                return tuple(inputs[key][indices] for key in ('energy', 'gradient', 'hessian'))
            values = gpu[indices]
            return values[:, 0], values[:, 1:5], values[:, 5:].reshape(-1, 4, 4)
        return evaluate

    originals = [stepper.energy_model.compute_energy_grad_hess_batch for stepper in system.steppers]
    try:
        for strip, (stepper, indices) in enumerate(zip(system.steppers, row_indices)):
            stepper.energy_model.compute_energy_grad_hess_batch = MethodType(recorded_material(strip, indices, False), stepper.energy_model)
        replay = system.evaluate(robots, plate, lengths)
        for strip, (stepper, indices) in enumerate(zip(system.steppers, row_indices)):
            stepper.energy_model.compute_energy_grad_hess_batch = MethodType(recorded_material(strip, indices, True), stepper.energy_model)
        assembled = system.evaluate(robots, plate, lengths)
    finally:
        for stepper, original in zip(system.steppers, originals):
            stepper.energy_model.compute_energy_grad_hess_batch = original
    assert calls == [2]*8, calls
    per_strip = []
    for strip, ((g0, _), (g1, _), free) in enumerate(zip(baseline['reactions'], assembled['reactions'], system.free)):
        difference = np.abs(g1-g0)
        per_strip.append(dict(strip=strip,
            max_node_force_component_error_n=float(np.max(difference[:99])),
            max_free_node_force_component_error_n=float(np.max(difference[free[free < 99]])),
            max_free_twist_moment_error_nm=float(np.max(difference[free[free >= 99]]))))
    indices = np.concatenate(row_indices)
    steel_reference = float(inputs['energy'][indices] @ inputs['physical_energy_scale'][indices])
    steel_gpu = float(gpu[indices, 0] @ inputs['physical_energy_scale'][indices])
    assert abs(steel_reference-sum(energy for _, energy in baseline['reactions'])) < 1e-12
    cable_gradient = np.asarray(baseline['loads']['gradient'])

    def residuals(evaluated):
        return dict(free_force_n=evaluated['free_force'], free_twist_moment_nm=evaluated['free_moment'],
            plate_potential_gradient_force_n=float(evaluated['gp'][0]),
            plate_potential_gradient_moment_nm=float(evaluated['gp'][1]),
            scaled_equilibrium_error=evaluated['error'],
            meets_existing_equilibrium_tolerances=evaluated['error'] <= 1)

    replay_error = max(float(np.max(np.abs(g0-g1))) for (g0, _), (g1, _) in zip(baseline['reactions'], replay['reactions']))
    report = dict(status='completed_with_verified_input_matching', frame_index=args.frame,
        frame=dict(progress=frame['progress'], compression_m=frame['compression_m'], yaw_rad=frame['yaw_rad']),
        scope=f'Actual recorded GPU {args.precision} material E/g/H in CPU float64 storage; restored geometry, chain rule, cables and assembly stay float64. One evaluation per variant, including a CPU64 record replay control; no Newton solve or time stepping.',
        energy_note='The material E returned to the gradient/Hessian interface is discarded upstream. Reported GPU steel energy is independently summed from the actual GPU E using unchanged physical scales; system.evaluate energy remains the CPU forward value and is not presented as GPU energy.',
        sign_convention='Internal residuals are +grad(E); physical internal forces and steel-on-plate wrench are their negatives. Absolute error metrics do not depend on this sign.',
        gpu=gpu_summary['gpu'], device=gpu_summary['device'], upstream_commit=revision,
        gpu_computation_dtype=args.precision, gpu_output_storage_dtype=storage_dtype,
        gpu_executed_source_sha256=gpu_summary['source_sha256'],
        gpu_current_source_matches_executed=digest(HERE/'gpu_probe.py') == gpu_summary['source_sha256'],
        run_sha256=digest(HERE/'run.py'), source_compatibility_exceptions=source_compatibility_exceptions,
        rows_used=len(indices), restored_input_exact_match=True, assembly_input_checks=input_checks,
        assembly_input_tolerance='64*float64 epsilon*max(1,abs(x)), checked elementwise; all differences reported',
        material_calls_per_strip=calls, cpu64_record_replay_max_generalized_gradient_error=replay_error,
        cpu64=residuals(baseline), cpu64_record_replay=residuals(replay), actual_gpu=residuals(assembled),
        errors={key: max(item[key] for item in per_strip) for key in per_strip[0] if key != 'strip'},
        per_strip=per_strip,
        steel_on_plate_wrench_cpu64_n_nm=(-(baseline['gp']-cable_gradient)).tolist(),
        steel_on_plate_wrench_gpu_n_nm=(-(assembled['gp']-cable_gradient)).tolist(),
        plate_force_error_n=float(abs(assembled['gp'][0]-baseline['gp'][0])),
        plate_moment_error_nm=float(abs(assembled['gp'][1]-baseline['gp'][1])),
        steel_energy_reference_j=steel_reference, steel_energy_actual_gpu_j=steel_gpu,
        steel_energy_absolute_error_j=abs(steel_gpu-steel_reference),
        existing_tolerances=dict(free_force_n=1e-6, free_twist_moment_nm=1e-7, plate_force_n=1e-5, plate_moment_nm=1e-6),
        sources=[dict(path=str(path.resolve()), sha256=digest(path)) for path in
                 [Path(__file__), inputs_path, args.probe/'inputs.json', gpu_path, summary_path,
                  args.input, parameters, Path(provenance['source_urdf']), gpu_source, *dependencies]])
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('status', 'cpu64', 'actual_gpu', 'errors',
                     'plate_force_error_n', 'plate_moment_error_nm', 'steel_energy_absolute_error_j')}, indent=2))
    print(output)


if __name__ == '__main__':
    main()

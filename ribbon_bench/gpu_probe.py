"""GPU experiment on real Sano material inputs, separate from the solver.

Run export_gpu_inputs.py in the original environment first. Run this file in
the isolated DirectML environment. Timings include completion/readback, not
just asynchronous GPU submission. No claim of whole-robot GPU acceleration.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace
import warnings

import numpy as np
import torch

from fast_sano import energy_grad_hess_batch


FIELDS = ('EA', 'EI1', 'EI2', 'GJ', 'delta_l', 'inv_dl', 'scaling',
          'energy_norm', 'zeta', 'h')


def coefficients(model):
    c0 = .5 * model.EA * model.delta_l / model.energy_norm
    factor = .5 * model.inv_dl * model.scaling**2 / model.energy_norm
    c2 = (1 / (model.zeta * model.inv_dl * model.scaling))**2
    return np.column_stack((c0, factor*model.EI1, factor*model.EI2,
                            factor*model.GJ, c2))


def material(x, c):
    """Same closed-form material E/g/H as fast_sano, packed into 21 columns."""
    eps, k1, k2, tau = x.unbind(1)
    c0, b1, b2, bt, c2 = c.unbind(1)
    inv = 1 / (k1*k1 + c2)
    tau2 = tau*tau
    tau3, tau4 = tau2*tau, tau2*tau2
    inv2, inv3 = inv*inv, inv*inv*inv
    energy = c0*eps*eps + b1*k1*k1 + b2*k2*k2 + bt*tau2 + b1*tau4*inv
    g0 = 2*c0*eps
    g1 = 2*b1*k1 - 2*b1*k1*tau4*inv2
    g2 = 2*b2*k2
    g3 = 2*bt*tau + 4*b1*tau3*inv
    h00 = 2*c0
    h11 = 2*b1 + 2*b1*tau4*(3*k1*k1-c2)*inv3
    h22 = 2*b2
    h33 = 2*bt + 12*b1*tau2*inv
    h13 = -8*b1*k1*tau3*inv2
    zero = torch.zeros_like(eps)
    return torch.stack((energy, g0, g1, g2, g3,
                        h00, zero, zero, zero,
                        zero, h11, zero, h13,
                        zero, zero, h22, zero,
                        zero, h13, zero, h33), dim=1)


def packed_reference(model, x):
    e, g, h = energy_grad_hess_batch(model, x)
    return np.column_stack((e, g, h.reshape(-1, 16)))


def errors(actual, reference, model):
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    scale = np.column_stack((np.full(len(actual), .001),
                             np.repeat((model.h/model.zeta)[:, None], 3, axis=1)))
    c = coefficients(model)
    natural = 2*c[:, :4]*scale**2
    transforms = (np.ones((len(actual), 1)), scale,
                  (scale[:, :, None]*scale[:, None, :]).reshape(-1, 16))
    floors = (natural.sum(1)[:, None], natural,
              np.sqrt(natural[:, :, None]*natural[:, None, :]).reshape(-1, 16))
    output = {}
    for name, slc, transform, floor in zip(('energy', 'gradient', 'hessian'),
            (slice(0, 1), slice(1, 5), slice(5, 21)), transforms, floors):
        difference = np.abs(actual[:, slc]-reference[:, slc])
        output[name] = {'max_absolute_normalized_units': float(difference.max()),
                       'max_scaled': float((difference*transform /
                                    (np.abs(reference[:, slc]*transform)+floor)).max())}
    return output


def timed(call, repeats):
    for _ in range(3):
        assert np.isfinite(call()).all()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()  # GPU paths return complete CPU arrays: implicit device wait.
        samples.append(time.perf_counter()-start)
        assert np.isfinite(result).all()
    return {'median_ms': statistics.median(samples)*1000,
            'samples_ms': [s*1000 for s in samples]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).parent/'gpu_probe_20261001/inputs.npz')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'gpu_probe_20261001/directml')
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--backend', choices=('directml', 'cuda'), default='directml')
    parser.add_argument('--compile', action='store_true', help='Fuse CUDA material operations with torch.compile')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('Need at least three timed repeats')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    packages = ['torch', 'numpy', 'scipy']
    if args.backend == 'directml':
        import torch_directml
        assert torch_directml.device_count() > 0, 'No DirectML GPU detected'
        device = torch_directml.device(0)
        gpu_name = torch_directml.device_name(0).rstrip('\0')
        packages.append('torch-directml')
    else:
        assert torch.cuda.is_available(), 'No CUDA GPU detected'
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
    if args.compile and args.backend != 'cuda':
        parser.error('--compile is only tested with CUDA')
    if args.compile:
        # Validation plus four timed shapes, each in two dtypes, need ten graphs.
        torch._dynamo.config.cache_size_limit = 16
    gpu_material = torch.compile(material, fullgraph=True, dynamic=False) if args.compile else material
    data = dict(np.load(args.inputs, allow_pickle=False))
    x = data['x']
    model = SimpleNamespace(**{key: data[key] for key in FIELDS})
    reference = np.column_stack((data['energy'], data['gradient'], data['hessian'].reshape(-1, 16)))
    local_reference = packed_reference(model, x)
    np.testing.assert_allclose(local_reference, reference, rtol=3e-13, atol=1e-18)
    report = {'status': 'running', 'scope': 'Sano material E/gradient/Hessian only; geometry, assembly, Newton and contact not on GPU',
              'gpu': gpu_name, 'device': str(device), 'backend': args.backend,
              'compiled': args.compile, 'cuda_version': torch.version.cuda,
              'platform': platform.platform(), 'python': platform.python_version(),
              'versions': {n: importlib.metadata.version(n) for n in packages},
              'torch_cpu_threads': torch.get_num_threads(), 'repeats': args.repeats,
              'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fast_sano_sha256': hashlib.sha256((Path(__file__).parent/'fast_sano.py').read_bytes()).hexdigest(),
              'validation': {}, 'timings': [],
              'timing_scope': 'Exclude initialization/compilation/constant coefficient upload; three warmups; every sample returns all E/g/H as float64 CPU arrays. End-to-end also casts and uploads float64 strain input; resident-input still includes result readback. Large batches repeat real strain states, not independent environments.'}
    outputs = {}
    supported = []
    for dtype, tolerance in ((torch.float32, 2e-5), (torch.float64, 3e-10)):
        name = str(dtype).split('.')[-1]
        record = {'scaled_error_limit': tolerance}
        try:
            cpu_x, cpu_c = torch.as_tensor(x, dtype=dtype), torch.as_tensor(coefficients(model), dtype=dtype)
            cpu = material(cpu_x, cpu_c).numpy().astype(np.float64)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                gpu = gpu_material(cpu_x.to(device), cpu_c.to(device)).cpu().numpy().astype(np.float64)
                tiny = (torch.tensor([1., 1.+2**-40], dtype=dtype).to(device)-1).cpu().numpy()
            record.update(cpu_vs_float64=errors(cpu, reference, model),
                          gpu_vs_float64=errors(gpu, reference, model),
                          warnings=[str(w.message) for w in caught],
                          precision_probe=tiny.tolist())
            assert not record['warnings'], 'Backend warned; inspect possible CPU fallback before accepting timings'
            assert max(v['max_scaled'] for v in record['gpu_vs_float64'].values()) <= tolerance
            if dtype == torch.float64:
                assert tiny[1] == 2**-40, 'Float64 lost a representable increment'
            record['status'] = 'passed_material_check'
            supported.append(dtype)
            outputs[name] = gpu
        except Exception as error:
            record.update(status='failed_or_unsupported', error=str(error))
            if isinstance(error, UnicodeDecodeError):
                record['backend_message_decoded_gb18030'] = error.object.decode('gb18030', errors='replace')
        report['validation'][name] = record
        print(name, record['status'], record.get('error', ''), flush=True)
    np.savez_compressed(args.output/'gpu_outputs.npz', **outputs)
    # Timed paths fail on warnings so a CPU fallback cannot silently look like GPU speed.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for size in (31, 248, 7936, 63488):
            indices = np.resize(data['single_indices'] if size == 31 else data['peak_indices'], size)
            batch_x = x[indices]
            batch_model = SimpleNamespace(**{key: data[key][indices] for key in FIELDS})
            record = {'rows': size, 'numpy_float64': timed(lambda: packed_reference(batch_model, batch_x), args.repeats)}
            for dtype in supported:
                name = str(dtype).split('.')[-1]
                xc = torch.as_tensor(batch_x, dtype=dtype)
                cc = torch.as_tensor(coefficients(batch_model), dtype=dtype)
                xg, cg = xc.to(device), cc.to(device)
                cpu_call = lambda: material(torch.as_tensor(batch_x, dtype=dtype), cc).numpy().astype(np.float64)
                gpu_call = lambda: gpu_material(torch.as_tensor(batch_x, dtype=dtype).to(device), cg).cpu().numpy().astype(np.float64)
                resident_call = lambda: gpu_material(xg, cg).cpu().numpy().astype(np.float64)
                record[name] = {'torch_cpu': timed(cpu_call, args.repeats),
                                args.backend+'_end_to_end': timed(gpu_call, args.repeats),
                                args.backend+'_resident_input': timed(resident_call, args.repeats)}
                record[name]['speedup_vs_numpy64'] = record['numpy_float64']['median_ms']/record[name][args.backend+'_end_to_end']['median_ms']
            report['timings'].append(record)
            print('rows', size, json.dumps(record), flush=True)
    report['status'] = 'completed' if supported else 'no_supported_material_precision'
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()

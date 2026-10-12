"""Hybrid full-path probe: CUDA float64 Sano material batches, CPU solver.

Run with --backend cpu|cuda [--compile] --nodes 33 --steps 6 --output NEW_DIR.
Other arguments are forwarded to actuate.main; its fast solver is kept intact.
Compare: --compare CPU/results.json CUDA/results.json --output NEW_REPORT.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def counters():
    return dict(cuda_calls=0, material_rows=0, coefficient_uploads=0,
                coefficient_upload_seconds=0., total_adapter_seconds=0.,
                h2d_bytes=0, d2h_bytes=0)


class CudaMaterial:
    """An instance-local adapter, with a cache invalidated by model parameters."""
    def __init__(self, model, device, function, stats):
        import torch
        from gpu_probe import FIELDS, coefficients
        self.torch, self.fields, self.coefficients = torch, FIELDS, coefficients
        self.model, self.device, self.function, self.stats = model, device, function, stats
        self.key = None
        self.refresh_constants()

    def refresh_constants(self):
        key = tuple(float(getattr(self.model, field)) for field in self.fields)
        if key != self.key:
            started = time.perf_counter()
            self.constants = self.torch.as_tensor(self.coefficients(self.model),
                                                 dtype=self.torch.float64, device=self.device)
            self.torch.cuda.synchronize(self.device)
            self.stats['coefficient_upload_seconds'] += time.perf_counter()-started
            self.stats['coefficient_uploads'] += 1
            self.key = key

    def __call__(self, x):
        started = time.perf_counter()
        self.refresh_constants()
        if hasattr(x, 'detach'):
            if x.device.type != 'cpu':
                raise ValueError('Probe requires the actual CPU strain-input path')
            x = x.detach().numpy()
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError('Expected normalized material strains with shape (batch, 4)')
        x_gpu = self.torch.as_tensor(x).to(self.device)
        with self.torch.no_grad():
            packed_gpu = self.function(x_gpu, self.constants.expand(len(x), -1))
        if packed_gpu.dtype != self.torch.float64 or packed_gpu.device.type != 'cuda':
            raise RuntimeError('Material kernel did not return CUDA float64')
        # The CPU result transfer waits for the complete material computation.
        packed = packed_gpu.cpu().numpy()
        self.stats['cuda_calls'] += 1
        self.stats['material_rows'] += len(x)
        self.stats['h2d_bytes'] += x.nbytes
        self.stats['d2h_bytes'] += packed.nbytes
        self.stats['total_adapter_seconds'] += time.perf_counter()-started
        if packed.shape != (len(x), 21) or not np.isfinite(packed).all():
            raise FloatingPointError('Invalid CUDA material output')
        return packed[:, 0], packed[:, 1:5], packed[:, 5:].reshape(-1, 4, 4)


def compare_results(reference_path, candidate_path):
    """Require the requested 8-strip, 33-node, 25-state complete cable path."""
    from benchmark_solver import compare
    reference, candidate = [json.loads(Path(p).read_text(encoding='utf-8'))
                            for p in (reference_path, candidate_path)]
    for result in (reference, candidate):
        assert result['metadata']['nodes'] == 33, 'Comparison requires 33 nodes'
        assert result['metadata']['solver_backend'] == 'fast', 'Keep the same CPU solver'
        assert len(result['actuation_frames']) == 25, 'Comparison requires all 25 states'
        assert all(len(c['frames']) == 25 for c in result['cases'])
    probe = candidate['metadata']['material_probe']
    assert probe['backend'] == 'cuda' and probe['status'] == 'completed'
    assert probe['path_counters']['cuda_calls'] > 0, 'No measured CUDA calls'
    assert reference['metadata'].get('material_probe', {}).get('backend', 'cpu') == 'cpu'
    report = compare(reference, candidate)
    # Validate both inputs against the same original equilibrium thresholds.
    compare(candidate, reference)
    seconds = {name: result['metadata']['wall_seconds']
               for name, result in [('cpu', reference), ('cuda', candidate)]}
    report.update(nodes=33, strips=8, states=25, path_seconds=seconds,
                  single_run_speedup=seconds['cpu']/seconds['cuda'],
                  timing_scope='actuate path timer; excludes initialization/CUDA warmup/rendering; CUDA calls complete on CPU result readback',
                  reference_sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--compile', action='store_true', help='Compile only the CUDA material kernel')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--solver', choices=('fast',), default='fast')
    parser.add_argument('--output', type=Path, required=True, help='New run directory, or new comparison JSON file')
    parser.add_argument('--compare', nargs=2, metavar=('CPU_RESULTS', 'CUDA_RESULTS'))
    args, forwarded = parser.parse_known_args()
    if args.output.exists():
        parser.error('Choose a new output path to preserve earlier results')
    if args.compare:
        if forwarded:
            parser.error('No actuate arguments are accepted in comparison mode')
        report = compare_results(*args.compare)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    if args.compile and args.backend != 'cuda':
        parser.error('--compile applies only to --backend cuda')

    import actuate
    import torch
    stats = counters()
    info = dict(backend=args.backend, compiled=args.compile, dtype='float64', status='running',
                scope='Hybrid: batch material E/g/H on CUDA only; standalone forward energy, geometry, assembly, banded blocks, Schur, Newton and line search on CPU' if args.backend == 'cuda' else 'Unchanged fast CPU solver and NumPy material adapter',
                platform=platform.platform(), python=sys.version, torch=torch.__version__, numpy=np.__version__,
                thread_environment={key: os.environ.get(key) for key in
                                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                counter_timing_scope='End-to-end adapter wall time; CPU input upload plus material computation plus complete E/g/H CPU readback; no intermediate CUDA synchronizations; constant uploads tracked separately',
                source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                               ('cuda_solver_probe.py', 'actuate.py', 'fast_sano.py', 'gpu_probe.py', 'benchmark_solver.py')})
    original_class, original_argv = actuate.PlateSystem, sys.argv
    if args.backend == 'cuda':
        from gpu_probe import material
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; this probe never falls back to CPU')
        device = torch.device('cuda', args.device)
        setup = time.perf_counter()
        tiny = (torch.tensor([1., 1.+2**-40], dtype=torch.float64, device=device)-1).cpu().numpy()
        if tiny[1] != 2**-40:
            raise RuntimeError('CUDA float64 precision probe failed')
        function = torch.compile(material, fullgraph=True, dynamic=False) if args.compile else material
        info.update(device=str(device), gpu=torch.cuda.get_device_name(device),
                    cuda_version=torch.version.cuda, precision_probe=tiny.tolist(),
                    device_setup_seconds=time.perf_counter()-setup,
                    compile_options={'fullgraph': True, 'dynamic': False} if args.compile else None)

        class ProbeSystem(original_class):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                started = time.perf_counter()
                adapters = []
                for stepper in self.steppers:
                    adapter = CudaMaterial(stepper.energy_model, device, function, stats)
                    stepper.energy_model.compute_energy_grad_hess_batch = adapter
                    adapters.append(adapter)
                # Exactly N-2 triplet material rows, matching the actual solver.
                for adapter in adapters:
                    adapter(np.zeros((self.nodes-2, 4), dtype=np.float64))
                torch.cuda.synchronize(device)
                info.update(warmup_seconds=time.perf_counter()-started,
                            warmup_rows_per_call=self.nodes-2, warmup_counters=dict(stats))
                stats.clear()
                stats.update(counters())

        actuate.PlateSystem = ProbeSystem
    started = time.perf_counter()
    try:
        sys.argv = [str(HERE/'actuate.py'), *forwarded, '--solver', 'fast', '--output', str(args.output)]
        actuate.main()
        info['status'] = 'completed'
    except BaseException as error:
        info.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        actuate.PlateSystem, sys.argv = original_class, original_argv
        info.update(wrapper_seconds=time.perf_counter()-started, path_counters=dict(stats))
        # Also label any completed checkpoint after a solver failure, without
        # altering its solved states, residuals, geometry, or path timer.
        if args.output.is_dir():
            for name in ('results.json', 'loading_results.json', 'checkpoint.json'):
                path = args.output/name
                if path.exists():
                    result = json.loads(path.read_text(encoding='utf-8'))
                    result['metadata']['material_probe'] = info
                    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            (args.output/'probe_metadata.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()

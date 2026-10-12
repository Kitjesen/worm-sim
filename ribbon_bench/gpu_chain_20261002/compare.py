"""Check the same five-segment N17 trajectory on CPU and RTX5090 CUDA."""
import json
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
cpu, gpu = (np.load(here/name/'trajectory.npz') for name in ('chain_n17_cpu_check', 'chain_cuda_probe'))
a, b = (json.loads((here/name/'summary.json').read_text(encoding='utf-8')) for name in ('chain_n17_cpu_check', 'chain_cuda_probe'))
for key in ('nodes', 'segments', 'dt_s', 'duration_s', 'command_amplitude_mm', 'gait', 'max_command_step_mm', 'parameters_sha256', 'urdf_sha256'):
    assert a['metadata'][key] == b['metadata'][key], key
errors = {key: float(np.max(np.abs(cpu[key]-gpu[key]))) for key in ('q', 'body_com', 'body_R', 'width_directors', 'cable_routes_m')}
assert max(errors.values()) < 1e-8, errors
for key in ('tensions_n', 'steel_energy_j', 'cable_energy_j', 'max_penetration_m', 'contact_normal_force_n'):
    errors[key] = float(np.max(np.abs(np.array([row[key] for row in a['frames']])-np.array([row[key] for row in b['frames']]))))
assert errors['tensions_n'] < 1e-5 and errors['steel_energy_j'] < 1e-8
assert [row.get('status_counts') for row in a['frames']] == [row.get('status_counts') for row in b['frames']]
result = dict(status='passed', segments=5, strips=40, nodes=17, frames=len(a['frames']),
              max_absolute_errors=errors, same_contact_status_counts=True,
              scope='Same 6 ms five-segment dynamic trajectory; Windows CPU vs remote CUDA; timings are not a controlled speed benchmark.')
(here/'parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

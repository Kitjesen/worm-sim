"""Validate the completed resumed path without rendering or changing prior runs."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path[:0] = [str(BENCH), str(BENCH/'prescribed_30_unload_diagnostic')]
from run import read_project
from actuate import PlateSystem
restore = importlib.import_module('continue').restore

path = HERE/'results.json'
data = json.loads(path.read_text())
cases = data['cases']
assert len(cases) == 8 and {c['strip'] for c in cases} == set(range(8))
progress = np.array([f['progress'] for f in cases[0]['frames']])
assert len(progress) == 205 and np.all(np.diff(progress) > 0)
assert progress[0] == 0 and progress[-1] == 4
assert all(np.array_equal(progress, [f['progress'] for f in c['frames']]) for c in cases)
assert len(data['actuation_frames']) == len(progress)
frames = [f for c in cases for f in c['frames']]
max_force = max(f['free_force_residual_n'] for f in frames)
max_moment = max(f['free_moment_residual_nm'] for f in frames)
assert max_force <= 1e-6 and max_moment <= 1e-7
assert all(f['compression_m'] == 0 and f['yaw_rad'] == 0 for f in [c['frames'][-1] for c in cases])
p, delta, _ = read_project(BENCH/'prescribed_30_refined/parameters.snapshot.json')
system = PlateSystem(p, delta, 33)
robots = restore(system, data)
recovered = system.evaluate(robots, np.zeros(2), np.ones(4))
assert recovered['free_force'] <= 1e-6 and recovered['free_moment'] <= 1e-7
shape_error = max(float(np.max(np.linalg.norm(np.array(c['frames'][-1]['nodes_m'])-c['rest_nodes_m'], axis=1))) for c in cases)
director_error = max(float(np.max(np.linalg.norm(np.array(c['frames'][-1]['width_directors'])-c['frames'][0]['width_directors'], axis=1))) for c in cases)
log = json.loads((HERE/'diagnostic.json').read_text())
baseline = json.loads((BENCH/'prescribed_30_unload_diagnostic/diagnostic.json').read_text())
baseline_success = [x for x in baseline['steps'] if 'failure' not in x]
loading_seconds = json.loads((BENCH/'prescribed_30_refined/checkpoint.json').read_text())['metadata']['wall_seconds']
unloading_seconds = baseline_success[-1]['wall_seconds'] + log[-1]['wall_seconds']
metadata = data['metadata']
metadata.update(loading_path_complete=True, unloading_converged=True, actual_duration='quasi-static',
    path_status='complete', loading_wall_seconds=loading_seconds,
    baseline_successful_unloading_wall_seconds=baseline_success[-1]['wall_seconds'],
    tangent_continuation_wall_seconds=log[-1]['wall_seconds'], unloading_wall_seconds=unloading_seconds,
    wall_seconds=loading_seconds+unloading_seconds,
    timing_note='Sum of elapsed successful path segments in separate runs; excludes failed retries and diagnostics; concurrent tasks were active',
    path_note='Original 13 loading states plus 192 independently solved unloading states; no reversed or interpolated equilibrium states',
    corrector_source_sha256=hashlib.sha256((BENCH/'actuate.py').read_bytes()).hexdigest())
path.write_text(json.dumps(data, indent=2, allow_nan=False))
report = dict(status='complete_path_passed', strips=8, nodes=33, saved_states_per_strip=len(progress),
    unloading_states=192, max_free_force_n=max_force, max_free_moment_nm=max_moment,
    final_recovered_free_force_n=recovered['free_force'], final_recovered_free_moment_nm=recovered['free_moment'],
    final_energy_j=recovered['energy'], final_shape_error_m=shape_error, final_width_director_error=director_error,
    unloading_wall_seconds=unloading_seconds, loading_wall_seconds=loading_seconds,
    total_path_wall_seconds=loading_seconds+unloading_seconds, tangent_newton_iterations=sum(r['iterations'] for r in log))
(HERE/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report, indent=2))

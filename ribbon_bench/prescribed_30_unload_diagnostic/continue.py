"""Resume prescribed unloading from saved material frames; keep source data intact."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run import read_project, refresh, predict, impose, pose
from actuate import PlateSystem

PEAK_SHA256 = 'dfe56c6a5e9ebcbdfa21b1e45d326c5c7bc362c1e6356ce5d546b007304dd845'
SEED_SHA256 = '8103bc4c6226f1994e0c8cbf1e2acaff20fb1de971b207ce725361547ed26928'


def verified_json(path, expected_sha256):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f'Checkpoint SHA-256 mismatch: {path}; restore the recorded input before continuing')
    return json.loads(payload)


def restore(system, data):
    robots = []
    for robot, case in zip(system.robots, data['cases']):
        frame = case['frames'][-1]
        q = robot.state.q.copy()
        xyz = np.asarray(frame['nodes_m'])
        q[:3*system.nodes] = xyz.ravel()
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        tangent = np.diff(xyz, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        m1 = np.cross(frame['width_directors'], tangent)
        q[3*system.nodes:] = np.unwrap(np.arctan2(np.sum(m1*a2, axis=1), np.sum(m1*a1, axis=1)))
        robots.append(refresh(robot, q))
    return robots


def main():
    source = HERE.parent/'prescribed_30_refined/checkpoint.json'
    data = verified_json(source, PEAK_SHA256)
    p, delta, _ = read_project(source.parent/'parameters.snapshot.json')
    system = PlateSystem(p, delta, data['metadata']['nodes'])
    robots = restore(system, data)
    comp = data['metadata']['target_compression_mm']/1000
    yaw = np.deg2rad(data['metadata']['target_yaw_deg'])
    begin = data['cases'][0]['frames'][-1]['progress']
    plate = np.asarray(pose(begin, comp, yaw)[:2])
    checked = system.evaluate(robots, plate, np.ones(4))
    saved_energy = sum(c['frames'][-1]['energy_j'] for c in data['cases'])
    assert abs(saved_energy-checked['energy']) < 1e-12
    assert checked['free_force'] < 1e-6 and checked['free_moment'] < 1e-7
    log = dict(source=str(source), recovered_energy_j=checked['energy'],
               recovered_free_force_n=checked['free_force'], recovered_free_moment_nm=checked['free_moment'], steps=[])
    print('Restored peak:', {k: v for k, v in log.items() if k != 'steps'}, flush=True)
    # Keep the recorded 28.75-degree seed immutable when repeating this experiment.
    output = HERE/'baseline_reproduction'
    output.mkdir(exist_ok=True)
    started = time.perf_counter()
    # 0.3125 degree yaw increments; then compression increments of 0.4167 mm.
    for end in np.r_[np.linspace(2, 3, 97)[1:], np.linspace(3, 4, 97)[1:]]:
        plate = np.asarray(pose(end, comp, yaw)[:2])
        candidates = [impose(predict(r, system.center, begin, end, comp, yaw), rest, width,
                            system.center, end, comp, yaw)
                      for r, rest, width in zip(robots, system.rest, system.widths)]
        entry = dict(begin=float(begin), end=float(end), yaw_deg=float(np.rad2deg(plate[1])),
                     compression_mm=float(1000*plate[0]))
        try:
            robots, plate, evaluated, iterations = system.equilibrate(candidates, plate, np.ones(4), prescribed=True)
            entry.update(iterations=iterations, free_force_n=evaluated['free_force'], free_moment_nm=evaluated['free_moment'],
                         energy_j=evaluated['energy'], wall_seconds=time.perf_counter()-started)
            frames, controls = system.frame(robots, plate, evaluated, end, np.ones(4))
            for case, frame in zip(data['cases'], frames):
                case['frames'].append(frame)
            data['actuation_frames'].append(controls)
            data['metadata'].update(path_status='unloading_in_progress', unloading_converged=False,
                                    continued_from=str(source), unloading_wall_seconds=entry['wall_seconds'])
            (output/'checkpoint.json').write_text(json.dumps(data, indent=2, allow_nan=False))
        except RuntimeError as exc:
            entry.update(failure=str(exc), wall_seconds=time.perf_counter()-started)
            log['steps'].append(entry)
            (output/'diagnostic.json').write_text(json.dumps(log, indent=2, allow_nan=False))
            print(json.dumps(entry), flush=True)
            raise
        log['steps'].append(entry)
        (output/'diagnostic.json').write_text(json.dumps(log, indent=2, allow_nan=False))
        print(json.dumps(entry), flush=True)
        begin = end
    data['metadata'].update(path_status='complete', unloading_converged=True)
    for case in data['cases']:
        case['final_shape_error_m'] = float(np.max(np.linalg.norm(
            np.asarray(case['frames'][-1]['nodes_m'])-case['rest_nodes_m'], axis=1)))
    (output/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

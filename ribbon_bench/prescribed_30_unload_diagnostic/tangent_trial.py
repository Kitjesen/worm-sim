"""Test a consistent equilibrium-tangent predictor with the original corrector."""
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from actuate import PlateSystem
from run import read_project, pose
recovery = importlib.import_module('continue')
restore = recovery.restore


def tangent_predict(system, robots, plate, target):
    value = system.evaluate(robots, plate, np.ones(4))
    dp = target-plate
    steps = [-np.linalg.solve(a, c@(dp/system.plate_scale))*scale
             for (a, c, _, g), scale in zip(value['blocks'], system.scales)]
    return system.moved(robots, plate, steps, dp)[0]


def main():
    source = HERE/'checkpoint.json'
    recovery.verified_json(HERE.parent/'prescribed_30_refined/checkpoint.json', recovery.PEAK_SHA256)
    data = recovery.verified_json(source, recovery.SEED_SHA256)
    p, delta, _ = read_project(HERE.parent/'prescribed_30_refined/parameters.snapshot.json')
    system = PlateSystem(p, delta, 33)
    robots = restore(system, data)
    begin = data['cases'][0]['frames'][-1]['progress']
    plate = np.array(pose(begin, .04, np.deg2rad(30))[:2])
    started = time.perf_counter()
    log = []
    # Start at the last true converged state, never substitute reversed loading data.
    schedule = np.r_[np.linspace(begin, 3, round((3-begin)*96)+1)[1:], np.linspace(3, 4, 97)[1:]]
    output = HERE.parent/'prescribed_30_tangent_unload'
    output.mkdir(exist_ok=True)
    for end in schedule:
        target = np.array(pose(end, .04, np.deg2rad(30))[:2])
        candidates = tangent_predict(system, robots, plate, target)
        initial = system.evaluate(candidates, target, np.ones(4))
        entry = dict(begin=float(begin), end=float(end), initial_free_force_n=initial['free_force'])
        try:
            robots, plate, value, iterations = system.equilibrate(candidates, target, np.ones(4), prescribed=True)
        except RuntimeError as exc:
            entry['failure'] = str(exc)
            log.append(entry)
            (output/'diagnostic.json').write_text(json.dumps(log, indent=2))
            print(entry, flush=True)
            raise
        entry.update(iterations=iterations, free_force_n=value['free_force'], free_moment_nm=value['free_moment'],
                     yaw_deg=float(np.rad2deg(plate[1])), compression_mm=float(plate[0]*1000),
                     energy_j=value['energy'], wall_seconds=time.perf_counter()-started)
        frames, controls = system.frame(robots, plate, value, end, np.ones(4))
        for case, frame in zip(data['cases'], frames):
            case['frames'].append(frame)
        data['actuation_frames'].append(controls)
        data['metadata'].update(path_status='unloading_in_progress', unloading_converged=False,
                               unloading_solver='Equilibrium-tangent predictor; unchanged energy-line-search Newton corrector',
                               continued_from=str(source), continuation_wall_seconds=entry['wall_seconds'])
        (output/'checkpoint.json').write_text(json.dumps(data, indent=2, allow_nan=False))
        log.append(entry)
        (output/'diagnostic.json').write_text(json.dumps(log, indent=2))
        print(json.dumps(entry), flush=True)
        begin = end
    data['metadata'].update(path_status='complete', unloading_converged=True)
    for case in data['cases']:
        case['final_shape_error_m'] = float(np.max(np.linalg.norm(
            np.asarray(case['frames'][-1]['nodes_m'])-case['rest_nodes_m'], axis=1)))
    (output/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

"""Check derivatives and line-search behavior at a restored unloading state."""
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from actuate import PlateSystem
from run import read_project, impose, predict, pose
restore = importlib.import_module('continue').restore


class TracedSystem(PlateSystem):
    def __init__(self, *args):
        super().__init__(*args)
        self.trace = []
        self.last_alpha = None
        self.iteration = -1

    def evaluate(self, robots, plate, rest_lengths):
        result = super().evaluate(robots, plate, rest_lengths)
        self.last_robots = robots
        self.trace.append(dict(iteration=self.iteration, alpha=self.last_alpha, energy=result['energy'],
            force=result['free_force'], moment=result['free_moment'],
            per_strip_force=[float(np.max(np.abs(g[f[f < 3*self.nodes]])))
                             for (g, _), f in zip(result['reactions'], self.free)]))
        return result

    def direction(self, evaluated, regularization, prescribed=False):
        if regularization == 0:
            self.iteration += 1
            self.last_base = (self.last_robots, evaluated)
        steps, dp = super().direction(evaluated, regularization, prescribed)
        self.trace.append(dict(iteration=self.iteration, regularization=regularization,
                               slope=float(sum(b[3]@(s/scale) for b, s, scale in zip(evaluated['blocks'], steps, self.scales)))))
        return steps, dp

    def moved(self, robots, plate, increments, plate_increment, fraction=1.):
        self.last_alpha = float(fraction)
        return super().moved(robots, plate, increments, plate_increment, fraction)


def derivative_checks(system, robots, plate):
    base = system.evaluate(robots, plate, np.ones(4))
    output = []
    rng = np.random.default_rng(1024)
    for index, ((a, _, _, g), scale, free) in enumerate(zip(base['blocks'], system.scales, system.free)):
        eig, vec = np.linalg.eigh((a+a.T)/2)
        for label, v in [('softest', vec[:, 0]), ('random', rng.normal(size=len(g)))]:
            v /= np.linalg.norm(v)
            increments = [np.zeros_like(s) for s in system.scales]
            increments[index] = v*scale
            for eps in [1e-5, 1e-6, 1e-7]:
                plus, _ = system.moved(robots, plate, increments, np.zeros(2), eps)
                minus, _ = system.moved(robots, plate, increments, np.zeros(2), -eps)
                ep = system.evaluate(plus, plate, np.ones(4)); em = system.evaluate(minus, plate, np.ones(4))
                fd = (ep['blocks'][index][3]-em['blocks'][index][3])/(2*eps)
                hv = a@v
                output.append(dict(strip=index, direction=label, eps=eps, min_eigenvalue=float(eig[0]),
                    max_eigenvalue=float(eig[-1]), hessian_asymmetry=float(np.linalg.norm(a-a.T)/np.linalg.norm(a)),
                    hessian_vector_rel_error=float(np.linalg.norm(fd-hv)/max(np.linalg.norm(fd), 1e-15)),
                    hessian_vector_abs_error=float(np.linalg.norm(fd-hv)), fd_norm=float(np.linalg.norm(fd)),
                    energy_slope_fd=float((ep['energy']-em['energy'])/(2*eps)), energy_slope_gradient=float(g@v)))
    return output


def main():
    source = HERE/'checkpoint.json'
    data = json.loads(source.read_text())
    p, delta, _ = read_project(HERE.parent/'prescribed_30_refined/parameters.snapshot.json')
    system = TracedSystem(p, delta, 33)
    robots = restore(system, data)
    begin = data['cases'][0]['frames'][-1]['progress']
    end, comp, yaw = begin+1/96, .04, np.deg2rad(30.)
    plate = np.array(pose(end, comp, yaw)[:2])
    robots = [impose(predict(r, system.center, begin, end, comp, yaw), rest, widths,
                    system.center, end, comp, yaw)
              for r, rest, widths in zip(robots, system.rest, system.widths)]
    started = time.perf_counter()
    system.trace = []
    try:
        rs, pl, value, iteration = system.equilibrate(robots, plate, np.ones(4), prescribed=True)
        outcome = dict(iterations=iteration, force=value['free_force'], moment=value['free_moment'])
    except RuntimeError as exc:
        outcome = dict(failure=str(exc))
    outcome.update(wall_seconds=time.perf_counter()-started, trace=system.trace)
    (HERE/'stall_line_search_trace.json').write_text(json.dumps(outcome, indent=2))
    print({k: v for k, v in outcome.items() if k != 'trace'}, flush=True)
    stalled, evaluated = system.last_base
    checks = derivative_checks(system, stalled, plate)
    (HERE/'stall_derivative_checks.json').write_text(json.dumps(checks, indent=2))
    frames, controls = system.frame(stalled, plate, evaluated, end, np.ones(4))
    for case, frame in zip(data['cases'], frames):
        case['frames'] = [frame]
    data['actuation_frames'] = [controls]
    data['metadata'].update(path_status='NOT_CONVERGED_DIAGNOSTIC', unloading_converged=False)
    (HERE/'stalled_state.json').write_text(json.dumps(data, indent=2))
    print('Stalled-state derivative checks written', flush=True)


if __name__ == '__main__':
    main()

"""Isolate a descent-direction repair without changing the geometric predictor."""
import json
from pathlib import Path
import time
import numpy as np
from diagnose import TracedSystem, restore, read_project, predict, impose, pose

HERE = Path(__file__).resolve().parent


class AdaptiveRegularization(TracedSystem):
    def direction(self, evaluated, regularization, prescribed=False):
        steps, dp = super().direction(evaluated, regularization, prescribed)
        slope = sum(b[3]@(s/scale) for b, s, scale in zip(evaluated['blocks'], steps, self.scales))
        if prescribed and regularization == 100. and slope >= 0:
            eigmin = min(float(np.linalg.eigvalsh((a+a.T)/2)[0]) for a, _, _, _ in evaluated['blocks'])
            shift = max(100., -eigmin + max(1e-4, abs(eigmin)*.01))
            self.trace.append(dict(adaptive_shift=shift, min_eigenvalue=eigmin))
            steps, dp = super().direction(evaluated, shift, prescribed)
        return steps, dp


def main():
    data = json.loads((HERE/'checkpoint.json').read_text())
    p, delta, _ = read_project(HERE.parent/'prescribed_30_refined/parameters.snapshot.json')
    system = AdaptiveRegularization(p, delta, 33)
    robots = restore(system, data)
    begin = data['cases'][0]['frames'][-1]['progress']
    end, comp, yaw = begin+1/96, .04, np.deg2rad(30.)
    plate = np.array(pose(end, comp, yaw)[:2])
    robots = [impose(predict(r, system.center, begin, end, comp, yaw), rest, widths,
                    system.center, end, comp, yaw)
              for r, rest, widths in zip(robots, system.rest, system.widths)]
    started = time.perf_counter()
    try:
        robots, plate, value, iterations = system.equilibrate(robots, plate, np.ones(4), prescribed=True)
        outcome = dict(iterations=iterations, free_force_n=value['free_force'], free_moment_nm=value['free_moment'], energy_j=value['energy'])
        frames, controls = system.frame(robots, plate, value, end, np.ones(4))
        for case, frame in zip(data['cases'], frames):
            case['frames'] = [frame]
        data['actuation_frames'] = [controls]
        data['metadata'].update(path_status='single_step_converged', unloading_converged=False)
        (HERE/'regularization_converged_step.json').write_text(json.dumps(data, indent=2))
    except RuntimeError as exc:
        outcome = dict(failure=str(exc))
    outcome.update(wall_seconds=time.perf_counter()-started, trace=system.trace)
    (HERE/'regularization_trial.json').write_text(json.dumps(outcome, indent=2))
    print({k:v for k,v in outcome.items() if k!='trace'}, flush=True)


if __name__ == '__main__':
    main()

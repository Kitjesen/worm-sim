"""Small repeatable coupled equilibrium and virtual-work check (no rendered data)."""
import json
import numpy as np

from actuate import PlateSystem
from run import HERE, read_project
from cable_loads import self_check as check_loads


def check():
    check_loads()
    p, delta, _ = read_project(HERE/'output/parameters.snapshot.json')
    system = PlateSystem(p, delta, nodes=9)
    initial = system.loads.evaluate([0., 0.], np.ones(4))['lengths_m']
    command = system.loads.evaluate([.005, .035], np.ones(4))['lengths_m']
    robots, plate, solved, _ = system.equilibrate(system.robots, np.zeros(2), command)
    assert solved['error'] <= 1
    assert plate[0] > .004 and plate[1] > 0
    assert abs(solved['gp'][0]) < 1e-5 and abs(solved['gp'][1]) < 1e-6
    assert np.all(solved['loads']['tensions_n'] >= 0)

    rng = np.random.default_rng(31)
    shifts = [rng.normal(size=len(f))*np.where(f < 27, 2e-6, 2e-4) for f in system.free]
    robots, plate = system.moved(robots, plate, shifts, np.array([2e-5, .003]))
    base = system.evaluate(robots, plate, command)
    # Both vector (prescribed) and multi-RHS (coupled Schur) banded solves
    # must reproduce the independent dense direction at a deformed state.
    for prescribed in (False, True):
        fast_internal, fast_plate = system.direction(base, 1e-5, prescribed)
        system.solver = 'reference'
        try:
            dense_internal, dense_plate = system.direction(base, 1e-5, prescribed)
        finally:
            system.solver = 'fast'
        for fast, dense in zip(fast_internal+[fast_plate], dense_internal+[dense_plate]):
            assert np.allclose(fast, dense, rtol=1e-8, atol=1e-11), (fast, dense)
    directions = [rng.normal(size=len(f))*np.where(f < 27, 1e-4, .01) for f in system.free]
    errors = []
    for internal, dp in (([np.zeros(len(f)) for f in system.free], np.array([.01, 0.])),
                         ([np.zeros(len(f)) for f in system.free], np.array([0., .1])),
                         (directions, np.array([.001, .02]))):
        derivative = base['gp']@dp + sum(block[3]@(v/scale) for block, v, scale
                                       in zip(base['blocks'], internal, system.scales))
        h = 1e-4
        plus, pp = system.moved(robots, plate, internal, dp, h)
        minus, pm = system.moved(robots, plate, internal, dp, -h)
        fd = (system.evaluate(plus, pp, command)['energy']-system.evaluate(minus, pm, command)['energy'])/(2*h)
        assert np.isclose(fd, derivative, rtol=2e-5, atol=1e-7), (fd, derivative)
        errors.append(float(abs(fd-derivative)))

    unloaded, plate0, rest, _ = system.equilibrate(robots, plate, initial)
    assert rest['error'] <= 1 and np.linalg.norm(plate0) < 1e-6
    return_error = max(float(np.max(np.linalg.norm(r.state.q[:27].reshape(9, 3)-x, axis=1)))
                       for r, x in zip(unloaded, system.rest))
    assert return_error < 1e-6
    report = dict(status='passed', nodes=9, strips=8, checks='cable/stop derivatives; coupled equilibrium; coupled and prescribed banded/dense directions; full virtual work; cable release',
                  virtual_work_absolute_errors=errors, return_shape_error_m=return_error,
                  plate_force_residual_n=float(rest['gp'][0]), plate_moment_residual_nm=float(rest['gp'][1]))
    (HERE/'check_actuated.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    check()

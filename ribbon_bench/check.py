"""One runnable integration check: rest state, objectivity and energy/force consistency."""
import json
import math
from pathlib import Path

import numpy as np
from run import read_project, strip_geometry, make_robot, impose, predict, refresh, elastic_gradient, HERE


def check():
    snapshot = HERE / "output/parameters.snapshot.json"
    p, delta, _ = read_project(snapshot)
    report = {}
    for model in ("sano", "kirchhoff"):
        rest, widths = strip_geometry(p, delta, 0, 13)
        robot, stepper = make_robot(p, rest, widths, model)
        assert np.linalg.norm(elastic_gradient(stepper, robot)) < 1e-8
        # A rigid translation/rotation must not generate strain energy or force.
        angle = .4
        rotation = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                             [0, math.sin(angle), math.cos(angle)]])
        q = robot.state.q.copy()
        q[:3*len(rest)] = (rest@rotation.T + [.01, -.02, .03]).ravel()
        rotated = robot.update(q=q, a1=robot.state.a1@rotation.T, a2=robot.state.a2@rotation.T,
                               m1=robot.state.m1@rotation.T, m2=robot.state.m2@rotation.T)
        assert abs(float(stepper.compute_total_elastic_energy(rotated.state))) < 1e-12
        # Small compression followed by yaw exercises strong/weak axes and twist.
        center = delta + p["back_plate_center_m"]
        before_prediction = robot.state.q.copy()
        guessed = predict(robot, center, 0, 1, .002, math.radians(.5))
        guessed_points = guessed.state.q[:3*len(rest)].reshape(-1, 3)
        assert np.allclose(guessed_points[:2], rest[:2], rtol=0, atol=1e-12)
        assert np.allclose(guessed_points[-2:], rest[-2:]+[.002, 0, 0], rtol=0, atol=1e-12)
        assert np.array_equal(robot.state.q, before_prediction)
        for progress in np.linspace(.1, 2, 20):
            robot, _, _ = stepper.step(impose(robot, rest, widths, center, progress, .002, math.radians(.5)))
        grad = elastic_gradient(stepper, robot)
        assert np.isfinite(grad).all()
        assert np.max(np.abs(grad[robot.state.free_dof])) < 1e-6
        assert np.linalg.norm(grad[:3*len(rest)].reshape(-1, 3).sum(0)) < 1e-6
        dx = 1e-7
        energies = []
        for sign in (-1, 1):
            q = robot.state.q.copy()
            q[3*(len(rest)-2):3*len(rest):3] += sign*dx
            energies.append(float(stepper.compute_total_elastic_energy(refresh(robot, q).state)))
        energy_derivative = (energies[1]-energies[0])/(2*dx)
        support_force = float(grad[3*(len(rest)-2):3*len(rest):3].sum())
        assert np.isclose(energy_derivative, support_force, rtol=2e-4, atol=2e-6), (energy_derivative, support_force)
        report[model] = dict(passed=True, support_force_n=support_force,
                              energy_derivative_n=energy_derivative,
                              free_residual=float(np.max(np.abs(grad[robot.state.free_dof]))))
    path = HERE / "check.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check()

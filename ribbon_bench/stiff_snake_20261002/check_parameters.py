"""Construct and check the isolated 0.20 mm steel experiment; no time stepping.

From ribbon_bench: PYTHONPATH=vendor/discrete-elastic-ribbon/src python
stiff_snake_20261002/check_parameters.py
Torch CUDA formulas are exercised on CPU here; remote trajectory is separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

from cuda_sano import _material, install_cuda_sano
from fast_sano import energy_grad_hess_batch
from full_robot import FullRobot
from gpu_geometry_contact import surface_geometry_batch
from ribbon_contact_geometry import surface_geometry
from run import read_project


def main():
    original = BENCH / 'output/parameters.snapshot.json'
    modified = HERE / 'parameters_t020mm.json'
    raw = original.read_bytes()
    token = b'"strip_thickness_m": 0.00015'
    assert raw.count(token) == 1
    altered = raw.replace(token, b'"strip_thickness_m": 0.0002')
    if modified.exists():
        assert modified.read_bytes() == altered, 'Candidate parameters were edited'
    else:
        modified.write_bytes(altered)
    old, delta, _ = read_project(original)
    new, new_delta, _ = read_project(modified)
    assert [k for k in old if old[k] != new[k]] == ['strip_thickness_m']
    np.testing.assert_array_equal(delta, new_delta)
    models = []
    constructed = []
    for p in (old, new):
        sim = FullRobot(p, delta, nodes=9, dt=.02, segments=5, unlocked_joints=True,
                        joint_kp=200., joint_kd=0., joint_torque_limit=20.,
                        joint_passive_damping=5.)
        w, h, e, nu, rho = (p[k] for k in ('strip_width_m', 'strip_thickness_m',
                              'youngs_modulus_pa', 'poisson_ratio', 'steel_density_kg_m3'))
        torsion = w*h**3*(1/3-.21*h/w*(1-h**4/(12*w**4)))
        expected = dict(EA=e*w*h, EI1=e*w*h**3/12, EI2=e*h*w**3/12,
                        GJ=e/(2*(1+nu))*torsion, h=h,
                        zeta=np.sqrt((1-nu)*w**4/(60*h*h)))
        for robot, stepper in zip(sim.robots, sim.steppers):
            for name, value in expected.items():
                np.testing.assert_allclose(getattr(stepper.energy_model, name), value,
                                           rtol=1e-13, atol=0.)
            assert abs(float(stepper.compute_total_elastic_energy(robot.state))) < 1e-12
        np.testing.assert_allclose(sim.mass_steel[:3*sim.nodes:3].sum(),
                                   rho*w*h*sim.robots[0].ref_len.sum(), rtol=1e-13)
        np.testing.assert_allclose(sim.mass_steel[3*sim.nodes:],
                                   rho*sim.robots[0].ref_len*(w*h**3+h*w**3)/12,
                                   rtol=1e-13)
        models.append(sim)
        constructed.append(dict(**expected,
            steel_total_mass_kg=float(sim.nstrips*sim.mass_steel[:3*sim.nodes:3].sum()),
            rigid_total_mass_kg=float(sim.body_mass.sum()),
            reference_length_m=float(sim.robots[0].ref_len.sum())))
    np.testing.assert_array_equal(models[0].body_mass, models[1].body_mass)
    sim = models[1]
    robot = sim.robots[0]
    state = robot.state
    elastic = sim.steppers[0]._TimeStepper__elastic_energies[0]
    cpu = elastic.grad_hess_energy_linear_elastic(state, sparse=False)
    restore = install_cuda_sano(elastic, 'cpu')
    try:
        cuda_formula = elastic.grad_hess_energy_linear_elastic(state, sparse=False)
    finally:
        elastic.grad_hess_energy_linear_elastic = restore
    np.testing.assert_allclose(cuda_formula[0], cpu[0], rtol=3e-10, atol=3e-9)
    np.testing.assert_allclose(cuda_formula[1], cpu[1], rtol=3e-9, atol=2e-6)
    strain = np.array([[.001, .002, -.003, .004], [0., -.003, .002, -.005]])
    fast = energy_grad_hess_batch(sim.steppers[0].energy_model, strain)
    cuda_material = _material(sim.steppers[0].energy_model,
                             torch.tensor(strain, dtype=torch.float64), torch)
    for a, b in zip(fast, cuda_material):
        np.testing.assert_allclose(a, b.numpy(), rtol=2e-12, atol=1e-12)
    points, _ = surface_geometry_batch(robot, torch.tensor(sim.q, dtype=torch.float64),
                    new['strip_width_m'], new['strip_thickness_m'], device='cpu',
                    reference_q=torch.tensor(sim.q, dtype=torch.float64),
                    reference_a1=torch.tensor(np.stack([r.state.a1 for r in sim.robots]),
                                               dtype=torch.float64))
    cpu_points = np.stack([surface_geometry(r, q, new['strip_width_m'],
                                new['strip_thickness_m'])[0]
                         for r, q in zip(sim.robots, sim.q)])
    np.testing.assert_allclose(points.numpy(), cpu_points, rtol=0., atol=2e-14)
    ratios = {k: constructed[1][k]/constructed[0][k] for k in constructed[0]}
    np.testing.assert_allclose(ratios['EI1'], (4/3)**3, rtol=1e-13)
    result = dict(status='passed',
        baseline_parameters='output/parameters.snapshot.json',
        candidate_parameters='stiff_snake_20261002/parameters_t020mm.json',
        baseline_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_sha256=hashlib.sha256(altered).hexdigest(),
        changed_keys=['strip_thickness_m'], nodes=9, segments=5, strips=40,
        thickness_m=[old['strip_thickness_m'], new['strip_thickness_m']],
        nominal_mass_and_axial_stiffness_ratio=4/3,
        nominal_weak_axis_bending_stiffness_ratio=(4/3)**3,
        constructed_material_and_mass=constructed, ratios=ratios,
        cpu_cuda_surface_max_error_m=float(np.max(np.abs(points.numpy()-cpu_points))),
        cpu_cuda_force_max_error_n=float(np.max(np.abs(cuda_formula[0]-cpu[0]))),
        cpu_cuda_hessian_max_error=float(np.max(np.abs(cuda_formula[1]-cpu[1]))),
        torch_check_device='cpu',
        scope='No dynamics here; CUDA numerical formulas checked on CPU. Thickness reaches Sano material, node/edge inertia and steel contact surfaces. Ground-law stiffness/friction and rigid CAD masses are unchanged.',
        geometry_note='Thickness offsets both clamp endpoints by half thickness; reference chord lengths and actual steel mass change slightly beyond nominal fixed-length ratios.',
        fold_scope='Elastic model only: no yield stress, hardening or plastic crease variable. N9 ribbon rendering is piecewise linear. Existing attachment-fold geometry is an idealized prescribed CAD tab, not newly predicted plastic folding.')
    (HERE/'stiffness_config.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

"""Closed-form derivatives of the pinned vendor's actual Sano.forward formula.

Only the material energy/strain derivative calculation is replaced. Natural
strains, geometry derivatives, assembly, constraints and Newton remain upstream.
Run ``python fast_sano.py`` for autodiff, finite-difference and assembly checks.
"""
from types import MethodType

import numpy as np


def energy_grad_hess_batch(model, x_batch):
    """Return float64 NumPy E[B], dE/dx[B,4], d2E/dx2[B,4,4].

    x is the upstream normalized *delta* strain [eps, k1, k2, tau], after
    natural strain subtraction. Accept NumPy or CPU Torch tensors. Read current
    model parameters on every call, including changes made by update_params().

    In normalized coordinates the actual forward has the nonlinear term
    b1*tau**4/(k1**2+c**2): EI1/k1, not GJ/k2 from its stale docstring.
    """
    if hasattr(x_batch, "detach"):
        x_batch = x_batch.detach().numpy()
    x = np.asarray(x_batch, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("Sano normalized delta strain must have shape (batch, 4)")

    # Use exactly the coefficients of forward, including its cached scaling.
    c0 = .5 * model.EA * model.delta_l / model.energy_norm
    factor = .5 * model.inv_dl * model.scaling**2 / model.energy_norm
    b1, b2, bt = factor * np.array([model.EI1, model.EI2, model.GJ])
    c2 = (1. / (model.zeta * model.inv_dl * model.scaling))**2
    eps, k1, k2, tau = x.T
    denom = k1*k1 + c2
    inv = 1. / denom
    tau2, tau3, tau4 = tau*tau, tau**3, tau**4

    energy = c0*eps*eps + b1*k1*k1 + b2*k2*k2 + bt*tau2 + b1*tau4*inv
    grad = np.empty_like(x)
    grad[:, 0] = 2*c0*eps
    grad[:, 1] = 2*b1*k1 - 2*b1*k1*tau4*inv**2
    grad[:, 2] = 2*b2*k2
    grad[:, 3] = 2*bt*tau + 4*b1*tau3*inv
    hess = np.zeros((len(x), 4, 4), dtype=np.float64)
    hess[:, 0, 0] = 2*c0
    hess[:, 1, 1] = 2*b1 + 2*b1*tau4*(3*k1*k1-c2)*inv**3
    hess[:, 2, 2] = 2*b2
    hess[:, 3, 3] = 2*bt + 12*b1*tau2*inv
    hess[:, 1, 3] = hess[:, 3, 1] = -8*b1*k1*tau3*inv**2
    return energy, grad, hess


def install_fast_sano(stepper):
    """Replace this stepper's Sano batch method; return its previous method.

    Usage: ``original = install_fast_sano(stepper)`` after make_robot().
    Restore with ``stepper.energy_model.compute_energy_grad_hess_batch = original``.
    The existing GeneralElasticEnergySano holds the same model object. Its
    forward() and natural-strain cache are deliberately left unchanged.
    """
    from dismech.elastics.analytical_sanos_elastic_energy import AnalyticalSanosElasticEnergy

    model = stepper.energy_model
    if type(model) is not AnalyticalSanosElasticEnergy:
        raise TypeError("Closed-form adapter only supports the pinned analytical Sano model")
    original = model.compute_energy_grad_hess_batch
    model.compute_energy_grad_hess_batch = MethodType(energy_grad_hess_batch, model)
    return original


def self_check():
    """Independent Torch derivatives and scaled finite differences; no file writes."""
    import json
    import math
    from pathlib import Path
    import subprocess
    import torch
    from dismech.elastics.analytical_sanos_elastic_energy import AnalyticalSanosElasticEnergy

    here = Path(__file__).resolve().parent
    vendor = here / "vendor/discrete-elastic-ribbon"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    assert revision == "c9d341164e2927fc24b2c43dff97fcfb492cf700"
    assert vendor.resolve() in Path(__import__("dismech").__file__).resolve().parents

    rng = np.random.default_rng(617)
    explicit = np.array([[0, 0, 0, 0], [.1, .1, .1, .1], [-.1, -.1, -.1, -.1],
                         [1, 5, -4, 3], [-1, -5, 4, -3], [0, 0, 0, 6],
                         [0, 0, 0, -6], [0, 5, 0, 0], [0, -5, 0, 0],
                         [0, 0, 5, 0], [0, 0, -5, 0]])
    y = np.vstack((explicit, rng.uniform(-6, 6, (85, 4))))
    worst = {name: 0. for name in ("autodiff_energy", "autodiff_gradient", "autodiff_hessian",
                                  "fd_gradient", "fd_hessian")}
    samples = 0
    # Thin steel strips spanning plausible widths, thicknesses and segment lengths.
    configurations = ((.016, .00015, 200e9, .3, .003),
                      (.008, .00008, 170e9, .22, .0015),
                      (.025, .0004, 210e9, .38, .012))
    for width, thickness, young, nu, length in configurations:
        torsion = width*thickness**3*(1/3-.21*(thickness/width)*(1-thickness**4/(12*width**4)))
        model = AnalyticalSanosElasticEnergy(
            EA=young*width*thickness, EI1=young*width*thickness**3/12,
            EI2=young*thickness*width**3/12, GJ=young/(2*(1+nu))*torsion,
            delta_l=length, zeta=math.sqrt((1-nu)*width**4/(60*thickness**2)), h=thickness)
        # Test a second parameter set via the supported in-place update interface.
        for updated in (False, True):
            if updated:
                model.update_params(EA=model.EA*1.1, EI1=model.EI1*.9, EI2=model.EI2*1.2,
                                    GJ=model.GJ*.8, delta_l=length*.7, zeta=model.zeta*1.3,
                                    h=thickness*1.1)
            scale = np.array([.001, *([model.h/model.zeta]*3)])
            x = y*scale
            actual = energy_grad_hess_batch(model, x)
            expected = model.compute_energy_grad_hess_batch(torch.tensor(x, dtype=torch.float64))
            zero_h = energy_grad_hess_batch(model, np.zeros((1, 4)))[2][0]
            # Per-component natural stiffness scales avoid mixing strain units,
            # and retain meaningful denominators at exact zeros/cancellations.
            natural = np.diag(zero_h)*scale**2
            energy_scale = natural.sum()
            transforms = (1., scale, scale[:, None]*scale[None, :])
            floors = (energy_scale, natural, np.sqrt(natural[:, None]*natural[None, :]))
            for name, got, ref, transform, floor in zip(
                    ("energy", "gradient", "hessian"), actual, expected, transforms, floors):
                error = np.max(np.abs((got-ref)*transform)/(np.abs(ref*transform)+floor))
                worst["autodiff_"+name] = max(worst["autodiff_"+name], float(error))
                assert error < 3e-13, (name, updated, error)
            assert actual[0][0] == 0 and np.all(actual[1][0] == 0)
            assert np.array_equal(actual[2], actual[2].swapaxes(1, 2))
            one = energy_grad_hess_batch(model, torch.tensor(x[:1], requires_grad=True))
            assert [v.shape for v in one] == [(1,), (1, 4), (1, 4, 4)]
            # Central differences in dimensionless y=x/scale, with Richardson
            # cancellation of the leading O(h^2) term. Energy and gradients on
            # the FD side come from the original vendor, not this adapter.
            fd_g, fd_h = np.empty_like(actual[1]), np.empty_like(actual[2])
            for j in range(4):
                estimates = []
                for h in (.002, .001):
                    dx = np.zeros(4); dx[j] = h*scale[j]
                    ep, gp, _ = model.compute_energy_grad_hess_batch(torch.tensor(x+dx))
                    em, gm, _ = model.compute_energy_grad_hess_batch(torch.tensor(x-dx))
                    estimates.append(((ep-em)/(2*h), (gp-gm)*scale/(2*h)))
                fd_g[:, j] = (4*estimates[1][0]-estimates[0][0])/3
                fd_h[:, :, j] = (4*estimates[1][1]-estimates[0][1])/3
            gy, hy = actual[1]*scale, actual[2]*scale[:, None]*scale[None, :]
            gradient_error = np.max(np.abs(fd_g-gy)/(np.abs(gy)+natural))
            hessian_error = np.max(np.abs(fd_h-hy)/(np.abs(hy)+floors[2]))
            worst["fd_gradient"] = max(worst["fd_gradient"], float(gradient_error))
            worst["fd_hessian"] = max(worst["fd_hessian"], float(hessian_error))
            assert gradient_error < 2e-6 and hessian_error < 2e-8, (gradient_error, hessian_error)
            samples += len(x)

    # Check the real stepper path with cached nonzero natural curvature. Merely
    # patching a detached model would pass the material checks but fail this one.
    from run import read_project, strip_geometry, make_robot, impose
    p, delta, _ = read_project(here/"output/parameters.snapshot.json")
    rest, widths = strip_geometry(p, delta, 0, 9)
    robot, stepper = make_robot(p, rest, widths, "sano")
    energy = stepper._TimeStepper__elastic_energies[0]
    natural_before = energy._nat_strain.copy()
    states = [robot, impose(robot, rest, widths, delta+p["back_plate_center_m"],
                            2, .001, math.radians(2))]
    before = [energy.grad_hess_energy_linear_elastic(r.state) for r in states]
    original = install_fast_sano(stepper)
    assembly_errors = []
    try:
        assert energy.nn_model is stepper.energy_model
        for r, baseline in zip(states, before):
            for fast, ref in zip(energy.grad_hess_energy_linear_elastic(r.state), baseline):
                error = float(np.max(np.abs(fast-ref))/max(np.max(np.abs(ref)), 1.))
                assembly_errors.append(error)
                assert error < 3e-13, error
        assert np.array_equal(energy._nat_strain, natural_before)
    finally:
        stepper.energy_model.compute_energy_grad_hess_batch = original
    report = dict(status="passed", vendor_commit=revision, material_samples=samples,
                  parameter_sets=6, max_scaled_errors=worst,
                  max_assembled_relative_error=max(assembly_errors),
                  natural_strain_cache_unchanged=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    self_check()

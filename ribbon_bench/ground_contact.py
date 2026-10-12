"""CPU float64 plane penalty contact with history-dependent Coulomb friction.

The plane normal is world +Z. Material point IDs are array row indices. Keep
oldpoints/history fixed throughout a Newton solve; commit returned history only
after accepting the time step. This law has tangential elastic pre-slip and is
not a globally conservative contact potential. Run --self-check for verification.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def evaluate_contact(points, oldpoints, history=None, *, dt, ground_height,
                     normal_stiffness, tangential_stiffness=None, weights=1.0,
                     mu=0.5):
    """Return physical forces, d(force)/d(points), energies and candidate history.

    points/oldpoints: (P,3), metres, current trial/last accepted world positions.
    history: None, or {'elastic_slip_m': (P,2), 'active': (P,)} from the last
        accepted call, with unchanged material IDs and stiffness parameters.
    normal_stiffness/tangential_stiffness: positive scalars or (P,) arrays.
        Multiplying by nonnegative scalar/(P,) weights must yield N/m per point.
        For area weights [m^2], supply stiffness density [N/m^3]; for length
        weights [m], supply [N/m^2]. Zero weights disable a point.
    dt: positive seconds; the law is rate independent, using displacement, not
        velocity. mu is one nonnegative coefficient for both stick and slip.

    Jacobians are piecewise exact on open active/stick/slip branches. At branch
    boundaries this returns the selected branch tangent, not a smooth Hessian.
    All history arrays are new; the function never mutates its inputs.
    """
    points = np.asarray(points, dtype=np.float64)
    oldpoints = np.asarray(oldpoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or oldpoints.shape != points.shape:
        raise ValueError('points and oldpoints must both have shape (P,3)')
    if not np.isfinite(points).all() or not np.isfinite(oldpoints).all():
        raise ValueError('Point positions must be finite')
    if not np.isfinite([dt, ground_height, mu]).all() or dt <= 0 or mu < 0:
        raise ValueError('Need finite dt>0, ground_height and mu>=0')
    count = len(points)

    def per_point(value, name, positive):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 0:
            value = np.full(count, float(value))
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite scalar or shape (P,)')
        if np.any(value <= 0 if positive else value < 0):
            raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
        return value

    weight = per_point(weights, 'weights', False)
    kn = per_point(normal_stiffness, 'normal_stiffness', True)*weight
    kt = per_point(normal_stiffness if tangential_stiffness is None else
                   tangential_stiffness, 'tangential_stiffness', True)*weight
    if not np.isfinite(kn).all() or not np.isfinite(kt).all():
        raise ValueError('Weighted stiffness overflow')
    old_elastic = np.zeros((count, 2))
    old_active = np.zeros(count, dtype=bool)
    if history is not None:
        old_elastic = np.array(history['elastic_slip_m'], dtype=np.float64, copy=True)
        old_active = np.array(history['active'], dtype=bool, copy=True)
        if old_elastic.shape != (count, 2) or old_active.shape != (count,):
            raise ValueError('History shape must match fixed material point IDs')
        if not np.isfinite(old_elastic).all():
            raise ValueError('History must be finite')
        if np.any(old_elastic[~old_active] != 0):
            raise ValueError('Inactive history must have zero elastic slip')

    gap = points[:, 2]-float(ground_height)
    normal = kn*np.maximum(-gap, 0.)
    active = normal > 0
    trial = old_elastic + points[:, :2]-oldpoints[:, :2]
    trial_norm = np.linalg.norm(trial, axis=1)
    limit = float(mu)*normal
    slip = active & (mu > 0) & (kt*trial_norm > limit)
    stick = active & (mu > 0) & ~slip
    elastic = np.zeros_like(trial)
    force = np.zeros_like(points)
    jacobian = np.zeros((count, 3, 3))
    force[:, 2] = normal
    jacobian[active, 2, 2] = -kn[active]

    elastic[stick] = trial[stick]
    force[stick, :2] = -kt[stick, None]*trial[stick]
    jacobian[stick, 0, 0] = -kt[stick]
    jacobian[stick, 1, 1] = -kt[stick]
    if np.any(slip):
        direction = trial[slip]/trial_norm[slip, None]
        elastic[slip] = limit[slip, None]/kt[slip, None]*direction
        force[slip, :2] = -limit[slip, None]*direction
        projection = np.eye(2)[None]-direction[:, :, None]*direction[:, None, :]
        jacobian[slip, :2, :2] = -limit[slip, None, None]/trial_norm[slip, None, None]*projection
        # Increasing height reduces N, hence reduces the magnitude of friction.
        # The reciprocal normal/tangential entry is zero: this is not a Hessian.
        jacobian[slip, :2, 2] = float(mu)*kn[slip, None]*direction

    plastic_increment = np.zeros_like(trial)
    plastic_increment[active] = trial[active]-elastic[active]
    dissipation = limit*np.linalg.norm(plastic_increment, axis=1)
    detached = old_active & ~active
    reset_loss = .5*kt[detached]*np.sum(old_elastic[detached]**2, axis=1)
    status = np.full(count, 'separated', dtype='<U12')
    status[stick] = 'stick'
    status[slip] = 'slip'
    status[active & (mu == 0)] = 'frictionless'
    return dict(force=force, jacobian=jacobian,
                normal_energy_j=float(.5*np.sum(kn*np.minimum(gap, 0.)**2)),
                tangential_energy_j=float(.5*np.sum(kt*np.sum(elastic**2, axis=1))),
                friction_dissipation_j=float(np.sum(dissipation)),
                history_reset_dissipation_j=float(np.sum(reset_loss)),
                history=dict(elastic_slip_m=elastic, active=active.copy()),
                gap_m=gap, normal_force_n=normal, status=status,
                plastic_increment_m=plastic_increment)


def self_check():
    """Small deterministic law tests only; no ribbon simulation or file writes."""
    # A supported block: 2 kg, four equal patches, 20 micrometre target penalty
    # penetration. kn_total=mg/delta, independent of point count via area weights.
    mass, gravity, delta, mu = 2., 9.81, 20e-6, .4
    old = np.array([[-.1, -.1, 0.], [-.1, .1, 0.], [.1, -.1, 0.], [.1, .1, 0.]])
    points = old.copy()
    points[:, 2] = -delta
    settings = dict(dt=.001, ground_height=0., normal_stiffness=mass*gravity/delta,
                    tangential_stiffness=mass*gravity/delta, weights=np.full(4, .25), mu=mu)
    support = evaluate_contact(points, old, **settings)
    np.testing.assert_allclose(support['force'].sum(0), [0., 0., mass*gravity], rtol=1e-14)
    assert np.all(support['status'] == 'stick')
    np.testing.assert_allclose(support['normal_energy_j'], .5*mass*gravity*delta)

    # Below threshold: displacement-controlled load has finite elastic pre-slip.
    # Hold that configuration for another step: static friction must not vanish.
    kn_total = mass*gravity/delta
    threshold = mu*mass*gravity
    moved = points.copy()
    moved[:, 0] += .5*threshold/kn_total
    stored_before = support['history']['elastic_slip_m'].copy()
    sticking = evaluate_contact(moved, points, support['history'], **settings)
    np.testing.assert_allclose(-sticking['force'][:, 0].sum(), .5*threshold, rtol=1e-10)
    assert np.all(sticking['status'] == 'stick')
    assert sticking['friction_dissipation_j'] == 0.
    held = evaluate_contact(moved, moved, sticking['history'], **settings)
    np.testing.assert_allclose(held['force'], sticking['force'])
    np.testing.assert_array_equal(support['history']['elastic_slip_m'], stored_before)
    assert not np.shares_memory(sticking['history']['elastic_slip_m'], support['history']['elastic_slip_m'])
    at_threshold = points.copy()
    at_threshold[:, 0] += threshold/kn_total
    transition = evaluate_contact(at_threshold, points, support['history'], **settings)
    np.testing.assert_allclose(-transition['force'][:, 0].sum(), threshold, rtol=1e-10)

    sliding_points = moved.copy()
    sliding_points[:, :2] += [3*threshold/kn_total, 2*threshold/kn_total]
    sliding = evaluate_contact(sliding_points, moved, sticking['history'], **settings)
    assert np.all(sliding['status'] == 'slip')
    np.testing.assert_allclose(np.linalg.norm(sliding['force'][:, :2], axis=1),
                               mu*sliding['normal_force_n'], rtol=1e-14)
    du = sliding_points[:, :2]-moved[:, :2]
    contact_work = float(np.sum(sliding['force'][:, :2]*du))
    assert contact_work < 0 and sliding['friction_dissipation_j'] > 0
    # Backward-Euler endpoint work bounds stored energy plus plastic dissipation.
    energy_balance = (-contact_work - (sliding['tangential_energy_j']-
                      sticking['tangential_energy_j'])-sliding['friction_dissipation_j'])
    assert energy_balance >= -1e-12

    lifted = sliding_points.copy()
    lifted[:, 2] = delta
    separated = evaluate_contact(lifted, sliding_points, sliding['history'], **settings)
    assert not np.any(separated['force']) and not np.any(separated['jacobian'])
    assert not np.any(separated['history']['elastic_slip_m'])
    assert not np.any(separated['history']['active'])
    assert separated['normal_energy_j'] == separated['tangential_energy_j'] == 0.
    np.testing.assert_allclose(separated['history_reset_dissipation_j'], sliding['tangential_energy_j'])
    recontacted = lifted.copy()
    recontacted[:, 2] = -delta
    reset = evaluate_contact(recontacted, lifted, separated['history'], **settings)
    assert not np.any(reset['force'][:, :2])

    # Central differences strictly inside separated, stick and sliding branches.
    # Sliding has nonzero x/y components and checks the d(T)/d(N) z column.
    max_scaled_jacobian_error = 0.
    for current, previous, history in ((points, old, None),
            (moved, points, support['history']),
            (sliding_points, moved, sticking['history']),
            (lifted, sliding_points, sliding['history'])):
        exact = evaluate_contact(current, previous, history, **settings)
        fd = np.zeros_like(exact['jacobian'])
        step = 1e-9
        for j in range(3):
            plus, minus = current.copy(), current.copy()
            plus[:, j] += step
            minus[:, j] -= step
            ep = evaluate_contact(plus, previous, history, **settings)
            em = evaluate_contact(minus, previous, history, **settings)
            assert np.array_equal(ep['status'], exact['status'])
            assert np.array_equal(em['status'], exact['status'])
            fd[:, :, j] = (ep['force']-em['force'])/(2*step)
        error = float(np.max(np.abs(fd-exact['jacobian']))/kn_total)
        max_scaled_jacobian_error = max(max_scaled_jacobian_error, error)
        assert error < 1e-7, error
    assert np.any(sliding['jacobian'][:, :2, 2] != 0)
    assert not np.any(sliding['jacobian'][:, 2, :2])
    # Zero friction and masked quadrature points have no hidden tangential force.
    frictionless = evaluate_contact(sliding_points, moved, sticking['history'], **{**settings, 'mu': 0.})
    assert not np.any(frictionless['force'][:, :2])
    stationary_frictionless = evaluate_contact(points, old, **{**settings, 'mu': 0.})
    assert not np.any(stationary_frictionless['jacobian'][:, :2, :])
    assert np.all(stationary_frictionless['status'] == 'frictionless')
    masked = evaluate_contact(points, old, **{**settings, 'weights': 0.})
    assert not np.any(masked['force']) and not np.any(masked['jacobian'])
    report = dict(status='passed', supported_weight_n=mass*gravity,
                  target_penetration_m=delta, stick_slip_threshold_n=threshold,
                  max_scaled_force_jacobian_fd_error=max_scaled_jacobian_error,
                  sliding_contact_work_j=contact_work,
                  sliding_friction_dissipation_j=sliding['friction_dissipation_j'],
                  sliding_incremental_numerical_loss_j=energy_balance,
                  immutable_trial_history=True, separation_resets_history=True,
                  scope='Plane contact constitutive-law tests; no ribbon or full dynamic validation')
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if not args.self_check:
        parser.error('Use --self-check to run the contact-law verification')
    self_check()

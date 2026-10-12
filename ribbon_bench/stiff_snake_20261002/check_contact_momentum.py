"""Small CPU drive/adaptive/rollback check for observational momentum logging."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from full_robot import FullRobot
from gpu_joint_wave_20261002.check import snapshot_module
from run import read_project


def main():
    p, delta, _ = read_project(HERE/'output/parameters.snapshot.json')
    sim = FullRobot(p, delta, nodes=9, segments=2, unlocked_joints=True,
                    dt=.002, contact_samples=12)
    mass = float(sim.body_mass.sum()+sim.nstrips*sim.mass_steel[:3*sim.nodes:3].sum())
    reports = []

    def check(info, before):
        np.testing.assert_allclose(info['dt_s'], .002, atol=1e-18, rtol=0)
        np.testing.assert_allclose(info['linear_momentum_before_kg_m_s'], before, atol=1e-14, rtol=0)
        np.testing.assert_allclose(info['linear_momentum_after_kg_m_s'], sim._linear_momentum(), atol=1e-14, rtol=0)
        np.testing.assert_allclose(info['gravity_impulse_world_ns'], [0., 0., -9.81*mass*.002], atol=1e-14, rtol=0)
        error = (np.asarray(info['linear_momentum_after_kg_m_s'])-before
                 - info['ground_impulse_world_ns']-info['gravity_impulse_world_ns'])
        np.testing.assert_allclose(error, info['linear_momentum_balance_error_ns'], atol=1e-14, rtol=0)
        assert np.max(abs(error)) <= info['linear_momentum_balance_tolerance_ns']+1e-12
        assert info['substep_max_linear_momentum_balance_ratio'] <= 1.+1e-6
        assert info['contact_force_evaluated']
        np.testing.assert_allclose(info['ground_force_world_n'][2], info['contact_normal_force_n'], atol=1e-10, rtol=0)
        reports.append(dict(command_substeps=info['command_substeps'], recovery_substeps=info['recovery_substeps'],
                            balance_error_ns=error.tolist(), tolerance_ns=info['linear_momentum_balance_tolerance_ns'],
                            worst_substep_ratio=info['substep_max_linear_momentum_balance_ratio']))

    before = sim._linear_momentum().copy()
    first = sim.step_adaptive(sim.base_cable_lengths+.01, max_rest_step_m=.02, joint_targets=np.array([.05]))
    check(first, before)
    assert first['command_substeps'] >= 2 and abs(first['angles_rad'][0]) > 1e-8

    frozen = snapshot_module('pre_momentum_full_robot', HERE/'snake_wave_20261002/snake_n9_cuda/full_robot.snapshot.py')
    reference = frozen.FullRobot(p, delta, nodes=9, segments=2, unlocked_joints=True,
                                 dt=.002, contact_samples=12)
    reference.step_adaptive(reference.base_cable_lengths+.01, max_rest_step_m=.02, joint_targets=np.array([.05]))
    parity = {}
    for name in ('q', 'u', 'body_com', 'body_R', 'body_v', 'body_omega'):
        parity[name] = float(np.max(np.abs(getattr(sim, name)-getattr(reference, name))))
        np.testing.assert_array_equal(getattr(sim, name), getattr(reference, name))

    # Fail once after a provisional commit: the retry must see the backed-up state.
    real_step = sim.step
    before = sim._linear_momentum().copy()
    q_before = sim.q.copy()
    elastic_before = [None if x is None else x['elastic_slip_m'].copy() for x in sim.steel_history]
    calls = 0
    def fail_once(rest, angles):
        nonlocal calls
        calls += 1
        if calls == 2:
            np.testing.assert_array_equal(sim.q, q_before)
            np.testing.assert_array_equal(sim._linear_momentum(), before)
            for old, current in zip(elastic_before, sim.steel_history):
                if old is not None:
                    np.testing.assert_array_equal(old, current['elastic_slip_m'])
        info = real_step(rest, angles)
        if calls == 1:
            raise RuntimeError('intentional failure after provisional acceptance')
        return info
    sim.step = fail_once
    try:
        second = sim.step_adaptive(sim.base_cable_lengths+.01, max_rest_step_m=.02, joint_targets=np.array([-.01]))
    finally:
        sim.step = real_step
    check(second, before)
    assert second['recovery_substeps'] >= 1
    report = dict(status='passed', backend='cpu', segments=2, nodes=9,
                          driven_joint=True, post_commit_rollback_checked=True,
                          unchanged_physics_snapshot_parity=parity,
                          scope='Log aggregation and linear momentum balance; not locomotion validation',
                          steps=reports)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    Path(__file__).with_name('contact_momentum_check.json').write_text(serialized+'\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()

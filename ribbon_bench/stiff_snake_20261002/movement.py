"""Audit saved chain translation without rerunning mechanics; plot raw saved frames."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('saved_chain_audit', HERE.parent/'gpu_joint_wave_20261002/check.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def center_of_mass(c, xyz, rigid_mass, node_mass):
    mass = rigid_mass.sum()+xyz.shape[1]*node_mass.sum()
    return (np.einsum('tbi,b->ti', c, rigid_mass)+np.einsum('tsni,n->ti', xyz, node_mass))/mass


def audit(folder, output):
    folder, output = Path(folder), Path(output)
    summary, data = previous.load(folder)
    meta, frames = summary['metadata'], summary['frames']
    times = np.asarray([f['time_s'] for f in frames])
    nt, strips, nq = data['q'].shape
    nodes = (nq+1)//4
    assert nq == 4*nodes-1 and nodes == meta['nodes'] and strips == meta['strips']
    assert nt > 1 and times[0] == 0 and np.all(np.diff(times) > 0)
    np.testing.assert_allclose(np.diff(times), meta['dt_s'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(times[-1], meta['duration_s'], atol=1e-12, rtol=0)
    xyz = data['q'][..., :3*nodes].reshape(nt, strips, nodes, 3)
    c, R = data['body_com'], data['body_R']
    masses, node_mass = data['body_mass_kg'], data['steel_node_mass_kg']
    assert c.shape == (nt, len(masses), 3) and R.shape == (nt, len(masses), 3, 3)
    assert node_mass.shape == (nodes,) and np.all(masses > 0) and np.all(node_mass > 0)
    com = center_of_mass(c, xyz, masses, node_mass)
    # Independently concatenate every material point, avoiding the einsum path.
    point_mass = np.r_[masses, np.tile(node_mass, strips)]
    points = np.concatenate((c, xyz.reshape(nt, strips*nodes, 3)), axis=1)
    independent_com = (points*point_mass[None, :, None]).sum(axis=1)/point_mass.sum()
    com_error = float(np.max(abs(com-independent_com)))
    np.testing.assert_allclose(com, independent_com, atol=1e-14, rtol=0)
    shift = np.array([.012, -.027, .019])
    np.testing.assert_allclose(center_of_mass(c+shift, xyz+shift, masses, node_mass), com+shift, atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.asarray([f['body_com'] for f in frames]), c, atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.asarray([f.get('body_R', R[i]) for i, f in enumerate(frames)]), R, atol=1e-13, rtol=0)
    b = data['plate_body_indices']
    plates = c[:, b]+np.einsum('tpij,pj->tpi', R[:, b], data['plate_centers_local_m']-data['body_com_local_m'][b])
    assert plates.shape == (nt, 2*meta['segments'], 3)
    modules = (plates[:, ::2]+plates[:, 1::2])/2
    lengths = np.linalg.norm(plates[:, ::2]-plates[:, 1::2], axis=2)
    np.testing.assert_allclose(lengths, [f['segment_lengths_m'] for f in frames], atol=1e-13, rtol=0)
    head, tail = plates[:, 0], plates[:, -1]
    initial_direction = (head[0]-plates[0, 1]).copy()
    initial_direction[2] = 0
    assert np.linalg.norm(initial_direction) > 1e-12
    initial_direction /= np.linalg.norm(initial_direction)
    lateral = np.array([-initial_direction[1], initial_direction[0], 0.])
    basis = np.array([initial_direction, lateral, [0., 0., 1.]])
    planar_delta = (com-com[0])@basis.T*1000
    local_y = np.einsum('tsi,i->ts', modules-com[:, None], lateral)*1000
    angles, targets = data['joint_angles_rad'], data['joint_targets_rad']
    period = meta['drive_period_s']
    assert period > 0
    phase_indices = []
    for t in np.arange(int(np.floor(times[-1]/period+1e-12))+1)*period:
        i = int(np.argmin(abs(times-t)))
        if abs(times[i]-t) <= 1e-10:
            phase_indices.append(i)
    phase_records = [dict(frame_index=i, time_s=float(times[i]), total_com_world_mm=(com[i]*1000).tolist(),
        total_com_delta_initial_heading_mm=planar_delta[i].tolist(), module_centers_world_mm=(modules[i]*1000).tolist(),
        head_plate_center_world_mm=(head[i]*1000).tolist(), tail_plate_center_world_mm=(tail[i]*1000).tolist(),
        joint_actual_deg=np.rad2deg(angles[i]).tolist(), joint_target_deg=np.rad2deg(targets[i]).tolist()) for i in phase_indices]
    cycles = []
    for i, j in zip(phase_indices, phase_indices[1:]):
        same_target = float(np.max(abs(targets[j]-targets[i]), initial=0)) < 1e-10
        # Shape agreement after removing arbitrary rigid planar translation and yaw.
        a, z = modules[i, :, :2], modules[j, :, :2]
        a, z = a-a.mean(0), z-z.mean(0)
        u, _, vh = np.linalg.svd(a.T@z)
        correction = np.eye(2); correction[-1, -1] = np.linalg.det(u@vh)
        aligned = a@(u@correction@vh)
        steel_a, steel_z = xyz[i].reshape(-1, 3), xyz[j].reshape(-1, 3)
        steel_a, steel_z = steel_a-steel_a.mean(0), steel_z-steel_z.mean(0)
        su, _, sv = np.linalg.svd(steel_a.T@steel_z)
        scorrection = np.eye(3); scorrection[-1, -1] = np.linalg.det(su@sv)
        steel_error = steel_a@(su@scorrection@sv)-steel_z
        cycle_delta = (com[j]-com[i])@basis.T*1000
        cycles.append(dict(start_s=float(times[i]), end_s=float(times[j]), same_commanded_shape=same_target,
            includes_startup=bool(times[i] < .4 and meta.get('joint_drive', {}).get('preset') == 'phase-sine'),
            com_delta_initial_heading_mm=cycle_delta.tolist(), com_mean_planar_velocity_mm_s=(cycle_delta[:2]/(times[j]-times[i])).tolist(),
            module_centers_delta_initial_heading_mm=((modules[j]-modules[i])@basis.T*1000).tolist(),
            head_delta_initial_heading_mm=((head[j]-head[i])@basis.T*1000).tolist(),
            tail_delta_initial_heading_mm=((tail[j]-tail[i])@basis.T*1000).tolist(),
            max_actual_joint_shape_difference_deg=float(np.rad2deg(np.max(abs(angles[j]-angles[i]), initial=0))),
            module_center_shape_rms_after_planar_alignment_mm=float(np.sqrt(np.mean(np.sum((aligned-z)**2, axis=1)))*1000),
            steel_node_shape_rms_after_3d_alignment_mm=float(np.sqrt(np.mean(np.sum(steel_error**2, axis=1)))*1000),
            steel_node_shape_max_after_3d_alignment_mm=float(np.linalg.norm(steel_error, axis=1).max()*1000),
            shape_metric_scope='Unweighted, corresponding saved material nodes after a proper rigid 3D fit; discrete shape recurrence, not mesh/step convergence.'))
    status_frames = [f['status_counts'] for f in frames[1:]]
    keys = sorted(set().union(*(f.keys() for f in status_frames)))
    status = {k: np.asarray([f.get(k, 0) for f in status_frames]) for k in keys}
    totals = np.array([sum(f.values()) for f in status_frames])
    assert np.all(totals == totals[0]) and totals[0] > 0
    counts = {k: dict(min=int(v.min()), max=int(v.max()), mean=float(v.mean())) for k, v in status.items()}
    active = totals-status.get('separated', np.zeros_like(totals))
    force = np.asarray([f['contact_normal_force_n'] for f in frames[1:]])
    assert np.all(force >= 0)
    path = float(np.linalg.norm(np.diff(planar_delta[:, :2], axis=0), axis=1).sum())
    net = float(np.linalg.norm(planar_delta[-1, :2]))
    momentum = audit_momentum(frames, times, com, float(point_mass.sum()), strips, nodes, len(masses))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(status='passed', run_path=str(folder), frames=nt, nodes=nodes, segments=meta['segments'],
        input_sha256={name: sha(folder/name) for name in ('summary.json', 'trajectory.npz')},
        checks=dict(independent_mass_weighted_com_max_error_m=com_error, global_translation_invariance=True,
            summary_npz_body_coordinates_match=True, segment_length_reconstruction_match=True),
        mass=dict(total_kg=float(point_mass.sum()), rigid_kg=float(masses.sum()), steel_kg=float(strips*node_mass.sum()),
            included='All saved CAD rigid masses (including locked wheels) and every translational ribbon node mass; cables are massless.'),
        coordinate_definition=dict(world='Saved inertial world coordinates; no camera transform or recentering.',
            initial_forward_world=initial_direction.tolist(), initial_lateral_world=lateral.tolist(),
            module_center='Arithmetic midpoint of that module\u0027s two geometric plate centers, not its mass center.',
            head_tail='Geometric centers of the first front plate and last back plate, not extreme steel/contact points.'),
        motion=dict(com_net_delta_world_mm=((com[-1]-com[0])*1000).tolist(), com_net_delta_initial_heading_mm=planar_delta[-1].tolist(),
            com_planar_path_length_mm=path, com_planar_net_distance_mm=net, com_planar_net_to_path_ratio=net/path if path > 0 else 0.,
            path_definition='Sum of horizontal Euclidean COM increments over every consecutive saved main frame; excludes vertical movement.',
            com_planar_component_peak_to_peak_mm=np.ptp(planar_delta[:, :2], axis=0).tolist(),
            com_height_peak_to_peak_mm=float(np.ptp(com[:, 2])*1000),
            module_center_lateral_peak_to_peak_relative_com_mm=np.ptp(local_y, axis=0).tolist(),
            max_module_contraction_from_initial_mm=((lengths[0]-lengths).max(0)*1000).tolist()),
        phase_samples=phase_records, cycle_displacements=cycles, momentum_balance=momentum,
        contact=dict(scope='Saved main-step endpoints t>0 only; t=0 contains synthetic initialization values and has no solved contact counts.',
            samples_per_frame=int(totals[0]), sample_status_counts=counts,
            active_samples_min=int(active.min()), active_samples_max=int(active.max()),
            normal_force_n=dict(min=float(force.min()), max=float(force.max()), mean=float(force.mean())),
            substep_max_penetration_m=float(max(f.get('substep_max_penetration_m', f['max_penetration_m']) for f in frames[1:])),
            interpretation='Counts are steel-surface and plate-rim samples, not contact area fractions. Aggregates cannot identify sample-by-sample stick/slip transitions or which component contacted.'),
        interpretation='Endpoint net COM shift is resolved in the saved trajectory. Startup settling and changing shape must be separated with same-phase samples. One post-startup cycle cannot establish steady locomotion, numerical convergence or experimentally correct friction.',
        limitations=['Per-point ground forces, contact identities and every accepted substep velocity are not saved; detailed component contact trajectories cannot be reconstructed.',
            'No no-drive control, dt/mesh convergence or forward/reverse comparison is contained in this one-run movement audit.',
            'CAD servo/connector meshes and wheel contacts are absent from this steel/plate contact model.'],
        time_series=dict(time_s=times.tolist(), com_world_mm=(com*1000).tolist(), com_delta_initial_heading_mm=planar_delta.tolist(),
            module_centers_world_mm=(modules*1000).tolist(), head_world_mm=(head*1000).tolist(), tail_world_mm=(tail*1000).tolist()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    plot(output.with_suffix(''), times, planar_delta, local_y, phase_indices)
    print(json.dumps(dict(saved=str(output), motion=result['motion'], cycle_displacements=cycles), ensure_ascii=False))
    return result


def audit_momentum(frames, times, com, total_mass, strips, nodes, bodies):
    """Audit optional macro-step impulses; baseline snapshots did not record them."""
    if 'linear_momentum_after_kg_m_s' not in frames[1]:
        return dict(status='not_recorded', reason='This run predates world-force, momentum and accepted-substep impulse logging; no retrospective force balance is possible.')
    def values(key):
        a = np.asarray([f[key] for f in frames])
        assert np.isfinite(a).all(), key
        return a
    before, after = values('linear_momentum_before_kg_m_s'), values('linear_momentum_after_kg_m_s')
    ground, gravity = values('ground_impulse_world_ns'), values('gravity_impulse_world_ns')
    saved_error = values('linear_momentum_balance_error_ns')
    tolerance = values('linear_momentum_balance_tolerance_ns')[1:]
    duration = np.diff(times)
    assert before.shape == after.shape == ground.shape == gravity.shape == saved_error.shape == (len(times), 3)
    np.testing.assert_allclose(after[0], np.zeros(3), atol=1e-14, rtol=0)
    np.testing.assert_allclose(before[1:], after[:-1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(values('dt_s')[1:], duration, atol=1e-12, rtol=0)
    expected_gravity = np.zeros_like(gravity[1:]); expected_gravity[:, 2] = -9.81*total_mass*duration
    np.testing.assert_allclose(gravity[1:], expected_gravity, atol=1e-12, rtol=0)
    expected_tolerance = duration*1e-5*(strips*(nodes-4)+bodies)
    np.testing.assert_allclose(tolerance, expected_tolerance, atol=1e-14, rtol=0)
    error = after-before-ground-gravity
    np.testing.assert_allclose(error, saved_error, atol=1e-12, rtol=0)
    ratio = np.max(abs(error[1:]), axis=1)/tolerance
    assert ratio.max() <= 1.01, 'Macro-step force/momentum closure exceeds the residual-based bound'
    substep_ratio = values('substep_max_linear_momentum_balance_ratio')[1:]
    assert substep_ratio.max() <= 1.01, 'An accepted substep exceeds the residual-based momentum bound'
    single = np.asarray([f.get('command_substeps', 1) == 1 for f in frames[1:]])
    ground_force = values('ground_force_world_n')
    assert ground_force.shape == (len(times), 3)
    assert frames[0]['contact_force_evaluated'] is False
    assert all(f['contact_force_evaluated'] is True for f in frames[1:])
    normal = values('contact_normal_force_n')
    assert np.all(normal >= 0) and np.all(ground_force[:, 2] >= 0) and np.all(ground[:, 2] >= 0)
    np.testing.assert_allclose(ground_force[:, 2], normal, atol=1e-10, rtol=0)
    # CLI runs use FullRobot's fixed mu=0.4; this tests aggregate cones, not contact histories.
    mu = .4
    force_cone_excess = np.linalg.norm(ground_force[:, :2], axis=1)-mu*ground_force[:, 2]
    impulse_cone_excess = np.linalg.norm(ground[:, :2], axis=1)-mu*ground[:, 2]
    assert np.all(force_cone_excess <= 1e-12*(1+ground_force[:, 2])), 'Aggregate ground force exceeds the Coulomb cone'
    assert np.all(impulse_cone_excess <= 1e-12*(1+ground[:, 2])), 'Macro-step ground impulse exceeds the Coulomb cone'
    np.testing.assert_allclose(ground[1:][single], ground_force[1:][single]*duration[single, None], atol=1e-12, rtol=0)
    # With one accepted substep, saved positions independently give implicit endpoint velocity.
    position_momentum = total_mass*np.diff(com, axis=0)/duration[:, None]
    position_error = np.max(abs(position_momentum[single]-after[1:][single]), initial=0)
    np.testing.assert_allclose(position_momentum[single], after[1:][single], atol=1e-10, rtol=0)
    global_error = after[-1]-after[0]-ground.sum(0)-gravity.sum(0)
    np.testing.assert_allclose(global_error, error.sum(0), atol=1e-11, rtol=0)
    return dict(status='passed', accepted_main_steps=len(times)-1, single_substep_position_checks=int(single.sum()),
        single_substep_position_momentum_max_error_kg_m_s=float(position_error),
        max_macro_component_balance_error_ns=float(np.max(abs(error[1:]))), max_macro_error_to_bound_ratio=float(ratio.max()),
        max_accepted_substep_error_to_bound_ratio=float(substep_ratio.max()),
        aggregate_contact_checks=dict(initial_contact_unevaluated=True, later_contact_evaluated=True,
            force_normal_sum_match=True, nonnegative_normal_force_and_impulse=True, coulomb_mu=mu,
            max_force_cone_excess_n=float(force_cone_excess.max()), max_impulse_cone_excess_ns=float(impulse_cone_excess.max()),
            scope='Saved final accepted-substep force and accumulated accepted-substep impulse only; no point-level contact-history audit.'),
        accumulated_ground_impulse_world_ns=ground.sum(0).tolist(), accumulated_gravity_impulse_world_ns=gravity.sum(0).tolist(),
        whole_run_balance_error_world_ns=global_error.tolist(),
        scope='Impulses sum only accepted substeps. Ground force is the last substep endpoint force, not a macro-step average. Momentum includes every rigid mass and translational steel node mass; baseline has no such fields.')


def self_check():
    """Three-frame free fall checks the optional impulse path and its failure guard."""
    dt, mass = .02, 3.
    times, frames, com = np.arange(3)*dt, [], np.zeros((3, 3))
    momentum = np.zeros(3)
    for i in range(3):
        impulse = np.array([0., 0., -9.81*mass*dt]) if i else np.zeros(3)
        before, momentum = momentum.copy(), momentum+impulse
        if i: com[i] = com[i-1]+momentum*dt/mass
        frames.append(dict(dt_s=dt if i else 0., command_substeps=1, contact_force_evaluated=bool(i), contact_normal_force_n=0.,
            ground_force_world_n=[0., 0., 0.], ground_impulse_world_ns=[0., 0., 0.], gravity_impulse_world_ns=impulse.tolist(),
            linear_momentum_before_kg_m_s=before.tolist(), linear_momentum_after_kg_m_s=momentum.tolist(),
            linear_momentum_balance_error_ns=[0., 0., 0.], linear_momentum_balance_tolerance_ns=dt*1e-5*210 if i else 0.,
            substep_max_linear_momentum_balance_ratio=0.))
    result = audit_momentum(frames, times, com, mass, 40, 9, 10)
    assert result['status'] == 'passed' and result['single_substep_position_checks'] == 2
    frames[1]['ground_impulse_world_ns'][0] = .01
    try:
        audit_momentum(frames, times, com, mass, 40, 9, 10)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect impulse was accepted')
    return dict(status='passed', free_fall_momentum_closure=True, incorrect_impulse_rejected=True)


def plot(stem, times, delta, local_y, phase_indices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 220})
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    ax[0, 0].plot(times, delta[:, 0], label='Forward', color='#245c86')
    ax[0, 0].plot(times, delta[:, 1], label='Lateral', color='#b85546')
    ax[0, 0].set(xlabel='Time (s)', ylabel='COM displacement (mm)'); ax[0, 0].legend(frameon=False)
    ax[0, 1].plot(times, delta[:, 2], color='#455561')
    ax[0, 1].set(xlabel='Time (s)', ylabel='COM vertical displacement (mm)')
    ax[1, 0].plot(delta[:, 1], delta[:, 0], color='#245c86', linewidth=1.2)
    for k, i in enumerate(phase_indices):
        ax[1, 0].scatter(delta[i, 1], delta[i, 0], s=22, zorder=3, label=f'{times[i]:g} s', color=plt.cm.viridis(k/max(1, len(phase_indices)-1)))
    ax[1, 0].set(xlabel='Lateral COM displacement (mm)', ylabel='Forward COM displacement (mm)', aspect='equal')
    ax[1, 0].legend(frameon=False, ncol=len(phase_indices), fontsize=8)
    for k in range(local_y.shape[1]):
        ax[1, 1].plot(times, local_y[:, k]-local_y[0, k], label=f'Module {k+1}', linewidth=1.)
    ax[1, 1].set(xlabel='Time (s)', ylabel='Module lateral motion relative to COM (mm)')
    ax[1, 1].legend(frameon=False, ncol=2, fontsize=8)
    for label, axis in zip('(a) (b) (c) (d)'.split(), ax.flat):
        axis.text(0, 1.02, label, transform=axis.transAxes)
    fig.savefig(stem.with_suffix('.png')); fig.savefig(stem.with_suffix('.svg')); plt.close(fig)


def compare(baseline_file, current_file):
    """One figure compares the final complete same-command cycle in two audits."""
    import matplotlib.pyplot as plt
    baseline, current = [json.loads(Path(f).read_text(encoding='utf-8')) for f in (baseline_file, current_file)]
    cycle = current['cycle_displacements'][-1]
    assert cycle['same_commanded_shape'] and not cycle['includes_startup']
    start, end = cycle['start_s'], cycle['end_s']
    assert any(c['same_commanded_shape'] and c['start_s'] == start and c['end_s'] == end for c in baseline['cycle_displacements'])
    curves = []
    for result in (baseline, current):
        times = np.asarray(result['time_series']['time_s'])
        i = int(np.argmin(abs(times-start))); j = int(np.argmin(abs(times-end)))
        assert abs(times[i]-start) < 1e-12 and abs(times[j]-end) < 1e-12
        com = np.asarray(result['time_series']['com_delta_initial_heading_mm'])
        curves.append((times[i:j+1], com[i:j+1]-com[i]))
    np.testing.assert_allclose(curves[0][0], curves[1][0], atol=1e-12, rtol=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for (times, delta), label, color in zip(curves, ('Baseline', 'Thicker steel'), ('#6b747a', '#245c86')):
        for k, axis in enumerate(axes):
            axis.plot(times, delta[:, k], label=label, color=color, linewidth=1.5)
            axis.scatter(times[-1], delta[-1, k], color=color, s=16, zorder=3)
    for k, (axis, label) in enumerate(zip(axes, ('Forward', 'Lateral'))):
        axis.set(xlabel='Time (s)', ylabel=f'{label} COM displacement from {start:g} s (mm)')
        axis.axhline(0, color='#aaaaaa', linewidth=.5, zorder=0)
        axis.text(0, 1.02, '(a)' if k == 0 else '(b)', transform=axis.transAxes)
        axis.legend(frameon=False)
    stem = Path(current_file).parent/'cycle_comparison'
    fig.savefig(stem.with_suffix('.png')); fig.savefig(stem.with_suffix('.svg')); plt.close(fig)
    print(json.dumps(dict(comparison=str(stem), phase_window_s=[start, end])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare', type=Path, help='Existing baseline movement JSON; overlay the final complete same-command cycle')
    args = parser.parse_args()
    audit(args.run, args.output)
    if args.compare: compare(args.compare, args.output)

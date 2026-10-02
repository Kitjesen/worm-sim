"""Compare CPU/CUDA probes, or audit the saved five-segment articulated run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from full_body_loads import read_chain_inertias
from revolute_joints import RevoluteJoints


def load(folder):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed', 'Simulation must complete before validation'
    with np.load(folder/'trajectory.npz') as saved:
        data = {key: saved[key] for key in saved.files}
    for key, value in data.items():
        assert np.isfinite(value).all(), f'Non-finite trajectory field: {key}'
    assert len(summary['frames']) == len(data['q'])
    return summary, data


def save(name, result):
    target = HERE/name
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(saved=str(target), **result), ensure_ascii=False, allow_nan=False))
    return result


def compare(cpu, cuda):
    cs, cd = load(cpu)
    gs, gd = load(cuda)
    for key in ('nodes', 'strips', 'segments', 'plates', 'rigid_bodies', 'gait', 'duration_s', 'dt_s'):
        assert cs['metadata'][key] == gs['metadata'][key], f'Probe configurations differ: {key}'
    assert cs['metadata']['solve_backend'] == 'cpu' and gs['metadata']['solve_backend'] == 'cuda'
    errors = {}
    for key in ('q', 'body_com', 'body_R', 'width_directors', 'cable_routes_m', 'joint_angles_rad', 'joint_targets_rad', 'joint_pivots_m'):
        assert cd[key].shape == gd[key].shape, f'Probe shapes differ: {key}'
        errors[key] = float(np.max(np.abs(cd[key]-gd[key])))
        np.testing.assert_allclose(cd[key], gd[key], atol=1e-8, rtol=0, err_msg=key)
    cf, gf = cs['frames'], gs['frames']
    np.testing.assert_array_equal([r['time_s'] for r in cf], [r['time_s'] for r in gf])
    for key in ('tensions_n', 'torques_nm', 'contact_normal_force_n'):
        ca, ga = np.asarray([r[key] for r in cf]), np.asarray([r[key] for r in gf])
        errors[key] = float(np.max(np.abs(ca-ga)))
        np.testing.assert_allclose(ca, ga, atol=1e-5, rtol=0, err_msg=key)
    assert [r.get('status_counts', {}) for r in cf] == [r.get('status_counts', {}) for r in gf], 'Contact states differ'
    return save('parity.json', dict(status='passed', cpu_path=str(cpu), cuda_path=str(cuda),
        frames=len(cf), nodes=cs['metadata']['nodes'], absolute_tolerances=dict(coordinates_and_angles=1e-8, force_and_torque=1e-5),
        max_absolute_errors=errors, identical_contact_state_counts=True,
        wall_seconds=dict(cpu=cs['wall_seconds'], cuda=gs['wall_seconds']),
        scope='Short CPU/CUDA probe; numerical agreement is not experimental validation'))


def audit(folder):
    summary, data = load(folder)
    meta, frames = summary['metadata'], summary['frames']
    q, c, R = data['q'], data['body_com'], data['body_R']
    nt, strips, nq = q.shape
    nodes = (nq+1)//4
    assert (nt, strips, nodes) == (41, 40, 33), 'Expected the formal 41-frame, N33, five-segment run'
    assert meta['segments'] == 5 and meta['rigid_bodies'] == 10 and meta['gait'] == 'backward-wave'
    assert meta['wave_direction'] == 'head-to-tail' and meta['head_segment'] == 1
    np.testing.assert_allclose([meta['duration_s'], meta['drive_period_s'], meta['command_amplitude_mm']], [.4, .4, 6.], atol=1e-12, rtol=0)
    times = np.asarray([frame['time_s'] for frame in frames])
    np.testing.assert_allclose(times, np.arange(nt)*meta['dt_s'], atol=1e-14, rtol=0)
    parameters = HERE.parent/'output/parameters.snapshot.json'
    cad = HERE.parent/'publication/data/cad_reference.urdf'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(parameters) == meta['parameters_sha256'] and sha(cad) == meta['urdf_sha256']
    source_verification = {}
    for name in meta['source_hashes']:
        snapshot_name = name.replace('.py', '.snapshot.py')
        candidates = [Path(folder)/snapshot_name, Path(folder)/name, HERE/snapshot_name]
        snapshot = next((path for path in candidates if path.exists()), None)
        assert snapshot is not None, f'Missing immutable source snapshot: {snapshot_name}'
        actual = sha(snapshot)
        assert actual == meta['source_hashes'][name], f'Snapshot hash differs from simulated source: {name}'
        if name in ('full_body_loads.py', 'revolute_joints.py'):
            assert sha(HERE.parent/name) == actual, f'Audit model differs from simulated source: {name}'
        source_verification[name] = dict(path=str(snapshot), sha256=actual, matches_simulation=True)
    assert meta['source_sha256'] == meta['source_hashes']['full_robot.py']
    p = json.loads(parameters.read_text(encoding='utf-8'))
    body = read_chain_inertias(p, 5, cad, unlocked=True)
    np.testing.assert_array_equal(data['plate_body_indices'], np.arange(10))
    np.testing.assert_array_equal(data['strip_bodies'], np.repeat(np.arange(10).reshape(5, 2), 8, axis=0))
    np.testing.assert_allclose(data['body_mass_kg'], body['mass_kg'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(data['body_com_local_m'], body['com_local_m'], atol=1e-14, rtol=0)
    joint_meta = meta['joint_model']
    model = RevoluteJoints(p, body, kp=joint_meta['kp_nm_rad'], kd=joint_meta['kd_nms_rad'], torque_limit=joint_meta['motor_torque_limit_nm'])
    angles, targets = data['joint_angles_rad'], data['joint_targets_rad']
    assert angles.shape == targets.shape == (nt, 4)
    assert np.max(np.abs(targets)) <= np.deg2rad(meta['joint_amplitude_deg'])+1e-12
    pivot_errors, axis_errors, angle_errors = [], [], []
    for k in range(nt):
        ev = model.evaluate(c[k], R[k], np.zeros(20), targets[k], meta['dt_s'], np.zeros((10, 3)))
        pivot_errors.append(ev['pivot_error_m']); axis_errors.append(ev['axis_error'])
        angle_errors.append(np.max(abs(ev['angles_rad']-angles[k])))
        np.testing.assert_allclose(data['joint_pivots_m'][k], ev['joint_pivots_m'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(frames[k]['angles_rad'], angles[k], atol=1e-10, rtol=0)
        np.testing.assert_allclose(frames[k]['targets_rad'], targets[k], atol=1e-10, rtol=0)
    pivot_max, axis_max, angle_max = map(float, (np.max(pivot_errors), np.max(axis_errors), np.max(angle_errors)))
    assert pivot_max < 1e-7 and axis_max < 2e-6 and angle_max < 1e-8
    # The four clamped vertices and the two clamped width directors must follow
    # their own rigid body, rather than a neighbouring segment's body.
    xyz = q[..., :3*nodes].reshape(nt, strips, nodes, 3)
    clamp_max, width_max, cable_max = 0., 0., 0.
    widths = data['width_directors']
    routes = data['cable_routes_m'].reshape(nt, 5, 4, 2, 3)
    for strip, pair in enumerate(data['strip_bodies']):
        for side, (b, row, edge) in enumerate(zip(pair, ([0, 1], [nodes-2, nodes-1]), (0, nodes-2))):
            arms = (xyz[0, strip, row]-c[0, b])@R[0, b]
            expected = c[:, b, None, :]+np.einsum('tij,nj->tni', R[:, b], arms)
            clamp_max = max(clamp_max, float(np.max(abs(xyz[:, strip, row]-expected))))
            local_width = widths[0, strip, edge]@R[0, b]
            expected_width = np.einsum('tij,j->ti', R[:, b], local_width)
            width_max = max(width_max, float(np.max(abs(widths[:, strip, edge]-expected_width))))
    for segment, pair in enumerate(np.arange(10).reshape(5, 2)):
        for side, b in enumerate(pair):
            arms = (routes[0, segment, :, side]-c[0, b])@R[0, b]
            expected = c[:, b, None, :]+np.einsum('tij,nj->tni', R[:, b], arms)
            cable_max = max(cable_max, float(np.max(abs(routes[:, segment, :, side]-expected))))
    assert clamp_max < 1e-8 and width_max < 1e-8 and cable_max < 1e-8
    torque = np.asarray([frame['torques_nm'] for frame in frames])
    stops = np.asarray([frame['stop_torques_nm'] for frame in frames])
    excess = np.maximum(np.maximum(angles-model.limits[:, 1], model.limits[:, 0]-angles), 0.)
    np.testing.assert_allclose(joint_meta['motor_torque_limit_nm'], .5, atol=1e-14, rtol=0)
    assert np.max(abs(torque)) <= .5+1e-12
    np.testing.assert_allclose([frame['limit_excess_rad'] for frame in frames], excess, atol=1e-10, rtol=0)
    stop_expected = -joint_meta['limit_stiffness_nm_rad']*(angles-np.clip(angles, model.limits[:, 0], model.limits[:, 1]))
    np.testing.assert_allclose(stops, stop_expected, atol=1e-10, rtol=0)
    # Verify prescribed pulse order separately from the actual mechanical response.
    base = data['base_cable_lengths_m'].reshape(5, 4)
    rest = np.asarray([frame['cable_rest_m'] for frame in frames]).reshape(nt, 5, 4)
    command = (base[None]-rest).mean(axis=2)
    peaks = times[np.argmax(command, axis=0)]
    np.testing.assert_allclose(peaks, [.04, .12, .20, .28, .36], atol=1e-12, rtol=0)
    np.testing.assert_allclose(command.max(axis=0), .006, atol=1e-12, rtol=0)
    plate_com = data['body_com_local_m']
    centers = c[:, data['plate_body_indices']]+np.einsum('tpij,pj->tpi', R[:, data['plate_body_indices']], data['plate_centers_local_m']-plate_com[data['plate_body_indices']])
    lengths = np.linalg.norm(centers[:, 1::2]-centers[:, ::2], axis=2)
    head_direction = centers[0, 0]-centers[0, 1]
    head_direction /= np.linalg.norm(head_direction)
    assert head_direction[0] > .999999
    masses, node_masses = data['body_mass_kg'], data['steel_node_mass_kg']
    assert masses.shape == (10,) and node_masses.shape == (nodes,) and np.all(masses > 0) and np.all(node_masses > 0)
    total_mass = float(masses.sum()+strips*node_masses.sum())
    com = (np.einsum('tbi,b->ti', c, masses)+np.einsum('tsni,n->ti', xyz, node_masses))/total_mass
    displacement = com[-1]-com[0]
    rigid_only = np.einsum('tbi,b->ti', c, masses)/masses.sum()
    residual_max = max(frame['residual_max'] for frame in frames)
    assert residual_max <= 1.001e-5
    dx = float(displacement[0])
    return save('validation.json', dict(status='passed', run_path=str(folder), frames=nt, nodes=nodes, segments=5,
        source_snapshots=source_verification, parameters_sha256=meta['parameters_sha256'], urdf_sha256=meta['urdf_sha256'],
        joint_constraints=dict(max_pivot_error_m=pivot_max, max_axis_error=axis_max, saved_angle_error_rad=angle_max),
        clamp_body_consistency=dict(max_vertex_error_m=clamp_max, max_width_director_error=width_max, max_cable_attachment_error_m=cable_max),
        joint_motion=dict(max_actual_angle_deg=np.rad2deg(np.max(abs(angles), axis=0)).tolist(),
            max_target_angle_deg=np.rad2deg(np.max(abs(targets), axis=0)).tolist(), declared_target_amplitude_deg=meta['joint_amplitude_deg'],
            max_tracking_error_deg=np.rad2deg(np.max(abs(angles-targets), axis=0)).tolist(),
            motor_peak_torque_nm=float(np.max(abs(torque))), torque_limit_nm=.5,
            max_CAD_stop_excess_rad=float(excess.max()), saved_frames_outside_CAD_limits=int(np.any(excess > 0, axis=1).sum()),
            max_stop_torque_nm=float(np.max(abs(stops))), target_amplitude_is_not_an_actual_angle_constraint=True),
        commanded_wave=dict(direction='head-to-tail', peak_times_s=peaks.tolist(), peak_shortening_mm=(command.max(0)*1000).tolist()),
        segment_response=dict(max_contraction_mm=((lengths[0]-lengths).max(0)*1000).tolist(), peak_contraction_times_s=times[np.argmax(lengths[0]-lengths, axis=0)].tolist()),
        center_of_mass=dict(included='CAD rigid bodies including locked wheels plus all ribbon translational node masses; ideal massless cables',
            total_mass_kg=total_mass, rigid_mass_kg=float(masses.sum()), steel_mass_kg=float(strips*node_masses.sum()),
            displacement_mm=(displacement*1000).tolist(), head_initial_direction_world=head_direction.tolist(),
            classification='net_backward' if dx < -1e-8 else 'net_forward' if dx > 1e-8 else 'no_resolved_net_translation',
            rigid_only_displacement_mm=((rigid_only[-1]-rigid_only[0])*1000).tolist(), direction_definition='Initial head direction is +X; negative total COM delta X is net backward'),
        solver=dict(wall_seconds=summary['wall_seconds'], physical_duration_s=meta['duration_s'],
            accepted_substeps=sum(frame.get('command_substeps', 0) for frame in frames), recovery_substeps=sum(frame.get('recovery_substeps', 0) for frame in frames),
            max_physical_residual=residual_max, max_scaled_residual=max(frame.get('scaled_residual_max', 0) for frame in frames)),
        loads=dict(saved_peak_tension_n=float(np.max([frame['tensions_n'] for frame in frames])),
            substep_peak_tension_n=max(frame.get('substep_peak_tension_n', max(frame['tensions_n'])) for frame in frames),
            saved_max_penetration_m=max(frame['max_penetration_m'] for frame in frames),
            substep_max_penetration_m=max(frame.get('substep_max_penetration_m', frame['max_penetration_m']) for frame in frames),
            peak_ground_normal_force_n=max(frame['contact_normal_force_n'] for frame in frames), contact_states=[frame.get('status_counts', {}) for frame in frames]),
        scope='One simulated cycle; checks numerical consistency, commanded wave and net displacement, not stable locomotion or experimental calibration'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    comparison = commands.add_parser('compare'); comparison.add_argument('cpu', type=Path); comparison.add_argument('cuda', type=Path)
    validation = commands.add_parser('audit'); validation.add_argument('run', type=Path)
    args = parser.parse_args()
    compare(args.cpu, args.cuda) if args.command == 'compare' else audit(args.run)

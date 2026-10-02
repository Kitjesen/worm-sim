"""Audit the saved two-cycle, 30-degree CUDA snake run; no mechanics rerun."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
spec = importlib.util.spec_from_file_location('saved_joint_audit', BENCH/'gpu_joint_wave_20261002/check.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def audit(folder, output=HERE/'validation.json'):
    folder = Path(folder)
    summary, data = previous.load(folder)
    meta, frames = summary['metadata'], summary['frames']
    q, c, R = data['q'], data['body_com'], data['body_R']
    assert q.shape == (201, 40, 35) and c.shape == (201, 10, 3) and R.shape == (201, 10, 3, 3)
    expected = dict(nodes=9, strips=40, segments=5, plates=10, rigid_bodies=10,
                    physical_body_dof=60, joint_constraint_dof=20, gait='backward-wave', solve_backend='cuda')
    for key, value in expected.items():
        assert meta[key] == value, f'Unexpected configuration: {key}'
    assert meta['joint_drive']['preset'] == 'phase-sine'
    np.testing.assert_allclose([meta[k] for k in ('dt_s', 'duration_s', 'drive_period_s', 'command_amplitude_mm', 'joint_amplitude_deg', 'max_command_step_mm')],
                               [.02, 4., 2., 0., 30., 1.], atol=1e-14, rtol=0)
    times = np.asarray([f['time_s'] for f in frames])
    np.testing.assert_allclose(times, np.arange(201)*.02, atol=1e-13, rtol=0)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    parameters, cad = BENCH/'output/parameters.snapshot.json', BENCH/'publication/data/cad_reference.urdf'
    assert sha(parameters) == meta['parameters_sha256'] and sha(cad) == meta['urdf_sha256']
    names = ('full_robot.py', 'full_body_loads.py', 'revolute_joints.py', 'cuda_sano.py', 'gpu_geometry_contact.py', 'ground_contact.py')
    assert set(meta['source_hashes']) == set(names)
    sources = {}
    for name in names:
        candidates = [folder/name.replace('.py', '.snapshot.py'), folder/name]
        path = next((p for p in candidates if p.is_file()), None)
        assert path is not None, f'Missing execution source: {name}'
        actual = sha(path)
        assert actual == meta['source_hashes'][name], f'Execution source hash differs: {name}'
        sources[name] = dict(path=str(path), sha256=actual, matches_simulation=True)
    assert meta['source_sha256'] == sources['full_robot.py']['sha256']
    modules = {name: previous.snapshot_module(name, sources[name+'.py']['path'])
               for name in ('full_body_loads', 'revolute_joints', 'full_robot')}
    p = json.loads(parameters.read_text(encoding='utf-8'))
    body = modules['full_body_loads'].read_chain_inertias(p, 5, cad, unlocked=True)
    np.testing.assert_array_equal(data['plate_body_indices'], np.arange(10))
    np.testing.assert_array_equal(data['strip_bodies'], np.repeat(np.arange(10).reshape(5, 2), 8, axis=0))
    for saved, key in (('body_mass_kg', 'mass_kg'), ('body_com_local_m', 'com_local_m'),
                       ('plate_centers_local_m', 'plate_centers_local_m'), ('plate_R_local', 'plate_R_local')):
        np.testing.assert_allclose(data[saved], body[key], atol=1e-14, rtol=0)
    jm = meta['joint_model']
    joint_names = ['front3', 'front4', 'front5', 'front6']
    assert jm['joint_names'] == [j['name'] for j in body['joints']] == joint_names
    np.testing.assert_allclose([jm[k] for k in ('kp_nm_rad', 'kd_nms_rad', 'motor_torque_limit_nm', 'passive_damping_nms_rad')],
                               [200., 0., 20., 5.], atol=1e-14, rtol=0)
    model = modules['revolute_joints'].RevoluteJoints(p, body, kp=200., kd=0., torque_limit=20., passive_damping=5.)
    assert jm['limit_stiffness_nm_rad'] == model.limit_stiffness
    angles, targets = data['joint_angles_rad'], data['joint_targets_rad']
    assert angles.shape == targets.shape == (201, 4) and data['joint_pivots_m'].shape == (201, 4, 3)
    command = modules['full_robot']._joint_command
    expected_targets = np.asarray([command(4, float(t), 2., np.deg2rad(30.)) for t in times])
    target_error = float(np.max(abs(targets-expected_targets)))
    np.testing.assert_allclose(targets, expected_targets, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(targets[0], np.zeros(4))
    assert np.max(abs(targets)) <= np.deg2rad(30.)+1e-13
    seam_errors = {str(t): float(np.max(abs(command(4, t+1e-8, 2., np.deg2rad(30.))-
                                                 command(4, t-1e-8, 2., np.deg2rad(30.))))) for t in (.4, 1., 2., 3., 4.)}
    assert max(seam_errors.values()) < 1e-7, 'Archived command has a seam jump'

    def values(key, start=0):
        value = np.asarray([f[key] for f in frames[start:]])
        assert np.isfinite(value).all(), f'Nonfinite summary field: {key}'
        return value

    errors = dict(pivot_m=0., axis=0., angle_rad=0., saved_pivot_m=0.)
    for k in range(201):
        ev = model.evaluate(c[k], R[k], np.zeros(20), targets[k], .02, np.zeros((10, 3)))
        errors['pivot_m'] = max(errors['pivot_m'], float(np.max(ev['pivot_error_m'])))
        errors['axis'] = max(errors['axis'], float(np.max(ev['axis_error'])))
        errors['angle_rad'] = max(errors['angle_rad'], float(np.max(abs(ev['angles_rad']-angles[k]))))
        errors['saved_pivot_m'] = max(errors['saved_pivot_m'], float(np.max(abs(ev['joint_pivots_m']-data['joint_pivots_m'][k]))))
    assert errors['pivot_m'] < 1e-7 and errors['axis'] < 2e-6 and errors['angle_rad'] < 1e-8 and errors['saved_pivot_m'] < 1e-8
    np.testing.assert_allclose(values('angles_rad'), angles, atol=1e-10, rtol=0)
    np.testing.assert_allclose(values('targets_rad'), targets, atol=1e-10, rtol=0)
    xyz = q[..., :27].reshape(201, 40, 9, 3)
    widths, routes = data['width_directors'], data['cable_routes_m'].reshape(201, 5, 4, 2, 3)
    assert widths.shape == (201, 40, 8, 3)

    def attachment_error(points, b):
        local = (points[0]-c[0, b])@R[0, b]
        bound = c[:, b, None]+np.einsum('tij,nj->tni', R[:, b], local)
        return float(np.max(abs(points-bound)))

    clamp_max, width_max, cable_max = 0., 0., 0.
    for s, pair in enumerate(data['strip_bodies']):
        for b, rows, edge in zip(pair, ([0, 1], [7, 8]), (0, 7)):
            clamp_max = max(clamp_max, attachment_error(xyz[:, s, rows], b))
            bound = np.einsum('tij,j->ti', R[:, b], widths[0, s, edge]@R[0, b])
            width_max = max(width_max, float(np.max(abs(widths[:, s, edge]-bound))))
    for s, pair in enumerate(np.arange(10).reshape(5, 2)):
        for side, b in enumerate(pair):
            cable_max = max(cable_max, attachment_error(routes[:, s, :, side], b))
    assert max(clamp_max, width_max, cable_max) < 1e-8
    base = data['base_cable_lengths_m']
    assert base.shape == (20,)
    np.testing.assert_allclose(values('cable_rest_m'), np.broadcast_to(base, (201, 20)), atol=1e-14, rtol=0)
    torque, passive, stops = values('torques_nm'), values('passive_damping_torques_nm'), values('stop_torques_nm')
    assert torque.shape == passive.shape == stops.shape == (201, 4) and np.max(abs(torque)) <= 20.+1e-12
    np.testing.assert_allclose(torque, np.clip(jm['kp_nm_rad']*(targets-angles), -20., 20.), atol=1e-10, rtol=0)
    excess = np.maximum(np.maximum(angles-model.limits[:, 1], model.limits[:, 0]-angles), 0.)
    np.testing.assert_allclose(values('limit_excess_rad'), excess, atol=1e-10, rtol=0)
    np.testing.assert_allclose(stops, -jm['limit_stiffness_nm_rad']*(angles-np.clip(angles, model.limits[:, 0], model.limits[:, 1])), atol=1e-10, rtol=0)
    scaled, residual = values('scaled_residual_max', 1), values('residual_max')
    assert scaled.shape == (200,) and np.max(scaled) <= 1. and np.max(residual) <= 1.001e-5
    accepted, recovery = values('command_substeps', 1), values('recovery_substeps', 1)
    assert np.all(accepted >= 1) and np.all(recovery >= 0)
    masses, node_masses = data['body_mass_kg'], data['steel_node_mass_kg']
    assert masses.shape == (10,) and node_masses.shape == (9,) and np.all(masses > 0) and np.all(node_masses > 0)
    total_mass = float(masses.sum()+40*node_masses.sum())
    com = (np.einsum('tbi,b->ti', c, masses)+np.einsum('tsni,n->ti', xyz, node_masses))/total_mass
    delta = com[-1]-com[0]
    centers = c+np.einsum('tbij,bj->tbi', R, data['plate_centers_local_m']-data['body_com_local_m'])
    lengths = np.linalg.norm(centers[:, 1::2]-centers[:, ::2], axis=2)
    head = centers[0, 0]-centers[0, 1]; head /= np.linalg.norm(head)
    assert head[0] > .999999
    tensions = values('tensions_n')
    assert tensions.shape == (201, 20) and np.min(tensions) >= -1e-12
    result = dict(status='passed', run_path=str(folder), frames=201, nodes=9, segments=5,
        source_snapshots=sources, parameters_sha256=meta['parameters_sha256'], urdf_sha256=meta['urdf_sha256'],
        command=dict(preset='phase-sine', amplitude_deg=30., period_s=2., duration_s=4., cycles=2,
                     startup_s=.4, startup='Existing linear ramp; C0 continuous', saved_target_error_rad=target_error,
                     seam_margin_s=1e-8, seam_target_differences_rad=seam_errors, cable_shortening_mm=0.),
        joint_constraints=dict(max_pivot_error_m=errors['pivot_m'], max_axis_error=errors['axis'], saved_angle_error_rad=errors['angle_rad'], saved_pivot_error_m=errors['saved_pivot_m']),
        clamp_body_consistency=dict(max_vertex_error_m=clamp_max, max_width_director_error=width_max, max_cable_attachment_error_m=cable_max),
        joint_motion=dict(names=joint_names, max_actual_angle_deg=np.rad2deg(np.max(abs(angles), axis=0)).tolist(),
            max_target_angle_deg=np.rad2deg(np.max(abs(targets), axis=0)).tolist(), max_tracking_error_deg=np.rad2deg(np.max(abs(angles-targets), axis=0)).tolist(),
            motor_peak_torque_nm=float(np.max(abs(torque))), torque_limit_nm=20., saved_peak_passive_damping_torque_nm=float(np.max(abs(passive))),
            max_CAD_stop_excess_rad=float(excess.max()), saved_frames_outside_CAD_limits=int(np.any(excess > 0, axis=1).sum()), max_stop_torque_nm=float(np.max(abs(stops)))),
        center_of_mass=dict(included='All CAD rigid masses including locked wheels, all ribbon translational node masses; massless cables',
            total_mass_kg=total_mass, rigid_mass_kg=float(masses.sum()), steel_mass_kg=float(40*node_masses.sum()), displacement_mm=(delta*1000).tolist(),
            head_initial_direction_world=head.tolist(), classification='net_backward' if delta[0] < -1e-8 else 'net_forward' if delta[0] > 1e-8 else 'no_resolved_net_translation'),
        segment_response=dict(max_contraction_mm=((lengths[0]-lengths).max(0)*1000).tolist()),
        solver=dict(wall_seconds=summary['wall_seconds'], physical_duration_s=4., accepted_substeps=int(accepted.sum()), recovery_substeps=int(recovery.sum()),
            max_physical_residual=float(np.max(residual)), max_scaled_residual=float(np.max(scaled)),
            residual_scope='All 200 saved accepted main-step endpoints; intermediate accepted substep residuals were not individually saved'),
        loads=dict(saved_peak_tension_n=float(tensions.max()), substep_peak_tension_n=float(values('substep_peak_tension_n', 1).max()),
            saved_max_penetration_m=float(values('max_penetration_m').max()), substep_max_penetration_m=float(values('substep_max_penetration_m', 1).max()),
            peak_ground_normal_force_n=float(values('contact_normal_force_n').max())),
        scope='Two cycles including startup on CAD modules 2..6; not steady locomotion, a complete V6 robot, convergence study or experimentally validated backward motion. CAD meshes are visual; ground contact uses steel surfaces and plate-rim samples, without servo/connector mesh or wheel collision.')
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(saved=str(output), **result), ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=HERE/'validation.json')
    args = parser.parse_args()
    audit(args.run, args.output)

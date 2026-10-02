"""Pure snake parameter study; frozen SOFA plant, no trained policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sofa_worm_env as physics
import numpy as np

HERE = Path(__file__).resolve().parent


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    env = physics.SofaWormEnv(randomize=False)
    original_contact = physics.wheel_contact
    contact = dict(samples=0, slip_sum=0., slip_max=0., saturated=0)

    def measure(*positional, **kwargs):
        result = original_contact(*positional, **kwargs)
        if positional[3] > .05:
            slip = float(np.linalg.norm(result[4]))
            contact['samples'] += 1
            contact['slip_sum'] += slip
            contact['slip_max'] = max(contact['slip_max'], slip)
            contact['saturated'] += int(result[5])
        return result

    rows, poses, servos, wheel_angles, wheel_speeds, actions, joints = [], [], [], [], [], [], []
    try:
        env.reset(seed=7)
        initial = env.dofs.position.array().copy()
        initial_velocity = env.dofs.velocity.array().copy()
        initial_hash = hashlib.sha256(b''.join(a.tobytes() for a in
            (initial, initial_velocity, env.servo, env.wheel_speed, env.contact_memory))).hexdigest()
        physics.wheel_contact = measure
        started = time.perf_counter()
        failed = False
        for step in range(round(args.seconds/.02)):
            t = step*.02
            ramp = .5*(1-math.cos(math.pi*min(t, 1.)))
            target = math.radians(args.amplitude)*ramp*np.sin(
                2*np.pi*t/args.period-(3-np.arange(4))*math.radians(args.phase))
            action = np.r_[np.zeros(10), np.clip((target-env.yaw_target)/.03, -1, 1)]
            _, _, done, truncated, info = env.step(action)
            x = env.dofs.position.array().copy()
            yaw = []
            for j in range(4):
                R = physics.rotation(x[2*j+1, 3:]).T@physics.rotation(x[2*j+2, 3:])
                yaw.append(math.atan2(R[1, 0], R[0, 0]))
            info.update(t_s=(step+1)*.02,
                mean_plate_forward_m=float(initial[:10, 0].mean()-x[:10, 0].mean()),
                mean_plate_lateral_m=float(x[:10, 1].mean()-initial[:10, 1].mean()),
                yaw_target_rad=target.tolist(), applied_yaw_target_rad=env.yaw_target.tolist())
            rows.append(info); poses.append(x); servos.append(env.servo.copy())
            wheel_angles.append(env.wheel_angle.copy()); wheel_speeds.append(env.wheel_speed.copy())
            actions.append(action); joints.append(yaw)
            if (step+1)%200 == 0:
                print(args.out.name, step+1, info['mean_plate_forward_m'], flush=True)
            if done:
                failed = True
                break
            if truncated:
                break
        assert np.isfinite(poses).all() and np.isfinite(wheel_speeds).all()
        assert np.all(np.asarray(servos) == 0) and contact['samples'] > 0
        report = dict(failed=failed, physical_s=len(rows)*.02, wall_s=time.perf_counter()-started,
            settings=dict(amplitude_deg=args.amplitude, period_s=args.period, phase_deg=args.phase,
                          startup_s=1, dt_s=.0005, control_dt_s=.02, seed=7, randomize=False),
            initial_state_sha256=initial_hash,
            source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()
                           for n in ['sofa_worm_env.py', 'parameters_sofa_candidate.json', 'snake_conditions.py']},
            mean_plate_forward_m=rows[-1]['mean_plate_forward_m'],
            mean_plate_lateral_m=rows[-1]['mean_plate_lateral_m'],
            max_joint_deg=float(np.max(np.abs(np.degrees(joints)))),
            loaded_wheel_slip_mean_m_s=contact['slip_sum']/contact['samples'],
            loaded_wheel_slip_max_m_s=contact['slip_max'],
            loaded_friction_limit_fraction=contact['saturated']/contact['samples'],
            contact_accumulators=contact,
            max_connector_error_m=max(r['connector_error_m'] for r in rows),
            min_clearance_m=min(r['body_clearance_m'] for r in rows),
            max_tension_n=max(r['max_tension_n'] for r in rows),
            policy='Clock-driven sine, no neural policy or observation-dependent control',
            scope='Chosen-duration input comparison, not equal cycles or equal power. Contact aggregates use all loaded substeps; raw wheel states saved at 50 Hz.')
        np.savez_compressed(args.out/'whole_rollout.npz', poses=poses, initial_poses=initial,
            initial_velocity=initial_velocity, servo=servos, phases=['snake']*len(rows),
            strip_nodes=env.strip_nodes, anchors=env.anchors, guides=env.guides,
            wheel_mounts=env.wheel_mounts, wheel_holes=env.wheel_holes,
            wheel_angle=wheel_angles, wheel_speed=wheel_speeds, actions=actions, joint_angle_rad=joints,
            width=env.params['strip_width_m'], thickness=env.params['strip_thickness_m'],
            head_xy=np.array(poses)[:,9,:2],
            reference_xy=np.column_stack([np.linspace(-1.7,.15,200),np.zeros(200)]), policy_steps=0)
        (args.out/'comparison.json').write_text(json.dumps(report, indent=2))
        (args.out/'replay_metrics.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        physics.wheel_contact = original_contact
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--amplitude', type=float, required=True)
    p.add_argument('--period', type=float, required=True)
    p.add_argument('--phase', type=float, default=90)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=12)
    args = p.parse_args()
    if not 0 < args.amplitude <= 20 or not 2 <= args.period <= 6 or not 0 <= args.phase <= 120:
        p.error('Bounds: amplitude (0,20], period [2,6], phase [0,120]')
    if not 0 < args.seconds <= 12 or not math.isclose(args.seconds/.02, round(args.seconds/.02)):
        p.error('Duration must be a multiple of 0.02 s, up to 12 s')
    run(args)

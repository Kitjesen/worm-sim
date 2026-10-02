"""Matched, deterministic SOFA worm/snake experiment with substep wheel logging."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import sofa_worm_env as physics  # Load SOFA before numerical policy libraries.
import numpy as np

HERE=Path(__file__).resolve().parent


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    env=physics.SofaWormEnv(randomize=False,dt=args.dt)
    env.episode_steps=round(args.seconds/.02)
    rows=[];poses=[];controls=[];wheel_angles=[];actions=[];joint_angles=[];gaps=[];gap_targets=[]
    contact_steps=[];contact=np.zeros((20,12),dtype=np.float32)
    original_contact=physics.wheel_contact
    original_step=env.physics
    initial=None
    columns=['normal_n','long_force_n','lat_force_n','long_slip_m_s','lat_slip_m_s',
             'spin_rad_s','friction_limited','long_carrier_velocity_m_s','lat_carrier_velocity_m_s',
             'long_memory_m','lat_memory_m','in_contact']

    def logged_contact(velocity,spin,memory,normal,dt,**kwargs):
        result=original_contact(velocity,spin,memory,normal,dt,**kwargs)
        # The frozen physics loop owns the wheel id; inspect it only for logging.
        wheel=sys._getframe(1).f_locals['wheel']
        force,omega,new_memory,reaction,slip,sliding=result
        contact[wheel]=[normal,*force,*slip,omega,float(sliding),*velocity,*new_memory,1.]
        return result

    def logged_physics():
        contact.fill(0.)
        original_step()
        contact_steps.append(contact.copy())

    try:
        obs,info=env.reset(seed=7)
        assert obs.shape==(77,) and env.action_space.shape==(14,)
        initial=env.dofs.position.array().copy()
        initial_velocity=env.dofs.velocity.array().copy()
        start_servo=env.servo.copy()
        initial_hash=hashlib.sha256(b''.join(a.tobytes() for a in
            (initial,initial_velocity,env.servo,env.wheel_speed,env.contact_memory))).hexdigest()
        physics.wheel_contact=logged_contact
        env.physics=logged_physics
        integral=np.zeros(5)
        start=time.perf_counter();failed=False
        for step in range(round(args.seconds/.02)):
            t=step*.02
            ramp=.5*(1-math.cos(math.pi*min(t,1.)))
            x=env.dofs.position.array()
            current_gap=np.linalg.norm(x[1:10:2,:3]-x[0:10:2,:3],axis=1)
            desired_gap=np.full(5,np.nan)
            motor=start_servo.copy();yaw=np.zeros(4)
            if args.mode=='worm':
                age=t-(4-np.arange(5))*.65
                pulse=np.where(age>=0,.5*(1-np.cos(2*np.pi*np.maximum(age,0)/4.)),0.)*ramp
                desired_gap=.1175-.035*pulse
                error=current_gap-desired_gap
                integral=np.clip(integral+error*.02,-.02,.02)
                motor=np.repeat(np.clip(1.5*pulse+35*error+8*integral,0.,2.3),2)
            elif args.mode=='snake':
                yaw=math.radians(args.amplitude)*ramp*np.sin(2*np.pi*t/4.-(3-np.arange(4))*np.pi/2)
            action=np.r_[np.clip((motor-env.servo)/.08,-1,1),np.clip((yaw-env.yaw_target)/.03,-1,1)]
            obs,_,done,truncated,info=env.step(action)
            x=env.dofs.position.array().copy()
            q=[]
            for j in range(4):
                relative=physics.rotation(x[2*j+1,3:]).T@physics.rotation(x[2*j+2,3:])
                q.append(math.atan2(relative[1,0],relative[0,0]))
            info.update(t_s=(step+1)*.02,mean_plate_forward_m=float(initial[:10,0].mean()-x[:10,0].mean()),
                        mean_plate_lateral_m=float(x[:10,1].mean()-initial[:10,1].mean()),
                        head_lateral_m=float(x[9,1]-initial[9,1]),
                        tail_forward_m=float(initial[0,0]-x[0,0]),yaw_target_rad=yaw.tolist())
            rows.append(info);poses.append(x);controls.append(env.servo.copy());wheel_angles.append(env.wheel_angle.copy())
            actions.append(action);joint_angles.append(q);gaps.append(np.linalg.norm(x[1:10:2,:3]-x[0:10:2,:3],axis=1));gap_targets.append(desired_gap)
            if (step+1)%100==0:print(json.dumps({'mode':args.mode,'time_s':(step+1)*.02,'advance_m':info['mean_plate_forward_m'],'slip_m_s':info['mean_contact_slip_m_s'],'failed':done}),flush=True)
            if done:failed=True;break
            if truncated:break
        elapsed=time.perf_counter()-start
        logs=np.asarray(contact_steps)
        loaded=logs[:,:,0]>.05
        slip=np.linalg.norm(logs[:,:,3:5],axis=-1)
        assert np.isfinite(logs).all() and np.isfinite(poses).all()
        assert np.all(logs[:,:,5]>=0.)
        assert np.all(np.linalg.norm(logs[:,:,1:3],axis=-1)<=.8*logs[:,:,0]+2e-6)
        actual_time=len(rows)*.02
        plate_pos=np.array([r['mean_plate_forward_m'] for r in rows])
        time_axis=np.arange(1,len(rows)+1)*.02
        mask=time_axis>max(0,actual_time-4.)
        steady=float(np.polyfit(time_axis[mask],plate_pos[mask],1)[0]) if mask.sum()>1 else None
        last_before=np.flatnonzero(time_axis<=actual_time-4.+1e-9)
        cycle_advance=float(plate_pos[-1]-plate_pos[last_before[-1]]) if len(last_before) else None
        report=dict(mode=args.mode,failed=failed,physical_s=actual_time,wall_s=elapsed,
            dt_s=args.dt,control_dt_s=.02,seed=7,randomize=False,factors=env.factors.tolist(),
            initial_state_sha256=initial_hash,source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['sofa_worm_env.py','parameters_sofa_candidate.json','compare_gaits.py']},
            mean_plate_forward_m=float(plate_pos[-1]),head_forward_m=rows[-1]['leading_forward_m'],
            tail_forward_m=rows[-1]['tail_forward_m'],mean_plate_lateral_m=rows[-1]['mean_plate_lateral_m'],
            last_cycle_advance_m=cycle_advance,last_cycle_fitted_speed_m_s=steady,
            loaded_wheel_slip_mean_m_s=float(slip[loaded].mean()),loaded_wheel_slip_max_m_s=float(slip[loaded].max()),
            loaded_friction_limit_fraction=float(logs[:,:,6][loaded].mean()),
            loaded_low_slip_fraction=float((slip[loaded]<.001).mean()),
            contact_duty_fraction=float((logs[:,:,11]>0).mean()),
            max_joint_deg=float(np.abs(np.degrees(joint_angles)).max()),
            max_tension_n=max(r['max_tension_n'] for r in rows),max_connector_error_m=max(r['connector_error_m'] for r in rows),
            min_clearance_m=min(r['body_clearance_m'] for r in rows),
            gap_min_mm=float(np.min(gaps)*1000),gap_max_mm=float(np.max(gaps)*1000),
            settings=dict(worm_stroke_m=.035,wave_period_s=4.,worm_delay_s=.65,snake_amplitude_deg=args.amplitude,snake_phase_delay_rad=math.pi/2,startup_s=1.,mu=.8,rolling_coefficient=.015),
            observation_dimension=int(obs.size),policy='Designed deterministic controllers; no PPO',
            scope='Same nominal plant and initial state; chosen inputs, not equal-power or optimized-gait comparison. Plate mean is not mass-weighted COM.')
        np.savez_compressed(args.out/'whole_rollout.npz',poses=poses,initial_poses=initial,initial_velocity=initial_velocity,
            servo=controls,phases=[args.mode]*len(poses),strip_nodes=env.strip_nodes,anchors=env.anchors,guides=env.guides,
            wheel_mounts=env.wheel_mounts,wheel_holes=env.wheel_holes,wheel_angle=wheel_angles,
            width=env.params['strip_width_m'],thickness=env.params['strip_thickness_m'],policy_steps=0,
            actions=actions,segment_gap_m=gaps,segment_gap_target_m=gap_targets,joint_angle_rad=joint_angles,
            head_xy=np.array(poses)[:,9,:2],reference_xy=np.column_stack([np.linspace(-1.7,.15,200),np.zeros(200)]))
        np.savez_compressed(args.out/'wheel_substeps.npz',samples=logs,dt_s=args.dt,columns=columns)
        (args.out/'replay_metrics.json').write_text(json.dumps(rows,indent=2))
        (args.out/'comparison.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        physics.wheel_contact=original_contact
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['worm','snake','idle'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--dt',type=float,default=.0005)
    p.add_argument('--amplitude',type=float,default=15.)
    a=p.parse_args()
    if not 0<a.seconds<=30 or not 0<a.amplitude<=20 or not 0<a.dt<=.001:p.error('Invalid experiment bounds')
    if not math.isclose(.02/a.dt,round(.02/a.dt),abs_tol=1e-9):p.error('dt must divide control timestep')
    run(a)

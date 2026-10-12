"""Bounded SOFA path evaluation: PPO or designed retrograde wave with feedback."""
import argparse
import json
import math
import time
from pathlib import Path
from sofa_worm_env import SofaWormEnv, rotation, horn_takeup  # SOFA before numpy/torch
import numpy as np
from stable_baselines3 import PPO
import torch


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


def run_actions(args):
    """Four commanded states, or a diagnostic sequence; no automatic state cycle."""
    args.out.mkdir(parents=True,exist_ok=True)
    env=SofaWormEnv(count=1,randomize=False,fixed=True)
    angles=np.linspace(0,2.3,1001);takeup=horn_takeup(angles)
    def lengths():
        x=env.dofs.position.array()
        a=x[0,:3]+env.anchors@rotation(x[0,3:]).T
        b=x[1,:3]+env.guides@rotation(x[1,3:]).T
        return np.linalg.norm(b-a,axis=1).reshape(2,2).mean(1)
    # L/R are command channels; the hardware sign convention still needs checking.
    cases={'common': [('compress',[1,1]),('relax',[0,0])]*3}
    for side in range(2):
        for other in ('hold','release'):
            goal=[None if other=='hold' else 0.]*2;goal[side]=1.
            cases[f'{"left" if side==0 else "right"}_{other}']=[('preload',[.5,.5]),('differential',goal),('relax',[0,0])]
    if args.mode=='states':
        passive=None if args.opposite=='hold' else 0.
        goals={'relax':[0,0],'tight':[1,1],'left':[1,passive],'right':[passive,1]}
        cases={'commanded_states':[(state,goals[state]) for state in args.states]}
    results={}
    try:
        for name,sequence in cases.items():
            env.episode_steps=len(sequence)*50;env.reset(seed=args.seed)
            rest=lengths();initial=env.dofs.position.array().copy();rows=[];ends=[];failed=False
            previous_phase=None
            for phase,goal in sequence:
                begin=lengths();end=np.array([begin[i] if g is None else rest[i]-.035*g for i,g in enumerate(goal)])
                if args.mode=='states' and phase==previous_phase:begin=end.copy()
                previous_phase=phase
                integral=np.zeros(2)
                for tick in range(50):
                    u=(tick+1)/50;desired=begin+(end-begin)*.5*(1-np.cos(np.pi*u))
                    error=lengths()-desired;integral=np.clip(integral+error*.02,-.02,.02)
                    feed=np.interp(rest-desired,takeup,angles)
                    target=np.clip(feed+35*error+8*integral,0.,2.3)
                    _,_,done,_,info=env.step(np.clip((target-env.servo)/.08,-1,1))
                    x=env.dofs.position.array();relative=rotation(x[0,3:]).T@rotation(x[1,3:])
                    rows.append({'t_s':len(rows)*.02+.02,'phase':phase,
                                 'shortening_mm':((rest-lengths())*1000).tolist(),
                                 'target_shortening_mm':((rest-desired)*1000).tolist(),
                                 'yaw_deg':math.degrees(math.atan2(relative[1,0],relative[0,0])),
                                 'side_tension_n':env.tensions.reshape(2,2).sum(1).tolist(),
                                 'servo_deg':np.degrees(env.servo).tolist(),
                                 'clearance_mm':info['body_clearance_m']*1000,
                                 'gap_mm':info['min_plate_spacing_m']*1000,
                                 'max_beam_node_displacement_mm':float(np.linalg.norm(x[2:,:3]-initial[2:,:3],axis=1).max()*1000)})
                    if done:failed=True;break
                ends.append(rows[-1]);print(name,phase,json.dumps(rows[-1]),flush=True)
                if failed:break
            assert len(rows)<=len(sequence)*50
            results[name]={'failed':failed,'phase_ends':ends,'samples':rows}
    finally:env.close()
    report={'scope':'Uncalibrated SOFA candidate; front partition fixed, wheel-ground contact retained; no PPO or training changes',
            'transition_s':1.,'automatic_state_cycle':False,'mode':args.mode,
            'command_sequence':args.states if args.mode=='states' else 'diagnostic sequences only',
            'opposite_side_candidate':args.opposite if args.mode=='states' else 'hold and release compared',
            'requested_side_stroke_mm':35.,
            'hold_definition':'Opposite measured cable-path length held by feedback, not rigidly constrained',
            'cases':results}
    (args.out/'single_actions.json').write_text(json.dumps(report,indent=2))
    assert all(not c['failed'] for c in results.values()), 'Single-segment action check failed; inspect single_actions.json'
    assert all(len(c['samples'])==len(cases[n])*50 for n,c in results.items())


def run(args):
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    env=SofaWormEnv(randomize=True)
    policy=PPO.load(args.checkpoint,device='cpu') if args.mode!='wave' else None
    env.episode_steps=round(args.seconds/.02)
    obs,_=env.reset(seed=args.seed)
    initial=env.dofs.position.array()[9,:2].copy()
    station=np.arange(-1.2,args.path_length+2.005,.005)
    curve_station=np.clip(station,0,args.path_length)
    lateral=.12*(1-np.cos(2*np.pi*curve_station/args.path_length))
    reference=np.column_stack([initial[0]-station,initial[1]+lateral])
    poses=[];servos=[];wheels=[];records=[];actions=[];targets=np.zeros(4)
    gaps_history=[];gap_targets=[];joint_history=[];wave_integral=np.zeros(5)
    started=time.monotonic();failed=False;complete=False
    try:
        for step in range(round(args.seconds/.02)):
            x=env.dofs.position.array()
            action=policy.predict(obs,deterministic=True)[0] if policy is not None else np.zeros(14)
            desired_gap=np.full(5,np.nan)
            if args.mode=='wave':
                # Array order is tail to head: segment 4 leads along world -X.
                age=step*.02-(4-np.arange(5))*args.wave_delay
                phase=2*np.pi*np.maximum(age,0)/args.wave_period
                pulse=np.where(age>=0,.5*(1-np.cos(phase)),0.)
                desired_gap=.1175-.035*pulse
                gap=np.linalg.norm(x[1:10:2,:3]-x[0:10:2,:3],axis=1)
                error=gap-desired_gap
                wave_integral=np.clip(wave_integral+error*.02,-.02,.02)
                motor=np.clip(1.5*pulse+35*error+8*wave_integral,0.,2.3)
                action[:10]=np.clip((np.repeat(motor,2)-env.servo)/.08,-1,1)
            if args.mode in ('feedback','wave'):
                for joint in range(4):
                    parent=2*joint+1;child=parent+1
                    center=x[child:child+2,:2].mean(axis=0)
                    nearest=int(np.argmin(np.linalg.norm(reference-center,axis=1)))
                    target=min(len(station)-1,int(np.searchsorted(station,station[nearest]+args.lookahead)))
                    delta=reference[target]-center
                    desired=math.atan2(delta[1],delta[0])
                    axis=-rotation(x[parent,3:])[:2,0]
                    heading=math.atan2(axis[1],axis[0])
                    requested=np.clip(wrap(desired-heading),-.25,.25)
                    targets[joint]+=.02/(.15+.02)*(requested-targets[joint])
                action[10:]=np.clip((targets-env.yaw_target)/.03,-1,1)
            obs,_,done,truncated,info=env.step(action)
            x=env.dofs.position.array().copy();head=x[9,:2]
            gaps_history.append(np.linalg.norm(x[1:10:2,:3]-x[0:10:2,:3],axis=1))
            gap_targets.append(desired_gap)
            joints=[]
            for j in range(4):
                relative=rotation(x[2*j+1,3:]).T@rotation(x[2*j+2,3:])
                joints.append(math.atan2(relative[1,0],relative[0,0]))
            joint_history.append(joints)
            nearest=int(np.argmin(np.linalg.norm(reference-head,axis=1)))
            info.update(t_s=(step+1)*.02,head_xy_m=head.tolist(),path_station_m=float(station[nearest]),
                        path_error_m=float(np.linalg.norm(reference[nearest]-head)))
            poses.append(x);servos.append(env.servo.copy());wheels.append(env.wheel_angle.copy())
            records.append(info);actions.append(action.copy())
            # Continue through the exit straight until the rear partition clears the curve.
            complete=bool(np.all(initial[0]-x[:10,0]>=args.path_length) and abs(x[0,1]-initial[1])<.06)
            if (step+1)%100==0:print(json.dumps({'mode':args.mode,'steps':step+1,'error_m':info['path_error_m'],'progress_m':info['path_station_m']}),flush=True)
            if done:failed=True;break
            if args.complete_path and complete:break
            if truncated:break
        errors=np.array([r['path_error_m'] for r in records])
        controllers={'wave':'Designed head-to-tail wave + length feedback + joint/path feedback','feedback':'PPO propulsion + geometric joint feedback','policy':'PPO policy without path input'}
        result={'controller':controllers[args.mode],
                'mode':args.mode,'seed':args.seed,'checkpoint_steps':policy.num_timesteps if policy is not None else None,'lookahead_m':args.lookahead,
                'physical_s':len(records)*.02,'failed':failed,'path_error_rmse_m':float(np.sqrt(np.mean(errors**2))),
                'path_error_max_m':float(errors.max()),'path_error_final_m':float(errors[-1]),
                'path_progress_m':records[-1]['path_station_m'],'leading_forward_m':records[-1]['leading_forward_m'],
                'minimum_clearance_m':min(r['body_clearance_m'] for r in records),'wall_s':time.monotonic()-started,
                'whole_body_cleared_curve':complete,'curve_longitudinal_extent_m':args.path_length,
                'scope':'Uncalibrated SOFA candidate, spatial path-following demo; no timed-trajectory command, no new RL training'}
        joints=np.degrees(joint_history)
        result.update(actual_joint_min_deg=joints.min(0).tolist(),actual_joint_max_deg=joints.max(0).tolist(),joint_peak_to_peak_deg=np.ptp(joints,axis=0).tolist())
        if args.mode=='wave':
            gap_data=np.array(gaps_history);times=(np.arange(len(records))+1)*.02;peaks=[]
            for rank,seg in enumerate(range(4,-1,-1)):
                expected=args.wave_period/2+rank*args.wave_delay
                window=np.flatnonzero(abs(times-expected)<args.wave_period*.2)
                peaks.append(float(times[window[np.argmin(gap_data[window,seg])]]) if len(window) else None)
            result.update(wave_period_s=args.wave_period,head_to_tail_delay_s=args.wave_delay,
                          first_contraction_peak_times_head_to_tail_s=peaks,
                          head_to_tail_order_verified=bool(not failed and all(p is not None for p in peaks) and np.all(np.diff(peaks)>.1)),
                          length_tracking_rmse_mm=float(np.sqrt(np.mean((gap_data-np.array(gap_targets))**2))*1000))
        np.savez_compressed(args.out/'whole_rollout.npz',poses=poses,servo=servos,phases=[args.mode]*len(poses),
                            strip_nodes=env.strip_nodes,anchors=env.anchors,guides=env.guides,
                            wheel_mounts=env.wheel_mounts,wheel_holes=env.wheel_holes,wheel_angle=wheels,
                            width=env.params['strip_width_m'],thickness=env.params['strip_thickness_m'],policy_steps=policy.num_timesteps if policy is not None else 0,
                            reference_xy=reference[(station>=0)&(station<=args.path_length+(.9 if args.complete_path else 0))],head_xy=np.array([r['head_xy_m'] for r in records]),actions=actions,
                            curve_end_xy=[initial[0]-args.path_length,initial[1]],
                            segment_gap_m=gaps_history,segment_gap_target_m=gap_targets,joint_angle_rad=joint_history)
        (args.out/'replay_metrics.json').write_text(json.dumps(records,indent=2))
        (args.out/'path_report.json').write_text(json.dumps(result,indent=2))
        if args.mode=='wave' and args.seconds>=args.wave_period+4*args.wave_delay:
            assert not failed and result['head_to_tail_order_verified'], 'Retrograde wave check failed; inspect path_report.json'
        print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--mode',choices=['policy','feedback','wave','actions','states'],default='feedback')
    p.add_argument('--states',nargs='+',choices=['relax','tight','left','right'],default=['relax','tight','tight','left','right','relax','relax'])
    p.add_argument('--opposite',choices=['hold','release'],default='release',help='Candidate differential behavior; not identified from hardware')
    p.add_argument('--lookahead',type=float,default=.35);p.add_argument('--seed',type=int,default=100)
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--complete-path',action='store_true',help='Record until the whole body clears the curve, bounded by --seconds')
    p.add_argument('--wave-period',type=float,default=4.);p.add_argument('--wave-delay',type=float,default=.65)
    p.add_argument('--path-length',type=float,default=2.4)
    a=p.parse_args()
    if not .1<=a.lookahead<=1 or not 0<a.seconds<=60:p.error('Evaluation bounds: lookahead 0.1–1 m, duration at most 60 s')
    if a.mode in ('policy','feedback') and a.checkpoint is None:p.error('PPO modes require --checkpoint')
    if not 2<=a.wave_period<=6 or not .2<=a.wave_delay<a.wave_period/4 or not 1.2<=a.path_length<=2.4:p.error('Invalid wave or path geometry')
    assert abs(wrap(3*np.pi)-(-np.pi))<1e-12
    if len(a.states)>60:p.error('State evaluation is limited to 60 one-second commands')
    run_actions(a) if a.mode in ('actions','states') else run(a)

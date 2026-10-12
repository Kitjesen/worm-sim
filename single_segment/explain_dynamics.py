"""Reproduce constitutive plots, isolated wheel tests and the saved-run audit.

The contact functions are loaded from the frozen source AST without importing
SOFA. The wheel tests are new one-wheel calculations, not whole-body rollouts.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE/'sofa_dr_runs/retrograde_wave_20261002T050058Z'
OUT = HERE/'dynamics_explained_20261002'


def main():
    OUT.mkdir(exist_ok=True)
    source = (HERE/'sofa_worm_env.py').read_bytes()
    names = {'horn_takeup', 'horn_lever', 'wheel_contact'}
    selected = [node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    scope = dict(np=np, math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<original wheel/horn functions>', 'exec'), scope)
    wheel_contact, takeup = scope['wheel_contact'], scope['horn_takeup']
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.dpi':180})
    colors = ['#28698c', '#c46739', '#36816c', '#895ca3']

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT/(name+'.png')); fig.savefig(OUT/(name+'.svg'))
        svg=OUT/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',newline='\n')
        plt.close(fig)

    dt, duration, normal, mass, radius, mu = .0005, 2.5, 1.4, .14, .018, .8
    cases = [('Locked: -0.5 N', -.5, 0.), ('Overload: -1.15 N', -1.15, 0.),
             ('Free roll: +0.05 N', .05, 0.), ('Coast: initial 0.1 m/s', 0., .1)]
    tests, arrays = {}, {}
    for name, drive, initial in cases:
        velocity, position, spin, memory = initial, 0., initial/radius, np.zeros(2)
        rows = []
        for k in range(round(duration/dt)):
            f, spin, memory, reaction, slip, sliding = wheel_contact(
                np.array([velocity, 0.]), spin, memory, normal, dt)
            assert np.isfinite(np.r_[f, spin, memory, reaction, slip]).all()
            assert np.linalg.norm(f) <= mu*normal+1e-12 and spin >= 0
            velocity += dt*(drive+f[0])/mass
            position += dt*velocity
            rows.append([(k+1)*dt, position, velocity, radius*spin, slip[0], f[0], float(sliding), spin])
        a = np.asarray(rows); arrays[name] = a
        tests[name] = dict(final_position_m=position, final_speed_m_s=velocity,
                           peak_abs_slip_m_s=float(np.abs(a[:,4]).max()),
                           final_slip_m_s=float(a[-1,4]),
                           friction_limit_hit_fraction=float(a[:,6].mean()))
    assert abs(tests[cases[0][0]]['final_position_m']) < .001
    assert abs(tests[cases[0][0]]['final_speed_m_s']) < 1e-5
    assert tests[cases[1][0]]['final_speed_m_s'] < -.01
    assert tests[cases[2][0]]['final_speed_m_s'] > .05
    assert abs(tests[cases[3][0]]['final_speed_m_s']) < .002
    np.savez_compressed(OUT/'wheel_tests.npz', **{f'case{i}':a for i,a in enumerate(arrays.values())})
    fig, axes = plt.subplots(2,2,figsize=(11,6))
    for (name,a),color in zip(arrays.items(),colors):
        axes[0,0].plot(a[:,0],a[:,1],label=name,color=color)
        axes[0,1].plot(a[:,0],a[:,2],color=color)
        axes[1,0].plot(a[:,0],a[:,4],color=color)
        axes[1,1].plot(a[:,0],a[:,5],color=color)
    axes[0,0].set_ylabel('Wheel centre displacement (m)');axes[0,0].legend(fontsize=8)
    axes[0,1].set_ylabel('Wheel centre velocity (m/s)')
    axes[1,0].set_ylabel('Contact slip velocity (m/s)')
    axes[1,1].set_ylabel('Tangential reaction (N)')
    for cap in [-mu*normal,mu*normal]: axes[1,1].axhline(cap,color='grey',ls=':',lw=1)
    for ax in axes.ravel(): ax.set_xlabel('Time (s)'); ax.grid(alpha=.2)
    save(fig,'wheel_tests')

    # Fixed rulers in each panel: frames show the same physical time at 1x.
    frames=[]
    for k in range(0,round(duration/dt),100):
        im=Image.new('RGB',(1040,640),'white'); draw=ImageDraw.Draw(im)
        draw.font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),16)
        for i,((name,a),color) in enumerate(zip(arrays.items(),colors)):
            left,top=(i%2)*520,(i//2)*320
            draw.text((left+20,top+16),name,fill=color)
            draw.text((left+20,top+40),f't={a[k,0]:.2f} s  slip={a[k,4]:+.4f} m/s',fill='#263544')
            low,high=min(-.015,float(a[:,1].min())-.015),max(.035,float(a[:,1].max())+.015)
            scale=420/(high-low)
            def px(x): return left+50+(x-low)*scale
            floor=top+230
            draw.line((left+35,floor,left+490,floor),fill='#586a72',width=2)
            for x in np.linspace(low,high,5):
                xx=px(x); draw.line((xx,floor,xx,floor+5),fill='#586a72')
                draw.text((xx-15,floor+12),f'{x:.2f}m',fill='#586a72')
            cx=px(a[k,1]);cy=floor-20
            draw.ellipse((cx-20,cy-20,cx+20,cy+20),outline=color,width=3)
            angle=float(np.sum(a[:k+1,7])*dt)
            draw.line((cx,cy,cx+18*np.sin(angle),cy+18*np.cos(angle)),fill=color,width=3)
            draw.text((left+20,top+290),'Independent ruler; visual wheel radius enlarged.',fill='#586a72')
        frames.append(im)
    frames[0].save(OUT/'wheel_tests.gif',save_all=True,append_images=frames[1:],duration=50,loop=0)
    assert len(frames)==50

    angle=np.linspace(0,np.pi,301); extension=np.linspace(-.003,.006,301)
    error=np.linspace(-45,45,301)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    axes[0].plot(np.degrees(angle),1000*takeup(angle),color=colors[0]);axes[0].set(xlabel='Horn angle (deg)',ylabel='Cable take-up (mm)')
    axes[1].plot(extension*1000,np.maximum(0,2000*extension),color=colors[1]);axes[1].set(xlabel='Cable extension (mm)',ylabel='Tension at zero speed (N)')
    for speed,style in [(0,'-'),(1,'--')]:
        axes[2].plot(error,np.clip(2*np.radians(error)-.03*speed,-.5,.5),style,label=f'Relative speed {speed} rad/s')
    axes[2].set(xlabel='Joint angle error (deg)',ylabel='Yaw motor torque (N m)');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'constitutive_laws')
    assert abs(float(takeup(0))) < 1e-12 and np.all(np.diff(takeup(angle)) >= 0)

    # Isolated fixtures, not a replacement steel model: hanging mass and fixed-base yaw inertia.
    fixture_mass, inertia = .28, .0009
    position, velocity = .101+fixture_mass*9.81/2000, 0.
    yaw, yaw_speed = 0., 0.
    rows=[]
    for k in range(round(4/dt)):
        now=(k+1)*dt
        horn=math.radians(25)*(1-math.cos(2*math.pi*now/4))
        rest=.101-float(takeup(horn))
        extension=position-rest
        tension=max(0.,2000*extension+.5*velocity) if extension>=0 else 0.
        velocity+=dt*(fixture_mass*9.81-tension)/fixture_mass
        position+=dt*velocity
        target=math.radians(20) if .2<=now<2 else 0.
        torque=float(np.clip(2*(target-yaw)-.03*yaw_speed,-.5,.5))
        yaw_speed+=dt*torque/inertia
        yaw+=dt*yaw_speed
        rows.append([now,position,rest,tension,yaw,target,torque])
    fixture=np.array(rows)
    assert np.isfinite(fixture).all() and fixture[:,3].min()>=0
    assert np.abs(fixture[:,6]).max()<=.5 and abs(yaw)<math.radians(.01)
    np.savez_compressed(OUT/'isolated_fixtures.npz',samples=fixture,
                       columns=['t_s','mass_position_m','cable_rest_m','tension_n','yaw_rad','yaw_target_rad','yaw_torque_nm'])
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    axes[0,0].plot(fixture[:,0],1000*fixture[:,1],label='Hanging mass');axes[0,0].plot(fixture[:,0],1000*fixture[:,2],label='Cable free length');axes[0,0].legend();axes[0,0].set_ylabel('Length (mm)')
    axes[1,0].plot(fixture[:,0],fixture[:,3]);axes[1,0].axhline(fixture_mass*9.81,color='grey',ls=':',label='Weight');axes[1,0].legend();axes[1,0].set_ylabel('Cable tension (N)')
    axes[0,1].plot(fixture[:,0],np.degrees(fixture[:,4]),label='Calculated yaw');axes[0,1].plot(fixture[:,0],np.degrees(fixture[:,5]),'--',label='Target');axes[0,1].legend();axes[0,1].set_ylabel('Fixed-base joint angle (deg)')
    axes[1,1].plot(fixture[:,0],fixture[:,6]);axes[1,1].set_ylabel('Motor torque (N m)')
    for ax in axes.ravel():ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
    save(fig,'isolated_fixtures')

    records=json.loads((RUN/'replay_metrics.json').read_text())
    saved=np.load(RUN/'whole_rollout.npz');t=np.array([r['t_s'] for r in records])
    fields=['leading_forward_m','forward_m','forward_velocity_m_s','mean_contact_slip_m_s',
            'wheel_support_ratio','max_tension_n','connector_error_m']
    data={key:np.array([r[key] for r in records]) for key in fields}
    assert len(t)==800 and np.allclose(np.diff(t),.02)
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    axes[0,0].plot(t,data['leading_forward_m'],label='Head');axes[0,0].plot(t,data['forward_m'],label='Mean plate position');axes[0,0].set_ylabel('Forward displacement (m)');axes[0,0].legend()
    for i in range(5): axes[0,1].plot(t,saved['segment_gap_m'][:,i]*1000,label=f'Segment {i+1}')
    axes[0,1].set_ylabel('Plate separation (mm)');axes[0,1].legend(ncol=3,fontsize=7)
    axes[1,0].plot(t,1000*data['forward_velocity_m_s'],label='Head forward velocity')
    axes[1,0].plot(t,1000*data['mean_contact_slip_m_s'],label='Mean loaded-wheel slip')
    axes[1,0].set_ylabel('Speed (mm/s)');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(t,data['wheel_support_ratio']*20);axes[1,1].set_ylabel('Wheels with load > 0.05 N');axes[1,1].set_ylim(0,21)
    axes[2,0].plot(t,data['max_tension_n']);axes[2,0].set_ylabel('Maximum cable tension (N)')
    axes[2,1].plot(t,np.degrees(saved['joint_angle_rad']));axes[2,1].set_ylabel('Actual intermodule yaw (deg)')
    for ax in axes.ravel():ax.grid(alpha=.2);ax.set_xlabel('Time (s)')
    save(fig,'saved_run_diagnostics')
    stats={key:dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max()),final=float(v[-1])) for key,v in data.items()}
    cycles=[]
    for end in [4.,8.,12.,16.]:
        last=np.flatnonzero(t<=end+1e-10)[-1]
        previous=np.flatnonzero(t<=end-4+1e-10)
        initial=data['leading_forward_m'][previous[-1]] if len(previous) else 0.
        cycles.append(float(data['leading_forward_m'][last]-initial))
    report=dict(source_sha256=hashlib.sha256(source).hexdigest(),wheel_tests=tests,
        isolated_fixtures=dict(duration_s=4.,dt_s=dt,hanging_mass_kg=fixture_mass,
            yaw_inertia_kg_m2=inertia,peak_tension_n=float(fixture[:,3].max()),
            peak_yaw_deg=float(np.degrees(fixture[:,4]).max()),
            peak_yaw_torque_nm=float(np.abs(fixture[:,6]).max()),
            scope='One hanging mass driven by prescribed horn angle and one fixed-base yaw inertia; no steel or whole-robot dynamics.'),
        wheel_test_parameters=dict(dt_s=dt,duration_s=duration,mass_kg=mass,normal_n=normal,mu=mu),
        saved_run_stats=stats,head_advance_per_4s_m=cycles,
        mean_head_forward_speed_m_s=float(data['leading_forward_m'][-1]/t[-1]),
        new_whole_body_simulation=False,
        limitations=['Original records omit per-wheel force, instantaneous speed, friction saturation and contact memory.',
                     'Mean plate position is not whole-robot mass-weighted COM.',
                     'Whole-body source snapshot may postdate the saved run.',
                     'Wheel tests use constant normal load and nominal friction, not the randomized whole-body loads.'])
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

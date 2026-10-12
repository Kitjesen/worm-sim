"""Collect actual SOFA PPO states on Linux; render those states on Windows."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def collect(run):
    from train_sofa_dr import StripEnv, PPO
    import numpy as np
    env = StripEnv(True)
    policy = PPO.load(run/'final_model.zip', device='cpu')
    obs, _ = env.reset(seed=100)
    poses, shrink, commands = [], [], []
    for k in range(201):
        poses.append(np.array([x.position.array().copy() for x in env.dofs]))
        shrink.append(env.shortening())
        commands.append(env.command.copy())
        if k == 200:
            break
        obs, _, terminated, truncated, _ = env.step(policy.predict(obs, deterministic=True)[0])
        if terminated:
            raise RuntimeError('Policy rollout failed; refusing to present it as successful')
        if truncated and k != 199:
            raise RuntimeError('Unexpected rollout length')
    np.savez_compressed(run/'rollout.npz', poses=poses, shortening=shrink, command=commands,
                        target=env.target, bases=env.bases, factors=env.factors, time=np.arange(201)*.005,
                        width=env.params['strip_width_m'], thickness=env.params['strip_thickness_m'])
    env.close()
    print('Recorded 201 SOFA states; one physical second.', flush=True)


def collect_whole(run, checkpoint=None):
    from sofa_worm_env import SofaWormEnv
    from stable_baselines3 import PPO
    import numpy as np
    env = SofaWormEnv(randomize=True)
    checkpoint = checkpoint or run/'final_model.zip'
    policy = PPO.load(checkpoint, device='cpu')
    run.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=100)
    poses, controls, records, wheel_angles = [], [], [], []
    for frame in range(env.episode_steps):
        obs, _, done, truncated, info = env.step(policy.predict(obs, deterministic=True)[0])
        poses.append(env.dofs.position.array().copy()); controls.append(env.servo.copy()); records.append(info)
        wheel_angles.append(env.wheel_angle.copy())
        if done: raise RuntimeError('Whole-body replay terminated: '+str(info))
        if truncated: break
    np.savez_compressed(run/'whole_rollout.npz', poses=poses, servo=controls, phases=['policy']*len(poses),
                        strip_nodes=env.strip_nodes, anchors=env.anchors, guides=env.guides,
                        wheel_mounts=env.wheel_mounts, wheel_holes=env.wheel_holes, wheel_angle=wheel_angles,
                        width=env.params['strip_width_m'],
                        thickness=env.params['strip_thickness_m'], policy_steps=policy.num_timesteps)
    (run/'replay_metrics.json').write_text(json.dumps(records, indent=2))
    env.close()
    print('Recorded whole-body policy:', len(poses), records[-1], flush=True)


def collect_command(run, checkpoint, case='left_stop'):
    """Replay the frozen command policy, recording physics without changing its inputs."""
    import hashlib
    snapshot = checkpoint.parent/'source'
    config = json.loads((checkpoint.parent/'config.json').read_text())
    for name, digest in config['source_sha256'].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Training snapshot hash mismatch: '+name)
    sys.path.insert(0, str(snapshot))
    from move_control import MoveEnv, physics
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
    import torch
    torch.set_num_threads(1)
    raw = MoveEnv(randomize=False, case=case)
    vector = DummyVecEnv([lambda: raw])
    norm = VecNormalize.load(str(checkpoint/'normalization.pkl'), vector)
    norm.training = False; norm.norm_reward = False
    vector = VecFrameStack(norm, n_stack=4)
    policy = PPO.load(checkpoint/'model.zip', device='cpu')
    frames, servos, wheels, targets, angles, torques, slips, residuals = [], [], [], [], [], [], [], []
    original_step = raw.env.step

    def record_step(action):
        result = original_step(action)
        core = raw.unwrapped
        x = core.dofs.position.array().copy(); v = core.dofs.velocity.array()
        q, tau = [], []
        for j in range(4):
            a,b = 2*j+1,2*j+2
            R = physics.rotation(x[a,3:])
            relative = R.T@physics.rotation(x[b,3:])
            yaw = float(np.arctan2(relative[1,0],relative[0,0]))
            speed = float(np.dot(v[b,3:]-v[a,3:],R[:,2]))
            q.append(yaw)
            tau.append(float(np.clip(2*(core.yaw_target[j]-yaw)-.03*speed,
                -core.yaw_servo_torque_limit_nm, core.yaw_servo_torque_limit_nm)))
        frames.append(x); servos.append(core.servo.copy()); wheels.append(core.wheel_angle.copy())
        targets.append(core.yaw_target.copy()); angles.append(q); torques.append(tau)
        slips.append(core.contact_slip.copy()); residuals.append(raw.filtered.copy())
        return result

    raw.env.step = record_step
    vector.seed(1000)
    obs = vector.reset()
    core = raw.unwrapped
    initial = core.dofs.position.array().copy()
    trace = []
    run.mkdir(parents=True, exist_ok=True)
    try:
        for k in range(120):
            action = policy.predict(obs, deterministic=True)[0]
            obs, _, done, infos = vector.step(action)
            trace.extend(infos[0]['step_trace'])
            if (k+1)%30 == 0:
                print('Recorded command-policy steps:',k+1,flush=True)
            if done[0]:
                break
        if infos[0]['failure'] or len(frames) != 600:
            raise RuntimeError('Policy replay did not complete 12 physical seconds')
        trace = np.array(trace)
        expected = np.load(checkpoint.parent.parent/'learned'/f'{case}.npz')['trace']
        error = float(np.max(np.abs(trace-expected)))
        if error > 1e-7:
            raise RuntimeError('Replay differs from evaluated policy: '+str(error))
        np.savez_compressed(run/'whole_rollout.npz', poses=frames, initial_poses=initial,
            servo=servos, wheel_angle=wheels, phases=['policy']*len(frames),
            strip_nodes=core.strip_nodes, anchors=core.anchors, guides=core.guides,
            wheel_mounts=core.wheel_mounts, wheel_holes=core.wheel_holes,
            width=core.params['strip_width_m'], thickness=core.params['strip_thickness_m'],
            policy_steps=policy.num_timesteps, frame_dt_s=.02, trace=trace,
            joint_target_rad=targets, joint_actual_rad=angles, yaw_pd_torque_nm=torques,
            contact_slip_m_s=slips, filtered_residual=residuals)
        moving=(trace[:,0]>2)&(trace[:,0]<=8)
        summary = dict(case=case, physical_s=12, frame_count=600, frame_dt_s=.02,
            evaluated_trace_max_abs_difference=error, policy_steps=int(policy.num_timesteps),
            policy_sha256=hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest(),
            joint_mean_target_deg=np.degrees(np.array(targets)[moving].mean(0)).tolist(),
            joint_mean_actual_deg=np.degrees(np.array(angles)[moving].mean(0)).tolist(),
            sampled_yaw_pd_torque_peak_nm=np.abs(torques).max(axis=0).tolist(),
            sampled_torque_limit_fraction=(np.abs(torques)>=.999*core.yaw_servo_torque_limit_nm).mean(0).tolist(),
            whole_heading_change_deg=float(np.degrees(trace[-1,11])),
            expected_command_heading_deg=float(np.degrees(np.sum(trace[:,2])*.02)),
            scope='Actual frozen PPO/SOFA replay; 50Hz diagnostic samples',
            gait_prior=config.get('gait_prior', 'No imposed snake wave in this policy snapshot'))
        (run/'replay_diagnostics.json').write_text(json.dumps(summary,indent=2))
        # Keep the exact training geometry available to the visualizer.
        (run/'parameters_sofa_candidate.json').write_bytes((snapshot/'parameters_sofa_candidate.json').read_bytes())
        print(json.dumps(summary),flush=True)
    finally:
        vector.close()


def horn_visual_transform(angle, side):
    """Original CAD pivot and visual rotor orientation, in the back-plate frame.

    The display neutral puts the CAD Y-directed arm along Z. This phase is a
    rendering convention, not a measured hardware assembly angle.
    """
    import numpy as np
    pivot = np.array([-.01375, .02854903 if side == 0 else -.02845097, .00031017])
    phase = np.pi/2 + (1 if side == 0 else -1)*angle
    c, s = np.cos(phase), np.sin(phase)
    return pivot, np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def add_wheel_mount_seats(asset, body, wheel_holes, prefix):
    """Two-bolt removable carrier with a curved neck and rigid U fork.

    Candidate geometry only. Axle centres stay fixed; the old upper pivot is
    removed. Plate bores/counterbores are generated by trim_wheel_lugs.
    """
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.spatial import ConvexHull, Delaunay
    from assembly_visuals import _ring, _numbers
    def mesh(name, vertices, faces):
        vertices=np.asarray(vertices);faces=np.asarray(faces,dtype=int)
        volume=np.einsum('ij,ij->i',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6
        if volume<0: faces=faces[:,[0,2,1]]
        edges=np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        assert np.all(np.unique(edges,axis=0,return_counts=True)[1]==2), name
        ET.SubElement(asset,'mesh',name=name,vertex=_numbers(vertices),face=_numbers(faces))
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,rgba='.57 .61 .65 1',
                      mass='0',contype='0',conaffinity='0')
    def perforated_capsule(name,a,b,radius,holes,origin,u,v,normal,thickness):
        angles=np.arange(48)*2*np.pi/48
        circle=np.c_[np.cos(angles),np.sin(angles)]
        outer=np.vstack([np.asarray(a)+radius*circle,np.asarray(b)+radius*circle])
        outer=outer[ConvexHull(outer).vertices]
        rings=[outer]+[np.asarray(center)+r*circle for center,r in holes]
        points=np.vstack(rings);tri=Delaunay(points).simplices
        centers=points[tri].mean(1)
        if holes:
            tri=tri[np.all([np.linalg.norm(centers-center,axis=1)>r*np.cos(np.pi/48) for center,r in holes],axis=0)]
        # Check actual hole/outer boundaries; reject a triangulation that bridges a bore.
        edge=np.sort(np.vstack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
        unique,counts=np.unique(edge,axis=0,return_counts=True)
        boundary={tuple(e) for e in unique[counts==1]};expected=set();offset=0
        for ring in rings:
            expected.update(tuple(sorted((offset+i,offset+(i+1)%len(ring)))) for i in range(len(ring)))
            offset+=len(ring)
        assert boundary==expected,(name,'bore boundary')
        origin=np.asarray(origin);u=np.asarray(u);v=np.asarray(v);normal=np.asarray(normal)
        xyz=origin+points[:,0,None]*u+points[:,1,None]*v;n=len(points)
        verts=np.vstack([xyz-thickness/2*normal,xyz+thickness/2*normal])
        faces=[]
        for t in tri:
            p=points[t]
            if np.linalg.det(np.array([p[1]-p[0],p[2]-p[0]]))<0:t=t[[0,2,1]]
            faces.extend([t[[0,2,1]],t+n])
        offset=0
        for ri,ring in enumerate(rings):
            order=np.arange(len(ring)) if ri==0 else np.arange(len(ring))[::-1]
            for i,j in zip(order,np.roll(order,-1)):
                i+=offset;j+=offset
                faces.extend([[i,j,n+j],[i,n+j,n+i]])
            offset+=len(ring)
        mesh(name,verts,faces)
    def rounded_neck(name,control):
        control=np.asarray(control);rings=[]
        for t in np.linspace(0,1,18):
            center=(1-t)**3*control[0]+3*(1-t)**2*t*control[1]+3*(1-t)*t*t*control[2]+t**3*control[3]
            tangent=3*(1-t)**2*(control[1]-control[0])+6*(1-t)*t*(control[2]-control[1])+3*t*t*(control[3]-control[2])
            across=np.array([0.,-tangent[2],tangent[1]]);across/=np.linalg.norm(across)
            half_x=.00125+(.005-.00125)*(3*t*t-2*t*t*t); half_w=.0035
            fillet=min(.001,half_x*.8);ring=[]
            for sx,sy,start in [(1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)]:
                for angle in np.radians(np.linspace(start,start+90,5)):
                    x=sx*(half_x-fillet)+fillet*np.cos(angle)
                    y=sy*(half_w-fillet)+fillet*np.sin(angle)
                    ring.append(center+np.array([x,0,0])+across*y)
            rings.append(ring)
        verts=np.asarray(rings).reshape(-1,3);n=len(rings[0]);faces=[]
        for k in range(len(rings)-1):
            for i in range(n):
                j=(i+1)%n;a=k*n+i;b=k*n+j;c=(k+1)*n+j;d=(k+1)*n+i
                faces.extend([[a,b,c],[a,c,d]])
        for i in range(1,n-1):faces.extend([[0,i+1,i],[(len(rings)-1)*n,(len(rings)-1)*n+i,(len(rings)-1)*n+i+1]])
        mesh(name,verts,faces)
    records = []
    for side, hole in enumerate(np.asarray(wheel_holes)):
        sign = 1. if hole[1] > 0 else -1.
        root=np.array([sign*.03627443,hole[2]+.03678503]);outward=np.array([sign,-1.])/np.sqrt(2)
        inner=root-.010*outward;tip=root+.006*outward
        pad_x=hole[0]+.00375
        perforated_capsule(f'{prefix}_{side}_pad',inner,tip,.0035,
            [(inner,.0016),(root,.0016)],[pad_x,0,0],[0,1,0],[0,0,1],[1,0,0],.0025)
        def geom(suffix, color='.57 .61 .65 1', **kwargs):
            return ET.SubElement(body,'geom',name=f'{prefix}_{side}_{suffix}',
                                 rgba=color,mass='0',contype='0',conaffinity='0',**kwargs)
        spacer=_ring(asset,'carrier_step_spacer',.0035,.0016,.002)
        geom('step_spacer',type='mesh',mesh=spacer,pos=_numbers([hole[0]+.0015,*inner]),zaxis='1 0 0')
        rounded_neck(f'{prefix}_{side}_neck',[[pad_x,*tip],
            [pad_x,sign*.050,hole[2]+.027],
            [hole[0],hole[1],hole[2]+.021],
            [hole[0],hole[1],hole[2]+.009]])
        perforated_capsule(f'{prefix}_{side}_fork_bridge',
            [hole[1]-.0055,hole[2]+.008],[hole[1]+.0055,hole[2]+.008],.004,[],
            [hole[0],0,0],[0,1,0],[0,0,1],[1,0,0],.012)
        axle=hole+[0,0,-.045]
        for leg,dy in enumerate((-.008,.008)):
            perforated_capsule(f'{prefix}_{side}_fork{leg}',[axle[0],axle[2]],[hole[0],hole[2]],.006,
                [([axle[0],axle[2]],.002)], [0,hole[1]+dy,0],[1,0,0],[0,0,1],[0,-1,0],.003)
        # Flush low-head candidates on the -X counterbored face, nuts on +X.
        washer=_ring(asset,'carrier_m3_washer',.003,.0016,.0005)
        nut=_ring(asset,'carrier_m3_nut',.00275,.0015,.0024,outer_hex=True)
        head=_ring(asset,'carrier_low_head',.00275,.00125,.0015,inner_hex=True)
        bolts=[]
        for i,yz in enumerate((inner,root)):
            face=hole[0]-.0025
            geom(f'root{i}_head',type='mesh',mesh=head,pos=_numbers([face+.00085,*yz]),zaxis='1 0 0')
            geom(f'root{i}_shaft',type='cylinder',size='.00145 .005',pos=_numbers([face+.0066,*yz]),zaxis='1 0 0')
            geom(f'root{i}_washer',type='mesh',mesh=washer,pos=_numbers([hole[0]+.00525,*yz]),zaxis='1 0 0')
            geom(f'root{i}_nut',type='mesh',mesh=nut,pos=_numbers([hole[0]+.0067,*yz]),zaxis='1 0 0')
            bolts.append(dict(center_yz_m=yz.tolist(),new_through_hole=i==0,diameter_mm=3.2,
                              counterbore_diameter_mm=6.,counterbore_depth_mm=1.6,head_recess_mm=.1))
        records.append(dict(side=side,axle_centre_m=axle.tolist(),removed_upper_pin_m=hole.tolist(),
                            pad_bolts=bolts,pad_width_mm=7.,pad_thickness_mm=2.5,
                            root_step_compensation_mm=2.,rigid_fork=True,
                            no_added_steering_or_upper_pivot=True))
    return records


def render_whole(run, clean=False, preview=False, follow=False, trajectory=False, show_path=True, top=False, mechanism=False, compile_only=False, straight_lower=False, show_ground=True):
    """Render recorded SOFA rigid/beam states; MuJoCo is only the renderer."""
    import numpy as np
    import mujoco
    import xml.etree.ElementTree as ET
    from assembly_visuals import add_plate_clamps, add_back_servo_mounts, add_wheel_fasteners, trim_wheel_lugs
    saved = np.load(run/'whole_rollout.npz')
    segment_count = len(saved['strip_nodes'])//8
    plate_count = 2*segment_count
    has_horns = 'servo' in saved.files
    if has_horns and saved['servo'].shape != (len(saved['poses']), 2*segment_count):
        raise ValueError('Recorded horn angles must have one independent value per side and segment')
    if mechanism and not has_horns:
        raise ValueError('Mechanism view requires recorded actual servo angles; no invented rotation')
    if trajectory and not {'reference_xy','head_xy'}.issubset(saved.files):
        raise ValueError('Trajectory inset requires recorded reference_xy and head_xy')
    params_path = run/'parameters_sofa_candidate.json'
    params = (json.loads(str(saved['parameters_json'].item())) if 'parameters_json' in saved.files else
              json.loads((params_path if params_path.exists() else HERE/'parameters_sofa_candidate.json').read_text()))
    assembly_version = str(saved['assembly_geometry_version'].item()) if 'assembly_geometry_version' in saved.files else 'legacy'
    if assembly_version not in ('legacy', 'cad_holes_v1'):
        raise ValueError('Unknown recorded assembly geometry version: '+assembly_version)
    new_assembly = assembly_version == 'cad_holes_v1'
    assembly = None
    if new_assembly:
        from motor_case_route import upper_horn_route
        required = {'parameters_json','assembly_json','strip_endpoints_local_m','back_exits',
                    'external_guides','front_outer_faces','front_hole_ids','back_hole_ids','rope_radius_m','wheel_holes'}
        if not required.issubset(saved.files):
            raise ValueError('New assembly recording misses snapshot fields: '+str(sorted(required-set(saved.files))))
        assembly = json.loads(str(saved['assembly_json'].item()))
        if not np.isfinite(saved['servo']).all() or np.min(saved['servo']) < -1e-8 or np.max(saved['servo']) > np.pi+1e-8:
            raise ValueError('Candidate CAD cord reconstruction requires actual recorded angles in 0..180 degrees.')
        for key,shape in (('strip_endpoints_local_m',(8,2,3)),('back_exits',(4,3)),
                          ('external_guides',(4,3)),('front_outer_faces',(4,3))):
            if saved[key].shape != shape or not np.isfinite(saved[key]).all():
                raise ValueError('Invalid recorded assembly geometry: '+key)
        if len(assembly['routes']) != 4 or not has_horns:
            raise ValueError('New assembly display requires four recorded routes and actual servo angles')
        for key in ('front','back'):
            if saved[key+'_hole_ids'].tolist() != [r[key+'_hole_id'] for r in assembly['routes']]:
                raise ValueError('Recorded bore order differs from assembly snapshot: '+key)
        for field,key in (('anchors','front_anchor_m'),('guides','back_entry_m'),
                          ('back_exits','back_exit_m'),('external_guides','external_guide_m'),
                          ('front_outer_faces','front_outer_face_m')):
            if not np.allclose(saved[field], [r[key] for r in assembly['routes']],rtol=0,atol=1e-10):
                raise ValueError('Recorded route coordinates differ from assembly snapshot: '+field)
        rope_radius = float(saved['rope_radius_m'])
        if not 0 < rope_radius < .001 or not np.isclose(rope_radius,assembly['rope_radius_m']):
            raise ValueError('Invalid or inconsistent recorded rope radius')
    else:
        rope_radius = .0006  # Preserve the old display and never retrofit new routing into old states.
    assembly_manifest = {'geometry_version':assembly_version,
                         'lower_route_display_override': False,
                         'straight_lower_requested': bool(straight_lower),
                         'override_scope':'CAD cord attachment geometry is reconstructed at the real 3 mm arm bores with supported lip curves. Recorded SOFA poses, angles, forces and original routing snapshot are unchanged; no new physical validation is implied.',
                         'parameters_source':'recorded parameters_json' if 'parameters_json' in saved.files else str(params_path if params_path.exists() else HERE/'parameters_sofa_candidate.json'),
                         'fasteners':[], 'fastener_scope':'Visual candidates only: zero added mass/collision, no threads or clamp-strength model; dimensions not measured hardware specifications',
                         'routing_scope':'Candidate 0..180 degree supported bore-lip and actual motor-case envelope geometry for all four real eccentric holes; lower direct cords are not used because they cross the motor at large angles. No free-space external guide or oversized tie marker. Rendered route length is not the SOFA circular-rim takeup law. This is geometric reconstruction, not rope contact dynamics or transmission calibration.' if new_assembly else 'Historical anchors and guides retained; legacy inferred guide-to-hole display only',
                         'rope_radius_m':rope_radius,
                         'source_assembly_snapshot':assembly}
    if new_assembly:
        assembly_manifest['wheel_mount_redesign'] = {
            'design':'v2: two-bolt removable pad, curved rigid neck and integrated U fork; no upper hinge pin',
            'scope':'Visual design candidate; wheel centres and dynamics unchanged. New inward bores and flush-head counterbores require plate machining. No material, strength or mass validation.',
            'mesh_trim':[], 'seats':[]}
    def mat(q):
        result = np.zeros(9); mujoco.mju_quat2Mat(result, q[[3, 0, 1, 2]])
        return result.reshape(3, 3)
    xml = ET.fromstring('''<mujoco>
      <visual><global offwidth="1280" offheight="800"/><headlight ambient=".5 .5 .5" diffuse=".6 .6 .6"/></visual>
      <asset><texture name="grid" type="2d" builtin="checker" rgb1=".64 .69 .73" rgb2=".72 .76 .79" width="512" height="512"/>
      <material name="ground" texture="grid" texrepeat="30 30" reflectance="0"/>
      <texture type="skybox" builtin="gradient" rgb1=".68 .75 .81" rgb2=".90 .93 .95" width="512" height="512"/></asset>
      <worldbody><light pos="0 -1 2" dir="0 .2 -1" diffuse=".8 .8 .8"/>
      <geom type="plane" size="3 3 .01" material="ground" contype="0" conaffinity="0"/></worldbody></mujoco>''')
    if clean:
        xml.find('visual/global').set('offwidth','1600'); xml.find('visual/global').set('offheight','900')
        xml.find('visual/headlight').attrib.update(ambient='.45 .45 .45',diffuse='.4 .4 .4',specular='.12 .12 .12')
        ET.SubElement(xml.find('visual'),'quality',shadowsize='4096',offsamples='4')
        for texture in xml.findall('asset/texture'):
            if texture.get('name')=='grid': texture.attrib.update(rgb1='.55 .57 .59',rgb2='.54 .56 .58')
            else: texture.attrib.update(rgb1='.71 .76 .80',rgb2='.86 .89 .91')
        xml.find('asset/material').attrib.update(texrepeat='2 2',reflectance='0',specular='.12',shininess='.15')
        xml.find('worldbody/geom').set('size','0 0 .01')
        xml.find('worldbody/light').attrib.update(pos='0 1.5 2',dir='0 -.5 -1',diffuse='.45 .44 .42',specular='.25 .25 .25',directional='true')
        ET.SubElement(xml.find('worldbody'),'light',pos='0 -1 1.5',dir='0 .4 -1',diffuse='.18 .20 .23',castshadow='false',directional='true')
    if not show_ground:
        xml.find('worldbody').remove(xml.find('worldbody/geom'))
    for name in ('front2_Link_structure', 'back2_Link_structure', 'back2_Link_terminal_structure',
                 'front2_Link_motors', 'back2_Link_motors'):
        path=HERE/'cad_visual_assets'/(name+'.stl')
        if new_assembly and name.endswith('structure'):
            root_z=-.03219261 if name.startswith('front') else -.033237885
            holes=[];counterbores=[]
            for sign in (-1,1):
                root=np.array([sign*.03627443,root_z]);inner=root+np.array([-sign,1.])*.010/np.sqrt(2)
                holes.append({'center_yz_m':inner.tolist(),'radius_m':.0016})
                counterbores.extend([{'center_yz_m':p.tolist(),'radius_m':.003,'depth_m':.0016} for p in (inner,root)])
            assembly_manifest['wheel_mount_redesign']['mesh_trim'].append(trim_wheel_lugs(xml.find('asset'),name,path,drill_holes=holes,counterbores=counterbores))
        else:
            ET.SubElement(xml.find('asset'), 'mesh', name=name, file=str(path))
    for plate in range(plate_count):
        body=ET.SubElement(xml.find('worldbody'), 'body', name=f'plate{plate}', mocap='true')
        kind = 'front2_Link' if plate%2==0 else 'back2_Link'
        structure = 'back2_Link_terminal_structure' if plate==plate_count-1 else kind+'_structure'
        ET.SubElement(body, 'geom', type='mesh', mesh=structure, rgba='.68 .72 .77 1', contype='0', conaffinity='0')
        if plate!=0:
            ET.SubElement(body, 'geom', type='mesh', mesh=kind+'_motors', rgba='.035 .04 .05 1', contype='0', conaffinity='0')
        fasteners = {'plate':plate, 'steel':add_plate_clamps(xml.find('asset'),body,params,
                     'front' if plate%2==0 else 'back',prefix=f'plate{plate}_clamp',include_tabs=new_assembly)}
        if plate%2:
            fasteners['servo'] = add_back_servo_mounts(xml.find('asset'),body,prefix=f'plate{plate}_servo')
        if 'wheel_holes' in saved.files:
            mounts = saved['wheel_mounts'][plate%2] if saved['wheel_mounts'].ndim==3 else saved['wheel_mounts']
            if new_assembly:
                assembly_manifest['wheel_mount_redesign']['seats'].append({
                    'plate':plate,'wheel_centres_m':mounts.tolist(),
                    'parts':add_wheel_mount_seats(xml.find('asset'),body,saved['wheel_holes'][plate%2],f'plate{plate}_seat')})
                fasteners['carrier_mount']={'bolt_sets':4,'scope':'Two new M3-class mounting bolts per carrier; flush low-head candidates, not measured hardware.'}
            fasteners['wheel'] = add_wheel_fasteners(xml.find('asset'),body,saved['wheel_holes'][plate%2],mounts,prefix=f'plate{plate}_wheel',include_upper=not new_assembly)
        assembly_manifest['fasteners'].append(fasteners)
    if has_horns:
        for side, label in enumerate(('left', 'right')):
            name = 'back2_Link_horn_'+label
            ET.SubElement(xml.find('asset'), 'mesh', name=name, file=str(HERE/'cad_visual_assets'/(name+'.stl')))
            pivot, _ = horn_visual_transform(0., side)
            for seg in range(segment_count):
                body = ET.SubElement(xml.find('worldbody'), 'body', name=f'horn_{seg}_{side}', mocap='true')
                ET.SubElement(body, 'geom', type='mesh', mesh=name,
                              pos=' '.join(map(str, -pivot)), rgba='.025 .028 .034 1', contype='0', conaffinity='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    assembly_manifest['bolt_sets_total'] = sum(part['bolt_sets'] for plate in assembly_manifest['fasteners']
                                                for name,part in plate.items() if name!='plate')
    assembly_manifest['compiled_geom_count'] = int(model.ngeom)
    assembly_manifest['compiled_mocap_count'] = int(model.nmocap)
    (run/'assembly_visual_manifest.json').write_text(json.dumps(assembly_manifest,indent=2),encoding='utf-8')
    if compile_only:
        print(json.dumps({'compiled':True,'bolt_sets':assembly_manifest['bolt_sets_total'],
                          'geoms':int(model.ngeom),'assembly_version':assembly_version}))
        return assembly_manifest
    sys.path.append(str(HERE/'.video_tools'))
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    cam = mujoco.MjvCamera(); cam.distance=1.45; cam.azimuth=115; cam.elevation=-22
    cam.lookat[:] = [(saved['poses'][:,:plate_count,0].min()+saved['poses'][:,:plate_count,0].max())/2, 0, .12]
    if clean:
        cam.distance=1.48; cam.azimuth=105; cam.elevation=-20
        model.vis.global_.fovy=35
        if 'reference_xy' in saved:
            bounds=np.vstack([saved['reference_xy'],saved['poses'][:,:10,:2].reshape(-1,2)])
            low,high=bounds.min(axis=0),bounds.max(axis=0)
            cam.lookat[:]=[*((low+high)/2),.05]
            cam.distance=max(1.48,float(np.max(high-low))*1.05)
            cam.azimuth=90; cam.elevation=-60
    detail_cam=mujoco.MjvCamera(); detail_cam.distance=.32; detail_cam.azimuth=125; detail_cam.elevation=-22
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    if follow:
        cam.distance=1.3; cam.azimuth=100; cam.elevation=-38
    if top:
        cam.azimuth=90; cam.elevation=-89; cam.distance=1.55
        cam.lookat[2]=0
    if segment_count == 1:
        cam.distance=.56; cam.azimuth=115; cam.elevation=-25 if not top else -89
        cam.lookat[:]=saved['poses'][:,:2,:3].mean(axis=(0,1))
    if mechanism:
        cam.azimuth=-15; cam.elevation=-18
        if segment_count == 1: cam.distance=.60
    stem='whole_sofa_follow' if follow else ('whole_sofa_studio' if clean else 'whole_sofa_cad')
    if top: stem+='_top'
    if mechanism: stem+='_mechanism'
    if preview: stem+='_preview'
    if trajectory:
        stem+='_trajectory'
        map_size=(480,225); map_origin=(1098,20)
        all_xy=np.vstack([saved['reference_xy'],saved['poses'][:,:10,:2].reshape(-1,2)])*[-1,1]
        map_center=(all_xy.min(0)+all_xy.max(0))/2
        map_scale=min(432/max(np.ptp(all_xy[:,0]),.01),120/max(np.ptp(all_xy[:,1]),.01))
        def map_points(xy):
            points=(np.asarray(xy)*[-1,1]-map_center)*[map_scale,-map_scale]+[240,126]
            return [tuple(v) for v in points]
        map_base=Image.new('RGB',map_size,'#f4f7fa');md=ImageDraw.Draw(map_base)
        md.rounded_rectangle((0,0,479,224),radius=12,outline='#a3b3c2',width=2)
        md.line(map_points(saved['reference_xy']),fill='#246ad2',width=3)
        if 'curve_end_xy' in saved:
            gx,gy=map_points(saved['curve_end_xy'][None,:])[0]
            md.rectangle((gx-4,gy-4,gx+4,gy+4),fill='#246ad2')
        small_font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
        for label,color,x in [('目标','#246ad2',22),('已走','#e77d24',137),('机身','#576776',252)]:
            md.line((x,202,x+23,202),fill=color,width=4)
            md.text((x+30,189),label,font=small_font,fill='#35465a')
        md.line((385,202,385+.5*map_scale,202),fill='#576776',width=2)
        md.text((387,176),'0.5 m',font=small_font,fill='#576776')
    writer = imageio_ffmpeg.write_frames(str(run/(stem+'.mp4')), (1600,900) if clean else (1280, 800), fps=25,
                                       codec='libx264', quality=8, pix_fmt_out='yuv420p', macro_block_size=2, output_params=['-movflags','+faststart'])
    writer.send(None)
    cord_paths = {}
    cord_metadata = {}
    with mujoco.Renderer(model, 900 if clean else 620, 1600 if clean else 960) as renderer:
        frame_indices = [0,len(saved['poses'])//2,len(saved['poses'])-2] if preview else range(0,len(saved['poses']),2)
        for frame in frame_indices:
            poses=saved['poses'][frame]
            if follow: cam.lookat[:]=poses[:10,:3].mean(axis=0)
            data.mocap_pos[:plate_count]=poses[:plate_count,:3]; data.mocap_quat[:plate_count]=poses[:plate_count,3:][:,[3,0,1,2]]
            if has_horns:
                for seg in range(segment_count):
                    plate = 2*seg+1
                    plate_rotation = mat(poses[plate,3:])
                    for side in range(2):
                        pivot, rotor = horn_visual_transform(float(saved['servo'][frame,2*seg+side]), side)
                        if new_assembly: pivot = np.asarray(assembly['horn_pivots_m'][side])
                        mocap = model.body(f'horn_{seg}_{side}').mocapid[0]
                        data.mocap_pos[mocap] = poses[plate,:3]+plate_rotation@pivot
                        mujoco.mju_mat2Quat(data.mocap_quat[mocap], (plate_rotation@rotor).ravel())
            mujoco.mj_forward(model,data)
            renderer.update_scene(data, cam); scn=renderer.scene
            def geom(kind, pos, size, R, color):
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, kind, np.asarray(size), np.asarray(pos), np.asarray(R).ravel(), np.asarray(color))
                if clean: g.specular=.35; g.shininess=.45
                scn.ngeom += 1
                return g
            def line(a, b, radius, color):
                g=geom(mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3), color)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
            R = [mat(p[3:]) for p in poses]
            for plate in range(plate_count):
                mounts=saved['wheel_mounts'][plate%2] if saved['wheel_mounts'].ndim==3 else saved['wheel_mounts']
                for side,mount in enumerate(mounts):
                    center=poses[plate,:3]+R[plate]@mount
                    geom(mujoco.mjtGeom.mjGEOM_CYLINDER, center, [.018,.006,0], np.column_stack([R[plate][:,0],-R[plate][:,2],R[plate][:,1]]), [.025,.027,.03,1])
                    if 'wheel_holes' in saved and not new_assembly:
                        hole=saved['wheel_holes'][plate%2,side]
                        for shift in (-.008,.008):
                            midpoint=(hole+mount)/2+np.array([0,shift,0])
                            geom(mujoco.mjtGeom.mjGEOM_BOX,poses[plate,:3]+R[plate]@midpoint,
                                 [.006,.0015,(hole[2]-mount[2])/2+.004],R[plate],[.45,.50,.56,1])
                        # The plate-local visual bolts already supply both upper pins and axles.
                    elif 'wheel_holes' not in saved:
                        line(poses[plate,:3]+R[plate]@np.array([0,mount[1],-.03]),center,.003,[.3,.34,.38,1])
                    if 'wheel_angle' in saved:
                        angle=float(saved['wheel_angle'][frame,2*plate+side])
                        face=mount+np.array([0,np.sign(mount[1])*.0065,0])
                        for theta in (angle,angle+np.pi/2):
                            spoke=np.array([.013*np.cos(theta),0,.013*np.sin(theta)])
                            line(poses[plate,:3]+R[plate]@(face-spoke),poses[plate,:3]+R[plate]@(face+spoke),.0008,[.8,.82,.85,1])
            for strip, ids in enumerate(saved['strip_nodes']):
                ids=ids.astype(int); seg=strip//8; j=strip%8
                points=poses[ids,:3].copy()
                front_end = saved['strip_endpoints_local_m'][j,0] if new_assembly else np.mean(params['front_clamp_hole_pairs_m'][j],axis=0)
                back_end = saved['strip_endpoints_local_m'][j,1] if new_assembly else np.mean(params['back_clamp_hole_pairs_m'][j],axis=0)
                points[0]+=R[2*seg]@front_end
                points[-1]+=R[2*seg+1]@back_end
                for k in range(len(ids)-1):
                    d=points[k+1]-points[k]; length=np.linalg.norm(d); d/=length
                    width=R[ids[max(1,k)]][:,1].copy(); width-=d*np.dot(width,d); width/=np.linalg.norm(width)
                    geom(mujoco.mjtGeom.mjGEOM_BOX, (points[k]+points[k+1])/2,
                         [length/2,float(saved['width'])/2,.00015], np.column_stack([d,width,np.cross(d,width)]), [.40,.47,.53,1] if clean else [.25,.34,.44,1])
            for seg in range(segment_count):
                a,b=2*seg,2*seg+1
                for index,(anchor,guide) in enumerate(zip(saved['anchors'],saved['guides'])):
                    route = assembly['routes'][index] if new_assembly else None
                    side = route['servo_slot'] if new_assembly else index//2
                    color = [.96,.46,.16,1] if side == 0 else [.13,.47,.91,1]
                    guide_world = poses[b,:3]+R[b]@guide
                    anchor_world = poses[a,:3]+R[a]@anchor
                    line(anchor_world,guide_world,rope_radius,color)
                    if has_horns:
                        pivot, rotor = horn_visual_transform(float(saved['servo'][frame,2*seg+side]), side)
                        if new_assembly:
                            pivot = np.asarray(assembly['horn_pivots_m'][side])
                            angle = float(saved['servo'][frame,2*seg+side])
                            key = (index,angle)
                            if key not in cord_paths:
                                local_path,meta = upper_horn_route(angle,route,assembly)
                                if not np.allclose(local_path[0],saved['back_exits'][index],rtol=0,atol=1e-10):
                                    raise ValueError('Cord path must start at the recorded back bore exit')
                                cord_paths[key] = np.vstack([guide,local_path])
                                if frame == 0: cord_metadata[route['name']] = meta
                            path = poses[b,:3]+cord_paths[key]@R[b].T
                            hole = path[-1]
                            for p0,p1 in zip(path,path[1:]): line(p0,p1,rope_radius,color)
                            line(poses[a,:3]+R[a]@saved['front_outer_faces'][index],anchor_world,rope_radius,color)
                        else:
                            # ponytail: preserve historical display; no retrofit of the new candidate routing.
                            offset = np.array([-.003, (.0235 if index%2 == 0 else -.0235), 0.])
                            hole = poses[b,:3]+R[b]@(pivot+rotor@offset)
                            line(guide_world,hole,rope_radius,color)
                        if not new_assembly:
                            geom(mujoco.mjtGeom.mjGEOM_SPHERE, hole, [.0017,0,0], np.eye(3),color)
                        if not new_assembly and index%2 == 0:
                            surface = assembly['horn_surface_tie_offset_x_m'] if new_assembly else -.003
                            hub = poses[b,:3]+R[b]@(pivot+rotor@np.array([surface,0.,0.]))
                            line(hub,hole,.0006,[.84,.87,.91,1])
                if seg<segment_count-1: line(poses[b,:3], poses[b+1,:3], .007, [.12,.15,.18,1])
            if new_assembly and mechanism and frame == 0:
                # Raw close-ups of the same recorded initial state; no hiding or moving parts.
                closeups = [
                    ('assembly_front',0,np.asarray(params['front_plate_center_m'])+[0,0,-.01],.48,165.,-12.),
                    ('assembly_back',1,np.asarray(params['back_plate_center_m'])+[0,0,-.01],.48,-15.,-12.),
                    ('rope_bore_detail',1,saved['back_exits'][0],.078,-15.,-10.),
                    ('lower_rope_detail',1,saved['back_exits'][1],.13,-15.,-10.),
                    ('horn_hole_detail',1,cord_paths[(1,float(saved['servo'][frame,0]))][-1],.035,-25.,-15.),
                    ('wheel_mount_detail',0,saved['wheel_holes'][0,0]+[0,-.010,-.014],.22,145.,-14.),
                    ('steel_fastener_detail',0,np.mean(params['front_clamp_hole_pairs_m'][0],axis=0),.073,-35.,-16.)]
                detail_files = []
                for label,plate,local,distance,azimuth,elevation in closeups:
                    close_cam = mujoco.MjvCamera()
                    close_cam.lookat[:] = poses[plate,:3]+R[plate]@local
                    close_cam.distance = distance
                    close_cam.azimuth = azimuth+np.degrees(np.arctan2(R[plate][1,0],R[plate][0,0]))
                    close_cam.elevation = elevation
                    mujoco.mjv_updateCamera(model,data,close_cam,scn)
                    filename = label+'.png'
                    Image.fromarray(renderer.render()).save(run/filename)
                    detail_files.append(filename)
                mujoco.mjv_updateCamera(model,data,cam,scn)
                assembly_manifest['detail_images'] = {'files':detail_files,
                    'source':'First recorded SOFA frame; same complete visible geometry, no removed occluders or exploded-part transforms'}
                assembly_manifest['cord_geometry_at_first_frame'] = cord_metadata
                (run/'assembly_visual_manifest.json').write_text(json.dumps(assembly_manifest,indent=2),encoding='utf-8')
            if show_path and 'reference_xy' in saved:
                reference=saved['reference_xy']
                for j in range(0,len(reference)-2,3):
                    line(np.r_[reference[j],.002],np.r_[reference[j+2],.002],.003,[.05,.35,.8,1])
                trace=saved['head_xy'][:frame+1:4]
                for a,b in zip(trace[:-1],trace[1:]):line(np.r_[a,.005],np.r_[b,.005],.0035,[.95,.35,.05,1])
            # Fixed overview and moving close-up use the same SOFA state, with no motion scaling.
            if clean:
                pixels=renderer.render()
                if trajectory:
                    panel=map_base.copy();md=ImageDraw.Draw(panel)
                    md.text((20,12),f'全程轨迹   {(frame+1)*.02:.1f} s',font=font,fill='#263c50')
                    if frame>0: md.line(map_points(saved['head_xy'][:frame+1]),fill='#e77d24',width=3)
                    md.line(map_points(poses[:10,:2]),fill='#576776',width=4)
                    hx,hy=map_points(poses[9:10,:2])[0]
                    md.ellipse((hx-5,hy-5,hx+5,hy+5),fill='#e77d24')
                    canvas=Image.fromarray(pixels);canvas.paste(panel,map_origin);pixels=np.asarray(canvas)
                writer.send(pixels)
                if not preview and frame%200==0: print('Rendered frames:',frame//2+1,'/',len(frame_indices),flush=True)
                if preview or frame in (0,(len(saved['poses'])//4)*2,frame_indices[-1]): Image.fromarray(pixels).save(run/f'{stem}_{frame:03d}.png')
                continue
            canvas=Image.new('RGB',(1280,800),'#e6edf2'); canvas.paste(Image.fromarray(renderer.render()),(0,110))
            detail_cam.lookat[:]=poses[:2,:3].mean(axis=0)
            mujoco.mjv_updateCamera(model,data,detail_cam,scn)
            detail=Image.fromarray(renderer.render()).crop((280,70,680,550)).resize((300,360))
            canvas.paste(detail,(970,110)); draw=ImageDraw.Draw(canvas)
            phase=str(saved['phases'][frame])
            title = {'policy':'SOFA + PPO · 实际策略回放 · 正常速度',
                     'feedback':'SOFA · PPO 推进 + 路径反馈转向 · 正常速度',
                     'wave':'SOFA · 后退波与长度反馈 · 设计控制器，非学习策略'}.get(phase,'SOFA · 收缩／回弹验证 · 预设动作，非训练策略')
            draw.text((20,18),title,font=font,fill='#172d42')
            phases={'rest':'静止承重','contract':'共同收缩','release':'松绳回弹','steer':'节间转向','policy':'策略输出驱动','cycle':'周期收缩—回弹验证','feedback':'路径闭环','wave':'收缩波由头向尾传播'}
            draw.text((20,52),f"{phases[str(saved['phases'][frame])]}  |  t = {(frame+1)*.02:.2f} s  |  CAD 隔板显示，SOFA 候选动力学",font=font,fill='#526778')
            gap=np.linalg.norm(poses[1,:3]-poses[0,:3])*1000
            initial=saved['poses'][0,:10,:3].mean(axis=0)
            displacement=(initial[0]-poses[:10,0].mean())*1000
            draw.text((980,485),'第一体节放大',font=font,fill='#172d42')
            draw.text((980,535),f'隔板间距 {gap:.1f} mm',font=font,fill='#172d42')
            draw.text((980,575),f'整机前进 {displacement:.1f} mm',font=font,fill='#172d42')
            draw.text((20,750),'左：固定相机看整机位移    右：近景看隔板、钢片和收缩；没有放大运动幅度',font=font,fill='#526778')
            writer.send(np.asarray(canvas))
            if frame in (0,(len(saved['poses'])//4)*2,frame_indices[-1]): canvas.save(run/f'whole_sofa_cad_{frame:03d}.png')
    writer.close()
    check=imageio_ffmpeg.read_frames(str(run/(stem+'.mp4'))); next(check)
    assert sum(1 for _ in check)==len(frame_indices)
    if has_horns:
        (run/(stem+'_mechanism.json')).write_text(json.dumps({
            'pose_and_angle_source':'whole_rollout.npz: poses and actual servo angles; no motion scaling',
            'horn_geometry':'Original CAD double-arm meshes; display neutral rotates the CAD Y arm onto Z',
            'rope_display':assembly_manifest['routing_scope'],
            'side_convention':'slot0=+Y=robot right (orange); slot1=-Y=robot left (blue), nominal forward=-X',
            'actual_horn_angle_range_deg':np.degrees([saved['servo'].min(0),saved['servo'].max(0)]).tolist()
        },indent=2),encoding='utf-8')
    print('Rendered and decoded whole-body frames:',len(frame_indices))


def render(run):
    import numpy as np
    sys.path.insert(0, str(HERE/'.video_tools'))
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    import mujoco
    with np.load(run/'rollout.npz') as saved:
        states = {key: saved[key].copy() for key in saved.files}
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="1000" offheight="600"/><headlight ambient=".4 .4 .4" diffuse=".5 .5 .5"/></visual>
      <asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".10 .15 .22" rgb2=".32 .40 .48" width="512" height="512"/></asset>
      <worldbody><light pos="0 -.5 1" dir="0 .2 -1" diffuse=".8 .8 .8"/>
      <geom type="plane" size="1 1 .01" pos="0 0 -.13" rgba=".22 .28 .34 1" contype="0" conaffinity="0"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.06, 0., 0.]
    camera.distance, camera.azimuth, camera.elevation = .39, 120, -24
    font = {n: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', n) for n in (18, 22, 28)}
    colors = [(0.20, .82, .73, 1.), (.98, .64, .26, 1.)]
    writer = imageio_ffmpeg.write_frames(str(run/'sofa_policy.mp4'), (1280, 800), fps=25,
                                       codec='libx264', quality=8, pix_fmt_out='yuv420p',
                                       output_params=['-movflags', '+faststart'])
    writer.send(None)
    with mujoco.Renderer(model, 560, 960) as renderer:
        for frame, poses in enumerate(states['poses']):
            renderer.update_scene(data, camera)
            scn = renderer.scene
            def box(pos, half, mat, rgba):
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half), np.asarray(pos), np.asarray(mat).ravel(), np.asarray(rgba))
                scn.ngeom += 1
            for strip, pose in enumerate(poses):
                side = int(states['bases'][strip, 1] < 0)
                for j in range(len(pose)-1):
                    a, b = pose[j, :3], pose[j+1, :3]
                    tangent = b-a
                    length = np.linalg.norm(tangent)
                    tangent /= length
                    mat = np.zeros(9)
                    q = pose[j, 3:][[3, 0, 1, 2]]
                    mujoco.mju_quat2Mat(mat, q)
                    lateral = mat.reshape(3, 3)[:, 1]
                    lateral -= tangent*np.dot(lateral, tangent)
                    lateral /= np.linalg.norm(lateral)
                    box((a+b)/2, [length/2, float(states['width'])/2, float(states['thickness'])/2],
                        np.column_stack([tangent, lateral, np.cross(tangent, lateral)]), colors[side])
                box(pose[0, :3], [.002, .004, .004], np.eye(3), [.7, .72, .75, 1])
                # Straight test cable: geometry from actual clamp and tip positions.
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.85, .35, .25, 1]))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .00025, pose[0, :3], pose[-1, :3])
                scn.ngeom += 1
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            canvas = Image.new('RGB', (1280, 800), '#111b27')
            canvas.paste(Image.fromarray(renderer.render()), (0, 90))
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 15), 'SOFA + PPO：已训练钢片端点控制', font=font[28], fill='white')
            draw.text((24, 56), '实际 SOFA 状态回放 · 8 倍慢放 · 独立悬臂台架，尚非完整体节', font=font[22], fill='#efbd70')
            labels = [('仿真时间', f"{states['time'][frame]:.3f} s"),
                      ('左侧目标 / 实际', f"{states['target'][0]*1000:.2f} / {states['shortening'][frame,0]*1000:.2f} mm"),
                      ('右侧目标 / 实际', f"{states['target'][1]*1000:.2f} / {states['shortening'][frame,1]*1000:.2f} mm"),
                      ('左 / 右每片拉力', f"{states['command'][frame,0]:.4f} / {states['command'][frame,1]:.4f} N")]
            for j, (label, value) in enumerate(labels):
                y = 130+j*100
                draw.text((970, y), label, font=font[18], fill='#91a8bb')
                draw.text((970, y+32), value, font=font[22], fill='white')
            for side, color in enumerate(('#33d1ba', '#faa342')):
                x0, y0, w, h = 40+side*640, 695, 550, 72
                draw.text((x0, y0-35), ('左' if side==0 else '右')+'侧收缩量：实线实际，虚线目标', font=font[18], fill=color)
                draw.line((x0, y0, x0, y0+h, x0+w, y0+h), fill='#7690a5')
                target_y = y0+h-float(states['target'][side])/.004*h
                for x in range(x0, x0+w, 14):
                    draw.line((x, target_y, x+7, target_y), fill=color)
                points = [(x0+i/200*w, y0+h-float(v)/.004*h) for i,v in enumerate(states['shortening'][:frame+1, side])]
                if len(points) > 1:
                    draw.line(points, fill=color, width=3)
            writer.send(np.asarray(canvas))
            if frame == 200:
                canvas.save(run/'sofa_policy.png')
    writer.close()
    frames = imageio_ffmpeg.read_frames(str(run/'sofa_policy.mp4'))
    meta = next(frames)
    count = sum(1 for _ in frames)
    assert count == 201
    (run/'video_check.json').write_text(json.dumps({'frames': count, 'size': meta['size'], 'physical_time_s': 1., 'playback_s': count/25}, indent=2))
    print('Rendered and decoded 201 frames.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--whole', action='store_true')
    p.add_argument('--clean', action='store_true', help='Text-free studio view for presentation')
    p.add_argument('--preview', action='store_true', help='Render start, middle and end for visual review')
    p.add_argument('--follow', action='store_true', help='Fixed-zoom camera follows the body; requires --whole --clean')
    p.add_argument('--trajectory', action='store_true', help='Global reference and actual trajectory inset; requires --whole --clean')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--command-case', choices=['straight_stop','left_stop','right_stop','switch'])
    p.add_argument('--top', action='store_true', help='Overhead view of unscaled SOFA poses')
    p.add_argument('--mechanism', action='store_true', help='Face the actual rotating CAD horns; reconstructed guide-to-hole display, not cable contact')
    p.add_argument('--compile-only', action='store_true', help='Compile replay geometry and save its assembly manifest without video')
    p.add_argument('--straight-lower', action='store_true', help='Remove lower free-space guide kinks from the display, preserving recorded physics')
    args = p.parse_args()
    if args.follow and (not args.whole or not args.clean): p.error('--follow requires --whole --clean')
    if args.trajectory and (not args.whole or not args.clean): p.error('--trajectory requires --whole --clean')
    if args.command_case:
        if not args.collect or not args.checkpoint: p.error('--command-case needs --collect and --checkpoint')
        collect_command(args.run,args.checkpoint,args.command_case)
    elif args.whole:
        collect_whole(args.run, args.checkpoint) if args.collect else render_whole(args.run,args.clean,args.preview,args.follow,args.trajectory,top=args.top,mechanism=args.mechanism,compile_only=args.compile_only,straight_lower=args.straight_lower)
    else:
        collect(args.run) if args.collect else render(args.run)

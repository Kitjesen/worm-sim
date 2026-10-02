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


def render_whole(run, clean=False, preview=False, follow=False, trajectory=False):
    """Render recorded SOFA rigid/beam states; MuJoCo is only the renderer."""
    import numpy as np
    import mujoco
    import xml.etree.ElementTree as ET
    sys.path.insert(0, str(HERE/'.video_tools'))
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    saved = np.load(run/'whole_rollout.npz')
    if trajectory and not {'reference_xy','head_xy'}.issubset(saved.files):
        raise ValueError('Trajectory inset requires recorded reference_xy and head_xy')
    params = json.loads((HERE/'parameters_sofa_candidate.json').read_text())
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
    for name in ('front2_Link_structure', 'back2_Link_structure', 'back2_Link_terminal_structure',
                 'front2_Link_motors', 'back2_Link_motors'):
        ET.SubElement(xml.find('asset'), 'mesh', name=name, file=str(HERE/'cad_visual_assets'/(name+'.stl')))
    for plate in range(10):
        body=ET.SubElement(xml.find('worldbody'), 'body', name=f'plate{plate}', mocap='true')
        kind = 'front2_Link' if plate%2==0 else 'back2_Link'
        structure = 'back2_Link_terminal_structure' if plate==9 else kind+'_structure'
        ET.SubElement(body, 'geom', type='mesh', mesh=structure, rgba='.68 .72 .77 1', contype='0', conaffinity='0')
        if plate!=0:
            ET.SubElement(body, 'geom', type='mesh', mesh=kind+'_motors', rgba='.035 .04 .05 1', contype='0', conaffinity='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera(); cam.distance=1.45; cam.azimuth=115; cam.elevation=-22
    cam.lookat[:] = [(saved['poses'][:,:10,0].min()+saved['poses'][:,:10,0].max())/2, 0, .12]
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
    stem='whole_sofa_follow' if follow else ('whole_sofa_studio' if clean else 'whole_sofa_cad')
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
    with mujoco.Renderer(model, 900 if clean else 620, 1600 if clean else 960) as renderer:
        frame_indices = [0,len(saved['poses'])//2,len(saved['poses'])-2] if preview else range(0,len(saved['poses']),2)
        for frame in frame_indices:
            poses=saved['poses'][frame]
            if follow: cam.lookat[:]=poses[:10,:3].mean(axis=0)
            data.mocap_pos[:]=poses[:10,:3]; data.mocap_quat[:]=poses[:10,3:][:,[3,0,1,2]]
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
            for plate in range(10):
                mounts=saved['wheel_mounts'][plate%2] if saved['wheel_mounts'].ndim==3 else saved['wheel_mounts']
                for side,mount in enumerate(mounts):
                    center=poses[plate,:3]+R[plate]@mount
                    geom(mujoco.mjtGeom.mjGEOM_CYLINDER, center, [.018,.006,0], np.column_stack([R[plate][:,0],-R[plate][:,2],R[plate][:,1]]), [.025,.027,.03,1])
                    if 'wheel_holes' in saved:
                        hole=saved['wheel_holes'][plate%2,side]
                        for shift in (-.008,.008):
                            midpoint=(hole+mount)/2+np.array([0,shift,0])
                            geom(mujoco.mjtGeom.mjGEOM_BOX,poses[plate,:3]+R[plate]@midpoint,
                                 [.006,.0015,(hole[2]-mount[2])/2+.004],R[plate],[.45,.50,.56,1])
                        for anchor in (hole,mount):
                            line(poses[plate,:3]+R[plate]@(anchor+np.array([0,-.011,0])),
                                 poses[plate,:3]+R[plate]@(anchor+np.array([0,.011,0])),.002,[.2,.23,.26,1])
                    else:
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
                points[0]+=R[2*seg]@np.mean(params['front_clamp_hole_pairs_m'][j],axis=0)
                points[-1]+=R[2*seg+1]@np.mean(params['back_clamp_hole_pairs_m'][j],axis=0)
                for k in range(len(ids)-1):
                    d=points[k+1]-points[k]; length=np.linalg.norm(d); d/=length
                    width=R[ids[max(1,k)]][:,1].copy(); width-=d*np.dot(width,d); width/=np.linalg.norm(width)
                    geom(mujoco.mjtGeom.mjGEOM_BOX, (points[k]+points[k+1])/2,
                         [length/2,float(saved['width'])/2,.00015], np.column_stack([d,width,np.cross(d,width)]), [.40,.47,.53,1] if clean else [.25,.34,.44,1])
            for seg in range(5):
                a,b=2*seg,2*seg+1
                for anchor,guide in zip(saved['anchors'],saved['guides']):
                    line(poses[a,:3]+R[a]@anchor,poses[b,:3]+R[b]@guide,.0004,[.85,.32,.12,1])
                if seg<4: line(poses[b,:3], poses[b+1,:3], .007, [.12,.15,.18,1])
            if 'reference_xy' in saved:
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
    args = p.parse_args()
    if args.follow and (not args.whole or not args.clean): p.error('--follow requires --whole --clean')
    if args.trajectory and (not args.whole or not args.clean): p.error('--trajectory requires --whole --clean')
    if args.whole:
        collect_whole(args.run, args.checkpoint) if args.collect else render_whole(args.run,args.clean,args.preview,args.follow,args.trajectory)
    else:
        collect(args.run) if args.collect else render(args.run)

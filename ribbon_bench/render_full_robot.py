"""Paper-style render for a saved full_robot.py trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_body_loads import FullBodyCables, read_body_inertias
from run import HERE, read_project


def _ring(center, R, radius, thickness, count=64):
    a = np.linspace(0., 2*np.pi, count)
    yz = np.column_stack((np.zeros(count), radius*np.cos(a), radius*np.sin(a)))
    return np.stack((center+(yz+[-thickness/2, 0, 0])@R.T,
                     center+(yz+[thickness/2, 0, 0])@R.T))


def render(input_dir, output, parameters):
    input_dir, output = Path(input_dir), Path(output)
    data = np.load(input_dir/'trajectory.npz')
    q, body_com, body_R = data['q'], data['body_com'], data['body_R']
    p, delta, provenance = read_project(Path(parameters))
    bodies = read_body_inertias(p, provenance['source_urdf'])
    cable = FullBodyCables(p, bodies['com_local_m'])
    summary = json.loads((input_dir/'summary.json').read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'svg.fonttype': 'none'})
    colors = ['#517f9b', '#668da4', '#7aa0b2', '#8eb2bf', '#ad5b4f', '#bd746a', '#ca8c82', '#d7a49a']
    anchors = np.asarray(p['tendon_anchors_m']).reshape(4, 3)
    fig = plt.figure(figsize=(6.2, 4.8), dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    def draw(k, save=False):
        ax.clear()
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for i in range(8):
            n = (len(q[k, i])+1)//4
            xyz = q[k, i, :3*n].reshape(n, 3)
            ax.plot(xyz[:, 0]*1000, xyz[:, 1]*1000, xyz[:, 2]*1000, color=colors[i], lw=1.25)
            lo = np.minimum(lo, xyz.min(0)); hi = np.maximum(hi, xyz.max(0))
        for b in range(2):
            center = body_com[k, b]-body_R[k, b]@bodies['com_local_m'][b]
            ring = _ring(center, body_R[k, b], p['plate_stop_radius_m'], p['plate_stop_thickness_m'])
            for side in ring:
                ax.plot(side[:, 0]*1000, side[:, 1]*1000, side[:, 2]*1000, color='#4d5961', lw=1.0)
            lo = np.minimum(lo, ring.reshape(-1,3).min(0)); hi = np.maximum(hi, ring.reshape(-1,3).max(0))
        routes = cable.evaluate(body_com[k], body_R[k], np.ones(4))['routes_m']
        for route in routes:
            ax.plot(route[:, 0]*1000, route[:, 1]*1000, route[:, 2]*1000, '--', color='#9b823a', lw=.7)
        ax.plot([lo[0]*1000-10, hi[0]*1000+10], [lo[1]*1000-10, hi[1]*1000+10], [0, 0], color='#b9b9b9', lw=.7)
        span = np.maximum(hi-lo, .02); mid=(lo+hi)/2
        ax.set_xlim((mid[0]-span[0]*.6)*1000, (mid[0]+span[0]*.6)*1000)
        ax.set_ylim((mid[1]-span[1]*.6)*1000, (mid[1]+span[1]*.6)*1000)
        ax.set_zlim(max(0, (mid[2]-span[2]*.6)*1000), (mid[2]+span[2]*.6)*1000)
        ax.set_xlabel('x (mm)', labelpad=1); ax.set_ylabel('y (mm)', labelpad=1); ax.set_zlabel('z (mm)', labelpad=1)
        ax.view_init(elev=22, azim=-63)
        ax.grid(False)
        if save:
            fig.savefig(output, bbox_inches='tight', pad_inches=.03)
    draw(len(q)-1, save=True)
    if output.suffix.lower() == '.png':
        fig.savefig(output.with_suffix('.svg'), bbox_inches='tight', pad_inches=.03)
    gif = output.with_suffix('.gif')
    ani = FuncAnimation(fig, lambda k: draw(k), frames=len(q), interval=120, blit=False)
    ani.save(gif, writer=PillowWriter(fps=8), dpi=110)
    plt.close(fig)
    metric = output.with_name(output.stem+'_metrics.png')
    frames = summary['frames']; t=np.array([f['time_s'] for f in frames]); pen=np.array([f.get('max_penetration_m',0) for f in frames])*1e6
    force=np.array([f.get('contact_normal_force_n',0) for f in frames]); E=np.array([f.get('energy_j',0) for f in frames])
    fig, axes=plt.subplots(3,1,figsize=(5.5,5),sharex=True,constrained_layout=True)
    axes[0].plot(t*1000,pen,color='#517f9b'); axes[0].set_ylabel('penetration (µm)')
    axes[1].plot(t*1000,force,color='#ad5b4f'); axes[1].set_ylabel('normal force (N)')
    axes[2].plot(t*1000,E,color='#877293'); axes[2].set_ylabel('cable + steel (J)'); axes[2].set_xlabel('time (ms)')
    for axx in axes: axx.spines[['top','right']].set_visible(False)
    fig.savefig(metric,dpi=300); fig.savefig(metric.with_suffix('.svg')); plt.close(fig)
    return dict(status='rendered', static=str(output), animation=str(gif), metrics=str(metric), frames=len(q))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--parameters',type=Path,default=HERE/'output/parameters.snapshot.json'); args=ap.parse_args(); print(json.dumps(render(args.input,args.output,args.parameters),indent=2))

"""Paper-style render of saved, physically solved ribbon/body trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_body_loads import FullBodyCables, read_body_inertias
from run import HERE, make_robot, read_project, refresh, strip_geometry


def _ring(center, R, radius, thickness, count=48):
    a = np.linspace(0., 2*np.pi, count, endpoint=False)
    yz = np.column_stack((np.zeros(count), radius*np.cos(a), radius*np.sin(a)))
    return np.stack((center+(yz+[-thickness/2, 0, 0])@R.T,
                     center+(yz+[thickness/2, 0, 0])@R.T))


def _ribbon_faces(xyz, widths, width):
    offset = .5*width*widths
    return np.stack((xyz[..., :-1, :]-offset, xyz[..., 1:, :]-offset,
                     xyz[..., 1:, :]+offset, xyz[..., :-1, :]+offset), axis=-2)


def render(input_dir, output, parameters):
    input_dir, output = Path(input_dir), Path(output)
    summary = json.loads((input_dir/'summary.json').read_text(encoding='utf-8'))
    metadata = summary['metadata']
    p, delta, provenance = read_project(Path(parameters))
    with np.load(input_dir/'trajectory.npz') as saved:
        data = {key: saved[key] for key in saved.files}
    q, body_com, body_R = data['q'], data['body_com'], data['body_R']
    nt, strips, nq = q.shape
    n = (nq+1)//4
    segments = int(metadata.get('segments', strips//int(p['strip_count'])))
    if nt == 0 or nq != 4*n-1 or not np.isfinite(q).all():
        raise ValueError('Expected nonempty finite ribbon trajectory')
    width = float(data.get('strip_width_m', p['strip_width_m']))
    xyz = q[..., :3*n].reshape(nt, strips, n, 3)
    if 'width_directors' in data and data['width_directors'].ndim == 4:
        widths = data['width_directors']
    else:
        # Old trajectories omitted material frames: replay temporal transport,
        # using the existing frame update without re-solving any mechanics.
        widths = np.empty((nt, strips, n-1, 3))
        for s in range(strips):
            rest = data['rest_nodes_m'][s]
            seed = strip_geometry(p, delta, s % int(p['strip_count']), n)[1]
            robot, _ = make_robot(p, rest, seed, 'sano')
            for k in range(nt):
                robot = refresh(robot, q[k, s])
                widths[k, s] = robot.state.m2
    if widths.shape != (nt, strips, n-1, 3) or not np.isfinite(widths).all():
        raise ValueError('Material width directors must match every saved edge')
    if 'body_com_local_m' in data:
        com_local = data['body_com_local_m']
    else:
        com_local = read_body_inertias(p, provenance['source_urdf'])['com_local_m']
    if 'plate_body_indices' in data:
        plate_body = data['plate_body_indices'].astype(int)
        plate_local = data['plate_centers_local_m']
        plate_R_local = data['plate_R_local']
    else:
        plate_body = np.arange(len(com_local))
        plate_local = np.zeros_like(com_local)
        plate_R_local = np.repeat(np.eye(3)[None], len(com_local), axis=0)
    plate_R = np.einsum('tpij,pjk->tpik', body_R[:, plate_body], plate_R_local)
    plate_centers = body_com[:, plate_body]+np.einsum(
        'tpij,pj->tpi', body_R[:, plate_body], plate_local-com_local[plate_body])
    if 'cable_routes_m' in data:
        routes = data['cable_routes_m'].reshape(nt, -1, 2, 3)
    else:
        cable = FullBodyCables(p, com_local)
        routes = np.array([cable.evaluate(c, R, np.ones(4))['routes_m']
                           for c, R in zip(body_com, body_R)])
    if len(plate_body) == 2*segments:
        paired = plate_centers.reshape(nt, segments, 2, 3)
        links = [(2*s+1, 2*s+2) for s in range(segments-1)]
    elif len(plate_body) == segments+1:
        paired = np.stack((plate_centers[:, :-1], plate_centers[:, 1:]), axis=2)
        links = []
    else:
        raise ValueError('Plate mapping must contain segment pairs or shared endplates')
    lengths = np.linalg.norm(paired[:, :, 1]-paired[:, :, 0], axis=-1)
    faces = _ribbon_faces(xyz, widths, width)
    rings = np.array([[_ring(c, R, p['plate_stop_radius_m'],
                             p['plate_stop_thickness_m'])
                       for c, R in zip(centers, rotations)]
                      for centers, rotations in zip(plate_centers, plate_R)])
    lo = np.minimum(faces.reshape(-1, 3).min(0), rings.reshape(-1, 3).min(0))
    hi = np.maximum(faces.reshape(-1, 3).max(0), rings.reshape(-1, 3).max(0))
    ground = float(metadata.get('ground_height_m', 0.))
    lo[2] = min(lo[2], ground)
    span = np.maximum(hi-lo, .02)
    margin = np.maximum(span*.045, .005)
    limits = np.column_stack((lo-margin, hi+margin))*1000
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                         'svg.fonttype': 'none'})
    colors = ['#68899b', '#7c9a8b', '#b2936c', '#a57973', '#8a83a3']
    size = (9.2, 4.) if segments > 1 else (6.2, 5.2)
    fig = plt.figure(figsize=size, dpi=160)
    if segments > 1:
        # Axes3D insists on a square viewport. Let it extend outside the
        # landscape canvas, keeping the whole chain inside the visible crop.
        height = size[0]/size[1]
        ax = fig.add_axes([0., .5-height/2, 1., height], projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    floor = np.array([[limits[0,0], limits[1,0], ground*1000],
                      [limits[0,1], limits[1,0], ground*1000],
                      [limits[0,1], limits[1,1], ground*1000],
                      [limits[0,0], limits[1,1], ground*1000]])
    def draw(k):
        ax.clear()
        ax.add_collection3d(Poly3DCollection([floor], facecolor='#f1f2f3',
                                             edgecolor='#d7dade', linewidth=.35, alpha=.28))
        for s in range(strips):
            color = colors[(s//int(p['strip_count'])) % len(colors)]
            ax.add_collection3d(Poly3DCollection(faces[k, s]*1000,
                facecolor=color, edgecolor='#61747d', linewidth=.18, alpha=.88))
        for ring in rings[k]*1000:
            sides = np.stack((ring[0], np.roll(ring[0], -1, axis=0),
                              np.roll(ring[1], -1, axis=0), ring[1]), axis=1)
            ax.add_collection3d(Poly3DCollection([*ring, *sides],
                facecolor='#e1e5e7', edgecolor='#69757e', linewidth=.25, alpha=.82))
        for route in routes[k]*1000:
            ax.plot(*route.T, color='#946843', lw=.7, linestyle=(0, (4, 2)))
        for a, b in links:
            line = plate_centers[k, [a,b]]*1000
            ax.plot(*line.T, color='#56616a', lw=2.)
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_zlim(limits[2])
        ax.set_box_aspect(limits[:,1]-limits[:,0])
        ax.view_init(elev=19, azim=-68)
        ax.set_axis_off()
    draw(nt-1)
    save_options = {} if segments > 1 else dict(bbox_inches='tight', pad_inches=.015)
    fig.savefig(output, **save_options)
    if output.suffix.lower() == '.png':
        fig.savefig(output.with_suffix('.svg'), **save_options)
    gif = output.with_suffix('.gif')
    # Playback is slower than simulated time; report both durations explicitly.
    playback = list(range(nt))+[nt-1]*10
    ani = FuncAnimation(fig, draw, frames=playback, interval=100, blit=False)
    ani.save(gif, writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)

    metric = output.with_name(output.stem+'_metrics.png')
    frames = summary['frames']
    t = np.array([f['time_s'] for f in frames])
    if len(t) != nt:
        raise ValueError('Summary times must match trajectory frames')
    tension = np.array([f.get('tensions_n', np.zeros(segments*4)) for f in frames]).reshape(nt, segments, 4)
    pen = np.array([f.get('max_penetration_m', 0) for f in frames])*1e6
    normal = np.array([f.get('contact_normal_force_n', 0) for f in frames])
    fig, axes = plt.subplots(4, 1, figsize=(6.1, 6.6), sharex=True, constrained_layout=True)
    for s in range(segments):
        color = colors[s % len(colors)]
        axes[0].plot(t, (lengths[0,s]-lengths[:,s])*1000, color=color, label=f'{s+1}')
        axes[1].plot(t, tension[:,s].max(1), color=color)
    axes[0].set_ylabel('Contraction (mm)')
    axes[0].legend(title='Segment', ncol=segments, loc='upper left', frameon=False)
    axes[1].set_ylabel('Peak cable tension (N)')
    axes[2].plot(t, normal, color='#68899b'); axes[2].set_ylabel('Ground normal force (N)')
    axes[3].plot(t, pen, color='#a57973'); axes[3].set_ylabel('Penetration (µm)')
    axes[3].set_xlabel('Simulated time (s)')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color='#eeeeee', linewidth=.6)
    fig.savefig(metric, dpi=300)
    fig.savefig(metric.with_suffix('.svg'))
    plt.close(fig)
    return dict(status='rendered', static=str(output), animation=str(gif),
                metrics=str(metric), frames=nt, segments=segments, strips=strips,
                playback_fps=10, simulated_duration_s=float(t[-1]-t[0]),
                max_segment_contraction_mm=np.max((lengths[0]-lengths)*1000, axis=0).tolist())


def self_check():
    xyz = np.array([[0., 0., 0.], [.02, 0., 0.], [.04, .001, 0.]])
    widths = np.array([[0., 1., 0.], [0., 0., 1.]])
    faces = _ribbon_faces(xyz, widths, .016)
    assert faces.shape == (2, 4, 3)
    assert np.allclose(np.linalg.norm(faces[:,3]-faces[:,0], axis=1), .016)
    assert np.allclose((faces[:,0]+faces[:,3])/2, xyz[:-1])
    assert np.allclose((faces[:,1]+faces[:,2])/2, xyz[1:])
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert np.allclose(_ribbon_faces(xyz@R.T, widths@R.T, .016), faces@R.T)
    return dict(status='passed', check='surface width, endpoints, and rigid rotation')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--parameters', type=Path, default=HERE/'output/parameters.snapshot.json')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    if args.self_check:
        result = self_check()
    elif args.input and args.output:
        result = render(args.input, args.output, args.parameters)
    else:
        ap.error('--input and --output are required for rendering')
    print(json.dumps(result, indent=2))

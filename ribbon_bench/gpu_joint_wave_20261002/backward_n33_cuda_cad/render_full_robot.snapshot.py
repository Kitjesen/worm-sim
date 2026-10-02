"""Paper-style render of saved, physically solved ribbon/body trajectories."""
from __future__ import annotations

import argparse
import hashlib
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


def _joint_lines(centers, pivots, links):
    if len(pivots) != len(links):
        raise ValueError('Each intersegment connector needs one saved joint pivot')
    return np.array([[centers[a], pivot, centers[b]]
                     for (a, b), pivot in zip(links, pivots)])


def _stl_triangles(raw):
    """Read the project's binary STL in metres, without a mesh dependency."""
    if len(raw) < 84:
        raise ValueError('Truncated binary STL')
    count = int.from_bytes(raw[80:84], 'little')
    if not count or len(raw) != 84+50*count:
        raise ValueError('Expected complete binary STL')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                      ('attribute', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)['vertices'].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError('Nonfinite STL vertex')
    return triangles


def _cluster_mesh(triangles, tolerance):
    """Collapse nearby vertices together; retain all nondegenerate surfaces."""
    # ponytail: sub-cell CAD details can merge; reduce tolerance for a close-up.
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    if tolerance == 0:
        return triangles, 0.
    _, groups = np.unique(np.rint(vertices/tolerance).astype(np.int64),
                          axis=0, return_inverse=True)
    counts = np.bincount(groups)
    clustered = np.column_stack([np.bincount(groups, weights=vertices[:, i])/counts
                                 for i in range(3)])
    faces = groups[inverse].reshape(-1, 3)
    valid = (faces[:,0] != faces[:,1]) & (faces[:,1] != faces[:,2]) & (faces[:,0] != faces[:,2])
    faces = faces[valid]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    result = clustered[faces[np.sort(unique)]]
    maximum_error = float(np.linalg.norm(clustered[groups]-vertices, axis=1).max())
    if not len(result):
        raise ValueError('Mesh clustering removed every face; reduce the tolerance')
    return result, maximum_error


def _cad_meshes(p, segments, directory, tolerance):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('CAD mesh LOD tolerance must be finite and nonnegative')
    meshes, sources = [], []
    for segment in range(2, segments+2):
        for side in ('front', 'back'):
            path = Path(directory)/f'{side}{segment}_Link.STL'
            raw = path.read_bytes()
            original = _stl_triangles(raw)
            triangles, error = _cluster_mesh(original, tolerance)
            local_center = np.asarray(p[f'{side}_plate_center_m'])
            meshes.append(triangles-local_center)
            sources.append(dict(link=path.stem, source_path=str(path),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                original_triangles=len(original), rendered_triangles=len(triangles),
                                lod_tolerance_m=tolerance, max_vertex_displacement_m=error,
                                bounds_plate_local_m=[(original-local_center).reshape(-1, 3).min(0).tolist(),
                                                      (original-local_center).reshape(-1, 3).max(0).tolist()],
                                plate_center_local_m=local_center.tolist()))
    return meshes, sources


def render(input_dir, output, parameters, *, cad_meshes=False, mesh_lod_m=.001,
           static_only=False, frame_index=-1):
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
    pivots = data.get('joint_pivots_m')
    if pivots is not None and pivots.shape == (nt, 0, 3):
        pivots = None
    if pivots is not None:
        if pivots.shape != (nt, len(links), 3) or not np.isfinite(pivots).all():
            raise ValueError('Saved joint pivots must match trajectory and connector count')
        joint_lines = np.array([_joint_lines(c, pivot, links)
                                for c, pivot in zip(plate_centers, pivots)])
    else:
        joint_lines = plate_centers[:, np.array(links)].copy() if links else None
    faces = _ribbon_faces(xyz, widths, width)
    rings = np.array([[_ring(c, R, p['plate_stop_radius_m'],
                             p['plate_stop_thickness_m'])
                       for c, R in zip(centers, rotations)]
                      for centers, rotations in zip(plate_centers, plate_R)])
    lo = np.minimum(faces.reshape(-1, 3).min(0), rings.reshape(-1, 3).min(0))
    hi = np.maximum(faces.reshape(-1, 3).max(0), rings.reshape(-1, 3).max(0))
    meshes, mesh_sources = [], []
    if cad_meshes:
        if len(plate_body) != 2*segments:
            raise ValueError('CAD meshes require saved front/back plate pairs')
        meshes, mesh_sources = _cad_meshes(p, segments, HERE.parent/'meshes/longworm2', mesh_lod_m)
        for i, source in enumerate(mesh_sources):
            corners = np.array(np.meshgrid(*np.asarray(source['bounds_plate_local_m']).T,
                                            indexing='ij')).reshape(3, -1).T
            world = plate_centers[:, i, None]+np.einsum('tij,vj->tvi', plate_R[:, i], corners)
            lo = np.minimum(lo, world.reshape(-1, 3).min(0))
            hi = np.maximum(hi, world.reshape(-1, 3).max(0))
    if pivots is not None and pivots.size:
        lo = np.minimum(lo, pivots.reshape(-1, 3).min(0))
        hi = np.maximum(hi, pivots.reshape(-1, 3).max(0))
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
    from matplotlib.colors import to_rgb
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
    camera = [19, -68]
    def draw(k):
        ax.clear()
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_zlim(limits[2])
        ax.set_autoscale_on(False)
        ax.add_collection3d(Poly3DCollection([floor], facecolor='#f1f2f3',
                                             edgecolor='#d7dade', linewidth=.35, alpha=.28))
        for s in range(strips):
            color = colors[(s//int(p['strip_count'])) % len(colors)]
            ax.add_collection3d(Poly3DCollection(faces[k, s]*1000,
                facecolor=color, edgecolor='#61747d', linewidth=.18, alpha=.88))
        if meshes:
            for i, mesh in enumerate(meshes):
                world = plate_centers[k, i]+mesh@plate_R[k, i].T
                normals = np.cross(world[:,1]-world[:,0], world[:,2]-world[:,0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
                light = np.array([.6, -.4, .7]); light /= np.linalg.norm(light)
                # Exported overlapping parts may have opposite triangle winding;
                # two-sided lighting keeps coplanar plate faces the same shade.
                shades = .45+.55*np.abs(normals@light)
                color = np.asarray(to_rgb('#c9ced2' if i % 2 == 0 else '#8b989f'))
                ax.add_collection3d(Poly3DCollection(world*1000,
                    facecolors=shades[:, None]*color, edgecolor='none', linewidth=0, alpha=1.))
        else:
            for ring in rings[k]*1000:
                sides = np.stack((ring[0], np.roll(ring[0], -1, axis=0),
                                  np.roll(ring[1], -1, axis=0), ring[1]), axis=1)
                ax.add_collection3d(Poly3DCollection([*ring, *sides],
                    facecolor='#e1e5e7', edgecolor='#69757e', linewidth=.25, alpha=.82))
        for route in routes[k]*1000:
            ax.plot(*route.T, color='#946843', lw=.7, linestyle=(0, (4, 2)))
        if joint_lines is not None and not meshes:
            for line in joint_lines[k]*1000:
                ax.plot(*line.T, color='#56616a', lw=2.)
        if pivots is not None and pivots.shape[1]:
            ax.scatter(*pivots[k].T*1000, s=13, color='#d08035',
                       edgecolors='#74441c', linewidths=.35, depthshade=False)
        ax.set_box_aspect(limits[:,1]-limits[:,0])
        ax.view_init(elev=camera[0], azim=camera[1])
        ax.set_axis_off()
    if not -nt <= frame_index < nt:
        raise ValueError('Static frame index outside the saved trajectory')
    static_frame = frame_index % nt
    draw(static_frame)
    save_options = {} if segments > 1 else dict(bbox_inches='tight', pad_inches=.015)
    fig.savefig(output, **save_options)
    if output.suffix.lower() == '.png':
        fig.savefig(output.with_suffix('.svg'), **save_options)
    gif = output.with_suffix('.gif')
    # Playback is slower than simulated time; report both durations explicitly.
    playback = list(range(nt))+[nt-1]*10
    if not static_only:
        ani = FuncAnimation(fig, draw, frames=playback, interval=100, blit=False)
        ani.save(gif, writer=PillowWriter(fps=10), dpi=110)
    top = output.with_name(output.stem+'_top.png')
    if segments > 1:
        camera[:] = [90, -90]
        draw(static_frame)
        fig.savefig(top, **save_options)
        fig.savefig(top.with_suffix('.svg'), **save_options)
        if not static_only:
            ani = FuncAnimation(fig, draw, frames=playback, interval=100, blit=False)
            ani.save(top.with_suffix('.gif'), writer=PillowWriter(fps=10), dpi=110)
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
    extra = {}
    if segments > 1:
        extra.update(top_static=str(top), top_animation=None if static_only else str(top.with_suffix('.gif')))
    joint_angles = data.get('joint_angles_rad')
    if joint_angles is not None and joint_angles.shape == (nt, 0):
        joint_angles = None
    if segments > 1 and joint_angles is not None:
        actual = np.asarray(joint_angles)
        target = np.asarray(data.get('joint_targets_rad', actual))
        if (actual.shape != (nt, segments-1) or target.shape != actual.shape
                or not np.isfinite(actual).all() or not np.isfinite(target).all()):
            raise ValueError('Saved joint angles and targets must match the connectors')
        fig, axes = plt.subplots(segments-1, 1, figsize=(6.1, 1.5*(segments-1)),
                                 sharex=True, constrained_layout=True, squeeze=False)
        for j, axis in enumerate(axes[:, 0]):
            axis.plot(t, np.rad2deg(actual[:, j]), color=colors[j % len(colors)], label='Actual')
            axis.plot(t, np.rad2deg(target[:, j]), color='#6f777b', linestyle='--', label='Command')
            axis.set_ylabel(f'Joint {j+1} (°)')
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', color='#eeeeee', linewidth=.6)
        axes[0,0].legend(ncol=2, frameon=False, loc='upper left')
        axes[-1,0].set_xlabel('Simulated time (s)')
        joint_metric = output.with_name(output.stem+'_joint_metrics.png')
        fig.savefig(joint_metric, dpi=300); fig.savefig(joint_metric.with_suffix('.svg'))
        plt.close(fig)
        extra['joint_metrics'] = str(joint_metric)
        extra['peak_joint_angle_deg'] = np.rad2deg(np.max(np.abs(actual), axis=0)).tolist()
    rest = data.get('cable_rest_m')
    if rest is None and all('cable_rest_m' in frame for frame in frames):
        rest = np.array([frame['cable_rest_m'] for frame in frames])
    if rest is not None and nt > 1:
        rest = np.asarray(rest).reshape(nt, segments, 4)
        base = np.asarray(data.get('base_cable_lengths_m', rest[0])).reshape(segments, 4)
        shortening = (base[None]-rest).mean(axis=2)*1000
        fig, axis = plt.subplots(figsize=(6.1, 3.), constrained_layout=True)
        heat = axis.pcolormesh(t, np.arange(1, segments+1), shortening.T,
                              shading='nearest', cmap='YlGnBu', vmin=0.)
        axis.set_yticks(np.arange(1, segments+1))
        axis.set_ylim(segments+.5, .5)
        axis.set_xlim(t[0], t[-1])
        axis.set_ylabel('Segment (1: head; 5: tail)' if segments == 5 else 'Segment (head → tail)')
        axis.set_xlabel('Simulated time (s)')
        fig.colorbar(heat, ax=axis, label='Commanded cable shortening (mm)')
        wave = output.with_name('backward_wave.png')
        fig.savefig(wave, dpi=300); fig.savefig(wave.with_suffix('.svg'))
        plt.close(fig)
        extra['command_wave'] = str(wave)
    if 'body_mass_kg' in data:
        masses = np.asarray(data['body_mass_kg'])
        if masses.shape != (body_com.shape[1],) or masses.sum() <= 0:
            raise ValueError('Saved body masses must match the rigid bodies')
        rigid_com = np.einsum('tbi,b->ti', body_com, masses)/masses.sum()
        extra['rigid_body_com_displacement_mm'] = ((rigid_com[-1]-rigid_com[0])*1000).tolist()
    return dict(status='rendered', static=str(output), animation=None if static_only else str(gif),
                metrics=str(metric), frames=nt, segments=segments, strips=strips,
                static_frame=static_frame, cad_mesh_sources=mesh_sources,
                cad_mesh_transform='world = saved_plate_center + (STL_vertex - parameter_plate_center_local) @ saved_plate_R.T' if meshes else None,
                cad_mesh_scope=f'CAD front/back2..{segments+1}; wheel visuals and wheel spin omitted' if meshes else None,
                cad_mesh_role='Visual replay only; STL surfaces are not added as collision geometry' if meshes else None,
                playback_fps=10, simulated_duration_s=float(t[-1]-t[0]),
                max_segment_contraction_mm=np.max((lengths[0]-lengths)*1000, axis=0).tolist(),
                **extra)


def self_check():
    import struct
    xyz = np.array([[0., 0., 0.], [.02, 0., 0.], [.04, .001, 0.]])
    widths = np.array([[0., 1., 0.], [0., 0., 1.]])
    faces = _ribbon_faces(xyz, widths, .016)
    assert faces.shape == (2, 4, 3)
    assert np.allclose(np.linalg.norm(faces[:,3]-faces[:,0], axis=1), .016)
    assert np.allclose((faces[:,0]+faces[:,3])/2, xyz[:-1])
    assert np.allclose((faces[:,1]+faces[:,2])/2, xyz[1:])
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert np.allclose(_ribbon_faces(xyz@R.T, widths@R.T, .016), faces@R.T)
    centers = np.array([[0., 0., 0.], [.02, 0., 0.], [.04, .01, 0.]])
    pivot = np.array([[.03, .004, .002]])
    line = _joint_lines(centers, pivot, [(1, 2)])
    assert line.shape == (1, 3, 3)
    np.testing.assert_array_equal(line[0], [centers[1], pivot[0], centers[2]])
    triangle = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
    raw = bytes(80)+struct.pack('<I', 1)+struct.pack('<12fH', *([0., 0., 1.]+triangle.ravel().tolist()), 0)
    np.testing.assert_allclose(_stl_triangles(raw), triangle[None], atol=1e-9)
    clustered, error = _cluster_mesh(np.array([triangle, triangle+[1e-5, 0., 0.]]), .001)
    assert clustered.shape == (1, 3, 3) and error < 1e-5
    assert np.cross(clustered[0,1]-clustered[0,0], clustered[0,2]-clustered[0,0])[2] > 0
    # The mesh and ribbon clamp share the same saved plate transform.
    local_center = np.array([.02, .003, -.004])
    world_center = np.array([.2, -.03, .05])
    world_mesh = world_center+(triangle-local_center)@R.T
    np.testing.assert_allclose(world_mesh[0], world_center+R@(triangle[0]-local_center))
    return dict(status='passed', check='ribbon geometry, saved joint pivot, binary STL, surface LOD and plate transform')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--parameters', type=Path, default=HERE/'output/parameters.snapshot.json')
    ap.add_argument('--cad-meshes', action='store_true', help='Render original front/back CAD meshes')
    ap.add_argument('--mesh-lod-m', type=float, default=.001, help='Vertex clustering cell width; 0 keeps full STL')
    ap.add_argument('--static-only', action='store_true', help='Skip GIF generation for visual inspection')
    ap.add_argument('--frame-index', type=int, default=-1, help='Saved trajectory frame for static figures')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    if args.self_check:
        result = self_check()
    elif args.input and args.output:
        result = render(args.input, args.output, args.parameters,
                        cad_meshes=args.cad_meshes, mesh_lod_m=args.mesh_lod_m,
                        static_only=args.static_only, frame_index=args.frame_index)
    else:
        ap.error('--input and --output are required for rendering')
    print(json.dumps(result, indent=2))

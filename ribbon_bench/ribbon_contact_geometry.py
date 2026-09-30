"""Material surface samples and local analytic Jacobians for rectangular rods.

Each edge has 12 fixed IDs: s=(0, .5, 1), then (width, thickness) signs
(-,-), (-,+), (+,-), (+,+). Width is along m2; thickness is along m1.
Endpoint samples on neighboring edges remain separate material points.

The old robot.state.q/a1 is held fixed during differentiation. Minimal-rotation
transport followed by projection/normalization matches the vendor construction,
using its smooth geometric limit rather than its 1e-10 tangent/cross-product
cutoffs. Literal differentiation of those cutoffs loses derivatives at planar
states. Near-antiparallel trial edges are rejected; subdivide such a trial.
This is sampled contact geometry, not continuous collision detection.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


MATERIAL_COORDINATES = np.array([(s, sw, sh) for s in (0., .5, 1.)
                                for sw, sh in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))])


def surface_geometry_local(robot, q, width, thickness):
    """Return points[E,12,3], J[E,12,3,7], global_dofs[E,7].

    Local DOFs are xyz of edge node 0, xyz of edge node 1, edge angle.
    Width/thickness are positive scalars or arrays of length E, held constant.
    Supports the vendor's rod DOFs (one angle per edge), not shell edges.
    """
    q = np.asarray(q, dtype=np.float64)
    n = len(robot.node_dof_indices)
    ne = len(robot.state.a1)
    edges = np.asarray(robot.edges[:ne], dtype=int)
    if q.shape != (3*n+ne,) or not np.isfinite(q).all():
        raise ValueError('Expected finite rod q with xyz nodes followed by edge angles')
    w, h = [np.broadcast_to(np.asarray(v, dtype=float), (ne,)) for v in (width, thickness)]
    if not (np.isfinite(w).all() and np.isfinite(h).all() and np.all(w > 0) and np.all(h > 0)):
        raise ValueError('Width and thickness must be positive and finite')
    xyz, xyz_old = q[:3*n].reshape(n, 3), robot.state.q[:3*n].reshape(n, 3)
    edge = xyz[edges[:, 1]]-xyz[edges[:, 0]]
    edge_old = xyz_old[edges[:, 1]]-xyz_old[edges[:, 0]]
    length, length_old = np.linalg.norm(edge, axis=1), np.linalg.norm(edge_old, axis=1)
    if np.any(length < 1e-10) or np.any(length_old < 1e-10):
        raise ValueError('Degenerate edge in surface geometry')
    t, t0 = edge/length[:, None], edge_old/length_old[:, None]
    # Derivative arrays below are [edge, local input xyz, output xyz].
    dt = (np.eye(3)-t[:, :, None]*t[:, None, :])/length[:, None, None]
    u = np.asarray(robot.state.a1)
    v = np.cross(t0, t)
    denom = 1+np.sum(t0*t, axis=1)
    if np.any(denom <= 1e-8):
        raise ValueError('Near-antiparallel temporal transport; reduce the trial increment')
    vu = np.cross(v, u)
    vvu = np.cross(v, vu)
    transported = u+vu+vvu/denom[:, None]
    dv = np.cross(t0[:, None, :], dt)
    dvu = np.cross(dv, u[:, None, :])
    dc = np.sum(t0[:, None, :]*dt, axis=-1)
    dr = dvu+(np.cross(dv, vu[:, None, :])+np.cross(v[:, None, :], dvu))/denom[:, None, None]
    dr -= vvu[:, None, :]*dc[:, :, None]/denom[:, None, None]**2
    dot = np.sum(t*transported, axis=1)
    projected = transported-t*dot[:, None]
    magnitude = np.linalg.norm(projected, axis=1)
    if np.any(magnitude < 1e-10):
        raise ValueError('Degenerate reference material director')
    a1 = projected/magnitude[:, None]
    ddot = np.sum(dt*transported[:, None, :]+t[:, None, :]*dr, axis=-1)
    projected_d = dr-dt*dot[:, None, None]-t[:, None, :]*ddot[:, :, None]
    da1 = (projected_d-a1[:, None, :]*np.sum(a1[:, None, :]*projected_d, axis=-1)[:, :, None])/magnitude[:, None, None]
    a2 = np.cross(t, a1)
    da2 = np.cross(dt, a1[:, None, :])+np.cross(t[:, None, :], da1)
    cosine, sine = np.cos(q[3*n:]), np.sin(q[3*n:])
    m1 = cosine[:, None]*a1+sine[:, None]*a2
    m2 = -sine[:, None]*a1+cosine[:, None]*a2
    dm1 = cosine[:, None, None]*da1+sine[:, None, None]*da2
    dm2 = -sine[:, None, None]*da1+cosine[:, None, None]*da2
    s, sw, sh = MATERIAL_COORDINATES.T
    cw, ch = .5*w[:, None]*sw, .5*h[:, None]*sh
    centers = (1-s)[None, :, None]*xyz[edges[:, 0], None, :]+s[None, :, None]*xyz[edges[:, 1], None, :]
    points = centers+cw[:, :, None]*m2[:, None, :]+ch[:, :, None]*m1[:, None, :]
    offset_d = cw[:, :, None, None]*dm2[:, None, :, :]+ch[:, :, None, None]*dm1[:, None, :, :]
    offset_d = offset_d.swapaxes(-1, -2)
    jac = np.empty((ne, 12, 3, 7))
    jac[..., :3] = (1-s)[None, :, None, None]*np.eye(3)-offset_d
    jac[..., 3:6] = s[None, :, None, None]*np.eye(3)+offset_d
    jac[..., 6] = -cw[:, :, None]*m1[:, None, :]+ch[:, :, None]*m2[:, None, :]
    dofs = np.column_stack((3*edges[:, 0, None]+np.arange(3),
                            3*edges[:, 1, None]+np.arange(3), 3*n+np.arange(ne)))
    return points, jac, dofs


def surface_geometry(robot, q, width, thickness):
    """Return points[P,3], Jacobian[P,3,nDOF]; P=12*number_of_rod_edges."""
    points, local, dofs = surface_geometry_local(robot, q, width, thickness)
    jac = np.zeros((*points.shape, len(q)))
    np.put_along_axis(jac, dofs[:, None, None, :], local, axis=-1)
    return points.reshape(-1, 3), jac.reshape(-1, 3, len(q))


def self_check():
    from run import HERE, make_robot, read_project, strip_geometry
    p, delta, _ = read_project(HERE/'output/parameters.snapshot.json')
    w, h = p['strip_width_m'], p['strip_thickness_m']
    rng = np.random.default_rng(7401)
    checks = []
    for n in (9, 17):
        rest, widths = strip_geometry(p, delta, 0, n)
        robot, _ = make_robot(p, rest, widths, 'sano')
        original = robot.state.q.copy()
        for perturbed in (False, True):
            q = original.copy()
            if perturbed:
                q[:3*n] += rng.normal(size=3*n)*1e-4
                q[3*n:] += rng.uniform(-.8, .8, n-1)
            points, jac = surface_geometry(robot, q, w, h)
            a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
            m1, m2 = robot.compute_material_directors(q, a1, a2)
            s, sw, sh = MATERIAL_COORDINATES.T
            xyz = q[:3*n].reshape(n, 3)
            expected = ((1-s)[None, :, None]*xyz[:-1, None]+s[None, :, None]*xyz[1:, None]
                        +.5*w*sw[None, :, None]*m2[:, None]+.5*h*sh[None, :, None]*m1[:, None])
            vendor_error = float(np.max(np.abs(points.reshape(n-1, 12, 3)-expected)))
            fd = np.empty_like(jac)
            for column in range(len(q)):
                eps = 2e-7 if column < 3*n else 2e-6
                dq = np.zeros_like(q)
                dq[column] = eps
                upper = surface_geometry_local(robot, q+dq, w, h)[0]
                lower = surface_geometry_local(robot, q-dq, w, h)[0]
                fd[:, :, column] = ((upper-lower)/(2*eps)).reshape(-1, 3)
            fd_error = float(np.max(np.abs(fd-jac)))
            assert vendor_error < 1e-12 and fd_error < 2e-7, (vendor_error, fd_error)
            corners = points.reshape(n-1, 3, 4, 3)
            assert np.allclose(np.linalg.norm(corners[:, :, 2]-corners[:, :, 0], axis=-1), w, rtol=1e-12)
            assert np.allclose(np.linalg.norm(corners[:, :, 1]-corners[:, :, 0], axis=-1), h, rtol=1e-11)
            shift = np.array([.03, -.07, .02])
            moved = q.copy()
            moved[:3*n] += np.tile(shift, n)
            assert np.max(np.abs(surface_geometry(robot, moved, w, h)[0]-points-shift)) < 1e-14
            dq = np.r_[np.tile(shift, n), np.zeros(n-1)]
            assert np.max(np.abs(jac@dq-shift)) < 1e-14
            checks.append(dict(nodes=n, perturbed=perturbed, vendor_point_error_m=vendor_error,
                               finite_difference_jacobian_max_absolute_error=fd_error))
        points, jac = surface_geometry(robot, original, w, h)
        omega = np.array([.31, -.23, .41])
        tangent = np.diff(rest, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        dq = np.r_[np.cross(omega, rest).ravel(), tangent@omega]
        rotation_error = float(np.max(np.abs(jac@dq-np.cross(omega, points))))
        assert rotation_error < 1e-12, rotation_error
        angle = np.linalg.norm(omega)
        axis = omega/angle
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        rotation = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
        rotated = original.copy()
        rotated[:3*n] = (rest@rotation.T).ravel()
        a1, a2 = robot.compute_time_parallel(robot.state.a1, original, rotated)
        normal = robot.state.m1@rotation.T
        rotated[3*n:] = np.arctan2(np.sum(normal*a2, axis=1), np.sum(normal*a1, axis=1))
        finite_rotation_error = float(np.max(np.abs(surface_geometry(robot, rotated, w, h)[0]-points@rotation.T)))
        assert finite_rotation_error < 1e-12, finite_rotation_error
        twisted = original.copy()
        twisted[3*n:] += np.pi/2
        twist_points = surface_geometry(robot, twisted, w, h)[0].reshape(n-1, 12, 3)
        s, sw, sh = MATERIAL_COORDINATES.T
        expected = ((1-s)[None, :, None]*rest[:-1, None]+s[None, :, None]*rest[1:, None]
                    -.5*w*sw[None, :, None]*robot.state.m1[:, None]+.5*h*sh[None, :, None]*robot.state.m2[:, None])
        twist_error = float(np.max(np.abs(twist_points-expected)))
        assert twist_error < 1e-12 and np.max(np.abs(twist_points.reshape(-1, 3)-points)) > w/4
        started = time.perf_counter()
        for _ in range(200):
            surface_geometry_local(robot, original, w, h)
        checks.append(dict(nodes=n, rigid_rotation_jacobian_error=rotation_error,
                           finite_rigid_rotation_error_m=finite_rotation_error,
                           quarter_turn_surface_error_m=twist_error,
                           mean_local_call_ms=(time.perf_counter()-started)*5))
    return dict(status='passed', width_m=w, thickness_m=h,
                scope='Geometry and Jacobian only; finite differences used solely by this self-check', checks=checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = self_check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))

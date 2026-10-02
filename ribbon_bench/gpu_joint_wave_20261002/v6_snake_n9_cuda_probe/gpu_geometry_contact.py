"""Batched PyTorch geometry/contact kernels for the full ribbon model.

These functions mirror :mod:`ribbon_contact_geometry` and :mod:`ground_contact`
in float64.  They keep the expensive surface transport, Jacobian construction,
and plane penalty/friction law on one device.  The caller may convert the
small returned arrays to NumPy only at the material/Schur boundary.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # CPU-only installs may use the original NumPy path.
    torch = None

from ribbon_contact_geometry import MATERIAL_COORDINATES


def _require_torch():
    if torch is None:
        raise RuntimeError('PyTorch is required for GPU geometry/contact kernels')


def surface_geometry_batch(robot, q, width, thickness, *, device=None,
                           reference_q=None, reference_a1=None):
    """Return ``(points, local_jacobian)`` for a batch of rod states.

    ``q`` has shape ``(S, 3*n+ne)``.  The Jacobian uses seven local columns:
    the two endpoint translations and the edge twist.  This avoids a dense
    global Jacobian while preserving exactly the existing NumPy formula.
    """
    _require_torch()
    if device is None:
        device = q.device
    q = torch.as_tensor(q, dtype=torch.float64, device=device)
    if q.ndim != 2:
        raise ValueError("q must have shape (batch, dof)")
    n = len(robot.node_dof_indices)
    ne = len(robot.state.a1)
    if q.shape[1] != 3*n+ne:
        raise ValueError("q has the wrong rod dimension")
    edges = torch.as_tensor(robot.edges[:ne], dtype=torch.long, device=device)
    xyz = q[:, :3*n].reshape(-1, n, 3)
    if reference_q is None:
        xyz_old = torch.as_tensor(robot.state.q[:3*n].reshape(n, 3), dtype=torch.float64, device=device)[None].expand(q.shape[0], -1, -1)
    else:
        ref = torch.as_tensor(reference_q, dtype=torch.float64, device=device)
        xyz_old = ref[..., :3*n].reshape(q.shape[0], n, 3)
    edge = xyz[:, edges[:, 1]] - xyz[:, edges[:, 0]]
    edge_old = xyz_old[:, edges[:, 1]] - xyz_old[:, edges[:, 0]]
    length = torch.linalg.vector_norm(edge, dim=-1)
    length_old = torch.linalg.vector_norm(edge_old, dim=-1)
    if torch.any(length < 1e-10) or torch.any(length_old < 1e-10):
        raise ValueError("degenerate edge in surface geometry")
    t = edge/length[..., None]
    t0 = edge_old/length_old[..., None]
    eye = torch.eye(3, dtype=torch.float64, device=device)
    dt = (eye[None, None] - t[..., :, None]*t[..., None, :])/length[..., None, None]
    u0 = torch.as_tensor(robot.state.a1 if reference_a1 is None else reference_a1,
                         dtype=torch.float64, device=device)
    u = u0[None].expand(q.shape[0], -1, -1) if u0.ndim == 2 else u0
    v = torch.cross(t0, t, dim=-1)
    denom = 1 + torch.sum(t0*t, dim=-1)
    if torch.any(denom <= 1e-8):
        raise ValueError("near-antiparallel temporal transport")
    vu = torch.cross(v, u, dim=-1)
    vvu = torch.cross(v, vu, dim=-1)
    transported = u + vu + vvu/denom[..., None]
    dv = torch.cross(t0[:, :, None, :], dt, dim=-1)
    dvu = torch.cross(dv, u[:, :, None, :], dim=-1)
    dc = torch.sum(t0[:, :, None, :]*dt, dim=-1)
    dr = dvu + (torch.cross(dv, vu[:, :, None, :], dim=-1)
                + torch.cross(v[:, :, None, :], dvu, dim=-1))/denom[..., None, None]
    dr = dr - vvu[:, :, None, :]*dc[..., None]/denom[..., None, None]**2
    dot = torch.sum(t*transported, dim=-1)
    projected = transported - t*dot[..., None]
    magnitude = torch.linalg.vector_norm(projected, dim=-1)
    if torch.any(magnitude < 1e-10):
        raise ValueError("degenerate reference material director")
    a1 = projected/magnitude[..., None]
    ddot = torch.sum(dt*transported[:, :, None, :] + t[:, :, None, :]*dr, dim=-1)
    projected_d = dr - dt*dot[..., None, None] - t[:, :, None, :]*ddot[..., :, :, None]
    da1 = (projected_d-a1[:, :, None, :]*torch.sum(a1[:, :, None, :]*projected_d, dim=-1)[..., None])/magnitude[..., None, None]
    a2 = torch.cross(t, a1, dim=-1)
    da2 = torch.cross(dt, a1[:, :, None, :], dim=-1) + torch.cross(t[:, :, None, :], da1, dim=-1)
    angles = q[:, 3*n:]
    cosine, sine = torch.cos(angles), torch.sin(angles)
    m1 = cosine[:, :, None]*a1 + sine[:, :, None]*a2
    m2 = -sine[:, :, None]*a1 + cosine[:, :, None]*a2
    dm1 = cosine[:, :, None, None]*da1 + sine[:, :, None, None]*da2
    dm2 = -sine[:, :, None, None]*da1 + cosine[:, :, None, None]*da2
    coords = torch.as_tensor(MATERIAL_COORDINATES, dtype=torch.float64, device=device)
    s, sw, sh = coords.T
    w = torch.as_tensor(width, dtype=torch.float64, device=device)
    h = torch.as_tensor(thickness, dtype=torch.float64, device=device)
    w = torch.broadcast_to(w, (ne,)); h = torch.broadcast_to(h, (ne,))
    cw = .5*w[:, None]*sw[None, :]
    ch = .5*h[:, None]*sh[None, :]
    centers = ((1-s)[None, None, :, None]*xyz[:, edges[:, 0], None, :]
               + s[None, None, :, None]*xyz[:, edges[:, 1], None, :])
    points = centers + cw[None, :, :, None]*m2[:, :, None, :] + ch[None, :, :, None]*m1[:, :, None, :]
    offset = (cw[None, :, :, None, None]*dm2[:, :, None, :, :]
              + ch[None, :, :, None, None]*dm1[:, :, None, :, :]).transpose(-1, -2)
    local_jac = torch.empty((q.shape[0], ne, 12, 3, 7), dtype=torch.float64, device=device)
    local_jac[..., :3] = (1-s)[None, None, :, None, None]*eye[None, None, None] - offset
    local_jac[..., 3:6] = s[None, None, :, None, None]*eye[None, None, None] + offset
    local_jac[..., 6] = -cw[None, :, :, None]*m1[:, :, None, :] + ch[None, :, :, None]*m2[:, :, None, :]
    return points.reshape(q.shape[0], ne*12, 3), local_jac.reshape(q.shape[0], ne*12, 3, 7)


def contact_batch(points, oldpoints, history_elastic, history_active, *, dt, ground_height,
                  normal_stiffness, tangential_stiffness, weights, mu):
    """Batched float64 plane penalty + stick/slip contact law."""
    _require_torch()
    del dt  # rate-independent displacement law, retained for API parity
    p = points
    old = oldpoints
    w = torch.as_tensor(weights, dtype=torch.float64, device=p.device)
    kn = float(normal_stiffness)*w
    kt = float(tangential_stiffness)*w
    gap = p[..., 2] - float(ground_height)
    normal = kn*torch.relu(-gap)
    active = normal > 0
    old_elastic = torch.zeros_like(p[..., :2]) if history_elastic is None else history_elastic
    old_active = torch.zeros_like(active) if history_active is None else history_active
    trial = old_elastic + p[..., :2] - old[..., :2]
    trial_norm = torch.linalg.vector_norm(trial, dim=-1)
    slip = active & (float(mu) > 0) & (kt*trial_norm > float(mu)*normal)
    stick = active & (float(mu) > 0) & ~slip
    elastic = torch.zeros_like(trial)
    force = torch.zeros_like(p)
    jac = torch.zeros((*p.shape[:-1], 3, 3), dtype=p.dtype, device=p.device)
    force[..., 2] = normal
    jac[..., 2, 2] = torch.where(active, -kn, torch.zeros_like(kn))
    elastic = torch.where(stick[..., None], trial, elastic)
    force[..., :2] = torch.where(stick[..., None], -kt[..., None]*trial, force[..., :2])
    jac[..., 0, 0] = torch.where(stick, -kt, jac[..., 0, 0])
    jac[..., 1, 1] = torch.where(stick, -kt, jac[..., 1, 1])
    safe_norm = torch.clamp(trial_norm, min=1e-30)
    direction = trial/safe_norm[..., None]
    limit = float(mu)*normal
    slip_elastic = limit[..., None]/torch.clamp(kt[..., None], min=1e-30)*direction
    elastic = torch.where(slip[..., None], slip_elastic, elastic)
    force[..., :2] = torch.where(slip[..., None], -limit[..., None]*direction, force[..., :2])
    projection = torch.eye(2, dtype=p.dtype, device=p.device)[None, None] - direction[..., :, None]*direction[..., None, :]
    jac[..., :2, :2] = torch.where(slip[..., None, None], -limit[..., None, None]/safe_norm[..., None, None]*projection, jac[..., :2, :2])
    jac[..., :2, 2] = torch.where(slip[..., None], float(mu)*kn[..., None]*direction, jac[..., :2, 2])
    plastic = torch.where(active[..., None], trial-elastic, torch.zeros_like(trial))
    detached = old_active & ~active
    status_code = torch.zeros_like(active, dtype=torch.int8)
    status_code = torch.where(stick, torch.ones_like(status_code), status_code)
    status_code = torch.where(slip, torch.full_like(status_code, 2), status_code)
    if float(mu) == 0:
        status_code = torch.where(active, torch.full_like(status_code, 3), status_code)
    return dict(force=force, jacobian=jac, gap_m=gap, normal_force_n=normal,
                history_elastic_m=elastic, history_active=active,
                active=active, stick=stick, slip=slip,
                plastic_increment_m=plastic, detached=detached, status_code=status_code)


def self_check():
    """N9 one-strip NumPy parity check; runs on CPU or CUDA."""
    _require_torch()
    import numpy as np
    from ground_contact import evaluate_contact
    from ribbon_contact_geometry import surface_geometry, surface_geometry_local
    from run import HERE, make_robot, read_project, strip_geometry
    p, delta, _ = read_project(HERE/'output/parameters.snapshot.json')
    rest, width = strip_geometry(p, delta, 0, 9)
    robot, _ = make_robot(p, rest, width, 'sano')
    q = robot.state.q.copy()
    points, jac = surface_geometry(robot, q, p['strip_width_m'], p['strip_thickness_m'])
    _, local, _ = surface_geometry_local(robot, q, p['strip_width_m'], p['strip_thickness_m'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_points, gpu_jac = surface_geometry_batch(robot, torch.from_numpy(q[None]),
                                                 p['strip_width_m'], p['strip_thickness_m'], device=device)
    np.testing.assert_allclose(gpu_points[0].cpu().numpy(), points, rtol=0, atol=2e-14)
    np.testing.assert_allclose(gpu_jac[0].cpu().numpy(), local.reshape(-1, 3, 7), rtol=0, atol=2e-13)
    old = points.copy(); current = points.copy(); current[:, 2] -= 2e-5
    settings = dict(dt=.002, ground_height=0., normal_stiffness=1e8,
                    tangential_stiffness=2e7, weights=np.repeat(robot.ref_len/12., 12), mu=.4)
    ref = evaluate_contact(current, old, **settings)
    got = contact_batch(torch.from_numpy(current[None]).to(device), torch.from_numpy(old[None]).to(device),
                        None, None, **settings)
    np.testing.assert_allclose(got['force'][0].cpu().numpy(), ref['force'], rtol=0, atol=2e-14)
    np.testing.assert_allclose(got['jacobian'][0].cpu().numpy(), ref['jacobian'], rtol=0, atol=2e-13)
    return dict(status='passed', device=device, point_max_abs_error=float(np.max(np.abs(gpu_points[0].cpu().numpy()-points))),
                local_jacobian_max_abs_error=float(np.max(np.abs(gpu_jac[0].cpu().numpy()-local.reshape(-1, 3, 7)))),
                contact_force_max_abs_error=float(np.max(np.abs(got['force'][0].cpu().numpy()-ref['force']))),
                contact_jacobian_max_abs_error=float(np.max(np.abs(got['jacobian'][0].cpu().numpy()-ref['jacobian']))))


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), indent=2))

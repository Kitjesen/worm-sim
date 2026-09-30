"""Float64 CAD rigid-body inertials and four ideal straight tension-only cables.

Cable derivatives use x=(dc0,dphi0,dc1,dphi1), c'=c+dc and R'=Exp(dphi)R.
Both translation and left rotation increments are in world axes. Derivatives
are at x=0; forces/torques are minus the gradient, about each selected body
origin (plate center by default, or COM via origin_offsets_local).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
URDF = (HERE/'publication/data/cad_reference.urdf' if (HERE/'publication/data/cad_reference.urdf').exists()
        else HERE.parent/'worm-sim/meshes/longworm2/longworm2.SLDASM.urdf')


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=np.float64)


def _origin(element):
    if element is None:
        return np.zeros(3), np.eye(3)
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
        raise ValueError('Invalid URDF origin')
    return xyz, Rotation.from_euler('xyz', rpy).as_matrix()


def _inertial(link):
    inertial = link.find('inertial')
    if inertial is None:
        raise ValueError(f'Missing inertial for {link.get("name")}')
    com, rotation = _origin(inertial.find('origin'))
    mass = float(inertial.find('mass').get('value'))
    v = {key: float(value) for key, value in inertial.find('inertia').attrib.items()}
    inertia = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']],
                        [v['ixz'], v['iyz'], v['izz']]], dtype=np.float64)
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(inertia).all():
        raise ValueError('Invalid CAD mass or inertia')
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[0] <= 0 or eigenvalues[-1] > sum(eigenvalues[:2])*(1+1e-8):
        raise ValueError('CAD inertia is not a physical positive inertia tensor')
    return mass, com, rotation@inertia@rotation.T


def read_body_inertias(p, urdf_path=None, include_locked_wheels=True):
    """Merge selected CAD links at zero joint pose using the parallel-axis theorem.

    Body origins are p's plate centers, axes are the corresponding URDF link
    axes. COM inertias are returned separately: do not use them as inertia about
    the plate origin without the COM translation coupling.
    """
    path = Path(urdf_path) if urdf_path is not None else URDF
    raw = path.read_bytes()
    tree = ET.fromstring(raw)
    links = {link.get('name'): link for link in tree.findall('link')}
    joints = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
    roots = ('front2_Link', 'back2_Link')
    centers = np.asarray([p['front_plate_center_m'], p['back_plate_center_m']], dtype=np.float64)
    if centers.shape != (2, 3) or not np.isfinite(centers).all():
        raise ValueError('Need two finite plate-center vectors')
    masses, coms, inertias, components = [], [], [], []
    for index, root in enumerate(roots):
        names = [root]
        if include_locked_wheels:
            names += [f'w2-{i}_Link' for i in ((1, 2) if index == 0 else (3, 4))]
        parts = []
        for name in names:
            position, rotation = np.zeros(3), np.eye(3)
            if name != root:
                joint = joints[name]
                limit = joint.find('limit')
                locked = (joint.get('type') == 'fixed' or
                          (joint.get('type') in ('revolute', 'prismatic') and limit is not None
                           and float(limit.get('lower')) == float(limit.get('upper')) == 0.))
                if joint.find('parent').get('link') != root or not locked:
                    raise ValueError(f'{name} is not a direct fixed/zero-locked child of {root}')
                position, rotation = _origin(joint.find('origin'))
            mass, com, inertia = _inertial(links[name])
            parts.append((name, mass, position+rotation@com-centers[index], rotation@inertia@rotation.T))
        total = sum(part[1] for part in parts)
        com = sum(m*x for _, m, x, _ in parts)/total
        inertia = sum(I+m*(np.dot(x-com, x-com)*np.eye(3)-np.outer(x-com, x-com))
                      for _, m, x, I in parts)
        masses.append(total); coms.append(com); inertias.append(inertia)
        components.append([dict(link=name, mass_kg=m, com_local_m=x.tolist(), inertia_com_local_kgm2=I.tolist())
                           for name, m, x, I in parts])
    back_joint = joints['back2_Link']
    if back_joint.find('parent').get('link') != 'front2_Link':
        raise ValueError('Unexpected front2/back2 CAD relation')
    delta, back_rotation = _origin(back_joint.find('origin'))
    provenance = dict(urdf_path=str(path), urdf_sha256=hashlib.sha256(raw).hexdigest(),
        body_links=list(roots), components=components, include_locked_wheels=bool(include_locked_wheels),
        omitted_wheel_links=[] if include_locked_wheels else [f'w2-{i}_Link' for i in range(1, 5)],
        assumption='URDF wheel joints with zero lower/upper limits are locked rigidly to their own plate; no wheel spin or one-way clutch dynamics',
        omitted='Other robot segments, separate steel ribbons, unmodeled fasteners/cable/guide additions; servo internals remain as exported in plate-link inertials',
        aggregation='Rotate COM inertial tensors into plate axes, then parallel-axis sum at aggregate COM; no duplicate plate mass')
    return dict(mass_kg=np.asarray(masses), com_local_m=np.asarray(coms),
                inertia_com_local_kgm2=np.asarray(inertias),
                reference_centers_world_m=np.array([centers[0], delta+back_rotation@centers[1]]),
                reference_rotations=np.array([np.eye(3), back_rotation]), provenance=provenance)


class FullBodyCables:
    def __init__(self, p, origin_offsets_local=None):
        """Optional origins relative to plate centers; pass CAD COM offsets for COM coordinates.

        evaluate(c,R,...) then takes world positions of those selected origins,
        and gradient torques are about those origins rather than plate centers.
        """
        origins = np.zeros((2, 3)) if origin_offsets_local is None else np.asarray(origin_offsets_local, dtype=np.float64)
        if origins.shape != (2, 3) or not np.isfinite(origins).all():
            raise ValueError('Need finite origin_offsets_local(2,3)')
        self.front_offsets = np.asarray(p['tendon_anchors_m'], dtype=np.float64).reshape(4, 3)-p['front_plate_center_m']-origins[0]
        self.back_offsets = np.asarray(p['tendon_guides_m'], dtype=np.float64).reshape(4, -1, 3)[:, -1]-p['back_plate_center_m']-origins[1]
        self.k = float(p['effective_tendon_stiffness_n_m'])
        if not np.isfinite(self.k) or self.k <= 0 or not np.isfinite([self.front_offsets, self.back_offsets]).all():
            raise ValueError('Need finite attachment geometry and positive cable stiffness')

    def evaluate(self, c, R, rest_lengths):
        c, R, rest = np.asarray(c, dtype=np.float64), np.asarray(R, dtype=np.float64), np.asarray(rest_lengths, dtype=np.float64)
        if (c.shape != (2, 3) or R.shape != (2, 3, 3) or rest.shape != (4,)
                or not all(np.isfinite(x).all() for x in (c, R, rest)) or np.any(rest < 0)):
            raise ValueError('Need finite c(2,3), R(2,3,3), nonnegative rest_lengths(4)')
        if not np.allclose(R@R.swapaxes(1, 2), np.eye(3), atol=1e-10, rtol=0) or not np.allclose(np.linalg.det(R), 1., atol=1e-10, rtol=0):
            raise ValueError('R must contain proper orthonormal rotations')
        arms = np.array([self.front_offsets@R[0].T, self.back_offsets@R[1].T])
        points = c[:, None, :]+arms
        vectors = points[1]-points[0]
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError('Zero-length cable direction is undefined')
        extension = np.maximum(lengths-rest, 0.)
        tensions = self.k*extension
        gradient, hessian = np.zeros(12), np.zeros((12, 12))
        for i, (length, tension) in enumerate(zip(lengths, tensions)):
            if extension[i] == 0:  # Inactive-side Hessian at the tension boundary.
                continue
            n = vectors[i]/length
            endpoint_gradient = tension*n
            endpoint_hessian = self.k*np.outer(n, n)+(tension/length)*(np.eye(3)-np.outer(n, n))
            jac = np.column_stack((-np.eye(3), skew(arms[0, i]), np.eye(3), -skew(arms[1, i])))
            gradient += jac.T@endpoint_gradient
            hessian += jac.T@endpoint_hessian@jac
            for body, sign in ((0, -1), (1, 1)):
                r, g = arms[body, i], sign*endpoint_gradient
                rotation_hessian = .5*(np.outer(g, r)+np.outer(r, g))-np.dot(g, r)*np.eye(3)
                slc = slice(body*6+3, body*6+6)
                hessian[slc, slc] += rotation_hessian
        return dict(energy_j=float(.5*self.k*np.dot(extension, extension)), gradient=gradient, hessian=hessian,
                    tensions_n=tensions, lengths_m=lengths, routes_m=points.transpose(1, 0, 2))


def self_check():
    p = json.loads((HERE/'output/parameters.snapshot.json').read_text(encoding='utf-8'))
    bodies = read_body_inertias(p)
    load = FullBodyCables(p)
    c, R = bodies['reference_centers_world_m'].copy(), bodies['reference_rotations'].copy()
    R = Rotation.from_rotvec([[.07, -.03, .04], [-.04, .06, -.08]]).as_matrix()@R
    c[1] += [.005, .002, -.001]
    rest = load.evaluate(c, R, np.ones(4))['lengths_m']-[.002, .003, .004, .005]
    base = load.evaluate(c, R, rest)
    scale = np.tile([.05, .05, .05, 1., 1., 1.], 2)
    def chart(y):
        x = (y*scale).reshape(2, 6)
        return load.evaluate(c+x[:, :3], Rotation.from_rotvec(x[:, 3:]).as_matrix()@R, rest)['energy_j']
    step, zero = 2e-4, np.zeros(12)
    fdg, fdh = np.zeros(12), np.zeros((12, 12))
    for i in range(12):
        ei = np.eye(12)[i]*step
        fdg[i] = (chart(ei)-chart(-ei))/(2*step)
        fdh[i, i] = (chart(ei)-2*chart(zero)+chart(-ei))/step**2
        for j in range(i):
            ej = np.eye(12)[j]*step
            fdh[i, j] = fdh[j, i] = (chart(ei+ej)-chart(ei-ej)-chart(-ei+ej)+chart(-ei-ej))/(4*step**2)
    g, h = base['gradient']*scale, base['hessian']*scale[:, None]*scale
    ge = float(np.max(abs(fdg-g))/max(np.max(abs(g)), 1e-12))
    he = float(np.max(abs(fdh-h))/max(np.max(abs(h)), 1e-12))
    assert ge < 2e-6 and he < 2e-6, (ge, he)
    np.testing.assert_allclose(h, h.T, atol=1e-12, rtol=0)
    Q = Rotation.from_rotvec([.3, -.2, .4]).as_matrix()
    transformed = load.evaluate(c@Q.T+[.25, -.12, .5], Q@R, rest)
    np.testing.assert_allclose(transformed['energy_j'], base['energy_j'], atol=1e-13, rtol=1e-12)
    P = np.kron(np.eye(4), Q)
    np.testing.assert_allclose(transformed['gradient'], P@base['gradient'], atol=1e-11, rtol=1e-12)
    np.testing.assert_allclose(transformed['hessian'], P@base['hessian']@P.T, atol=1e-10, rtol=1e-12)
    wrenches = -base['gradient'].reshape(2, 6)
    force_balance = np.linalg.norm(wrenches[:, :3].sum(0))
    moment_balance = np.linalg.norm((wrenches[:, 3:]+np.cross(c, wrenches[:, :3])).sum(0))
    assert force_balance < 1e-12 and moment_balance < 1e-12
    slack = load.evaluate(c, R, base['lengths_m']+.001)
    assert slack['energy_j'] == 0 and not np.any(slack['gradient']) and not np.any(slack['hessian'])
    assert not np.any(slack['tensions_n'])
    com_load = FullBodyCables(p, bodies['com_local_m'])
    com_arms = np.einsum('bij,bj->bi', R, bodies['com_local_m'])
    about_com = com_load.evaluate(c+com_arms, R, rest)
    np.testing.assert_allclose(about_com['routes_m'], base['routes_m'], atol=1e-15, rtol=0)
    np.testing.assert_allclose(about_com['energy_j'], base['energy_j'], atol=1e-13, rtol=1e-12)
    com_wrenches = -about_com['gradient'].reshape(2, 6)
    np.testing.assert_allclose(com_wrenches[:, :3], wrenches[:, :3], atol=1e-11, rtol=1e-12)
    np.testing.assert_allclose(com_wrenches[:, 3:]+np.cross(com_arms, com_wrenches[:, :3]),
                               wrenches[:, 3:], atol=1e-12, rtol=1e-12)
    # Independent aggregate check: inertia translated to origin equals sum of parts.
    for mass, com, inertia, parts in zip(bodies['mass_kg'], bodies['com_local_m'], bodies['inertia_com_local_kgm2'], bodies['provenance']['components']):
        expected = np.zeros((3, 3))
        for part in parts:
            x = np.asarray(part['com_local_m'])
            expected += part['inertia_com_local_kgm2']+part['mass_kg']*(np.dot(x, x)*np.eye(3)-np.outer(x, x))
        np.testing.assert_allclose(inertia+mass*(np.dot(com, com)*np.eye(3)-np.outer(com, com)), expected, atol=1e-18, rtol=1e-12)
        assert np.linalg.eigvalsh(inertia)[0] > 0
    report = dict(status='passed', gradient_fd_relative_error=ge, hessian_fd_relative_error=he,
                  force_balance_n=float(force_balance), moment_balance_nm=float(moment_balance),
                  mass_kg=bodies['mass_kg'].tolist(), com_local_m=bodies['com_local_m'].tolist(),
                  inertia_com_local_kgm2=bodies['inertia_com_local_kgm2'].tolist(), provenance=bodies['provenance'])
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    self_check()

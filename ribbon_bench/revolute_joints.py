"""CAD revolute joints with five KKT constraints and configurable PD motors.

Rigid coordinates are COM translations and *world-left* rotation increments.
Each joint supplies three coincident-pivot constraints and two parallel-axis
constraints. The free signed angle follows the CAD joint axis, including a
negative CAD Z axis. Motor torque is equal and opposite on the two bodies.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from full_body_loads import skew


class JointPoseError(ValueError):
    """A Newton trial left the connected revolute-pose branch."""


def _rotations(value, count):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (count, 3, 3) or not np.isfinite(value).all()
            or not np.allclose(value @ value.swapaxes(1, 2), np.eye(3), atol=1e-9, rtol=0)
            or not np.allclose(np.linalg.det(value), 1., atol=1e-9, rtol=0)):
        raise ValueError('Need finite proper body rotations (bodies,3,3)')
    return value


class RevoluteJoints:
    def __init__(self, p, body_data, kp=1., kd=.03, torque_limit=.5, passive_damping=0.):
        self.nbody = len(body_data['mass_kg'])
        self.nphysical = 6*self.nbody
        self.joints = body_data.get('joints', [])
        self.njoints = len(self.joints)
        self.nconstraint = 5*self.njoints
        self.kp, self.kd, self.torque_limit = map(float, (kp, kd, torque_limit))
        self.passive_damping = float(passive_damping)
        if (not np.isfinite([self.kp, self.kd, self.torque_limit, self.passive_damping]).all()
                or self.kp < 0 or self.kd < 0 or self.torque_limit <= 0 or self.passive_damping < 0):
            raise ValueError('Need finite nonnegative motor gains and positive torque limit')
        self.limit_stiffness = 10.  # N m/rad; compliant stop, not a hard inequality.
        self.parent, self.child, self.pivot_parent, self.pivot_child = [], [], [], []
        self.axis_parent, self.axis_child, self.basis, self.child_reference, self.limits = [], [], [], [], []
        for joint in self.joints:
            parent, child = int(joint['parent_body']), int(joint['child_body'])
            if not 0 <= parent < self.nbody or not 0 <= child < self.nbody or parent == child:
                raise ValueError('Invalid revolute body indices')
            values = np.asarray([joint['pivot_parent_local_m'], joint['pivot_child_local_m'],
                                 joint['axis_parent_local'], joint['axis_child_local']], dtype=np.float64)
            if values.shape != (4, 3) or not np.isfinite(values).all():
                raise ValueError('Need finite pivot arms and local axes')
            pp, pc, ap, ac = values
            if min(np.linalg.norm(ap), np.linalg.norm(ac)) < 1e-12:
                raise ValueError('Revolute axes must be nonzero')
            ap, ac = ap/np.linalg.norm(ap), ac/np.linalg.norm(ac)
            reference = _rotations(np.asarray(joint['reference_rotation_parent_child'])[None], 1)[0]
            if not np.allclose(ap, reference @ ac, atol=1e-9, rtol=0):
                raise ValueError('CAD parent/child axes disagree in the zero pose')
            limits = np.asarray(joint['limits_rad'], dtype=np.float64)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Need ordered finite joint angle limits')
            seed = np.eye(3)[np.argmin(np.abs(ap))]
            u = seed-ap*np.dot(ap, seed)
            u /= np.linalg.norm(u)
            self.parent.append(parent); self.child.append(child)
            self.pivot_parent.append(pp); self.pivot_child.append(pc)
            self.axis_parent.append(ap); self.axis_child.append(ac)
            self.basis.append([u, np.cross(ap, u)])
            self.child_reference.append(reference.T @ u)
            self.limits.append(limits)
        self.parent, self.child = np.asarray(self.parent, dtype=int), np.asarray(self.child, dtype=int)
        self.pivot_parent, self.pivot_child = np.asarray(self.pivot_parent), np.asarray(self.pivot_child)
        self.axis_parent, self.axis_child = np.asarray(self.axis_parent), np.asarray(self.axis_child)
        self.basis, self.child_reference, self.limits = map(np.asarray, (self.basis, self.child_reference, self.limits))

    def provenance(self):
        return dict(joint_names=[j['name'] for j in self.joints],
                    constraints='Three coincident CAD-pivot coordinates plus two parallel CAD-axis coordinates per joint; Lagrange multipliers',
                    motor='Configurable torque-clipped PD on signed CAD-axis angle; equal/opposite body torques; gains not experimentally calibrated',
                    kp_nm_rad=self.kp, kd_nms_rad=self.kd, motor_torque_limit_nm=self.torque_limit,
                    passive_damping_nms_rad=self.passive_damping,
                    passive_damping='Independent joint damping, outside the actuator torque limit',
                    limits='Compliant angular stops outside CAD limits; hard unilateral stops are not implemented',
                    limit_stiffness_nm_rad=self.limit_stiffness,
                    tangent='Quasi-Newton: omit multiplier geometric Hessians and motor angle second derivatives; implicit velocity tangent approximated by 1/dt')

    def _angle(self, joint, R):
        parent, child = self.parent[joint], self.child[joint]
        a = R[parent] @ self.axis_parent[joint]
        u = R[parent] @ self.basis[joint, 0]
        v = R[child] @ self.child_reference[joint]
        x, y = float(np.dot(u, v)), float(np.dot(a, np.cross(u, v)))
        denom = x*x+y*y
        if denom < 1e-12:
            raise JointPoseError('Joint twist angle is undefined for this perpendicular-axis pose')
        # Exact derivative of atan2 under world-left rotation; parent is minus child.
        derivative = (x*(x*a-np.dot(a, v)*u)+y*np.cross(u, v))/denom
        return np.arctan2(y, x), derivative

    def evaluate(self, c, R, multipliers, targets, dt, omega_old, Rold=None):
        c = np.asarray(c, dtype=np.float64)
        R = _rotations(R, self.nbody)
        multipliers, targets, omega_old = map(lambda x: np.asarray(x, dtype=np.float64),
                                             (multipliers, targets, omega_old))
        if (c.shape != (self.nbody, 3) or multipliers.shape != (self.nconstraint,)
                or targets.shape != (self.njoints,) or omega_old.shape != c.shape
                or not all(np.isfinite(x).all() for x in (c, multipliers, targets, omega_old))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Invalid joint pose, multipliers, motor target, velocity or timestep')
        if self.njoints and (np.any(targets < self.limits[:, 0]) or np.any(targets > self.limits[:, 1])):
            raise ValueError('Joint motor target lies outside CAD angle limits')
        omega = omega_old if Rold is None else Rotation.from_matrix(R @ _rotations(Rold, self.nbody).swapaxes(1, 2)).as_rotvec()/dt
        g, J = np.zeros(self.nconstraint), np.zeros((self.nconstraint, self.nphysical))
        gradient, H = np.zeros(self.nphysical), np.zeros((self.nphysical, self.nphysical))
        angles, torques, passive_torques, stop_torques, excess, pivots, pivot_error, axis_error = [], [], [], [], [], [], [], []
        for joint, (parent, child) in enumerate(zip(self.parent, self.child)):
            pi, ci, row = 6*parent, 6*child, 5*joint
            rp, rc = R[parent] @ self.pivot_parent[joint], R[child] @ self.pivot_child[joint]
            pp, pc = c[parent]+rp, c[child]+rc
            g[row:row+3] = pc-pp
            J[row:row+3, pi:pi+3], J[row:row+3, ci:ci+3] = -np.eye(3), np.eye(3)
            J[row:row+3, pi+3:pi+6], J[row:row+3, ci+3:ci+6] = skew(rp), -skew(rc)
            a, ac = R[parent] @ self.axis_parent[joint], R[child] @ self.axis_child[joint]
            # The parallel branch is the CAD connected pose; anti-parallel axes are invalid.
            if np.dot(a, ac) <= 0:
                raise JointPoseError('Joint axes crossed the antiparallel constraint branch')
            for k, local in enumerate(self.basis[joint]):
                u = R[parent] @ local
                g[row+3+k] = np.dot(u, ac)
                d = np.cross(u, ac)
                J[row+3+k, pi+3:pi+6], J[row+3+k, ci+3:ci+6] = d, -d
            angle, angle_d = self._angle(joint, R)
            rate = np.dot(angle_d, omega[child]-omega[parent])
            requested = self.kp*(targets[joint]-angle)-self.kd*rate
            torque = float(np.clip(requested, -self.torque_limit, self.torque_limit))
            passive = -self.passive_damping*rate
            angle_row = np.zeros(self.nphysical)
            angle_row[pi+3:pi+6], angle_row[ci+3:ci+6] = -angle_d, angle_d
            limit_error = angle-np.clip(angle, *self.limits[joint])
            stop = -self.limit_stiffness*limit_error
            gradient -= (torque+passive+stop)*angle_row
            tangent = (self.kp+(self.kd/dt if Rold is not None else 0.)) if abs(requested) < self.torque_limit else 0.
            if Rold is not None:
                tangent += self.passive_damping/dt
            if limit_error:
                tangent += self.limit_stiffness
            # ponytail: a symmetric Gauss-Newton tangent; add geometric Hessians if large-angle convergence requires them.
            H += tangent*np.outer(angle_row, angle_row)
            angles.append(angle); torques.append(torque); passive_torques.append(passive); stop_torques.append(stop); excess.append(abs(limit_error))
            pivots.append((pp+pc)/2)
            pivot_error.append(np.linalg.norm(pc-pp)); axis_error.append(np.linalg.norm(np.cross(a, ac)))
        n = self.nphysical
        residual, hessian = np.zeros(n+self.nconstraint), np.zeros((n+self.nconstraint, n+self.nconstraint))
        residual[:n], residual[n:] = gradient+J.T @ multipliers, g
        hessian[:n, :n], hessian[:n, n:], hessian[n:, :n] = H, J.T, J
        return dict(residual=residual, hessian=hessian, constraint_values=g, constraint_jacobian=J,
                    angles_rad=np.asarray(angles), targets_rad=targets.copy(), torques_nm=np.asarray(torques),
                    passive_damping_torques_nm=np.asarray(passive_torques),
                    stop_torques_nm=np.asarray(stop_torques), limit_excess_rad=np.asarray(excess),
                    pivot_error_m=np.asarray(pivot_error), axis_error=np.asarray(axis_error),
                    joint_pivots_m=np.asarray(pivots).reshape(-1, 3))


def self_check():
    rng = np.random.default_rng(19)
    parent_R = Rotation.from_rotvec([.13, -.22, .17]).as_matrix()
    reference = Rotation.from_rotvec([-.05, .07, -.08]).as_matrix()
    a = np.array([0., 0., -1.])
    rp, rc = np.array([.03, -.01, .02]), np.array([-.02, .01, -.03])
    data = dict(mass_kg=[1., 2.], joints=[dict(name='cad_yaw', parent_body=0, child_body=1,
                 pivot_parent_local_m=rp, pivot_child_local_m=rc, axis_parent_local=a,
                 axis_child_local=reference.T @ a, reference_rotation_parent_child=reference,
                 limits_rad=[-1.57, 1.57])])
    model = RevoluteJoints({}, data, kp=1., kd=.03, torque_limit=.5)
    def pose(angle):
        R = np.array([parent_R, parent_R @ Rotation.from_rotvec(a*angle).as_matrix() @ reference])
        c = np.array([[.1, .2, .3], np.zeros(3)])
        c[1] = c[0]+R[0]@rp-R[1]@rc
        return c, R
    c, R = pose(.4)
    ev = model.evaluate(c, R, np.zeros(5), [.2], .02, np.zeros((2, 3)))
    assert np.max(np.abs(ev['constraint_values'])) < 1e-14
    assert abs(ev['angles_rad'][0]-.4) < 1e-14
    assert np.allclose(ev['residual'][3:6]+ev['residual'][9:12], 0., atol=1e-14)
    # A perturbed pose checks all 12 translation/rotation columns, not just yaw.
    c = c+rng.normal(0, .002, c.shape)
    R = Rotation.from_rotvec(rng.normal(0, .03, (2, 3))).as_matrix() @ R
    ev = model.evaluate(c, R, np.zeros(5), [.2], .02, np.zeros((2, 3)))
    eps, numeric_J, numeric_angle = 1e-7, np.zeros((5, 12)), np.zeros(12)
    for column in range(12):
        body, coord = divmod(column, 6)
        values, angles_fd = [], []
        for sign in (1., -1.):
            cn, Rn = c.copy(), R.copy()
            if coord < 3:
                cn[body, coord] += sign*eps
            else:
                increment = np.zeros(3); increment[coord-3] = sign*eps
                Rn[body] = Rotation.from_rotvec(increment).as_matrix() @ Rn[body]
            out = model.evaluate(cn, Rn, np.zeros(5), [.2], .02, np.zeros((2, 3)))
            values.append(out['constraint_values']); angles_fd.append(out['angles_rad'][0])
        numeric_J[:, column] = (values[0]-values[1])/(2*eps)
        numeric_angle[column] = (angles_fd[0]-angles_fd[1])/(2*eps)
    J_error = float(np.max(np.abs(numeric_J-ev['constraint_jacobian'])))
    assert J_error < 2e-8
    energy_gradient = (ev['angles_rad'][0]-.2)*numeric_angle
    assert np.allclose(energy_gradient, ev['residual'][:12], atol=2e-9, rtol=0)
    # Constraint reactions conserve net force/moment on the connected manifold.
    # Away from it, a world-coordinate pivot gap contributes gap cross multiplier.
    c, R = pose(.4)
    connected = model.evaluate(c, R, np.zeros(5), [.2], .02, np.zeros((2, 3)))
    reactions = connected['constraint_jacobian'].T @ rng.normal(size=5)
    forces, moments = reactions.reshape(2, 6)[:, :3], reactions.reshape(2, 6)[:, 3:]
    assert np.linalg.norm(forces.sum(axis=0)) < 1e-14
    assert np.linalg.norm((moments+np.cross(c, forces)).sum(axis=0)) < 1e-14
    c, R = pose(.4); _, Rold = pose(.37)
    ev = model.evaluate(c, R, np.zeros(5), [.2], .02, np.zeros((2, 3)), Rold=Rold)
    assert abs(ev['torques_nm'][0]-(-.2-.03*1.5)) < 1e-12
    saturated = model.evaluate(*pose(1.), np.zeros(5), [-1.], .02, np.zeros((2, 3)))
    assert saturated['torques_nm'][0] == -.5
    assert np.max(np.abs(saturated['hessian'][:12, :12])) == 0.
    damped = RevoluteJoints({}, data, kp=1., kd=0., torque_limit=.5, passive_damping=.07)
    saturated = damped.evaluate(*pose(1.), np.zeros(5), [-1.], .02, np.zeros((2, 3)), Rold=pose(.97)[1])
    assert saturated['torques_nm'][0] == -.5
    np.testing.assert_allclose(saturated['passive_damping_torques_nm'], [-.07*1.5], atol=1e-13)
    np.testing.assert_allclose(np.trace(saturated['hessian'][:12, :12]), 2*.07/.02, atol=1e-12)
    stop = model.evaluate(*pose(1.58), np.zeros(5), [0.], .02, np.zeros((2, 3)))
    assert abs(stop['limit_excess_rad'][0]-.01) < 1e-12
    return dict(constraint_J_max_abs_error=J_error, signed_CAD_yaw=True,
                force_moment_balance=True, implicit_PD=True, torque_clipping=True,
                passive_damping_outside_motor_limit=True, compliant_limit=True)


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), indent=2))

"""Minimal full worm-robot dynamics: Sano ribbons, cables, rigid plates, ground.

This is the first integrated verification core.  It uses local banded Newton
solves for each ribbon and a 12-DOF Schur solve for the coupled plate poses;
the steel/contact Jacobian is assembled analytically, while contact geometric
second derivatives are deliberately omitted (a documented quasi-Newton term).
Run ``python full_robot.py --self-check`` before a simulation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation

from full_body_loads import FullBodyCables, read_body_inertias, skew
from ground_contact import evaluate_contact
from ribbon_contact_geometry import surface_geometry
from run import COMMIT, HERE, make_robot, read_project, refresh, strip_geometry


def _mass(robot, p):
    """Diagonal translational/edge-angle mass used by the integrated model."""
    n = len(robot.node_dof_indices)
    out = robot.mass_matrix.copy()
    w, h, rho = p['strip_width_m'], p['strip_thickness_m'], p['steel_density_kg_m3']
    polar_area = (w*h**3 + h*w**3)/12.
    out[3*n:] = rho*robot.ref_len*polar_area
    return out


def _plate_contact(plate_radius, thickness, angles=24):
    """Rim points on the two plate faces; x is plate thickness direction."""
    theta = np.linspace(0., 2*np.pi, angles, endpoint=False)
    yz = np.column_stack((plate_radius*np.cos(theta), plate_radius*np.sin(theta)))
    local = np.vstack((np.column_stack((np.full(angles, -thickness/2), yz)),
                       np.column_stack((np.full(angles, thickness/2), yz))))
    return local, np.full(len(local), 1./len(local))


def _plate_points(c, R, local, com_local):
    arm = local-com_local[None, :]
    points = c[None, :]+arm@R.T
    jac = np.empty((len(local), 3, 6))
    jac[:, :, :3] = np.broadcast_to(np.eye(3), (len(local), 3, 3))
    jac[:, :, 3:] = -np.array([skew(x) for x in (arm@R.T)])
    return points, jac


class FullRobot:
    """Coupled implicit-Euler simulator with a deliberately small public API."""

    def __init__(self, parameters, delta, *, nodes=9, dt=.002, mu=.4,
                 normal_stiffness=1e8, tangential_stiffness=2e7,
                 damping=0., ground_height=0., clearance=2e-5,
                 cad_path=None, contact_samples=24, solve_backend='cpu', cuda_device=0):
        self.p, self.delta, self.nodes, self.dt = parameters, np.asarray(delta), nodes, dt
        self.mu, self.kn, self.kt, self.damping = float(mu), float(normal_stiffness), float(tangential_stiffness), float(damping)
        if solve_backend not in ('cpu', 'cuda'):
            raise ValueError('solve_backend must be cpu or cuda')
        self.solve_backend = solve_backend
        self.solve_seconds = 0.
        self.cuda_device = int(cuda_device)
        self.torch = None
        if solve_backend == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA solve backend requested but CUDA is unavailable')
            self.torch = torch
            self.device = torch.device('cuda', self.cuda_device)
        self.ground_height = float(ground_height)
        if nodes < 7 or dt <= 0 or mu < 0 or normal_stiffness <= 0 or tangential_stiffness <= 0:
            raise ValueError('Invalid nodes, dt, friction or contact stiffness')
        self.body = read_body_inertias(parameters, cad_path)
        self.cable = FullBodyCables(parameters, self.body['com_local_m'])
        self.robots, self.steppers, self.rest, self.widths = [], [], [], []
        # Lift the CAD reference so the circular plate rims begin just above Z=0.
        centers = self.body['reference_centers_world_m'].copy()
        lift = self.p['plate_stop_radius_m'] + self.p['plate_stop_thickness_m']/2 + clearance - np.min(centers[:, 2])
        self.shift = np.array([0., 0., lift])
        for strip in range(self.p['strip_count']):
            rest, width = strip_geometry(parameters, delta, strip, nodes)
            rest = rest + self.shift
            robot, stepper = make_robot(parameters, rest, width, 'sano')
            self.robots.append(robot); self.steppers.append(stepper)
            self.rest.append(rest); self.widths.append(width)
        self.rest, self.widths = np.asarray(self.rest), np.asarray(self.widths)
        # Some CAD ribbons hang below the plate rim.  Lift the whole assembly
        # once more so the lowest *actual* surface sample starts above z=0;
        # this avoids injecting a 4 cm artificial impact at t=0.
        min_surface = min(float(np.min(surface_geometry(r, r.state.q, parameters['strip_width_m'],
                                                        parameters['strip_thickness_m'])[0][:, 2]))
                          for r in self.robots)
        extra = max(0., clearance-min_surface)
        if extra:
            self.shift[2] += extra
            self.rest[:, :, 2] += extra
            for i, robot in enumerate(self.robots):
                qq = robot.state.q.copy()
                xyz = qq[:3*nodes].reshape(nodes, 3); xyz[:, 2] += extra
                self.robots[i] = refresh(robot, qq)
        self.nq = self.robots[0].n_dof
        self.free = self.robots[0].state.free_dof.copy()
        self.nfree = len(self.free)
        # Each of the eight steel strips has its own internal coordinates.
        self.nsteel = self.p['strip_count'] * self.nfree
        # Reorder each local block by node (x, y, z, twist).  The Sano
        # nearest-neighbour stencil then has a fixed half-bandwidth of ten;
        # the solver below keeps the original coordinate order at its API.
        keys = []
        for local, global_dof in enumerate(self.free):
            if global_dof < 3*nodes:
                keys.append((int(global_dof)//3, int(global_dof)%3, local))
            else:
                keys.append((int(global_dof)-3*nodes, 3, local))
        self.local_perm = np.asarray([local for _, _, local in sorted(keys)], dtype=int)
        self.local_bandwidth = min(10, self.nfree-1)
        self.mass_steel = _mass(self.robots[0], parameters)
        self.fixed = np.setdiff1d(np.arange(self.nq), self.free)
        self.body_mass = self.body['mass_kg']
        self.body_inertia_local = self.body['inertia_com_local_kgm2']
        self.body_plate_centers = centers + self.shift
        self.body_com = self.body_plate_centers + self.body['com_local_m']
        self.body_R = self.body['reference_rotations'].copy()
        self.body_v = np.zeros((2, 3)); self.body_omega = np.zeros((2, 3))
        self.q = np.array([robot.state.q for robot in self.robots])
        self.u = np.zeros_like(self.q)
        self.plate_local, self.plate_weights = _plate_contact(self.p['plate_stop_radius_m'], self.p['plate_stop_thickness_m'], contact_samples)
        self.steel_history = [None]*8
        self.plate_history = [None]*2
        self.steel_oldpoints = [self._steel_points(i, self.q[i]) for i in range(8)]
        self.plate_oldpoints = [self._plate_points(i)[0] for i in range(2)]
        c0, R0 = self.body_com.copy(), self.body_R.copy()
        self.base_cable_lengths = self.cable.evaluate(c0, R0, np.ones(4))['lengths_m']
        self.last = None

    def _steel_points(self, strip, q):
        points, _ = surface_geometry(self.robots[strip], q, self.p['strip_width_m'], self.p['strip_thickness_m'])
        return points

    def _plate_points(self, body):
        return _plate_points(self.body_com[body], self.body_R[body], self.plate_local,
                             self.body['com_local_m'][body])

    def _decode(self, z, qold, cold, Rold):
        """Map internal/free coordinates and plate increments to all ribbon q."""
        q = np.array(qold, copy=True)
        for strip in range(8):
            base = strip*self.nfree
            q[strip, self.free] = qold[strip, self.free] + z[base:base+self.nfree]
        body_delta = z[self.nsteel:].reshape(2, 6)
        c = cold + body_delta[:, :3]
        R = Rotation.from_rotvec(body_delta[:, 3:]).as_matrix() @ Rold
        B = np.zeros((8, self.nq, 12))
        for strip in range(8):
            for body, rows, edge in ((0, (0, 1), 0), (1, (self.nodes-2, self.nodes-1), self.nodes-2)):
                arm = self.rest[strip, list(rows)[0]] - self.body_plate_centers[body] - self.body['com_local_m'][body]
                arm2 = self.rest[strip, list(rows)[1]] - self.body_plate_centers[body] - self.body['com_local_m'][body]
                q[strip, 3*np.array(list(rows))[:, None]+np.arange(3)] = c[body] + np.array([R[body]@arm, R[body]@arm2])
                world_arm = np.array([R[body]@arm, R[body]@arm2])
                rows_xyz = 3*np.array(list(rows))[:, None]+np.arange(3)
                for endpoint, rowset in enumerate(rows_xyz):
                    for coordinate, row in enumerate(rowset):
                        B[strip, row, body*6+coordinate] = 1.
                        B[strip, row, body*6+3:body*6+6] = -skew(world_arm[endpoint])[coordinate]
                # Clamp end twist is the rigid-body rotation about the current tangent.
                tmp = q[strip].copy()
                a1, a2 = self.robots[strip].compute_time_parallel(self.robots[strip].state.a1,
                                                                    self.robots[strip].state.q, tmp)
                tangent = tmp[3*(edge+1):3*(edge+2)]-tmp[3*edge:3*(edge+1)]
                tangent /= np.linalg.norm(tangent)
                desired_width = R[body]@self.widths[strip, edge]
                normal = np.cross(desired_width, tangent)
                normal /= np.linalg.norm(normal)
                q[strip, 3*self.nodes+edge] = math.atan2(normal@a2[edge], normal@a1[edge])
                B[strip, 3*self.nodes+edge, body*6+3:body*6+6] = tangent
        return q, c, R, B

    def _elastic(self, strip, q):
        stepper, oldrobot = self.steppers[strip], self.robots[strip]
        with contextlib.redirect_stdout(io.StringIO()):
            stepper._compute_forces_and_jacobian(oldrobot, q, np.zeros(self.nq))
        grad, hess = stepper._forces.copy(), stepper._jacobian.copy()
        trial = refresh(oldrobot, q)
        energy = float(stepper.compute_total_elastic_energy(trial.state))
        return grad, hess, energy, trial

    def _evaluate(self, z, cable_rest, qold, uold, cold, Rold, commit=False, dense=False):
        q, c, R, B = self._decode(z, qold, cold, Rold)
        total_res = np.zeros(self.nsteel+12)
        strip_hessians, strip_upper, strip_lower = [], [], []
        body_H = np.zeros((12, 12)); body_r = np.zeros(12)
        steel_energy = 0.; max_pen = 0.; contact_force = 0.; statuses = []
        for strip in range(8):
            grad, hess, energy, trial = self._elastic(strip, q[strip])
            points, J = surface_geometry(self.robots[strip], q[strip], self.p['strip_width_m'], self.p['strip_thickness_m'])
            weights = np.repeat(self.robots[strip].ref_len/12., 12)
            contact = evaluate_contact(points, self.steel_oldpoints[strip], self.steel_history[strip], dt=self.dt,
                                       ground_height=self.ground_height, normal_stiffness=self.kn, tangential_stiffness=self.kt,
                                       weights=weights, mu=self.mu)
            Aq_strip = np.zeros((self.nq, self.nfree+12)); Aq_strip[self.free, :self.nfree] = np.eye(self.nfree); Aq_strip[self.fixed, self.nfree:] = B[strip, self.fixed]
            rq = grad + self.mass_steel*(q[strip]-qold[strip]-self.dt*uold[strip])/self.dt**2 - np.einsum('pij,pi->j', J, contact['force'])
            hq = hess + np.diag(self.mass_steel/self.dt**2)
            Jq = np.einsum('pij,jk->pik', J, Aq_strip)
            hcontact = np.einsum('pai,pab,pbj->ij', Jq, contact['jacobian'], Jq)
            local_res = Aq_strip.T@rq
            local_H = Aq_strip.T@hq@Aq_strip-hcontact
            lo = strip*self.nfree; hi = lo+self.nfree
            total_res[lo:hi] += local_res[:self.nfree]
            body_r += local_res[self.nfree:]
            body_H += local_H[self.nfree:, self.nfree:]
            strip_hessians.append(local_H[:self.nfree, :self.nfree])
            strip_upper.append(local_H[:self.nfree, self.nfree:])
            strip_lower.append(local_H[self.nfree:, :self.nfree])
            max_pen = max(max_pen, float(max(0., -np.min(contact['gap_m']))))
            contact_force += float(np.sum(contact['normal_force_n']))
            statuses.extend(contact['status'].tolist())
            if commit:
                if not hasattr(self, '_candidate_steel') or not isinstance(self._candidate_steel, list): self._candidate_steel = []
                self._candidate_steel.append(contact)
        cable = self.cable.evaluate(c, R, cable_rest)
        body_r += cable['gradient']
        for body in range(2):
            I = R[body]@self.body_inertia_local[body]@R[body].T
            dphi = z[self.nsteel+6*body+3:self.nsteel+6*body+6]
            body_r[6*body:6*body+3] += self.body_mass[body]*(c[body]-cold[body]-self.dt*self.body_v[body])/self.dt**2
            body_r[6*body+3:6*body+6] += I@(dphi/self.dt**2-self.body_omega[body]/self.dt) + np.cross(self.body_omega[body], I@self.body_omega[body])
            body_r[6*body+2] += self.body_mass[body]*9.81
            body_H[6*body:6*body+3, 6*body:6*body+3] += self.body_mass[body]/self.dt**2*np.eye(3)
            body_H[6*body+3:6*body+6, 6*body+3:6*body+6] += I/self.dt**2
            p, Jb = self._plate_points_at(body, c[body], R[body])
            bc = evaluate_contact(p, self.plate_oldpoints[body], self.plate_history[body], dt=self.dt,
                                  ground_height=self.ground_height, normal_stiffness=self.kn, tangential_stiffness=self.kt,
                                  weights=self.plate_weights, mu=self.mu)
            body_r[6*body:6*body+6] -= np.einsum('pij,pi->j', Jb, bc['force'])
            body_H[6*body:6*body+6, 6*body:6*body+6] -= np.einsum('pai,pab,pbj->ij', Jb, bc['jacobian'], Jb)
            max_pen = max(max_pen, float(max(0., -np.min(bc['gap_m'])))); contact_force += float(np.sum(bc['normal_force_n'])); statuses.extend(bc['status'].tolist())
            if commit: self._candidate_plate = getattr(self, '_candidate_plate', []); self._candidate_plate.append(bc)
        total_res[self.nsteel:] += body_r
        if not np.isfinite(total_res).all() or not all(np.isfinite(x).all() for x in strip_hessians+strip_upper+strip_lower+[body_H]):
            raise FloatingPointError('non-finite full-robot residual')
        total_H = None
        if dense:
            total_H = np.zeros((self.nsteel+12, self.nsteel+12))
            for strip in range(8):
                lo = strip*self.nfree; hi = lo+self.nfree
                total_H[lo:hi, lo:hi] = strip_hessians[strip]
                total_H[lo:hi, self.nsteel:] = strip_upper[strip]
                total_H[self.nsteel:, lo:hi] = strip_lower[strip]
            total_H[self.nsteel:, self.nsteel:] = body_H
        info = dict(residual=total_res, hessian=total_H, strip_hessians=strip_hessians,
                    strip_upper=strip_upper, strip_lower=strip_lower, body_hessian=body_H,
                    energy_j=steel_energy+cable['energy_j'], steel_energy_j=steel_energy,
                    cable_energy_j=cable['energy_j'], tensions_n=cable['tensions_n'], cable_lengths_m=cable['lengths_m'],
                    max_penetration_m=max_pen, contact_normal_force_n=contact_force, statuses=statuses, q=q, c=c, R=R, B=B)
        return info

    def _plate_points_at(self, body, c, R):
        return _plate_points(c, R, self.plate_local, self.body['com_local_m'][body])

    def _solve_schur(self, evaluation):
        """Solve the block-diagonal strip / 12-DOF plate Newton system.

        The eight strip blocks have no direct cross-coupling; all cross-strip
        coupling passes through the rigid plates.  Solving the local blocks
        first reduces the dense solve from ``(8*nfree+12)^3`` to eight local
        solves plus one 12-by-12 Schur solve.
        """
        if self.solve_backend == 'cuda':
            return self._solve_schur_cuda(evaluation)
        body = self.nsteel
        residual = evaluation['residual']
        strip_hessians = evaluation['strip_hessians']
        strip_upper = evaluation['strip_upper']
        strip_lower = evaluation['strip_lower']
        body_hessian = evaluation['body_hessian']
        local_solutions = []
        coupling = np.zeros((12, 12))
        body_rhs = -residual[body:].copy()

        def solve_dense(A, b):
            try:
                out = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                out = np.linalg.lstsq(A + 1e-8*np.eye(len(A)), b, rcond=None)[0]
            if not np.isfinite(out).all():
                raise np.linalg.LinAlgError('non-finite local Schur solve')
            return out

        for strip in range(8):
            lo = strip*self.nfree
            hi = lo+self.nfree
            A = strip_hessians[strip]
            C = strip_upper[strip]
            L = strip_lower[strip]
            p = self.local_perm
            Aperm = A[np.ix_(p, p)]
            n = len(p); bw = self.local_bandwidth
            ab = np.zeros((2*bw+1, n), dtype=A.dtype)
            for col in range(n):
                start, stop = max(0, col-bw), min(n, col+bw+1)
                ab[bw+start-col:bw+stop-col, col] = Aperm[start:stop, col]
            rhs = np.column_stack((-residual[lo:hi][p], C[p, :]))
            try:
                solved = solve_banded((bw, bw), ab, rhs, check_finite=False)
            except (np.linalg.LinAlgError, ValueError):
                solved = np.linalg.lstsq(Aperm + 1e-8*np.eye(n), rhs, rcond=None)[0]
            if not np.isfinite(solved).all():
                raise np.linalg.LinAlgError('non-finite local banded Schur solve')
            y0p, Xp = solved[:, 0], solved[:, 1:]
            local_solutions.append((y0p, Xp))
            Lp = L[:, p]
            coupling += Lp@Xp
            body_rhs -= Lp@y0p
        schur = body_hessian-coupling
        try:
            db = solve_dense(schur, body_rhs)
        except np.linalg.LinAlgError:
            # Keep a diagnostic fallback for nonsmooth contact states; the
            # normal path remains the local-block Schur solve.
            hessian = np.zeros((self.nsteel+12, self.nsteel+12))
            for strip in range(8):
                lo = strip*self.nfree; hi = lo+self.nfree
                hessian[lo:hi, lo:hi] = strip_hessians[strip]
                hessian[lo:hi, body:] = strip_upper[strip]
                hessian[body:, lo:hi] = strip_lower[strip]
            hessian[body:, body:] = body_hessian
            return np.linalg.lstsq(hessian + 1e-8*np.eye(len(hessian)), -residual, rcond=None)[0]
        dz_steel = np.zeros(self.nsteel)
        for strip in range(8):
            lo = strip*self.nfree
            hi = lo+self.nfree
            p = self.local_perm
            y0p, Xp = local_solutions[strip]
            local = y0p - Xp@db
            block = dz_steel[lo:hi]
            block[p] = local
            dz_steel[lo:hi] = block
        return np.r_[dz_steel, db]

    def _solve_schur_cuda(self, evaluation):
        """CUDA FP64 local-block solve; geometry and contact remain on CPU."""
        started = time.perf_counter()
        torch = self.torch
        body = self.nsteel
        residual = evaluation['residual']
        A = torch.as_tensor(np.stack(evaluation['strip_hessians']), dtype=torch.float64, device=self.device)
        C = torch.as_tensor(np.stack(evaluation['strip_upper']), dtype=torch.float64, device=self.device)
        L = torch.as_tensor(np.stack(evaluation['strip_lower']), dtype=torch.float64, device=self.device)
        rs = torch.as_tensor(residual[:self.nsteel].reshape(8, self.nfree), dtype=torch.float64, device=self.device)
        rb = torch.as_tensor(residual[body:], dtype=torch.float64, device=self.device)
        y0 = torch.linalg.solve(A, -rs[..., None])[..., 0]
        X = torch.linalg.solve(A, C)
        schur = torch.as_tensor(evaluation['body_hessian'], dtype=torch.float64, device=self.device) - torch.sum(L@X, dim=0)
        rhs = -rb - torch.sum(torch.matmul(L, y0[..., None])[..., 0], dim=0)
        db = torch.linalg.solve(schur, rhs)
        dz = torch.cat(((y0-torch.matmul(X, db[..., None])[..., 0]).reshape(-1), db)).cpu().numpy()
        torch.cuda.synchronize(self.device)
        self.solve_seconds += time.perf_counter()-started
        if not np.isfinite(dz).all():
            raise FloatingPointError('non-finite CUDA Schur increment')
        return dz

    def step(self, cable_rest):
        qold, uold, cold, Rold = self.q.copy(), self.u.copy(), self.body_com.copy(), self.body_R.copy()
        z = np.zeros(self.nsteel+12)
        # A full velocity predictor is useful for the internal coordinates.
        z[:self.nsteel] = self.dt*uold[:, self.free].reshape(-1)
        z[self.nsteel:self.nsteel+3] = self.dt*self.body_v[0]
        z[self.nsteel+6:self.nsteel+9] = self.dt*self.body_v[1]
        z[self.nsteel+3:self.nsteel+6] = self.dt*self.body_omega[0]
        z[self.nsteel+9:self.nsteel+12] = self.dt*self.body_omega[1]
        last = None
        for iteration in range(30):
            ev = self._evaluate(z, cable_rest, qold, uold, cold, Rold)
            scaled = max(np.max(np.abs(ev['residual'][:self.nsteel]))/1e-5,
                         np.max(np.abs(ev['residual'][self.nsteel:]))/1e-5)
            last = ev
            if scaled <= 1.:
                self._evaluate(z, cable_rest, qold, uold, cold, Rold, commit=True)
                self.q, self.u = ev['q'], (ev['q']-qold)/self.dt
                self.body_com, self.body_R = ev['c'], ev['R']
                self.body_v = (self.body_com-cold)/self.dt
                self.body_omega = Rotation.from_matrix(self.body_R@Rold.swapaxes(1,2)).as_rotvec()/self.dt
                self.robots = [refresh(r, self.q[i]).update(u=self.u[i], a=np.zeros(self.nq)) for i, r in enumerate(self.robots)]
                self.steel_history = [x['history'] for x in self._candidate_steel[-8:]]
                self.plate_history = [x['history'] for x in self._candidate_plate[-2:]]
                self.steel_oldpoints = [self._steel_points(i, self.q[i]) for i in range(8)]
                self.plate_oldpoints = [self._plate_points(i)[0] for i in range(2)]
                info = {k: v for k, v in ev.items() if k not in ('q','c','R','B','hessian','residual','statuses',
                                                                  'strip_hessians','strip_upper','strip_lower','body_hessian')}
                info.update(iterations=iteration, residual_max=float(np.max(np.abs(ev['residual']))),
                            body_com=self.body_com.tolist(), body_R=self.body_R.tolist(),
                            status_counts={name: ev['statuses'].count(name) for name in sorted(set(ev['statuses']))})
                self.last = info
                return info
            dz = self._solve_schur(ev)
            if not np.isfinite(dz).all():
                raise FloatingPointError('non-finite Newton increment')
            # Cap both strip displacement and rigid-body translation/rotation in one
            # line-search scale; contact stiffness can otherwise produce a large
            # plate jump before the omitted geometric contact Hessian is corrected.
            steel_step = np.max(np.abs(dz[:self.nsteel]))
            body_step = max(np.max(np.abs(dz[self.nsteel:self.nsteel+3])),
                            np.max(np.abs(dz[self.nsteel+6:self.nsteel+9])),
                            .1*max(np.max(np.abs(dz[self.nsteel+3:self.nsteel+6])),
                                    np.max(np.abs(dz[self.nsteel+9:self.nsteel+12]))))
            fraction = min(1., .002/max(steel_step, body_step, 1e-12))
            accepted = False
            for _ in range(12):
                trial = self._evaluate(z+fraction*dz, cable_rest, qold, uold, cold, Rold)
                if np.linalg.norm(trial['residual']) < np.linalg.norm(ev['residual']):
                    z += fraction*dz; accepted = True; break
                fraction *= .5
            if not accepted:
                raise RuntimeError(f'Newton line search stalled at residual {scaled:g}')
        raise RuntimeError(f'Newton limit at residual {max(np.abs(last["residual"])):g}')


def _command(base, t, duration, amplitude):
    phase = 2*np.pi*t/max(duration, 1e-12)
    shape = .5*(1+np.sin(phase+np.arange(4)*np.pi/2))
    return np.maximum(base-amplitude*shape, 0.)


def self_check():
    p, delta, provenance = read_project(HERE/'output/parameters.snapshot.json')
    sim = FullRobot(p, delta, nodes=9, dt=.002, contact_samples=12)
    assert len(sim.q) == 8 and sim.nfree == len(sim.free) and sim.nsteel == 8*sim.nfree
    qold, uold, cold, Rold = sim.q.copy(), sim.u.copy(), sim.body_com.copy(), sim.body_R.copy()
    probe = sim._evaluate(np.zeros(sim.nsteel+12), sim.base_cable_lengths+.01, qold, uold, cold, Rold, dense=True)
    dz_schur = sim._solve_schur(probe)
    dz_dense = np.linalg.solve(probe['hessian'], -probe['residual'])
    assert np.linalg.norm(dz_schur-dz_dense)/max(np.linalg.norm(dz_dense), 1e-12) < 1e-8
    zero = sim.step(sim.base_cable_lengths+.01)
    assert np.isfinite(zero['residual_max']) and np.isfinite(sim.body_com).all()
    assert zero['max_penetration_m'] < 1e-3
    report = dict(status='passed', nodes=9, strips=8, residual_n=float(zero['residual_max']),
                  max_penetration_m=float(zero['max_penetration_m']), contact_force_n=float(zero['contact_normal_force_n']),
                  masses_kg=sim.body_mass.tolist(), scope='One implicit full-robot step; no gait or calibration')
    print(json.dumps(report, indent=2)); return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters', type=Path, default=HERE/'output/parameters.snapshot.json')
    ap.add_argument('--nodes', type=int, default=9); ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--dt', type=float, default=.002); ap.add_argument('--duration', type=float, default=.02)
    ap.add_argument('--command-mm', type=float, default=.5); ap.add_argument('--output', type=Path, default=HERE/'full_robot_demo')
    ap.add_argument('--solve-backend', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--cuda-device', type=int, default=0)
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    if args.self_check: self_check(); return
    if args.steps < 1 or args.nodes < 7 or args.command_mm < 0: ap.error('invalid simulation arguments')
    p, delta, provenance = read_project(args.parameters)
    sim = FullRobot(p, delta, nodes=args.nodes, dt=args.dt, cad_path=provenance['source_urdf'],
                    solve_backend=args.solve_backend, cuda_device=args.cuda_device)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); rows = []; q_frames = []; body_frames = []; rotation_frames = []
    for i in range(args.steps+1):
        t = i*args.duration/max(args.steps, 1)
        rest = _command(sim.base_cable_lengths, t, args.duration, args.command_mm/1000.)
        if i: info = sim.step(rest)
        else: info = dict(iterations=0, residual_max=0., max_penetration_m=0., contact_normal_force_n=0.,
                          cable_energy_j=0., tensions_n=np.zeros(4), body_com=sim.body_com.tolist())
        rows.append(dict(time_s=t, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in info.items()}))
        q_frames.append(sim.q.copy()); body_frames.append(sim.body_com.copy()); rotation_frames.append(sim.body_R.copy())
        print(json.dumps(rows[-1], allow_nan=False), flush=True)
    result = dict(status='completed', metadata=dict(**provenance, nodes=args.nodes, strips=8,
        steel_dof_per_strip=sim.nfree, steel_dof_total=sim.nsteel, newton_dof=sim.nsteel+12,
        independent_strip_coordinates=True, dt_s=args.dt,
        duration_s=args.duration, command_amplitude_mm=args.command_mm, model='Sano steel + CAD rigid bodies + ideal tension-only cables + sampled plane penalty/friction contact',
        contact='12 surface samples per steel edge plus 2x rim samples per plate; no continuous CCD; quasi-Newton contact geometric Hessian',
        solve_backend=args.solve_backend, cuda_device=args.cuda_device,
        local_solver='scipy.solve_banded' if args.solve_backend == 'cpu' else 'torch.linalg.solve',
        local_bandwidth=sim.local_bandwidth,
        solve_seconds=sim.solve_seconds, vendor_commit=COMMIT,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), frames=rows,
        wall_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out/'trajectory.npz', q=np.asarray(q_frames), body_com=np.asarray(body_frames), body_R=np.asarray(rotation_frames),
                        rest_nodes_m=sim.rest, strip_width_m=p['strip_width_m'], strip_thickness_m=p['strip_thickness_m'])
    print(json.dumps({'saved': str(out/'summary.json'), 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()

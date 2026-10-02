"""Minimal full worm-robot dynamics: Sano ribbons, cables, rigid plates, ground.

This is the first integrated verification core.  It uses local banded Newton
solves for each ribbon and a Schur solve for the coupled rigid-body poses;
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

from full_body_loads import FullBodyCables, read_body_inertias, read_chain_inertias, skew
from fast_sano import install_fast_sano
from cuda_sano import install_cuda_sano
from gpu_geometry_contact import contact_batch, surface_geometry_batch
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
                 cad_path=None, contact_samples=24, solve_backend='cpu', cuda_device=0, segments=1):
        self.p, self.delta, self.nodes, self.dt = parameters, np.asarray(delta), nodes, dt
        self.mu, self.kn, self.kt, self.damping = float(mu), float(normal_stiffness), float(tangential_stiffness), float(damping)
        if solve_backend not in ('cpu', 'cuda'):
            raise ValueError('solve_backend must be cpu or cuda')
        self.solve_backend = solve_backend
        self.solve_seconds = 0.
        self.cuda_device = int(cuda_device)
        self.torch = None
        self._cuda_buffers = None
        if solve_backend == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA solve backend requested but CUDA is unavailable')
            self.torch = torch
            self.device = torch.device('cuda', self.cuda_device)
        self.ground_height = float(ground_height)
        if nodes < 7 or dt <= 0 or mu < 0 or normal_stiffness <= 0 or tangential_stiffness <= 0 or segments not in range(1, 6):
            raise ValueError('Invalid nodes, dt, friction or contact stiffness')
        self.segments = int(segments)
        self.nstrips = self.segments*self.p['strip_count']
        self.nbodies = self.segments+1
        self.nbody_dof = 6*self.nbodies
        self.strip_bodies = np.repeat(np.column_stack((np.arange(segments), np.arange(1, segments+1))), self.p['strip_count'], axis=0)
        self.body = read_chain_inertias(parameters, segments, cad_path)
        self.robots, self.steppers, self.rest, self.widths = [], [], [], []
        # Lift the CAD reference so the circular plate rims begin just above Z=0.
        centers = self.body['reference_centers_world_m'].copy()
        lift = self.p['plate_stop_radius_m'] + self.p['plate_stop_thickness_m']/2 + clearance - np.min(centers[:, 2])
        self.shift = np.array([0., 0., lift])
        for strip in range(self.nstrips):
            segment = strip//self.p['strip_count']
            rest, width = strip_geometry(parameters, delta, strip % self.p['strip_count'], nodes)
            rest = rest + self.body['segment_translation_world_m'][segment] + self.shift
            robot, stepper = make_robot(parameters, rest, width, 'sano')
            install_fast_sano(stepper)
            if self.solve_backend == 'cuda':
                energy = stepper._TimeStepper__elastic_energies[0]
                install_cuda_sano(energy, self.device)
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
        # Steel interiors are independent; neighbouring segments share a rigid connector.
        self.nsteel = self.nstrips * self.nfree
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
        self._surface_dofs = []
        for robot in self.robots:
            edges = np.asarray(robot.edges[:len(robot.state.a1)], dtype=int)
            self._surface_dofs.append(np.column_stack((
                3*edges[:, 0, None]+np.arange(3),
                3*edges[:, 1, None]+np.arange(3),
                3*len(robot.node_dof_indices)+np.arange(len(edges)))))
        self.body_mass = self.body['mass_kg']
        self.body_inertia_local = self.body['inertia_com_local_kgm2']
        self.body_plate_centers = centers + self.shift
        self.body_R = self.body['reference_rotations'].copy()
        self.body_com = self.body_plate_centers + np.einsum('bij,bj->bi', self.body_R, self.body['com_local_m'])
        self.body_v = np.zeros((self.nbodies, 3)); self.body_omega = np.zeros((self.nbodies, 3))
        self.plate_bodies = self.body['plate_body_indices']
        self.nplates = len(self.plate_bodies)
        self.clamp_arms = np.zeros((self.nstrips, 2, 2, 3))
        self.clamp_widths = np.zeros((self.nstrips, 2, 3))
        for strip, bodies in enumerate(self.strip_bodies):
            for side, (body, rows, edge) in enumerate(zip(bodies, ((0, 1), (nodes-2, nodes-1)), (0, nodes-2))):
                self.clamp_arms[strip, side] = (self.rest[strip, list(rows)]-self.body_plate_centers[body])@self.body_R[body]-self.body['com_local_m'][body]
                self.clamp_widths[strip, side] = self.body_R[body].T@self.widths[strip, edge]
        self.cables = []
        for segment in range(self.segments):
            pair = np.array([segment, segment+1])
            cable = FullBodyCables(parameters)
            translation = self.body['segment_translation_world_m'][segment]+self.shift
            front_world = np.asarray(parameters['tendon_anchors_m']).reshape(4, 3)+translation
            back_world = np.asarray(parameters['tendon_guides_m']).reshape(4, -1, 3)[:, -1]+self.delta+translation
            cable.front_offsets = (front_world-self.body_com[pair[0]])@self.body_R[pair[0]]
            cable.back_offsets = (back_world-self.body_com[pair[1]])@self.body_R[pair[1]]
            self.cables.append(cable)
        self.cable = self.cables[0]  # Preserve the one-segment inspection API.
        self.q = np.array([robot.state.q for robot in self.robots])
        self.u = np.zeros_like(self.q)
        self.plate_local, self.plate_weights = _plate_contact(self.p['plate_stop_radius_m'], self.p['plate_stop_thickness_m'], contact_samples)
        self.steel_history = [None]*self.nstrips
        self.plate_history = [None]*self.nplates
        self.steel_oldpoints = [self._steel_points(i, self.q[i]) for i in range(self.nstrips)]
        self.plate_oldpoints = [self._plate_points(i)[0] for i in range(self.nplates)]
        c0, R0 = self.body_com.copy(), self.body_R.copy()
        self.base_cable_lengths = np.concatenate([cable.evaluate(c0[s:s+2], R0[s:s+2], np.ones(4))['lengths_m'] for s, cable in enumerate(self.cables)])
        self._last_cable_rest = self.base_cable_lengths.copy()
        self.last = None

    def _steel_points(self, strip, q):
        points, _ = surface_geometry(self.robots[strip], q, self.p['strip_width_m'], self.p['strip_thickness_m'])
        return points

    def _plate_points(self, plate):
        body = self.plate_bodies[plate]
        return self._plate_points_at(plate, self.body_com[body], self.body_R[body])

    def _decode(self, z, qold, cold, Rold):
        """Map internal/free coordinates and plate increments to all ribbon q."""
        q = np.array(qold, copy=True)
        for strip in range(self.nstrips):
            base = strip*self.nfree
            q[strip, self.free] = qold[strip, self.free] + z[base:base+self.nfree]
        body_delta = z[self.nsteel:].reshape(self.nbodies, 6)
        c = cold + body_delta[:, :3]
        R = Rotation.from_rotvec(body_delta[:, 3:]).as_matrix() @ Rold
        B = np.zeros((self.nstrips, self.nq, self.nbody_dof))
        for strip in range(self.nstrips):
            for side, (body, rows, edge) in enumerate(zip(self.strip_bodies[strip], ((0, 1), (self.nodes-2, self.nodes-1)), (0, self.nodes-2))):
                arm, arm2 = self.clamp_arms[strip, side]
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
                desired_width = R[body]@self.clamp_widths[strip, side]
                normal = np.cross(desired_width, tangent)
                normal /= np.linalg.norm(normal)
                angle = math.atan2(normal@a2[edge], normal@a1[edge])
                dof = 3*self.nodes+edge
                q[strip, dof] = angle+2*np.pi*round((qold[strip, dof]-angle)/(2*np.pi))
                B[strip, 3*self.nodes+edge, body*6+3:body*6+6] = tangent
        return q, c, R, B

    def _elastic(self, strip, q):
        stepper, oldrobot = self.steppers[strip], self.robots[strip]
        with contextlib.redirect_stdout(io.StringIO()):
            stepper._compute_forces_and_jacobian(oldrobot, q, np.zeros(self.nq))
        # Newton uses only force and tangent; avoid a duplicate material
        # forward and temporary robot refresh on every line-search trial.
        return stepper._forces.copy(), stepper._jacobian.copy()

    def _cuda_strip_contact(self, q, qold, B, commit=False):
        """Evaluate all ribbon surfaces/contact on CUDA and reduce locally."""
        torch = self.torch
        device = self.device
        qd = torch.as_tensor(np.ascontiguousarray(q), dtype=torch.float64, device=device)
        qold_d = torch.as_tensor(np.ascontiguousarray(qold), dtype=torch.float64, device=device)
        reference_a1 = torch.as_tensor(np.stack([r.state.a1 for r in self.robots]),
                                        dtype=torch.float64, device=device)
        points, local = surface_geometry_batch(
            self.robots[0], qd, self.p['strip_width_m'], self.p['strip_thickness_m'],
            device=device, reference_q=qold_d, reference_a1=reference_a1)
        oldpoints = torch.as_tensor(np.ascontiguousarray(np.stack(self.steel_oldpoints)),
                                    dtype=torch.float64, device=device)
        history_elastic = torch.zeros((self.nstrips, points.shape[1], 2), dtype=torch.float64, device=device)
        history_active = torch.zeros((self.nstrips, points.shape[1]), dtype=torch.bool, device=device)
        for strip, history in enumerate(self.steel_history):
            if history is not None:
                history_elastic[strip] = torch.as_tensor(history['elastic_slip_m'], dtype=torch.float64, device=device)
                history_active[strip] = torch.as_tensor(history['active'], dtype=torch.bool, device=device)
        weights = torch.as_tensor(np.repeat(self.robots[0].ref_len/12., 12),
                                  dtype=torch.float64, device=device)
        contact = contact_batch(points, oldpoints, history_elastic, history_active,
                                dt=self.dt, ground_height=self.ground_height,
                                normal_stiffness=self.kn, tangential_stiffness=self.kt,
                                weights=weights, mu=self.mu)
        Aq = np.zeros((self.nstrips, self.nq, self.nfree+self.nbody_dof))
        Aq[:, self.free, :self.nfree] = np.eye(self.nfree)[None, :, :]
        Aq[:, self.fixed, self.nfree:] = B[:, self.fixed, :]
        local_maps = torch.as_tensor(np.stack([
            Aq[strip][self._surface_dofs[strip]] for strip in range(self.nstrips)]),
            dtype=torch.float64, device=device)
        local_jac = local.reshape(self.nstrips, -1, 12, 3, 7)
        Jq = torch.einsum('sepik,sekm->sepim', local_jac, local_maps)
        Jq = Jq.reshape(self.nstrips, -1, 3, self.nfree+self.nbody_dof)
        contact_residual = torch.einsum('spij,spi->sj', Jq, contact['force'])
        contact_hessian = torch.einsum('spai,spab,spbj->sij', Jq, contact['jacobian'], Jq)
        residual_np = contact_residual.cpu().numpy()
        hessian_np = contact_hessian.cpu().numpy()
        gap_np = contact['gap_m'].cpu().numpy()
        normal_np = contact['normal_force_n'].cpu().numpy()
        active_np = contact['active'].cpu().numpy()
        slip_np = contact['slip'].cpu().numpy()
        if commit:
            force_np = contact['force'].cpu().numpy()
            jacobian_np = contact['jacobian'].cpu().numpy()
            elastic_np = contact['history_elastic_m'].cpu().numpy()
            plastic_np = contact['plastic_increment_m'].cpu().numpy()
            detached_np = contact['detached'].cpu().numpy()
        reduced = []
        for strip in range(self.nstrips):
            active = active_np[strip]
            slip = slip_np[strip]
            if self.mu == 0.:
                status = np.where(active, 'frictionless', 'separated')
            else:
                status = np.where(slip, 'slip', np.where(active, 'stick', 'separated'))
            item = dict(
                contact_residual=residual_np[strip],
                contact_hessian=hessian_np[strip],
                gap_m=gap_np[strip],
                normal_force_n=normal_np[strip],
                status=status)
            if commit:
                item.update(
                    force=force_np[strip], jacobian=jacobian_np[strip],
                    history=dict(elastic_slip_m=elastic_np[strip],
                                 active=active.copy()),
                    plastic_increment_m=plastic_np[strip], detached=detached_np[strip])
            reduced.append(item)
        return reduced

    def _cuda_plate_contact(self, plate, c, R, commit=False):
        """Run the plate's small sampled contact law on the same CUDA kernel."""
        torch = self.torch
        device = self.device
        # Keep the plate sample geometry and its rigid-body Jacobian on device;
        # only the six-by-six reduction is copied back for the CPU Schur solve.
        body = self.plate_bodies[plate]
        local = torch.as_tensor(self._plate_local(plate), dtype=torch.float64, device=device)
        com_local = torch.as_tensor(self.body['com_local_m'][body], dtype=torch.float64, device=device)
        c_d = torch.as_tensor(c, dtype=torch.float64, device=device)
        R_d = torch.as_tensor(R, dtype=torch.float64, device=device)
        arm = local - com_local[None, :]
        world_arm = arm @ R_d.T
        current = (c_d[None, :] + world_arm)[None]
        x, y, z = world_arm.unbind(-1)
        zero = torch.zeros_like(x)
        skew_world = torch.stack((torch.stack((zero, -z, y), dim=-1),
                                  torch.stack((z, zero, -x), dim=-1),
                                  torch.stack((-y, x, zero), dim=-1)), dim=-2)
        eye = torch.eye(3, dtype=torch.float64, device=device).expand(len(self.plate_local), -1, -1)
        Jd = torch.cat((eye, -skew_world), dim=-1)
        old = torch.as_tensor(self.plate_oldpoints[plate][None], dtype=torch.float64, device=device)
        history = self.plate_history[plate]
        elastic = None if history is None else torch.as_tensor(history['elastic_slip_m'][None], dtype=torch.float64, device=device)
        active = None if history is None else torch.as_tensor(history['active'][None], dtype=torch.bool, device=device)
        weights = torch.as_tensor(self.plate_weights, dtype=torch.float64, device=device)
        contact = contact_batch(current, old, elastic, active, dt=self.dt,
                                ground_height=self.ground_height, normal_stiffness=self.kn,
                                tangential_stiffness=self.kt, weights=weights, mu=self.mu)
        force = contact['force'][0]
        jacobian = contact['jacobian'][0]
        residual = torch.einsum('pij,pi->j', Jd, force).cpu().numpy()
        hessian = torch.einsum('pai,pab,pbj->ij', Jd, jacobian, Jd).cpu().numpy()
        gap = contact['gap_m'][0].cpu().numpy()
        normal = contact['normal_force_n'][0].cpu().numpy()
        active_np = contact['active'][0].cpu().numpy()
        slip_np = contact['slip'][0].cpu().numpy()
        status = (np.where(active_np, 'frictionless', 'separated') if self.mu == 0.
                  else np.where(slip_np, 'slip', np.where(active_np, 'stick', 'separated')))
        out = dict(contact_residual=residual, contact_hessian=hessian,
                   gap_m=gap, normal_force_n=normal, status=status)
        if commit:
            out.update(force=force.cpu().numpy(), jacobian=jacobian.cpu().numpy(),
                       history=dict(elastic_slip_m=contact['history_elastic_m'][0].cpu().numpy(),
                                    active=active_np.copy()),
                       plastic_increment_m=contact['plastic_increment_m'][0].cpu().numpy(),
                       detached=contact['detached'][0].cpu().numpy())
        return out

    def _evaluate(self, z, cable_rest, qold, uold, cold, Rold, commit=False, dense=False):
        q, c, R, B = self._decode(z, qold, cold, Rold)
        cuda_contact = self._cuda_strip_contact(q, qold, B, commit) if self.solve_backend == 'cuda' else None
        total_res = np.zeros(self.nsteel+self.nbody_dof)
        strip_hessians, strip_upper, strip_lower = [], [], []
        body_H = np.zeros((self.nbody_dof, self.nbody_dof)); body_r = np.zeros(self.nbody_dof)
        steel_energy = 0.; max_pen = 0.; contact_force = 0.; statuses = []
        for strip in range(self.nstrips):
            grad, hess = self._elastic(strip, q[strip])
            Aq_strip = np.zeros((self.nq, self.nfree+self.nbody_dof)); Aq_strip[self.free, :self.nfree] = np.eye(self.nfree); Aq_strip[self.fixed, self.nfree:] = B[strip, self.fixed]
            if cuda_contact is None:
                points, J = surface_geometry(self.robots[strip], q[strip], self.p['strip_width_m'], self.p['strip_thickness_m'])
                weights = np.repeat(self.robots[strip].ref_len/12., 12)
                contact = evaluate_contact(points, self.steel_oldpoints[strip], self.steel_history[strip], dt=self.dt,
                                           ground_height=self.ground_height, normal_stiffness=self.kn, tangential_stiffness=self.kt,
                                           weights=weights, mu=self.mu)
                contact_residual = np.einsum('pij,pi->j', J, contact['force'], optimize=True)
                Jq = np.einsum('pij,jk->pik', J, Aq_strip)
                contact_hessian = np.einsum('pai,pab,pbj->ij', Jq, contact['jacobian'], Jq, optimize=True)
            else:
                contact = cuda_contact[strip]
                contact_residual = contact['contact_residual']
                contact_hessian = contact['contact_hessian']
            base_rq = grad + self.mass_steel*(q[strip]-qold[strip]-self.dt*uold[strip])/self.dt**2
            base_rq[2:3*self.nodes:3] += self.mass_steel[2:3*self.nodes:3]*9.81
            hq = hess + np.diag(self.mass_steel/self.dt**2)
            local_res = (Aq_strip.T@base_rq-contact_residual if cuda_contact is not None
                         else Aq_strip.T@(base_rq-contact_residual))
            local_H = Aq_strip.T@hq@Aq_strip-contact_hessian
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
                steel_energy += float(self.steppers[strip].compute_total_elastic_energy(refresh(self.robots[strip], q[strip]).state))
                if not hasattr(self, '_candidate_steel') or not isinstance(self._candidate_steel, list): self._candidate_steel = []
                self._candidate_steel.append(contact)
        cable_energy = 0.; tensions = []; cable_lengths = []; cable_routes = []
        for segment, cable_model in enumerate(self.cables):
            cable = cable_model.evaluate(c[segment:segment+2], R[segment:segment+2], cable_rest[4*segment:4*segment+4])
            sl = slice(6*segment, 6*segment+12)
            body_r[sl] += cable['gradient']
            body_H[sl, sl] += cable['hessian']
            cable_energy += cable['energy_j']
            tensions.extend(cable['tensions_n']); cable_lengths.extend(cable['lengths_m']); cable_routes.append(cable['routes_m'])
        for body in range(self.nbodies):
            I = R[body]@self.body_inertia_local[body]@R[body].T
            dphi = z[self.nsteel+6*body+3:self.nsteel+6*body+6]
            body_r[6*body:6*body+3] += self.body_mass[body]*(c[body]-cold[body]-self.dt*self.body_v[body])/self.dt**2
            body_r[6*body+3:6*body+6] += I@(dphi/self.dt**2-self.body_omega[body]/self.dt) + np.cross(self.body_omega[body], I@self.body_omega[body])
            body_r[6*body+2] += self.body_mass[body]*9.81
            body_H[6*body:6*body+3, 6*body:6*body+3] += self.body_mass[body]/self.dt**2*np.eye(3)
            body_H[6*body+3:6*body+6, 6*body+3:6*body+6] += I/self.dt**2
        for plate, body in enumerate(self.plate_bodies):
            if self.solve_backend == 'cuda':
                bc = self._cuda_plate_contact(plate, c[body], R[body], commit)
            else:
                p, Jb = self._plate_points_at(plate, c[body], R[body])
                bc = evaluate_contact(p, self.plate_oldpoints[plate], self.plate_history[plate], dt=self.dt,
                                      ground_height=self.ground_height, normal_stiffness=self.kn, tangential_stiffness=self.kt,
                                      weights=self.plate_weights, mu=self.mu)
                bc['contact_residual'] = np.einsum('pij,pi->j', Jb, bc['force'], optimize=True)
                bc['contact_hessian'] = np.einsum('pai,pab,pbj->ij', Jb, bc['jacobian'], Jb, optimize=True)
            body_r[6*body:6*body+6] -= bc['contact_residual']
            body_H[6*body:6*body+6, 6*body:6*body+6] -= bc['contact_hessian']
            max_pen = max(max_pen, float(max(0., -np.min(bc['gap_m'])))); contact_force += float(np.sum(bc['normal_force_n'])); statuses.extend(bc['status'].tolist())
            if commit: self._candidate_plate = getattr(self, '_candidate_plate', []); self._candidate_plate.append(bc)
        total_res[self.nsteel:] += body_r
        if not np.isfinite(total_res).all() or not all(np.isfinite(x).all() for x in strip_hessians+strip_upper+strip_lower+[body_H]):
            raise FloatingPointError('non-finite full-robot residual')
        total_H = None
        if dense:
            total_H = np.zeros((self.nsteel+self.nbody_dof, self.nsteel+self.nbody_dof))
            for strip in range(self.nstrips):
                lo = strip*self.nfree; hi = lo+self.nfree
                total_H[lo:hi, lo:hi] = strip_hessians[strip]
                total_H[lo:hi, self.nsteel:] = strip_upper[strip]
                total_H[self.nsteel:, lo:hi] = strip_lower[strip]
            total_H[self.nsteel:, self.nsteel:] = body_H
        info = dict(residual=total_res, hessian=total_H, strip_hessians=strip_hessians,
                    strip_upper=strip_upper, strip_lower=strip_lower, body_hessian=body_H,
                    energy_j=steel_energy+cable_energy, steel_energy_j=steel_energy,
                    cable_energy_j=cable_energy, tensions_n=np.asarray(tensions), cable_lengths_m=np.asarray(cable_lengths), cable_routes_m=np.asarray(cable_routes),
                    max_penetration_m=max_pen, contact_normal_force_n=contact_force, statuses=statuses, q=q, c=c, R=R, B=B)
        return info

    def _plate_local(self, plate):
        return self.plate_local@self.body['plate_R_local'][plate].T+self.body['plate_centers_local_m'][plate]

    def _plate_points_at(self, plate, c, R):
        body = self.plate_bodies[plate]
        return _plate_points(c, R, self._plate_local(plate), self.body['com_local_m'][body])

    def _solve_schur(self, evaluation):
        """Eliminate independent steel interiors, then solve the shared bodies."""
        if self.solve_backend == 'cuda':
            return self._solve_schur_cuda(evaluation)
        body = self.nsteel
        residual = evaluation['residual']
        strip_hessians = evaluation['strip_hessians']
        strip_upper = evaluation['strip_upper']
        strip_lower = evaluation['strip_lower']
        body_hessian = evaluation['body_hessian']
        local_solutions = []
        coupling = np.zeros((self.nbody_dof, self.nbody_dof))
        body_rhs = -residual[body:].copy()

        def solve_dense(A, b):
            try:
                out = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                out = np.linalg.lstsq(A + 1e-8*np.eye(len(A)), b, rcond=None)[0]
            if not np.isfinite(out).all():
                raise np.linalg.LinAlgError('non-finite local Schur solve')
            return out

        for strip in range(self.nstrips):
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
            hessian = np.zeros((self.nsteel+self.nbody_dof, self.nsteel+self.nbody_dof))
            for strip in range(self.nstrips):
                lo = strip*self.nfree; hi = lo+self.nfree
                hessian[lo:hi, lo:hi] = strip_hessians[strip]
                hessian[lo:hi, body:] = strip_upper[strip]
                hessian[body:, lo:hi] = strip_lower[strip]
            hessian[body:, body:] = body_hessian
            return np.linalg.lstsq(hessian + 1e-8*np.eye(len(hessian)), -residual, rcond=None)[0]
        dz_steel = np.zeros(self.nsteel)
        for strip in range(self.nstrips):
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
        """CUDA FP64 local-block solve; ribbon geometry/contact are batched on device."""
        started = time.perf_counter()
        torch = self.torch
        body = self.nsteel
        residual = evaluation['residual']
        # Reuse device allocations across Newton iterations.  These small
        # matrices are too cheap to justify repeated CUDA allocator work.
        s, f, b = self.nstrips, self.nfree, self.nbody_dof
        shapes = ((s, f, f), (s, f, b), (s, b, f), (s, f), (b,), (b, b), (s, f, b+1))
        cache = self._cuda_buffers
        if cache is None or cache['A'].shape != shapes[0]:
            cache = {name: torch.empty(shape, dtype=torch.float64, device=self.device)
                     for name, shape in zip(('A', 'C', 'L', 'rs', 'rb', 'body_hessian', 'rhs'), shapes)}
            self._cuda_buffers = cache
        # copy_ keeps the reusable device storage while accepting the NumPy
        # arrays produced by the CPU geometry/contact assembly.
        cache['A'].copy_(torch.from_numpy(np.ascontiguousarray(np.stack(evaluation['strip_hessians']))))
        cache['C'].copy_(torch.from_numpy(np.ascontiguousarray(np.stack(evaluation['strip_upper']))))
        cache['L'].copy_(torch.from_numpy(np.ascontiguousarray(np.stack(evaluation['strip_lower']))))
        cache['rs'].copy_(torch.from_numpy(np.ascontiguousarray(residual[:self.nsteel].reshape(self.nstrips, self.nfree))))
        cache['rb'].copy_(torch.from_numpy(np.ascontiguousarray(residual[body:])))
        cache['body_hessian'].copy_(torch.from_numpy(np.ascontiguousarray(evaluation['body_hessian'])))
        A, C, L = cache['A'], cache['C'], cache['L']
        rs, rb = cache['rs'], cache['rb']
        # Solve the residual and all 12 plate-coupling right-hand sides in
        # one batched factorization.  The CPU banded path already does this;
        # keeping the same RHS layout avoids factoring each local block twice.
        # ``solve`` performs an eager CUDA error check.  ``solve_ex`` keeps
        # the info tensor on-device; the final NumPy finite check below is the
        # single required host-side validation.
        cache['rhs'][..., 0].copy_(rs).neg_()
        cache['rhs'][..., 1:].copy_(C)
        solved, _ = torch.linalg.solve_ex(A, cache['rhs'], check_errors=False)
        y0 = solved[..., 0]
        X = solved[..., 1:]
        schur = cache['body_hessian'] - torch.sum(L@X, dim=0)
        rhs = -rb - torch.sum(torch.matmul(L, y0[..., None])[..., 0], dim=0)
        db, _ = torch.linalg.solve_ex(schur, rhs, check_errors=False)
        dz = torch.cat(((y0-torch.matmul(X, db[..., None])[..., 0]).reshape(-1), db)).cpu().numpy()
        self.solve_seconds += time.perf_counter()-started
        if not np.isfinite(dz).all():
            raise FloatingPointError('non-finite CUDA Schur increment')
        return dz

    def step(self, cable_rest):
        cable_rest = np.asarray(cable_rest, dtype=float)
        if cable_rest.shape != (4*self.segments,) or not np.isfinite(cable_rest).all() or np.any(cable_rest < 0):
            raise ValueError('Need four finite, nonnegative cable lengths per segment')
        qold, uold, cold, Rold = self.q.copy(), self.u.copy(), self.body_com.copy(), self.body_R.copy()
        z = np.zeros(self.nsteel+self.nbody_dof)
        # A full velocity predictor is useful for the internal coordinates.
        z[:self.nsteel] = self.dt*uold[:, self.free].reshape(-1)
        predictor = z[self.nsteel:].reshape(self.nbodies, 6)
        predictor[:, :3] = self.dt*self.body_v
        predictor[:, 3:] = self.dt*self.body_omega
        last = None
        ev = None
        for iteration in range(60 if self.solve_backend == 'cuda' else 30):
            # Reuse the accepted line-search evaluation.  Recomputing the
            # same residual at the top of the next Newton iteration doubles
            # the expensive eight-strip force/Hessian assembly.
            if ev is None:
                ev = self._evaluate(z, cable_rest, qold, uold, cold, Rold)
            scaled = max(np.max(np.abs(ev['residual'][:self.nsteel]))/1e-5,
                         np.max(np.abs(ev['residual'][self.nsteel:]))/1e-5)
            last = ev
            if scaled <= 1.:
                self._candidate_steel = []; self._candidate_plate = []
                ev = self._evaluate(z, cable_rest, qold, uold, cold, Rold, commit=True)
                self.q, self.u = ev['q'], (ev['q']-qold)/self.dt
                self.body_com, self.body_R = ev['c'], ev['R']
                self.body_v = (self.body_com-cold)/self.dt
                self.body_omega = Rotation.from_matrix(self.body_R@Rold.swapaxes(1,2)).as_rotvec()/self.dt
                self.robots = [refresh(r, self.q[i]).update(u=self.u[i], a=np.zeros(self.nq)) for i, r in enumerate(self.robots)]
                self.steel_history = [x['history'] for x in self._candidate_steel]
                self.plate_history = [x['history'] for x in self._candidate_plate]
                self.steel_oldpoints = [self._steel_points(i, self.q[i]) for i in range(self.nstrips)]
                self.plate_oldpoints = [self._plate_points(i)[0] for i in range(self.nplates)]
                info = {k: v for k, v in ev.items() if k not in ('q','c','R','B','hessian','residual','statuses',
                                                                  'strip_hessians','strip_upper','strip_lower','body_hessian','cable_routes_m')}
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
            rigid_step = dz[self.nsteel:].reshape(self.nbodies, 6)
            body_step = max(np.max(np.abs(rigid_step[:, :3])), .1*np.max(np.abs(rigid_step[:, 3:])))
            fraction = min(1., .002/max(steel_step, body_step, 1e-12))
            accepted = False
            for _ in range(12):
                trial = self._evaluate(z+fraction*dz, cable_rest, qold, uold, cold, Rold)
                if np.linalg.norm(trial['residual']) < np.linalg.norm(ev['residual']):
                    z += fraction*dz; ev = trial; accepted = True; break
                fraction *= .5
            if not accepted:
                raise RuntimeError(f'Newton line search stalled at residual {scaled:g}')
        raise RuntimeError(f'Newton limit at residual {max(np.abs(last["residual"])):g}')

    def step_adaptive(self, cable_rest, max_rest_step_m=1e-4):
        """Split commands; rollback and bisect a failed contact/dynamic step."""
        target = np.asarray(cable_rest, dtype=float)
        start = self._last_cable_rest
        if target.shape != start.shape or not np.isfinite(target).all() or np.any(target < 0):
            raise ValueError('Invalid cable command')
        if max_rest_step_m <= 0.:
            count = 1
        else:
            count = max(1, int(np.ceil(np.max(np.abs(target-start))/max_rest_step_m)))
        nominal_dt = self.dt
        accepted = []
        state_names = ('q', 'u', 'body_com', 'body_R', 'body_v', 'body_omega', 'robots',
                       'steel_history', 'plate_history', 'steel_oldpoints', 'plate_oldpoints', 'last')
        outer_backup = {name: getattr(self, name) for name in state_names}

        def advance(begin, end, interval, depth=0):
            backup = {name: getattr(self, name) for name in state_names}
            self.dt = interval
            try:
                info = self.step(end)
                accepted.append(info)
                return info
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                for name, value in backup.items(): setattr(self, name, value)
                if depth >= 6: raise
                middle = (begin+end)/2
                advance(begin, middle, interval/2, depth+1)
                return advance(middle, end, interval/2, depth+1)

        try:
            for index in range(1, count+1):
                begin = start + (target-start)*((index-1)/count)
                rest = start + (target-start)*(index/count)
                info = advance(begin, rest, nominal_dt/count)
        except Exception:
            for name, value in outer_backup.items(): setattr(self, name, value)
            raise
        finally:
            self.dt = nominal_dt
        self._last_cable_rest = target.copy()
        info['command_substeps'] = len(accepted)
        info['recovery_substeps'] = len(accepted)-count
        info['substep_peak_tension_n'] = max(float(np.max(x['tensions_n'])) for x in accepted)
        info['substep_max_penetration_m'] = max(x['max_penetration_m'] for x in accepted)
        return info


def _command(base, t, duration, amplitude, gait='differential'):
    phase = 2*np.pi*t/max(duration, 1e-12)
    segments = len(base)//4
    if gait == 'axial-wave':
        shape = np.repeat(.5*(1-np.cos(phase-np.arange(segments)*2*np.pi/segments)), 4)
        shape *= min(1., t/max(.2*duration, 1e-12))
    else:
        shape = .5*(1+np.sin(phase+np.tile(np.arange(4), segments)*np.pi/2))
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
    chain = FullRobot(p, delta, nodes=9, dt=.002, contact_samples=12, segments=2)
    old = chain.q.copy(), chain.u.copy(), chain.body_com.copy(), chain.body_R.copy()
    qdecode, cdecode, Rdecode, B = chain._decode(np.zeros(chain.nsteel+chain.nbody_dof), old[0], old[2], old[3])
    np.testing.assert_allclose(qdecode, old[0], atol=1e-13)
    # Both neighbouring steel segments apply force to the SAME connector columns.
    assert np.any(B[0, :, 6:12]) and np.any(B[8, :, 6:12])
    probe = chain._evaluate(np.zeros(chain.nsteel+chain.nbody_dof), chain.base_cable_lengths-.0001, *old, dense=True)
    dx = chain._solve_schur(probe)
    np.testing.assert_allclose(probe['hessian']@dx, -probe['residual'], atol=1e-7, rtol=1e-7)
    moved = chain.step(chain.base_cable_lengths-.0001)
    assert moved['steel_energy_j'] > 0 and len(moved['tensions_n']) == 8
    report = dict(status='passed', nodes=9, strips=8, chain_segments=2, chain_strips=16,
                  chain_residual=float(moved['residual_max']), shared_connector=True, residual_n=float(zero['residual_max']),
                  max_penetration_m=float(zero['max_penetration_m']), contact_force_n=float(zero['contact_normal_force_n']),
                  masses_kg=sim.body_mass.tolist(), scope='Single/two-segment coupled implicit steps; no experimental calibration')
    print(json.dumps(report, indent=2)); return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters', type=Path, default=HERE/'output/parameters.snapshot.json')
    ap.add_argument('--nodes', type=int, default=9); ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--dt', type=float, default=.002); ap.add_argument('--duration', type=float, default=.02)
    ap.add_argument('--command-mm', type=float, default=.5); ap.add_argument('--output', type=Path, default=HERE/'full_robot_demo')
    ap.add_argument('--max-command-step-mm', type=float, default=.1)
    ap.add_argument('--solve-backend', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--cuda-device', type=int, default=0)
    ap.add_argument('--segments', type=int, default=1, choices=range(1, 6))
    ap.add_argument('--gait', choices=('differential', 'axial-wave'), default='differential')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    if args.self_check: self_check(); return
    if (args.steps < 1 or args.nodes < 7 or args.command_mm < 0 or args.dt <= 0 or args.duration <= 0
            or not np.isfinite([args.command_mm, args.dt, args.duration, args.max_command_step_mm]).all()): ap.error('invalid simulation arguments')
    if not math.isclose(args.steps*args.dt, args.duration, rel_tol=1e-8, abs_tol=1e-12):
        ap.error('duration must equal steps*dt so saved timestamps match physical integration')
    p, delta, provenance = read_project(args.parameters)
    sim = FullRobot(p, delta, nodes=args.nodes, dt=args.dt, cad_path=provenance['source_urdf'],
                    solve_backend=args.solve_backend, cuda_device=args.cuda_device, segments=args.segments)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); rows = []; q_frames = []; body_frames = []; rotation_frames = []; width_frames = []; routes_frames = []
    for i in range(args.steps+1):
        t = i*args.duration/max(args.steps, 1)
        rest = _command(sim.base_cable_lengths, t, args.duration, args.command_mm/1000., args.gait)
        if i: info = sim.step_adaptive(rest, args.max_command_step_mm/1000.)
        else: info = dict(iterations=0, residual_max=0., max_penetration_m=0., contact_normal_force_n=0.,
                          energy_j=0., steel_energy_j=0., cable_energy_j=0., tensions_n=np.zeros(4*args.segments), body_com=sim.body_com.tolist())
        centers = np.array([sim.body_com[b]+sim.body_R[b]@(sim.body['plate_centers_local_m'][plate]-sim.body['com_local_m'][b]) for plate, b in enumerate(sim.plate_bodies)])
        info['segment_lengths_m'] = np.linalg.norm(centers[1::2]-centers[::2], axis=1)
        info['cable_rest_m'] = rest
        rows.append(dict(time_s=t, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in info.items()}))
        q_frames.append(sim.q.copy()); body_frames.append(sim.body_com.copy()); rotation_frames.append(sim.body_R.copy())
        width_frames.append(np.stack([r.state.m2 for r in sim.robots]))
        routes_frames.append(np.array([cable.evaluate(sim.body_com[s:s+2], sim.body_R[s:s+2], rest[4*s:4*s+4])['routes_m'] for s, cable in enumerate(sim.cables)]))
        print(json.dumps(rows[-1], allow_nan=False), flush=True)
    result = dict(status='completed', metadata=dict(**provenance, nodes=args.nodes, strips=sim.nstrips,
        segments=args.segments, plates=sim.nplates, rigid_bodies=sim.nbodies, gait=args.gait, body_provenance=sim.body['provenance'],
        steel_dof_per_strip=sim.nfree, steel_dof_total=sim.nsteel, newton_dof=sim.nsteel+sim.nbody_dof,
        independent_strip_coordinates=True, dt_s=args.dt,
        duration_s=args.duration, ground_height_m=sim.ground_height, command_amplitude_mm=args.command_mm, model='Sano steel + CAD rigid bodies + ideal tension-only cables + sampled plane penalty/friction contact',
        max_command_step_mm=args.max_command_step_mm,
        contact='12 surface samples per steel edge plus 2x rim samples per plate; no continuous CCD; quasi-Newton contact geometric Hessian',
        solve_backend=args.solve_backend, cuda_device=args.cuda_device,
        local_solver='scipy.solve_banded' if args.solve_backend == 'cpu' else 'torch.linalg.solve_ex(reused_buffers)',
        local_bandwidth=sim.local_bandwidth,
        material_derivative=('cuda_sano.closed_form_fp64+cuda_strain_derivatives'
                             if args.solve_backend == 'cuda' else 'fast_sano.closed_form_numpy'),
        material_assembly='cuda_sano.chain_rule_scatter' if args.solve_backend == 'cuda' else 'vendor_numpy_chain_rule',
        geometry_contact=('cuda.fp64.batch_surface_jacobian+plate_contact_history'
                          if args.solve_backend == 'cuda' else 'numpy.surface_geometry+evaluate_contact'),
        solve_seconds=sim.solve_seconds, vendor_commit=COMMIT,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), frames=rows,
        wall_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out/'trajectory.npz', q=np.asarray(q_frames), body_com=np.asarray(body_frames), body_R=np.asarray(rotation_frames),
                        width_directors=np.asarray(width_frames), cable_routes_m=np.asarray(routes_frames),
                        body_com_local_m=sim.body['com_local_m'], plate_body_indices=sim.plate_bodies,
                        plate_centers_local_m=sim.body['plate_centers_local_m'], plate_R_local=sim.body['plate_R_local'], strip_bodies=sim.strip_bodies,
                        rest_nodes_m=sim.rest, strip_width_m=p['strip_width_m'], strip_thickness_m=p['strip_thickness_m'])
    print(json.dumps({'saved': str(out/'summary.json'), 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()

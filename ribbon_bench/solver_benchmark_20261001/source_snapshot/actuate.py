"""Eight Sano ribbons coupled to four ideal take-up cables and a guided endplate.

The plate has two free coordinates (axial translation and world-Z yaw). All other
plate coordinates are held by an ideal guide. This is static equilibrium, not a
dynamic motor/rope simulation. Upstream steel energies are used unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.linalg import solve_banded

from run import HERE, COMMIT, VENDOR, read_project, strip_geometry, make_robot, impose, pose, predict
from cable_loads import CableLoads


class PlateSystem:
    def __init__(self, parameters, delta, nodes=33, solver='fast'):
        if solver not in ('reference', 'fast'):
            raise ValueError('solver must be reference or fast')
        self.solver = solver
        self.p, self.delta, self.nodes = parameters, delta, nodes
        self.center = delta + parameters['back_plate_center_m']
        self.loads = CableLoads(parameters, delta)
        self.rest, self.widths, self.steppers, self.robots = [], [], [], []
        for strip in range(8):
            rest, widths = strip_geometry(parameters, delta, strip, nodes)
            robot, stepper = make_robot(parameters, rest, widths, 'sano')
            if solver == 'fast':
                from fast_sano import install_fast_sano
                install_fast_sano(stepper)
            self.rest.append(rest)
            self.widths.append(widths)
            self.robots.append(robot)
            self.steppers.append(stepper)
        self.free = [r.state.free_dof for r in self.robots]
        self.scales = [np.where(f < 3*nodes, .05, 1.) for f in self.free]
        self.plate_scale = np.array([.05, 1.])
        # Interleave XYZ and edge angle to expose the triplet stencil bandwidth.
        self.permutations = [np.argsort(np.where(f < 3*nodes,
                             4*(f//3) + f%3, 4*(f-3*nodes)+3)) for f in self.free]

    def moved(self, robots, plate, increments, plate_increment, fraction=1.):
        target = plate + fraction*plate_increment
        changed = []
        for r, free, increment, rest, width in zip(robots, self.free, increments, self.rest, self.widths):
            q = r.state.q.copy()
            q[free] += fraction*increment
            # One transport from the base state to the final state, including clamps.
            changed.append(impose(r, rest, width, self.center, 2., *target, q=q))
        return changed, target

    def evaluate(self, robots, plate, rest_lengths):
        loads = self.loads.evaluate(plate, rest_lengths)
        energy = float(loads['energy_j'])
        gp = np.array(loads['gradient'], copy=True)
        hp = np.array(loads['hessian'], copy=True)
        blocks, reactions = [], []
        force_error = moment_error = 0.
        for robot, stepper, free, scale in zip(robots, self.steppers, self.free, self.scales):
            stepper._compute_forces_and_jacobian(robot, robot.state.q, np.zeros(robot.n_dof))
            g, h = stepper._forces.copy(), stepper._jacobian.copy()
            if hasattr(h, 'toarray'):
                h = h.toarray()
            n = self.nodes
            xyz = robot.state.q[:3*n].reshape(n, 3)
            relative = xyz[-2:] - self.center - [plate[0], 0, 0]
            tangent = xyz[-1]-xyz[-2]
            tangent /= np.linalg.norm(tangent)
            b = np.zeros((robot.n_dof, 2))
            b[3*(n-2):3*n, 0] = np.tile([1., 0, 0], 2)
            b[3*(n-2):3*n, 1] = np.cross([0., 0, 1.], relative).ravel()
            b[-1, 1] = tangent[2]
            gp += b.T @ g
            hp += b.T @ h @ b
            hp[1, 1] += np.sum(g[3*(n-2):3*n].reshape(2, 3) *
                              np.cross([0., 0, 1.], np.cross([0., 0, 1.], relative)))
            internal_energy = float(stepper.compute_total_elastic_energy(robot.state))
            energy += internal_energy
            force_error = max(force_error, np.max(np.abs(g[free[free < 3*n]])))
            moment_error = max(moment_error, np.max(np.abs(g[free[free >= 3*n]])))
            blocks.append((h[np.ix_(free, free)]*scale[:, None]*scale[None, :],
                           (h[free]@b)*scale[:, None]*self.plate_scale,
                           (b.T@h[:, free])*self.plate_scale[:, None]*scale,
                           g[free]*scale))
            reactions.append((g, internal_energy))
        error = max(force_error/1e-6, moment_error/1e-7, abs(gp[0])/1e-5, abs(gp[1])/1e-6)
        if not np.isfinite([energy, error]).all():
            raise FloatingPointError('Non-finite coupled energy or residual')
        return dict(energy=energy, gp=gp, hp=hp, blocks=blocks, loads=loads,
                    reactions=reactions, error=float(error), free_force=float(force_error),
                    free_moment=float(moment_error))

    def direction(self, evaluated, regularization, prescribed=False):
        if prescribed:
            return ([-self.solve_block(a, g, regularization, permutation)*scale
                     for (a, _, _, g), scale, permutation in
                     zip(evaluated['blocks'], self.scales, self.permutations)], np.zeros(2))
        sp = self.plate_scale
        schur = evaluated['hp']*sp[:, None]*sp + regularization*np.eye(2)
        rhs = evaluated['gp']*sp
        solved = []
        for (a, c, d, g), permutation in zip(evaluated['blocks'], self.permutations):
            solution = self.solve_block(a, np.column_stack((g, c)), regularization, permutation)
            schur -= d @ solution[:, 1:]
            rhs -= d @ solution[:, 0]
            solved.append(solution)
        dp_scaled = -np.linalg.solve(schur, rhs)
        internal = [(-s[:, 0]-s[:, 1:]@dp_scaled)*scale for s, scale in zip(solved, self.scales)]
        return internal, dp_scaled*sp

    def solve_block(self, a, rhs, regularization, permutation):
        if self.solver == 'reference':
            return np.linalg.solve(a + regularization*np.eye(len(a)), rhs)
        matrix = a[np.ix_(permutation, permutation)].copy()
        matrix.flat[::len(matrix)+1] += regularization
        rows, cols = np.nonzero(matrix)
        lower = max(0, int(np.max(rows-cols, initial=0)))
        upper = max(0, int(np.max(cols-rows, initial=0)))
        # ponytail: dense assembly is retained; direct band assembly if profiling warrants it.
        # Derive the actual band: never discard entries if the stencil changes.
        band = np.zeros((lower+upper+1, len(matrix)))
        band[upper+rows-cols, cols] = matrix[rows, cols]
        solved = solve_banded((lower, upper), band, rhs[permutation])
        result = np.empty_like(solved)
        result[permutation] = solved
        return result

    def equilibrate(self, robots, plate, rest_lengths, max_iter=100, prescribed=False):
        base = None
        for iteration in range(max_iter):
            if base is None or self.solver == 'reference':
                base = self.evaluate(robots, plate, rest_lengths)
            if prescribed:
                base['error'] = max(base['free_force']/1e-6, base['free_moment']/1e-7)
            if base['error'] <= 1:
                return robots, plate, base, iteration
            accepted = False
            for regularization in (0., 1e-8, 1e-6, 1e-4, .01, 1., 100.):
                try:
                    internal, dp = self.direction(base, regularization, prescribed)
                except np.linalg.LinAlgError:
                    continue
                slope = base['gp']@dp + sum(block[3]@(step/scale) for block, step, scale
                                           in zip(base['blocks'], internal, self.scales))
                if not np.isfinite(slope) or slope >= 0:
                    continue
                largest = max(abs(dp[0])/.006, abs(dp[1])/.1, 1.)
                for step, free in zip(internal, self.free):
                    largest = max(largest, np.max(np.abs(step[free < 3*self.nodes]))/.012)
                alpha = 1/largest
                for _ in range(14):
                    try:
                        candidates, candidate_plate = self.moved(robots, plate, internal, dp, alpha)
                        trial = self.evaluate(candidates, candidate_plate, rest_lengths)
                        if prescribed:
                            trial['error'] = max(trial['free_force']/1e-6, trial['free_moment']/1e-7)
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                        alpha *= .5
                        continue
                    if (trial['energy'] <= base['energy'] + 1e-4*alpha*slope + 1e-13
                            and trial['error'] < max(base['error']*5, 1)):
                        robots, plate = candidates, candidate_plate
                        if self.solver == 'fast':
                            base = trial
                        accepted = True
                        break
                    alpha *= .5
                if accepted:
                    break
            if not accepted:
                raise RuntimeError(f'Coupled Newton stalled: scaled residual {base["error"]:.3g}')
        raise RuntimeError(f'Coupled Newton reached {max_iter} iterations; residual {base["error"]:.3g}')

    def frame(self, robots, plate, evaluated, progress, rest_lengths):
        _, _, rotation = pose(2., *plate)
        frames = []
        for robot, (gradient, energy), free in zip(robots, evaluated['reactions'], self.free):
            n = self.nodes
            nodal = gradient[:3*n].reshape(n, 3)
            frames.append(dict(progress=float(progress), compression_m=float(plate[0]), yaw_rad=float(plate[1]),
                plate_rotation=rotation.tolist(), plate_translation_m=[float(plate[0]), 0., 0.],
                nodes_m=robot.state.q[:3*n].reshape(n, 3).tolist(), width_directors=robot.state.m2.tolist(),
                energy_j=energy, back_support_force_n=nodal[-2:].sum(0).tolist(),
                front_support_force_n=nodal[:2].sum(0).tolist(),
                free_force_residual_n=float(np.max(np.abs(gradient[free[free < 3*n]]))),
                free_moment_residual_nm=float(np.max(np.abs(gradient[free[free >= 3*n]]))),
                force_balance_n=float(np.linalg.norm(nodal.sum(0)))))
        loads = evaluated['loads']
        controls = {key: np.asarray(loads[key]).tolist() for key in
                    ('routes_m', 'tensions_n', 'lengths_m', 'plate_gap_m', 'contact_force_n', 'contact_energy_j')}
        controls.update(rest_lengths_m=np.asarray(rest_lengths).tolist(), total_energy_j=evaluated['energy'],
                        progress=float(progress),
                        plate_force_residual_n=float(evaluated['gp'][0]),
                        plate_moment_residual_nm=float(evaluated['gp'][1]))
        return frames, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, default=HERE/'output/parameters.snapshot.json')
    parser.add_argument('--nodes', type=int, default=33)
    parser.add_argument('--solver', choices=('reference', 'fast'), default='fast',
                        help='same Sano model; fast uses closed material derivatives, banded blocks and accepted-state reuse')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--compression-mm', type=float, default=40)
    parser.add_argument('--yaw-deg', type=float, default=30)
    parser.add_argument('--output', type=Path, default=HERE/'actuated_demo')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--prescribed', action='store_true', help='fixed-plate reference; cables slack')
    parser.add_argument('--loading-only', action='store_true', help='prescribed reference up to peak, without unloading')
    args = parser.parse_args()
    if args.nodes < 7 or args.steps < 1 or not np.isfinite([args.compression_mm, args.yaw_deg]).all():
        parser.error('Need finite targets, >=7 nodes and positive steps')
    if not 0 <= args.compression_mm < 85 or abs(args.yaw_deg) > 45:
        parser.error('This guided-plate demo supports 0..85 mm compression and at most 45 degree yaw')
    if args.loading_only and not args.prescribed:
        parser.error('--loading-only is for the prescribed reference')
    p, delta, provenance = read_project(args.parameters)
    if p['strip_count'] != 8:
        parser.error('This coupled benchmark requires eight strips')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() != COMMIT:
        raise RuntimeError('Vendor revision differs from the reviewed implementation')
    system = PlateSystem(p, delta, args.nodes, solver=args.solver)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output/'parameters.snapshot.json').write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = dict(**provenance, upstream_commit=COMMIT, nodes=args.nodes, width_m=p['strip_width_m'],
        thickness_m=p['strip_thickness_m'], youngs_modulus_pa=p['youngs_modulus_pa'], poisson_ratio=p['poisson_ratio'],
        plate_radius_m=p['plate_stop_radius_m'], front_plate_center_m=p['front_plate_center_m'],
        back_plate_center_world_m=system.center.tolist(), target_compression_mm=args.compression_mm,
        target_yaw_deg=args.yaw_deg, loading='Four ideal cable take-up lengths; plate translation and yaw are solved',
        reference='Provisional stress-free parabola; no measured assembly preload',
        limits='Two-DOF guided plate; massless straight tension-only cable spans; no motor mapping, sag, friction, steel contact, gravity or inertia',
        contact='Frictionless penalty between simplified solid plate envelopes; not exact CAD collision',
        solver='Block Newton / two-coordinate Schur complement, energy line search, adaptive load subdivision; upstream Hessian used as Newton approximation',
        solver_backend=args.solver,
        contact_stiffness_n_m=50000, cable_stiffness_n_m=p['effective_tendon_stiffness_n_m'])
    result = dict(metadata=metadata, cases=[dict(model='sano', strip=i, rest_nodes_m=rest.tolist(), frames=[])
                                           for i, rest in enumerate(system.rest)], actuation_frames=[])
    if args.prescribed:
        metadata['loading'] = 'Prescribed plate reference; cable lengths held slack; plate reactions are external'
    # Targets only generate cable lengths. No target plate coordinate is prescribed to the solve.
    def command(progress):
        if args.prescribed:
            return np.ones(4)
        d, a, _ = pose(progress, args.compression_mm/1000, math.radians(args.yaw_deg))
        return np.asarray(system.loads.evaluate([d, a], np.ones(4))['lengths_m'])
    subdivisions = 0
    def advance(robots, plate, begin, end, depth=0):
        nonlocal subdivisions
        if args.prescribed:
            _, a0, _ = pose(begin, args.compression_mm/1000, math.radians(args.yaw_deg))
            _, a1, _ = pose(end, args.compression_mm/1000, math.radians(args.yaw_deg))
            if abs(a1-a0) > math.radians(.5) and depth < 7:
                subdivisions += 1
                midpoint = (begin+end)/2
                middle, mp, _, _ = advance(robots, plate, begin, midpoint, depth+1)
                return advance(middle, mp, midpoint, end, depth+1)
        try:
            candidates, target = robots, plate
            if args.prescribed:
                compression, yaw = args.compression_mm/1000, math.radians(args.yaw_deg)
                d, a, _ = pose(end, compression, yaw)
                target = np.array([d, a])
                candidates = [impose(predict(r, system.center, begin, end, compression, yaw),
                              rest, widths, system.center, end, compression, yaw)
                              for r, rest, widths in zip(robots, system.rest, system.widths)]
            return system.equilibrate(candidates, target, command(end), prescribed=args.prescribed)
        except RuntimeError:
            if depth >= 7:
                raise
            subdivisions += 1
            midpoint = (begin+end)/2
            middle, mp, _, _ = advance(robots, plate, begin, midpoint, depth+1)
            return advance(middle, mp, midpoint, end, depth+1)
    robots, plate = system.robots, np.zeros(2)
    started = time.perf_counter()
    previous = 0.
    end_progress = 2 if args.loading_only else 4
    saved_states = end_progress*args.steps+1
    for index, progress in enumerate(np.linspace(0, end_progress, saved_states)):
        robots, plate, evaluated, iterations = advance(robots, plate, previous, progress)
        frames, controls = system.frame(robots, plate, evaluated, progress, command(progress))
        for case, frame in zip(result['cases'], frames):
            case['frames'].append(frame)
        result['actuation_frames'].append(controls)
        metadata.update(wall_seconds=time.perf_counter()-started, adaptive_subdivisions=subdivisions)
        (output/'checkpoint.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(f'{index+1}/{saved_states}: solved compression={plate[0]*1000:.3f} mm, '
              f'yaw={math.degrees(plate[1]):.3f} deg, Newton={iterations}, residual={evaluated["error"]:.3g}, '
              f'wall={metadata["wall_seconds"]:.1f}s', flush=True)
        previous = progress
    if args.loading_only:
        metadata.update(loading_path_complete=True, unloading_converged=None, path_status='loading_only',
                        path_note='Only the loading path was requested in this run; unloading was not attempted')
    else:
        for case in result['cases']:
            case['final_shape_error_m'] = float(np.max(np.linalg.norm(
                np.asarray(case['frames'][-1]['nodes_m'])-case['rest_nodes_m'], axis=1)))
    result_path = output/('loading_results.json' if args.loading_only else 'results.json')
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    if args.render:
        if args.prescribed:
            from render import render
        else:
            from render_actuated import render
        render(result_path)


if __name__ == '__main__':
    main()

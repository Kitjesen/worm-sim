"""One clamped Sano ribbon: static transverse preload, then implicit-Euler release.

No ground, ropes, moving plates, or contact. The time histories are integrated
states, not interpolation of static shapes. Vendor elastic energies are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.linalg import eigh

from run import HERE, COMMIT, VENDOR, read_project, strip_geometry, make_robot, refresh


class DynamicStrip:
    def __init__(self, p, delta, nodes=17, alpha=2., fast=True):
        self.p, self.nodes, self.alpha = p, nodes, alpha
        self.rest, widths = strip_geometry(p, delta, 0, nodes)
        self.robot, self.stepper = make_robot(p, self.rest, widths, 'sano')
        if fast:
            from fast_sano import install_fast_sano
            install_fast_sano(self.stepper)
        self.free = self.robot.state.free_dof
        self.scale = np.where(self.free < 3*nodes, .05, 1.)
        self.mass = self.robot.mass_matrix.copy()
        w, h, rho = p['strip_width_m'], p['strip_thickness_m'], p['steel_density_kg_m3']
        self.polar_area_m4 = (w*h**3+h*w**3)/12
        # Torsional stiffness keeps Saint-Venant J; rotational inertia uses I1+I2.
        self.mass[3*nodes:] = rho*self.robot.ref_len*self.polar_area_m4
        self.damping = alpha*self.mass  # kg/s for xyz, kg m^2/s for edge angles.
        expected_mass = rho*w*h*sum(self.robot.ref_len)
        assert np.isclose(sum(self.mass[:3*nodes:3]), expected_mass, rtol=1e-12)
        self.mid = nodes//2
        self.direction = self.rest[self.mid]-(self.rest[0]+self.rest[-1])/2
        self.direction /= np.linalg.norm(self.direction)

    def elastic(self, robot, q):
        self.stepper._compute_forces_and_jacobian(robot, q, np.zeros_like(q))
        changed = refresh(robot, q)
        return (self.stepper._forces.copy(), self.stepper._jacobian.copy(),
                float(self.stepper.compute_total_elastic_energy(changed.state)))

    def solve(self, robot, force, dt=None):
        old, velocity = robot.state.q, robot.state.u
        q = old.copy()
        diagonal = np.zeros_like(old) if dt is None else self.mass/dt**2+self.damping/dt
        predictor = old if dt is None else old+dt*velocity

        def evaluate(q):
            gradient, hessian, energy = self.elastic(robot, q)
            residual = gradient-force
            potential = energy-force@(q-old)
            if dt is not None:
                residual += self.mass*(q-predictor)/dt**2+self.damping*(q-old)/dt
                potential += .5*np.dot(self.mass, (q-predictor)**2)/dt**2
                potential += .5*np.dot(self.damping, (q-old)**2)/dt
            hessian[np.diag_indices_from(hessian)] += diagonal
            f = self.free[self.free < 3*self.nodes]
            m = self.free[self.free >= 3*self.nodes]
            error = max(np.max(abs(residual[f]))/1e-7, np.max(abs(residual[m]))/1e-10)
            return residual, hessian, energy, float(potential), float(error)

        for iteration in range(40):
            residual, hessian, energy, potential, error = evaluate(q)
            if error <= 1:
                changed = refresh(robot, q)
                v = np.zeros_like(q) if dt is None else (q-old)/dt
                a = np.zeros_like(q) if dt is None else (v-velocity)/dt
                return changed.update(u=v, a=a), dict(iterations=iteration, error=error,
                    force_residual_n=float(np.max(abs(residual[self.free[self.free < 3*self.nodes]]))),
                    moment_residual_nm=float(np.max(abs(residual[self.free[self.free >= 3*self.nodes]]))))
            a = hessian[np.ix_(self.free, self.free)]*self.scale[:, None]*self.scale
            g = residual[self.free]*self.scale
            shift = 0.
            accepted = False
            for attempt in range(2):
                dq = -np.linalg.solve(a+shift*np.eye(len(g)), g)*self.scale
                slope = residual[self.free]@dq
                if slope >= 0 or not np.isfinite(dq).all():
                    shift = max(1e-5, -float(np.linalg.eigvalsh((a+a.T)/2)[0])+1e-5)
                    continue
                fraction = min(1., .005/max(np.max(abs(dq[self.free < 3*self.nodes])), 1e-30))
                for _ in range(20):
                    trial = q.copy()
                    trial[self.free] += fraction*dq
                    _, _, _, trial_potential, trial_error = evaluate(trial)
                    if trial_potential <= potential+1e-4*fraction*slope+1e-15:
                        q, accepted = trial, True
                        break
                    fraction *= .5
                if accepted:
                    break
                shift = max(1e-5, -float(np.linalg.eigvalsh((a+a.T)/2)[0])+1e-5)
            if not accepted:
                raise RuntimeError(f'Newton line search stalled; scaled residual {error:g}')
        raise RuntimeError(f'Newton limit; scaled residual {error:g}')

    def history(self, initial, dt, duration):
        count = round(duration/dt)
        if not np.isclose(count*dt, duration):
            raise ValueError('Duration must be an integer multiple of dt')
        q, u, kinetic, elastic, residuals, iterations = [], [], [], [], [], []
        robot = initial
        dissipation = [0.]
        for index in range(count+1):
            if index:
                robot, info = self.solve(robot, np.zeros(robot.n_dof), dt)
                residuals.append([info['force_residual_n'], info['moment_residual_nm']])
                iterations.append(info['iterations'])
                dissipation.append(dissipation[-1]+dt*np.dot(self.damping, robot.state.u**2))
            q.append(robot.state.q.copy())
            u.append(robot.state.u.copy())
            kinetic.append(.5*np.dot(self.mass, robot.state.u**2))
            elastic.append(float(self.stepper.compute_total_elastic_energy(robot.state)))
        result = dict(time_s=np.arange(count+1)*dt, q=np.array(q), velocity=np.array(u),
                      kinetic_energy_j=np.array(kinetic), elastic_energy_j=np.array(elastic),
                      viscous_dissipation_j=np.array(dissipation), residuals=np.array(residuals),
                      newton_iterations=np.array(iterations), mass_diagonal=self.mass,
                      damping_diagonal=self.damping, rest_nodes_m=self.rest, fixed_dof=self.robot.fixed_dof)
        result['midpoint_displacement_m'] = ((result['q'][:, 3*self.mid:3*self.mid+3]-self.rest[self.mid])@self.direction)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, default=HERE/'output/parameters.snapshot.json')
    parser.add_argument('--nodes', type=int, default=17)
    parser.add_argument('--dt', type=float, default=.000005)
    parser.add_argument('--duration', type=float, default=.015)
    parser.add_argument('--force-n', type=float, default=.01)
    parser.add_argument('--alpha', type=float, default=2., help='Mass-proportional viscous damping coefficient, s^-1')
    parser.add_argument('--output', type=Path, default=HERE/'dynamic_strip_demo')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--autodiff', action='store_true', help='Use original vendor material derivatives')
    args = parser.parse_args()
    if (args.nodes < 7 or not np.isfinite([args.dt, args.duration, args.force_n, args.alpha]).all()
            or args.alpha < 0 or min(args.dt, args.duration, args.force_n) <= 0):
        parser.error('Need >=7 nodes, alpha>=0, and positive dt/duration/force')
    p, delta, provenance = read_project(args.parameters)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() != COMMIT:
        raise RuntimeError('Vendor revision differs from the reviewed implementation')
    system = DynamicStrip(p, delta, args.nodes, args.alpha, fast=not args.autodiff)
    started = time.perf_counter()
    g, h, _ = system.elastic(system.robot, system.robot.state.q)
    eigenvalues, modes = eigh(h[np.ix_(system.free, system.free)], np.diag(system.mass[system.free]))
    all_frequencies = np.sqrt(np.maximum(eigenvalues, 0))/(2*np.pi)
    frequencies = all_frequencies[:6]
    force = np.zeros(system.robot.n_dof)
    force[3*system.mid:3*system.mid+3] = args.force_n*system.direction
    preload, preload_info = system.solve(system.robot, force)
    initial_displacement = float((preload.state.q[3*system.mid:3*system.mid+3]-system.rest[system.mid])@system.direction)
    amplitudes = modes.T@(system.mass[system.free]*(preload.state.q-system.robot.state.q)[system.free])
    fractions = eigenvalues*amplitudes**2
    fractions /= sum(fractions)
    dominant = [dict(frequency_hz=float(all_frequencies[i]), linear_preload_energy_fraction=float(fractions[i]))
                for i in np.argsort(fractions)[-3:][::-1]]
    print(json.dumps(dict(natural_frequencies_hz=frequencies.tolist(), preload_displacement_m=initial_displacement,
                          preload_info=preload_info, dominant_linear_modes=dominant,
                          setup_seconds=time.perf_counter()-started)), flush=True)
    if args.setup_only:
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    zero = system.history(system.robot, args.dt, 3*args.dt)
    zero_motion = float(np.max(abs(zero['q'][:, :3*args.nodes]-system.robot.state.q[:3*args.nodes])))
    zero_twist = float(np.max(abs(zero['q'][:, 3*args.nodes:]-system.robot.state.q[3*args.nodes:])))
    assert zero_motion < 1e-10 and zero_twist < 1e-10
    assert np.max(abs(zero['velocity'][:, :3*args.nodes])) < 1e-8
    assert np.max(abs(zero['velocity'][:, 3*args.nodes:])) < 1e-8
    summary = dict(**provenance, vendor_commit=COMMIT, nodes=args.nodes, strip=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        material_backend='vendor_autodiff' if args.autodiff else 'closed_form_adapter',
        fast_sano_sha256=None if args.autodiff else hashlib.sha256((HERE/'fast_sano.py').read_bytes()).hexdigest(),
        method='Implicit Euler; diagonal physical mass; Sano elastic force/Hessian; incremental-potential line search',
        mass_kg=float(sum(system.mass[:3*args.nodes:3])), polar_area_m4=system.polar_area_m4,
        damping_alpha_per_s=args.alpha, preload_force_n=args.force_n, preload_displacement_m=initial_displacement,
        natural_frequencies_hz=frequencies.tolist(), zero_input_max_motion_m=zero_motion,
        dominant_linear_modes=dominant,
        zero_input_max_twist_rad=zero_twist, boundary='First and last two nodes and end twist angles fixed at natural-state clamps',
        limits='One ribbon; no ropes, moving plates, contact, gravity, measured preload or calibrated damping', runs=[])
    histories = []
    for label, dt in [('dt', args.dt), ('half_dt', args.dt/2)]:
        mark = time.perf_counter()
        result = system.history(preload, dt, args.duration)
        np.savez_compressed(output/f'{label}.npz', **result)
        energy = result['kinetic_energy_j']+result['elastic_energy_j']
        assert np.array_equal(result['q'][:, system.robot.fixed_dof],
                              np.broadcast_to(system.robot.state.q[system.robot.fixed_dof],
                                              result['q'][:, system.robot.fixed_dof].shape)), 'Clamp drift'
        assert not np.any(result['velocity'][:, system.robot.fixed_dof]), 'Clamp velocity'
        crosses = int(np.sum(result['midpoint_displacement_m'][1:]*result['midpoint_displacement_m'][:-1] < 0))
        assert np.max(np.diff(energy)) <= max(1e-13, energy[0]*1e-7), 'Release gained mechanical energy'
        assert np.max(result['kinetic_energy_j']) > energy[0]*.01, 'No energy transfer to motion'
        numerical_loss = energy[0]-energy-result['viscous_dissipation_j']
        assert min(numerical_loss) >= -max(1e-13, energy[0]*1e-7), 'Viscous work exceeds mechanical energy loss'
        entry = dict(label=label, dt_s=dt, duration_s=args.duration, steps=len(energy)-1,
                     wall_seconds=time.perf_counter()-mark, initial_energy_j=float(energy[0]),
                     final_energy_j=float(energy[-1]), max_kinetic_energy_j=float(max(result['kinetic_energy_j'])),
                     zero_crossings=crosses, max_energy_increase_j=float(max(np.diff(energy))),
                     final_viscous_dissipation_j=float(result['viscous_dissipation_j'][-1]),
                     final_numerical_energy_loss_j=float(numerical_loss[-1]),
                     max_force_residual_n=float(max(result['residuals'][:, 0])),
                     max_moment_residual_nm=float(max(result['residuals'][:, 1])))
        summary['runs'].append(entry)
        histories.append(result)
        print(json.dumps(entry), flush=True)
    error = histories[0]['midpoint_displacement_m']-histories[1]['midpoint_displacement_m'][::2]
    summary['dt_half_comparison'] = dict(max_displacement_difference_m=float(max(abs(error))),
        normalized_rms_difference=float(np.sqrt(np.mean(error**2))/abs(initial_displacement)))
    coarse_energy = histories[0]['kinetic_energy_j']+histories[0]['elastic_energy_j']
    fine_energy = histories[1]['kinetic_energy_j']+histories[1]['elastic_energy_j']
    normalized_energy_error = float(max(abs(coarse_energy-fine_energy[::2]))/coarse_energy[0])
    summary['dt_half_comparison'].update(max_energy_difference_over_initial=normalized_energy_error,
        rms_displacement_limit=.05, normalized_energy_difference_limit=.10)
    comparison_passed = (summary['dt_half_comparison']['normalized_rms_difference'] <= .05
                         and normalized_energy_error <= .10)
    summary['wall_seconds'] = time.perf_counter()-started
    summary['status'] = ('validated' if comparison_passed and min(r['zero_crossings'] for r in summary['runs']) >= 2
                         else 'timestep_or_duration_check_failed')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), constrained_layout=True)
    for result, label in zip(histories, ['dt', 'dt/2']):
        axes[0].plot(result['time_s']*1000, result['midpoint_displacement_m']*1e6, label=label)
        axes[1].plot(result['time_s']*1000,
                     (result['kinetic_energy_j']+result['elastic_energy_j'])*1e6, label=label)
    axes[0].set_ylabel('Midpoint displacement (µm)')
    axes[1].set_ylabel('Mechanical energy (µJ)')
    axes[1].set_xlabel('Time (ms)')
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output/'response.png', dpi=300)
    fig.savefig(output/'response.svg')
    plt.close(fig)
    print(json.dumps(summary['dt_half_comparison']), flush=True)
    if summary['status'] != 'validated':
        raise RuntimeError('Saved honest histories, but timestep/duration acceptance checks did not pass')


if __name__ == '__main__':
    main()

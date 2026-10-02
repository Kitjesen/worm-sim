"""Coupled SOFA whole-worm candidate: plates, clamped beams and 4 tendons/segment.

Candidate reductions: elastic massless tendons, circular horn-rim routing,
analytic flat-ground wheel contact with wheel spin and unilateral clutch.
These are explicit training approximations, not a calibrated hardware twin.
"""
import json
import math
from pathlib import Path
import Sofa
import SofaRuntime
import numpy as np
import gymnasium as gym
import mujoco  # rotation utilities only; no MuJoCo simulation

HERE = Path(__file__).resolve().parent


def rotation(q):
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(q)[[3, 0, 1, 2]])
    return m.reshape(3, 3)


def quat(m):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(m).ravel())
    return q[[1, 2, 3, 0]]


def horn_takeup(theta, radius=.0235, guide_radius=.035):
    """Shortest straight/tangent+arc route from a guide to a rim-bound eccentric tie.

    ponytail: circular rim proxy, not the CAD arm outline; replace with measured
    angle-to-takeup data or explicit cable/arm contact before hardware transfer.
    """
    theta = np.asarray(theta)
    alpha = math.acos(radius/guide_radius)
    direct = np.sqrt(guide_radius**2+radius**2-2*guide_radius*radius*np.cos(theta))
    wrapped = math.sqrt(guide_radius**2-radius**2)+radius*(theta-alpha)
    return np.where(theta <= alpha, direct, wrapped)-(guide_radius-radius)


def horn_lever(theta, radius=.0235, guide_radius=.035):
    alpha = math.acos(radius/guide_radius)
    chord = np.sqrt(guide_radius**2+radius**2-2*guide_radius*radius*np.cos(theta))
    return np.where(theta <= alpha, radius*guide_radius*np.sin(theta)/chord, radius)


def wheel_contact(velocity, spin, memory, normal, dt, radius=.018, mu=.8, rolling=.015):
    """Compliant sticking/sliding contact, implicit wheel spin, one-way clutch.

    Tangential spring stores bounded elastic displacement, so a stationary locked
    tire supports force. This is a regularized contact law, not exact rigid contact.
    """
    inertia = .5*.009*radius**2
    stiffness, damping = 2000., 8.
    resistance = min(rolling*normal*radius+2e-6*spin, inertia*spin/dt)
    gain = stiffness*dt+damping
    base = -stiffness*memory-gain*velocity
    free = (spin-dt*(radius*base[0]+resistance)/inertia)/(1+dt*radius**2*gain/inertia)
    omega = max(0., free)
    force = base.copy(); force[0] += gain*radius*omega
    cap = mu*normal
    sliding = np.linalg.norm(force) > cap
    if sliding:
        force *= cap/max(1e-15, np.linalg.norm(force))
    free = spin-dt*(radius*force[0]+resistance)/inertia
    omega = max(0., free)
    clutch = inertia*(omega-free)/dt
    slip = velocity-np.array([radius*omega, 0.])
    memory = -(force+damping*slip)/stiffness if sliding else memory+dt*slip
    return force, omega, memory, clutch-resistance, slip, sliding


class SofaWormEnv(gym.Env):
    metadata = {}

    def __init__(self, count=5, elements=8, randomize=True, dt=.0005, fixed=False):
        super().__init__()
        if not 1 <= count <= 5 or elements < 4 or not 0 < dt <= .001:
            raise ValueError('Invalid topology or time step')
        self.count, self.elements, self.randomize, self.dt, self.fixed = count, elements, randomize, dt, fixed
        self.ndrive = 3*count-1
        self.episode_steps = 600  # 12 s: multiple contraction/recovery cycles
        self.action_space = gym.spaces.Box(-1., 1., (self.ndrive,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15+4*count+2*(count-1)+4*count+self.ndrive,), dtype=np.float32)
        self.params = json.loads((HERE/'parameters_sofa_candidate.json').read_text())
        self.root = None

    def close(self):
        if self.root is not None:
            Sofa.Simulation.unload(self.root)
            self.root = None

    def build(self):
        SofaRuntime.importPlugin('BeamAdapter')
        root = self.root = Sofa.Core.Node('CoupledWorm')
        root.gravity = [0, 0, -9.81]
        root.dt = self.dt
        root.addObject('RequiredPlugin', pluginName=('BeamAdapter Sofa.Component.AnimationLoop '
            'Sofa.Component.StateContainer Sofa.Component.Topology.Container.Dynamic '
            'Sofa.Component.LinearSolver.Direct Sofa.Component.ODESolver.Backward '
            'Sofa.Component.Constraint.Projective Sofa.Component.Mass '
            'Sofa.Component.SolidMechanics.Spring Sofa.Component.Mapping.Linear'))
        root.addObject('DefaultAnimationLoop')
        root.addObject('EulerImplicitSolver', rayleighMass=.05, rayleighStiffness=.002*self.factors[1])
        root.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
        p = self.params
        self.plate_ids = np.arange(2*self.count)
        self.plate_mass = .28*self.factors[2]
        self.radius = .018
        # CAD axle holes on front/back partitions. A bolted 45 mm drop bracket
        # preserves steel clearance; its geometry is shared with the renderer.
        self.wheel_holes = np.array([[[-.014, .0617601286221, -.0689776402648],
                                     [-.014, -.0617601286221, -.0689776402648]],
                                    [[-.0005, .0617601286221, -.0700229145182],
                                     [-.0005, -.0617601286221, -.0700229145182]]])
        self.wheel_mounts = self.wheel_holes+np.array([0., 0., -.045])
        z = -self.wheel_mounts[0, 0, 2]+self.radius+.001
        poses = [[-i*.1755-j*.1175, 0., z+j*.0010452742534, 0., 0., 0., 1.] for i in range(self.count) for j in (0, 1)]
        edges, transform0, transform1, lengths = [], [], [], []
        self.strip_nodes = []
        identity = [0., 0., 0., 0., 0., 0., 1.]
        for seg in range(self.count):
            for strip in range(8):
                front = np.mean(p['front_clamp_hole_pairs_m'][strip], axis=0)
                back = np.mean(p['back_clamp_hole_pairs_m'][strip], axis=0)
                a = np.array(poses[2*seg][:3])+front
                b = np.array(poses[2*seg+1][:3])+back
                radial = np.array([0., front[1], front[2]])
                radial /= np.linalg.norm(radial)
                side = np.subtract(*p['front_clamp_hole_pairs_m'][strip])
                side /= np.linalg.norm(side)
                nodes, frames = [], []
                for j,u in enumerate(np.linspace(0., 1., self.elements+1)):
                    point = a+u*(b-a)+4*p['stress_free_bow_m']*u*(1-u)*radial
                    tangent = b-a+4*p['stress_free_bow_m']*(1-2*u)*radial
                    tangent /= np.linalg.norm(tangent)
                    width_axis = side-np.dot(side, tangent)*tangent
                    width_axis /= np.linalg.norm(width_axis)
                    frame = [*point, *quat(np.column_stack([tangent, width_axis, np.cross(tangent, width_axis)]))]
                    frames.append(frame)
                    if j == 0:
                        nodes.append(2*seg)
                    elif j == self.elements:
                        nodes.append(2*seg+1)
                    else:
                        nodes.append(len(poses)); poses.append(frame)
                for j in range(self.elements):
                    edges.append([nodes[j], nodes[j+1]])
                    # End sections belong directly to each plate's rigid DOF.
                    transform0.append([*front, *frames[0][3:]] if j == 0 else identity)
                    transform1.append([*back, *frames[-1][3:]] if j == self.elements-1 else identity)
                    lengths.append(float(np.linalg.norm(np.subtract(frames[j+1][:3], frames[j][:3]))))
                self.strip_nodes.append(nodes)
        self.rest = np.array(poses)
        self.total_mass = 2*self.count*self.plate_mass+sum(lengths)*p['strip_width_m']*p['strip_thickness_m']*p['steel_density_kg_m3']
        self.dofs = root.addObject('MechanicalObject', template='Rigid3d', name='DOFs', position=poses, rest_position=poses)
        root.addObject('EdgeSetTopologyContainer', name='Edges', edges=edges)
        root.addObject('BeamInterpolation', name='Beams', edgeList=list(range(len(edges))), lengthList=lengths,
                       DOF0TransformNode0=transform0, DOF1TransformNode1=transform1, dofsAndBeamsAligned=False,
                       straight=False, crossSectionShape='rectangular', lengthY=p['strip_width_m'],
                       lengthZ=p['strip_thickness_m'], defaultYoungModulus=p['youngs_modulus_pa']*self.factors[0],
                       defaultPoissonRatio=p['poisson_ratio'])
        root.addObject('AdaptiveBeamForceFieldAndMass', interpolation='@Beams', massDensity=p['steel_density_kg_m3'], computeMass=True)
        # Extra masses are assigned only to plate DOFs; beam mass is native FEM.
        root.addChild('HardwareMass').addObject('UniformMass', template='Rigid3d',
                           indices=self.plate_ids.tolist(),
                           vertexMass=f'{self.plate_mass} 1 .0016 0 0 0 .0009 0 0 0 .0009')
        if self.fixed:
            root.addObject('FixedProjectiveConstraint', indices=[0])
        self.anchors = np.array([[-.0165, y, zz] for y in (.0285, -.0285) for zz in (.039, -.039)])
        self.guides = self.anchors.copy(); self.guides[:, 0] = 0.
        self.rest_cable = .101
        Sofa.Simulation.init(root)
        assert self.dofs.position.array().shape == self.rest.shape

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.factors = self.np_random.uniform([.85, .8, .9, .8], [1.15, 1.2, 1.1, 1.2]) if self.randomize else np.ones(4)
        self.build()
        self.servo = np.zeros(2*self.count)
        self.yaw_target = np.zeros(self.count-1)
        self.wheel_speed = np.zeros(4*self.count)
        self.wheel_angle = np.zeros(4*self.count)
        self.contact_memory = np.zeros((4*self.count, 3))
        self.contact_slip = np.zeros((4*self.count, 2))
        self.previous = np.zeros(self.ndrive)
        self.steps = 0
        self.support = np.zeros(4*self.count)
        self.tensions = np.zeros((self.count, 4))
        for _ in range(round(.25/self.dt)):
            self.physics()
        self.start_x = float(self.dofs.position.array()[0, 0])
        self.start_center_x = float(self.dofs.position.array()[self.plate_ids, 0].mean())
        self.start_leading_x = float(self.dofs.position.array()[2*self.count-1, 0])
        return self.observation(), self.metrics()

    def physics(self):
        x = self.dofs.position.array()
        v = self.dofs.velocity.array()
        f = np.zeros_like(v)
        mats = np.array([rotation(x[i, 3:]) for i in self.plate_ids])
        # Axial tension-only elastic tendons, two per side, transmit equal/opposite forces.
        for seg in range(self.count):
            a, b = 2*seg, 2*seg+1
            ar = self.anchors @ mats[a].T
            br = self.guides @ mats[b].T
            vec = x[b, :3]+br-x[a, :3]-ar
            length = np.linalg.norm(vec, axis=1)
            direction = vec/length[:, None]
            va = v[a, :3]+np.cross(v[a, 3:], ar)
            vb = v[b, :3]+np.cross(v[b, 3:], br)
            rest = self.rest_cable-np.repeat(horn_takeup(self.servo[2*seg:2*seg+2]), 2)
            speed = np.sum((vb-va)*direction, axis=1)
            tension = np.where(length >= rest, np.maximum(0., 2000*(length-rest)+.5*speed), 0.)
            self.tensions[seg] = tension
            force = tension[:, None]*direction
            f[a, :3] += force.sum(axis=0); f[b, :3] -= force.sum(axis=0)
            f[a, 3:] += np.cross(ar, force).sum(axis=0)
            f[b, 3:] -= np.cross(br, force).sum(axis=0)
            # Candidate motor/plate stop: 10 mm motor clearance at 75 mm plate spacing.
            gap = np.dot(x[b, :3]-x[a, :3], -mats[a][:, 0])
            if gap < .075:
                push = 50000*(.075-gap)*(-mats[a][:, 0])
                f[b, :3] += push; f[a, :3] -= push
                f[a, 3:] -= np.cross(x[b, :3]-x[a, :3], push)
        for seg in range(self.count-1):
            a, b = 2*seg+1, 2*seg+2
            # Compliant revolute connector. Native JointSpringForceField lacks
            # addKToMatrix in this SOFA build; use explicit forces at small dt.
            arm = mats[a] @ np.array([-.058, 0., -.0010452742534])
            error = x[b, :3]-x[a, :3]-arm
            separation = x[b, :3]-x[a, :3]
            relative_v = v[b, :3]-v[a, :3]-np.cross(v[a, 3:], separation)
            force = -10000*error-10*relative_v
            f[b, :3] += force; f[a, :3] -= force
            f[a, 3:] -= np.cross(separation, force)
            relative = mats[a].T @ mats[b]
            yaw = math.atan2(relative[1, 0], relative[0, 0])
            axis = mats[a][:, 2]
            torque = np.clip(2*(self.yaw_target[seg]-yaw)-.03*np.dot(v[b, 3:]-v[a, 3:], axis), -.5, .5)*axis
            omega = v[b, 3:]-v[a, 3:]
            torque += 20*np.cross(mats[b][:, 2], axis)-.1*(omega-axis*np.dot(omega, axis))
            f[b, 3:] += torque; f[a, 3:] -= torque
        self.internal_force_residual = float(np.linalg.norm(f[:, :3].sum(axis=0)))
        self.internal_torque_residual = float(np.linalg.norm((np.cross(x[:, :3], f[:, :3])+f[:, 3:]).sum(axis=0)))
        # Analytic wheel/plane contact: real load-dependent forces, no root-position prescription.
        self.support.fill(0.)
        for plate in self.plate_ids:
            R = mats[plate]
            axle = R[:, 1]
            forward = np.cross([0., 0., 1.], axle); forward /= max(1e-9, np.linalg.norm(forward))
            lateral = np.cross([0., 0., 1.], forward)
            normal_in_plane = np.array([0., 0., 1.])-axle*axle[2]
            rim = -self.radius*normal_in_plane/max(1e-9, np.linalg.norm(normal_in_plane))
            for side, mount in enumerate(self.wheel_mounts[plate % 2]):
                wheel = 2*plate+side
                offset = R@mount
                center = x[plate, :3]+offset
                velocity = v[plate, :3]+np.cross(v[plate, 3:], offset+rim)
                penetration = -(center[2]+rim[2])
                if penetration <= 0:
                    self.contact_memory[wheel] = 0.; self.contact_slip[wheel] = 0.
                    self.wheel_speed[wheel] *= max(0., 1-self.dt*2e-6/(.5*.009*self.radius**2))
                    self.wheel_angle[wheel] += self.dt*self.wheel_speed[wheel]
                    continue
                normal = max(0., 5000*penetration-8*velocity[2])
                self.support[wheel] = normal
                basis = np.array([forward, lateral])
                tangent, omega, memory, reaction, slip, _ = wheel_contact(
                    basis@velocity, self.wheel_speed[wheel], basis@self.contact_memory[wheel],
                    normal, self.dt, radius=self.radius, mu=.8*self.factors[3])
                self.wheel_speed[wheel] = omega
                self.contact_memory[wheel] = memory@basis
                self.contact_slip[wheel] = slip
                self.wheel_angle[wheel] += self.dt*self.wheel_speed[wheel]
                force = tangent@basis+np.array([0., 0., normal])
                f[plate, :3] += force
                rim_torque = np.cross(rim, force)
                f[plate, 3:] += np.cross(offset, force)+rim_torque-axle*np.dot(rim_torque, axle)+reaction*axle
        self.dofs.externalForce = f.tolist()
        Sofa.Simulation.animate(self.root, self.dt)
        if not np.isfinite(self.dofs.position.array()).all():
            raise RuntimeError('SOFA nonfinite whole-body state')

    def metrics(self):
        x = self.dofs.position.array()
        plate_z = x[self.plate_ids, 2]
        beam_z = x[2*self.count:, 2]
        clearance = min(float(plate_z.min()-.055), float(beam_z.min()-.008))
        gaps = [float(np.linalg.norm(x[2*i+1, :3]-x[2*i, :3])) for i in range(self.count)]
        return {'wheel_support_ratio': float(np.mean(self.support > .05)), 'support_n': float(self.support.sum()),
                'body_clearance_m': clearance, 'min_plate_spacing_m': min(gaps),
                'forward_m': getattr(self, 'start_center_x', float(x[self.plate_ids, 0].mean()))-float(x[self.plate_ids, 0].mean()),
                'leading_forward_m': getattr(self, 'start_leading_x', float(x[2*self.count-1, 0]))-float(x[2*self.count-1, 0]),
                'mean_contact_slip_m_s': float(np.mean(np.linalg.norm(self.contact_slip[self.support>.05],axis=1))) if np.any(self.support>.05) else 0.,
                'max_tension_n': float(self.tensions.max()),
                'servo_max_deg': float(np.degrees(self.servo.max())),
                'internal_force_residual_n': self.internal_force_residual,
                'internal_torque_residual_nm': self.internal_torque_residual,
                'connector_error_m': max([float(np.linalg.norm(x[2*i+2, :3]-x[2*i+1, :3]-rotation(x[2*i+1, 3:])@np.array([-.058, 0., -.0010452742534]))) for i in range(self.count-1)] or [0.])}

    def observation(self):
        x, v = self.dofs.position.array(), self.dofs.velocity.array()
        shape, yaw, yaw_v = [], [], []
        for seg in range(self.count):
            a, b = 2*seg, 2*seg+1
            shape += [float(np.linalg.norm(x[b, :3]-x[a, :3]))/.12,
                      float(v[b, 0]-v[a, 0])/.1]
        for seg in range(self.count-1):
            a, b = 2*seg+1, 2*seg+2
            relative = rotation(x[a, 3:]).T @ rotation(x[b, 3:])
            yaw.append(math.atan2(relative[1, 0], relative[0, 0])); yaw_v.append(v[b, 5]-v[a, 5])
        root = np.r_[x[0, :3]-np.array([getattr(self, 'start_x', x[0, 0]), 0., 0.]), x[0, 3:], v[0]]
        phase = 2*np.pi*.25*self.steps*.02
        return np.r_[root, shape, self.servo/np.pi, yaw, yaw_v, self.wheel_speed*.01, self.previous,
                     np.sin(phase), np.cos(phase)].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.ndrive,) or not np.isfinite(action).all():
            raise ValueError('Invalid whole-body action')
        action = np.clip(action, -1, 1)
        target = np.clip(self.servo+action[:2*self.count]*.08, 0., np.pi)
        self.yaw_target = np.clip(self.yaw_target+action[2*self.count:]*.03, -.4, .4)
        before = self.dofs.position.array()[2*self.count-1, :3].copy()
        for _ in range(round(.02/self.dt)):
            load = self.tensions.reshape(-1, 2).sum(axis=1)*horn_lever(self.servo)
            increment = np.clip(target-self.servo, -2*self.dt, 2*self.dt)
            # Candidate .5 Nm motor limit: stall rather than silently clipping
            # cable force while the horn keeps rotating through a physical stop.
            increment *= np.where(increment > 0, np.clip(1-load/.5, 0., 1.), 1.)
            self.servo += increment
            self.physics()
        self.steps += 1
        info = self.metrics()
        delta = (self.dofs.position.array()[2*self.count-1, :3]-before)/.02
        vx = -float(delta[0])
        info['forward_velocity_m_s'] = vx
        bad = info['body_clearance_m'] < .005 or info['min_plate_spacing_m'] < .065
        bad = bad or info['connector_error_m'] > .005
        reward = vx-.1*abs(float(delta[1]))-.001*float(np.square(action-self.previous).sum())-float(bad)
        self.previous = action.copy()
        return self.observation(), reward, bool(bad), self.steps >= self.episode_steps, info


def validate(out):
    """Gate a candidate training run; no claim of experimental calibration."""
    import time
    import hashlib
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    env = SofaWormEnv(randomize=False)
    records, poses, controls, phases, wheel_angles = [], [], [], [], []
    report = {'passed': False, 'scope': 'Coupled SOFA candidate, uncalibrated parameters'}
    report['sha256'] = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                        for name in ('sofa_worm_env.py', 'parameters_sofa_candidate.json')}
    try:
        contact_tests = {}
        for name, drive, initial_velocity in [('locked', -.5, 0.), ('overload', -2., 0.), ('coast', 0., .1)]:
            velocity, position, spin, memory = initial_velocity, 0., initial_velocity/.018, np.zeros(2)
            for step in range(5000):
                force, spin, memory, _, _, _ = wheel_contact(np.array([velocity, 0.]), spin, memory, 1.4, .0005)
                velocity += .0005*(drive+force[0])/.14
                position += .0005*velocity
            contact_tests[name] = {'velocity_m_s': velocity, 'position_m': position}
        assert abs(contact_tests['locked']['velocity_m_s']) < 1e-5, contact_tests
        assert abs(contact_tests['locked']['position_m']) < .001, contact_tests
        assert contact_tests['overload']['velocity_m_s'] < -.01, contact_tests
        assert abs(contact_tests['coast']['velocity_m_s']) < .002, contact_tests
        report['wheel_unit_tests'] = contact_tests
        obs, _ = env.reset(seed=7)
        assert obs.shape == (77,) and env.action_space.shape == (14,)
        initial = env.dofs.position.array()[:10, :3].copy()
        for phase, steps in [('rest', 50), ('contract', 120), ('release', 120), ('steer', 40)]:
            for k in range(steps):
                action = np.zeros(14)
                if phase == 'contract': action[:10] = 1.
                if phase == 'release': action[:10] = -1.
                if phase == 'steer': action[10:] = [.2, -.2, .2, -.2]
                obs, _, done, _, info = env.step(action)
                records.append(info); poses.append(env.dofs.position.array().copy())
                controls.append(env.servo.copy()); phases.append(phase)
                wheel_angles.append(env.wheel_angle.copy())
                assert np.isfinite(obs).all() and not done, (phase, k, info)
                assert info['min_plate_spacing_m'] > .073, ('stop penetration', info)
                assert info['internal_force_residual_n'] < 1e-8 and info['internal_torque_residual_nm'] < 1e-8
                if k % 40 == 0: print(phase, k, info, flush=True)
            if phase == 'rest':
                report['rest_drift_m'] = abs(info['forward_m'])
                report['weight_n'] = env.total_mass*9.81
                report['support_n'] = info['support_n']
                assert report['rest_drift_m'] < .001
                assert abs(info['support_n']/report['weight_n']-1) < .05
            if phase == 'contract':
                report['contraction_m'] = .1175-info['min_plate_spacing_m']
                report['loaded_servo_deg'] = info['servo_max_deg']
                assert .025 < report['contraction_m'] < .045
            if phase == 'release':
                report['recovered_spacing_m'] = info['min_plate_spacing_m']
                assert abs(info['min_plate_spacing_m']-.1175) < .003
        x = env.dofs.position.array()
        turn = rotation(x[1, 3:]).T@rotation(x[2, 3:])
        report['steering_response_rad'] = math.atan2(turn[1, 0], turn[0, 0])
        assert abs(report['steering_response_rad']) > .01
        for seed in (11, 22):
            env.randomize = True
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            for k in range(80):
                if k % 10 == 0: action = rng.uniform(-1., 1., 14)
                obs, _, done, _, info = env.step(action)
                assert np.isfinite(obs).all() and not done, (seed, k, info)
        env.randomize = False
        obs, _ = env.reset(seed=7)
        demo_obs, demo_actions, cycle_poses, cycle_angles, cycle_servo, cycle_metrics = [], [], [], [], [], []
        cycle_distance, cycle_gaps = [], []
        last_distance = 0.
        for step in range(600):
            # A tested initialization demonstration, not a learned policy.
            target = .9*(1-np.cos(2*np.pi*.25*(step+1)*.02))
            action = np.r_[np.clip((target-env.servo)/.08, -1., 1.), np.zeros(4)]
            demo_obs.append(obs.copy()); demo_actions.append(action.copy())
            obs, _, done, _, info = env.step(action)
            assert not done and np.isfinite(obs).all(), ('cycle', step, info)
            cycle_poses.append(env.dofs.position.array().copy()); cycle_angles.append(env.wheel_angle.copy())
            cycle_servo.append(env.servo.copy()); cycle_metrics.append(info)
            if (step+1)%200 == 0:
                distance = info['leading_forward_m']
                cycle_distance.append(distance-last_distance); last_distance = distance
                cycle_gaps.append(info['min_plate_spacing_m'])
                print('CYCLE',step+1,info,flush=True)
        report['cycle_leading_advance_m'] = cycle_distance
        report['cycle_end_spacing_m'] = cycle_gaps
        assert min(cycle_distance) > .02, report
        assert max(abs(g-.1175) for g in cycle_gaps) < .004, report
        np.savez_compressed(out/'demonstration.npz', observations=demo_obs, actions=demo_actions)
        np.savez_compressed(out/'cycle_rollout.npz', poses=cycle_poses, servo=cycle_servo, phases=['cycle']*600,
                            strip_nodes=env.strip_nodes, anchors=env.anchors, guides=env.guides,
                            wheel_mounts=env.wheel_mounts, wheel_holes=env.wheel_holes, wheel_angle=cycle_angles,
                            width=env.params['strip_width_m'], thickness=env.params['strip_thickness_m'])
        (out/'cycle_metrics.json').write_text(json.dumps(cycle_metrics,indent=2))
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
    finally:
        report['wall_s'] = time.perf_counter()-start
        report['min_clearance_m'] = min([i['body_clearance_m'] for i in records] or [0.])
        report['max_connector_error_m'] = max([i['connector_error_m'] for i in records] or [0.])
        if poses:
            np.savez_compressed(out/'whole_rollout.npz', poses=poses, servo=controls, phases=phases,
                            strip_nodes=env.strip_nodes, anchors=env.anchors, guides=env.guides,
                            wheel_mounts=env.wheel_mounts, wheel_holes=env.wheel_holes, wheel_angle=wheel_angles,
                            width=env.params['strip_width_m'],
                            thickness=env.params['strip_thickness_m'])
        (out/'metrics.json').write_text(json.dumps(records, indent=2))
        (out/'gate.json').write_text(json.dumps(report, indent=2))
        env.close()
    print(json.dumps(report), flush=True)
    if not report['passed']: raise RuntimeError('Whole-body validation failed: '+report.get('error', ''))
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--validate', type=Path)
    args = p.parse_args()
    if args.validate:
        validate(args.validate)
        raise SystemExit(0)
    env = SofaWormEnv(count=args.count, randomize=False)
    obs, info = env.reset(seed=7)
    print('RESET', obs.shape, info, flush=True)
    for step in range(30):
        obs, reward, done, _, info = env.step(np.zeros(env.ndrive))
        if step % 5 == 0:
            print('STEP', step, info, flush=True)
        assert np.isfinite(obs).all() and not done, info
    env.close()

"""Sensor-only residual control of worm, snake and mixed five-module gaits."""
import sofa_worm_env as physics  # SOFA must load before Conda numerical libraries.
import math
import numpy as np
import gymnasium as gym
from sensor_observation import make_env


class SnakeResidualEnv(gym.Wrapper):
    """10 Hz policy, 50 Hz sensor/controller, 2 kHz physics.

    Tendon actions change the amplitude of a prescribed backward travelling wave;
    no true segment length enters the controller. Phase ordering is a prior, not
    a guarantee about physical contraction peaks under load.
    """
    def __init__(self, randomize=True, mode='snake', beam_model='native', rolling=.015, elements=8):
        if mode not in ('worm', 'snake', 'mixed'):
            raise ValueError('Unknown gait mode')
        super().__init__(make_env(randomize=randomize, beam_model=beam_model, rolling=rolling, elements=elements))
        self.mode = mode
        n = {'worm': 5, 'snake': 4, 'mixed': 9}[mode]
        self.action_space = gym.spaces.Box(-1., 1., (n,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (50+n,), dtype=np.float32)
        self.filtered = np.zeros(n)
        self.last_action = np.zeros(n)

    def reset(self, *, seed=None, options=None):
        # Keep measurement noise reproducible across vector-env automatic resets.
        if seed is None:
            seed = int(self.unwrapped.np_random.integers(0, 2**31-1))
        obs, info = self.env.reset(seed=seed, options=options)
        self.filtered.fill(0.); self.last_action.fill(0.)
        self.elapsed = 0.; self.slip_integral = 0.; self.work_proxy = 0.
        self.servo_work_proxy = 0.
        self.peak_tension = 0.; self.peak_joint = 0.; self.episode_return = 0.
        self.initial_y = float(self.unwrapped.dofs.position.array()[:10,1].mean())
        return np.r_[obs, self.filtered].astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid normalized gait residual commands')
        action = np.clip(action, -1, 1)
        reward = -.01*float(np.mean((action-self.last_action)**2))
        core = self.unwrapped
        before = core.dofs.position.array()[:10,:3].mean(axis=0)
        for _ in range(5):
            self.filtered += .1*(action-self.filtered)
            t = core.steps*.02
            ramp = .5*(1-math.cos(math.pi*min(t,1.)))
            target = np.zeros(4)
            motor = np.zeros(10)
            if self.mode != 'worm':
                base = math.radians(20)*ramp*np.sin(2*np.pi*t/4.-(3-np.arange(4))*np.pi/2)
                target = np.clip(base+math.radians(5)*self.filtered[-4:], -math.radians(20), math.radians(20))
            if self.mode != 'snake':
                age = t-(4-np.arange(5))*.65
                pulse = np.where(age>=0, .5*(1-np.cos(2*np.pi*np.maximum(age,0)/4.)), 0.)*ramp
                motor = np.repeat(1.5*pulse*(1+.2*self.filtered[:5]), 2)
            command = np.r_[np.clip((motor-core.servo)/.08,-1,1), np.clip((target-core.yaw_target)/.03, -1, 1)]
            old_servo = core.servo.copy()
            old_load = core.tensions.reshape(-1,2).sum(axis=1)*physics.horn_lever(core.servo)
            obs, _, done, truncated, info = self.env.step(command)
            x, v = core.dofs.position.array(), core.dofs.velocity.array()
            after = x[:10,:3].mean(axis=0)
            power = 0.; peak = 0.
            for j in range(4):
                a,b = 2*j+1,2*j+2
                R = physics.rotation(x[a,3:]); relative = R.T@physics.rotation(x[b,3:])
                yaw = math.atan2(relative[1,0],relative[0,0])
                speed = float(np.dot(v[b,3:]-v[a,3:],R[:,2]))
                torque = float(np.clip(2*(core.yaw_target[j]-yaw)-.03*speed,
                                       -core.yaw_servo_torque_limit_nm, core.yaw_servo_torque_limit_nm))
                power += abs(torque*speed); peak = max(peak,abs(yaw))
            slip = info['mean_contact_slip_m_s']
            load = core.tensions.reshape(-1,2).sum(axis=1)*physics.horn_lever(core.servo)
            servo_work = float(np.sum(np.abs(.5*(old_load+load)*(core.servo-old_servo))))
            # Reward-only simulator truth never enters the policy packet.
            reward += 20*(before[0]-after[0])-10*abs(after[1]-before[1])
            reward -= .02*(50*slip+.2*power+.05*float(np.mean(self.filtered**2)))
            reward -= .2*servo_work
            self.servo_work_proxy += servo_work
            self.elapsed += .02; self.slip_integral += .02*slip; self.work_proxy += .02*power
            self.peak_joint = max(self.peak_joint,peak)
            self.peak_tension = max(self.peak_tension,info['max_tension_n'])
            before = after.copy()
            if done or truncated:
                break
        reward -= 2*float(done)
        self.last_action = action.copy(); self.episode_return += reward
        info.update(physical_s=self.elapsed, mean_plate_forward_m=info['forward_m'],
            lateral_m=float(after[1]-self.initial_y),
            mean_sampled_slip_m_s=self.slip_integral/self.elapsed,
            joint_abs_work_proxy_j=self.work_proxy, peak_joint_deg=math.degrees(self.peak_joint),
            servo_abs_work_proxy_j=self.servo_work_proxy,
            total_abs_work_proxy_j=self.work_proxy+self.servo_work_proxy,
            peak_tension_n=self.peak_tension, task_return=self.episode_return,
            # A simulator evaluation criterion, not a hardware safety claim.
            success=bool(truncated and not done and info['forward_m']>.02 and abs(after[1]-self.initial_y)<.1))
        if self.mode == 'snake':
            assert np.all(core.servo == 0)
        if self.mode == 'worm':
            assert np.all(core.yaw_target == 0)
        return np.r_[obs,self.filtered].astype(np.float32), float(reward), done, truncated, info


def check(baseline, out):
    import hashlib
    import json
    from pathlib import Path
    from stable_baselines3.common.env_checker import check_env
    env = SnakeResidualEnv(randomize=False)
    try:
        obs,_ = env.reset(seed=7)
        initial = env.unwrapped.dofs.position.array().copy()
        truth = np.load(Path(baseline)/'whole_rollout.npz')
        initial_error=float(np.max(np.abs(initial-truth['initial_poses'])))
        # Loading the training numerical libraries changes rounding at ~4e-13.
        assert obs.shape==(54,) and initial_error<1e-10, initial_error
        max_error = 0.
        for k in range(120):
            obs,reward,done,truncated,info=env.step(np.zeros(4))
            assert np.isfinite(obs).all() and math.isfinite(reward) and not done
            max_error=max(max_error,float(np.max(np.abs(env.unwrapped.dofs.position.array()-truth['poses'][5*k+4]))))
        assert truncated and info['success'] and max_error<1e-8, (info,max_error)
        baseline_result=info.copy()
        # Real Gym/SB3 boundary check, including random actions and automatic reset.
        check_env(env,warn=True)
        for seed in (31,32):
            env.unwrapped.randomize=True
            obs,_=env.reset(seed=seed)
            rng=np.random.default_rng(seed)
            for _ in range(5):
                obs,reward,done,_,info=env.step(rng.uniform(-1,1,4))
                assert not done and np.isfinite(obs).all() and math.isfinite(reward)
        here=Path(__file__).resolve().parent
        report=dict(passed=True,initial_max_pose_error=initial_error,
            zero_residual_max_pose_error=max_error,baseline=baseline_result,
            raw_observation_dim=54,policy_history_dim=216,action_dim=4,
            source_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
                ['sofa_worm_env.py','sensor_observation.py','snake_rl_env.py','parameters_sofa_candidate.json']})
        Path(out).write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(); check(a.baseline,a.out)

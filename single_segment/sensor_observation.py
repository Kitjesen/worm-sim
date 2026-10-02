"""Deployable input boundary: encoder angles, one IMU, commands and clock only.

Run this file locally for the dependency-light check. On the SOFA host use
make_env(randomize=True) in place of SofaWormEnv for a NEW policy (50 inputs).
"""
from collections import deque
import importlib.util
import math
# On the SOFA host load its C++ runtime before NumPy/Gym load Conda libraries.
if importlib.util.find_spec('Sofa') is not None:
    import Sofa
import numpy as np


class SensorObservation:
    def __init__(self, dt=.02, delay_steps=1, angle_noise=.002,
                 gyro_noise=.01, accel_noise=.1):
        values = np.array([dt, angle_noise, gyro_noise, accel_noise], float)
        if not np.isfinite(values).all() or dt <= 0 or min(values[1:]) < 0:
            raise ValueError('Invalid sampling period or sensor noise')
        if not isinstance(delay_steps, int) or not 0 <= delay_steps <= 100:
            raise ValueError('Invalid sensor delay')
        self.dt, self.delay_steps = dt, delay_steps
        self.noise = np.r_[np.full(14, angle_noise), np.full(3, gyro_noise), np.full(3, accel_noise)]
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.queue = deque(maxlen=self.delay_steps+1)
        self.previous_angles = None

    def sample(self, angles, gyro, acceleration, previous_action, time_s):
        """Angles rad; body gyro rad/s; body specific force m/s²; action in [-1,1]."""
        arrays = [np.asarray(a, float) for a in (angles, gyro, acceleration, previous_action)]
        if [a.shape for a in arrays] != [(14,), (3,), (3,), (14,)]:
            raise ValueError('Expected 14 encoder angles, 3 gyro, 3 accel, 14 actions')
        if not np.isfinite(np.r_[*arrays, time_s]).all() or time_s < 0:
            raise ValueError('Invalid sensor packet')
        packet = np.r_[*arrays[:3]] + self.rng.normal(size=20)*self.noise
        if not self.queue:
            self.queue.extend([packet.copy() for _ in range(self.delay_steps)])
        self.queue.append(packet)
        delayed = self.queue[0]
        angles = delayed[:14]
        rate = np.zeros(14) if self.previous_angles is None else (angles-self.previous_angles)/self.dt
        # Rotary joint differences wrap; tendon servo angles have bounded travel.
        if self.previous_angles is not None:
            delta = angles[10:]-self.previous_angles[10:]
            rate[10:] = np.arctan2(np.sin(delta), np.cos(delta))/self.dt
        self.previous_angles = angles.copy()
        phase = 2*math.pi*time_s/4
        return np.r_[angles/math.pi, rate/5, delayed[14:17]/5,
                     delayed[17:20]/9.81, arrays[3], math.sin(phase), math.cos(phase)].astype(np.float32)


def make_env(*, sensor_settings=None, **physics_settings):
    """SOFA adapter. Privileged state remains available for rewards/logs, not policy input."""
    import gymnasium as gym
    from sofa_worm_env import SofaWormEnv, rotation

    class SensorEnv(gym.Wrapper):
        def __init__(self):
            super().__init__(SofaWormEnv(**physics_settings))
            if self.env.count != 5:
                raise ValueError('This sensor layout is for the five-module robot')
            self.sensors = SensorObservation(**(sensor_settings or {}))
            if self.sensors.dt != .02:
                raise ValueError('SOFA control sampling is fixed at 20 ms')
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (50,), dtype=np.float32)

        def packet(self, reset=False):
            env = self.env
            x, v = env.dofs.position.array(), env.dofs.velocity.array()
            R = rotation(x[0, 3:])
            angles = list(env.servo)
            for j in range(4):
                relative = rotation(x[2*j+1, 3:]).T@rotation(x[2*j+2, 3:])
                angles.append(math.atan2(relative[1, 0], relative[0, 0]))
            # ponytail: IMU at plate-0 origin, interval-average acceleration;
            # add measured mounting offset and sensor bandwidth for hardware transfer.
            acceleration = np.zeros(3) if reset else (v[0, :3]-self.last_velocity)/.02
            self.last_velocity = v[0, :3].copy()
            return self.sensors.sample(angles, R.T@v[0, 3:],
                R.T@(acceleration-np.array([0., 0., -9.81])), env.previous, env.steps*.02)

        def reset(self, *, seed=None, options=None):
            _, info = self.env.reset(seed=seed, options=options)
            self.sensors.reset(seed)
            return self.packet(reset=True), info

        def step(self, action):
            _, reward, done, truncated, info = self.env.step(action)
            return self.packet(), reward, done, truncated, info

    return SensorEnv()


def check():
    sensor = SensorObservation(delay_steps=1, angle_noise=0, gyro_noise=0, accel_noise=0)
    a, u = np.zeros(14), np.zeros(14)
    first = sensor.sample(a, [0, 0, 0], [0, 0, 9.81], u, 0)
    assert first.shape == (50,) and first[33] == 1 and np.isfinite(first).all()
    a[0] = .1
    delayed = sensor.sample(a, [1, 0, 0], [0, 0, 9.81], u, .02)
    received = sensor.sample(a, [1, 0, 0], [0, 0, 9.81], u, .04)
    assert delayed[0] == 0 and np.isclose(received[0], .1/math.pi)
    assert np.isclose(received[14], 1) and np.isclose(received[28], .2)
    sensor.reset(7)
    assert np.all(sensor.sample(a, [0]*3, [0]*3, u, 0)[14:28] == 0)
    try:
        sensor.sample(a[:-1], [0]*3, [0]*3, u, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed packet accepted')
    print('PASS: 50 sensor-only inputs, delay, differentiation, reset and validation')


if __name__ == '__main__':
    check()
    import sys
    if '--sofa-check' in sys.argv:
        env = make_env(randomize=False)
        try:
            observation, _ = env.reset(seed=7)
            assert observation.shape == (50,)
            for _ in range(20):
                observation, _, done, _, _ = env.step(np.zeros(14))
                assert observation.shape == (50,) and np.isfinite(observation).all() and not done
            print('PASS: SOFA integration, 20 control steps, 50 observations')
        finally:
            env.close()

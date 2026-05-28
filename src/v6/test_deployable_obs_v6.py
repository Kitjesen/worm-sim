"""
Smoke tests for the deployable WormEnvV6 observation contract.

This is intentionally lightweight: it validates the observation interface used
by the deployable RL policy without running training.
"""

import inspect
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from worm_env_v6 import (  # noqa: E402
    GAIT_BLENDS,
    NUM_ACTUATORS,
    NUM_IMUS,
    OBS_DIM,
    OBS_LAYOUT,
    PERISTALTIC_ACTUATION_PERIOD_S,
    PHASE_FREQ,
    WormEnvV6,
)


PAPER_TERRAINS = ("flat", "sand", "slope")
PAPER_MODES = ("worm", "snake", "mixed")


def assert_layout_contract():
    assert OBS_DIM == 80
    assert OBS_LAYOUT["phase_clock"].stop == OBS_DIM
    assert "base_linvel" not in OBS_LAYOUT
    assert np.isclose(PERISTALTIC_ACTUATION_PERIOD_S, 1.0)
    assert np.isclose(PHASE_FREQ, 1.0)

    source = inspect.getsource(WormEnvV6._get_obs)
    forbidden = ("base_linvel", "qvel[0:3]", "xpos[")
    for token in forbidden:
        assert token not in source, f"deployable obs leaks privileged token: {token}"


def assert_env_contract(terrain, mode):
    env = WormEnvV6(terrain=terrain, gait_mode=mode)
    try:
        obs, _ = env.reset(seed=123)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))

        cmd = obs[OBS_LAYOUT["command"]]
        assert np.isclose(cmd[2], GAIT_BLENDS[mode])

        joint_pos = obs[OBS_LAYOUT["joint_pos"]]
        joint_vel = obs[OBS_LAYOUT["joint_vel"]]
        last_action = obs[OBS_LAYOUT["previous_action"]]
        gravity = obs[OBS_LAYOUT["segment_gravity"]].reshape(NUM_IMUS, 3)
        gyro = obs[OBS_LAYOUT["segment_gyro"]].reshape(NUM_IMUS, 3)

        assert joint_pos.shape == (NUM_ACTUATORS,)
        assert joint_vel.shape == (NUM_ACTUATORS,)
        assert last_action.shape == (NUM_ACTUATORS,)
        assert gravity.shape == (NUM_IMUS, 3)
        assert gyro.shape == (NUM_IMUS, 3)
        assert np.allclose(np.linalg.norm(gravity, axis=1), 1.0, atol=1e-3)

        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))
        assert not truncated
        if terminated:
            obs, _ = env.reset(seed=456)
            assert obs.shape == (OBS_DIM,)
    finally:
        env.close()


def main():
    assert_layout_contract()
    for terrain in PAPER_TERRAINS:
        for mode in PAPER_MODES:
            assert_env_contract(terrain, mode)
            print(f"ok terrain={terrain} mode={mode}")
    env = WormEnvV6(
        terrain="flat", gait_mode="mixed",
        encoder_pos_noise_std=0.01, encoder_vel_noise_std=0.02,
        imu_gravity_noise_std=0.01, imu_gyro_noise_std=0.01,
        action_delay_steps=2, action_saturation=0.5)
    try:
        obs, _ = env.reset(seed=789)
        for _ in range(10):
            obs, _, _, _, _ = env.step(
                np.ones(env.action_space.shape, dtype=np.float32))
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))
        assert np.max(np.abs(env._last_action)) <= 0.5 + 1e-6
        print("ok noisy_sensor_action_delay_saturation")
    finally:
        env.close()
    print("deployable observation contract passed")


if __name__ == "__main__":
    main()

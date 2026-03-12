"""
Worm Robot V5.1 — Gymnasium RL Environment
============================================
Wraps the V5.1 MuJoCo worm model as a standard Gymnasium environment
for reinforcement learning training.

Action space:  8-dim continuous [-1, 1] → scaled to joint ranges
               [slide0..3, yaw0..3]

Observation:   25-dim: joint_pos(8) + joint_vel(8) + projected_gravity(3)
               + base_angvel(3) + base_linvel(3)

Reward:        forward velocity - lateral drift - energy - heading error
               - action smoothness + alive bonus
"""

import os
import sys
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# Import model builder and constants from V5.1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worm_v5_1 import (
    build_xml, inject_strips,
    NUM_SEGMENTS, SLIDE_RANGE, YAW_RANGE,
    SEG_SPACING_01, SEG_SPACING_REST,
)

# ─── Environment constants ────────────────────────────────────────────────────
N_ACTUATORS = (NUM_SEGMENTS - 1) * 2       # 4 slide + 4 yaw = 8
N_SLIDES    = NUM_SEGMENTS - 1              # 4
N_YAWS      = NUM_SEGMENTS - 1              # 4
OBS_DIM     = 27                            # 25 base + 2 phase clock
CTRL_DT     = 0.02                          # 50 Hz control frequency
PHASE_FREQ  = 0.5                           # Hz — locomotion rhythm clock
PHYSICS_DT  = 0.002                         # 500 Hz physics (from XML)
N_FRAMES    = int(CTRL_DT / PHYSICS_DT)     # 10 physics steps per control step
MAX_EP_TIME = 20.0                          # seconds of simulation per episode
MAX_EP_STEPS = int(MAX_EP_TIME / CTRL_DT)   # 1000 steps
SETTLE_STEPS = 250                          # 0.5s settle after reset

# Reward weights — forward velocity dominates, smoothness prevents vibration
W_FORWARD   = 10.0      # main signal: m/s × 10 → ~0.43 per step at snake speed
W_LATERAL   = 0.5       # penalize sideways drift
W_HEADING   = 0.1       # penalize heading deviation
W_ENERGY    = 0.001     # small energy cost to discourage wasteful vibration
W_SMOOTH    = 0.1       # penalize rapid action changes (anti-vibration)
W_ALIVE     = 0.0       # DISABLED: alive bonus dominates and masks forward signal
ACTION_EMA  = 0.3       # EMA filter: action = α*raw + (1-α)*prev  (lower = smoother)


class WormEnv(gym.Env):
    """MuJoCo worm robot environment for RL training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── Build model ──
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        mesh_dir = os.path.join(project_root, "meshes")
        xml_str = build_xml(mesh_dir)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # ── Locate actuated joint indices in qpos / qvel ──
        self._act_qpos_idx = np.zeros(N_ACTUATORS, dtype=int)
        self._act_qvel_idx = np.zeros(N_ACTUATORS, dtype=int)
        for i in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            self._act_qpos_idx[i] = self.model.jnt_qposadr[jnt_id]
            self._act_qvel_idx[i] = self.model.jnt_dofadr[jnt_id]

        # ── Body IDs ──
        self._seg_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"seg{i}")
            for i in range(NUM_SEGMENTS)
        ]
        self._root_body_id = self._seg_ids[0]

        # ── Natural spacings for strip rendering ──
        self._natural_spacings = (
            [SEG_SPACING_01] + [SEG_SPACING_REST] * (NUM_SEGMENTS - 2)
        )

        # ── Gym spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_ACTUATORS,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # ── State tracking ──
        self._last_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

        # ── Renderer (lazy init) ──
        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Small random noise on actuated joints (helps exploration)
        if self.np_random is not None:
            for i in range(N_ACTUATORS):
                idx = self._act_qpos_idx[i]
                if i < N_SLIDES:
                    self.data.qpos[idx] += self.np_random.uniform(
                        -0.003, 0.003)  # ~3mm noise on slides
                else:
                    self.data.qpos[idx] += self.np_random.uniform(
                        -0.05, 0.05)    # ~3° noise on yaws

        # Brief settle to stabilize on ground
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self._last_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # EMA filter — physically limit action change rate to prevent vibration
        action = ACTION_EMA * action + (1.0 - ACTION_EMA) * self._last_action

        # Scale action to joint ranges
        # action[0:4] → slide targets, action[4:8] → yaw targets
        ctrl = np.zeros(N_ACTUATORS)
        ctrl[:N_SLIDES] = action[:N_SLIDES] * SLIDE_RANGE
        ctrl[N_SLIDES:] = action[N_SLIDES:] * YAW_RANGE
        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(N_FRAMES):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self._step_count >= MAX_EP_STEPS

        self._last_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 720, 1280)
            self._cam_lookat_smooth = None

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 0.9
        cam.elevation = -25
        cam.azimuth = 135
        mid = np.mean([self.data.xpos[sid] for sid in self._seg_ids], axis=0)
        # Smooth camera tracking
        if self._cam_lookat_smooth is None:
            self._cam_lookat_smooth = mid.copy()
        else:
            self._cam_lookat_smooth += 0.05 * (mid - self._cam_lookat_smooth)
        cam.lookat[:] = self._cam_lookat_smooth

        self._renderer.update_scene(self.data, cam)
        inject_strips(
            self._renderer.scene, self.data,
            self._seg_ids, self._natural_spacings)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        # Actuated joint positions (8): slide/SLIDE_RANGE, yaw/YAW_RANGE
        joint_pos = np.zeros(N_ACTUATORS, dtype=np.float32)
        for i in range(N_ACTUATORS):
            joint_pos[i] = self.data.qpos[self._act_qpos_idx[i]]

        # Actuated joint velocities (8)
        joint_vel = np.zeros(N_ACTUATORS, dtype=np.float32)
        for i in range(N_ACTUATORS):
            joint_vel[i] = self.data.qvel[self._act_qvel_idx[i]]

        # Root body rotation matrix (world → body)
        root_xmat = self.data.xmat[self._root_body_id].reshape(3, 3)

        # Projected gravity in body frame (3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        proj_gravity = root_xmat.T @ gravity_world

        # Base velocities in body frame
        # freejoint: qvel[0:3] = linear vel (world), qvel[3:6] = angular vel (world)
        base_linvel_world = self.data.qvel[0:3].copy()
        base_angvel_world = self.data.qvel[3:6].copy()
        base_linvel = root_xmat.T @ base_linvel_world
        base_angvel = root_xmat.T @ base_angvel_world

        # Phase clock — periodic signal for locomotion rhythm
        t = self._step_count * CTRL_DT
        phase = 2.0 * math.pi * PHASE_FREQ * t
        phase_clock = np.array([math.sin(phase), math.cos(phase)],
                               dtype=np.float32)

        obs = np.concatenate([
            joint_pos,                          # 8
            joint_vel,                          # 8
            proj_gravity.astype(np.float32),    # 3
            base_angvel.astype(np.float32),     # 3
            base_linvel.astype(np.float32),     # 3
            phase_clock,                        # 2
        ])
        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action):
        # Forward velocity (Y direction, world frame)
        forward_vel = self.data.qvel[1]

        # Lateral drift (X direction)
        lateral_vel = abs(self.data.qvel[0])

        # Heading error: angle between body Y-axis and world Y-axis
        root_xmat = self.data.xmat[self._root_body_id].reshape(3, 3)
        fwd_dir = root_xmat[:, 1]  # body Y in world frame
        heading_error = abs(math.atan2(fwd_dir[0], fwd_dir[1]))

        # Energy cost: sum |ctrl * joint_vel| for actuated joints
        energy = 0.0
        for i in range(self.model.nu):
            energy += abs(self.data.ctrl[i] * self.data.qvel[self._act_qvel_idx[i]])

        # Action smoothness
        action_rate = float(np.sum(np.square(action - self._last_action)))

        # Alive bonus
        alive = 1.0

        reward = (
            + W_FORWARD  * forward_vel
            - W_LATERAL  * lateral_vel
            - W_HEADING  * heading_error
            - W_ENERGY   * energy
            - W_SMOOTH   * action_rate
            + W_ALIVE    * alive
        )
        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # Termination
    # ──────────────────────────────────────────────────────────────────────

    def _check_termination(self):
        # Root body too low (fell over or collapsed)
        z = self.data.xpos[self._root_body_id][2]
        if z < 0.03:
            return True

        # Body flipped (Z-axis of body should point up)
        root_xmat = self.data.xmat[self._root_body_id].reshape(3, 3)
        up_component = root_xmat[2, 2]  # world-Z component of body-Z axis
        if up_component < 0.3:
            return True

        # NaN check
        if np.any(np.isnan(self.data.qpos)):
            return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== WormEnv self-test ===")
    env = WormEnv()
    print(f"  obs_space:    {env.observation_space.shape}")
    print(f"  action_space: {env.action_space.shape}")
    print(f"  model: bodies={env.model.nbody}, nv={env.model.nv}, nu={env.model.nu}")

    obs, info = env.reset(seed=42)
    print(f"  reset obs shape: {obs.shape}")
    print(f"  reset obs range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Run 100 random steps
    total_reward = 0.0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}: terminated={terminated}")
            obs, info = env.reset()
            break

    print(f"  100 random steps: total_reward={total_reward:.3f}")
    print(f"  final obs range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Check obs is finite
    assert np.all(np.isfinite(obs)), "Non-finite obs!"
    print("  All checks passed.")
    env.close()

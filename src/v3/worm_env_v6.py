"""
Worm Robot V6 — Gymnasium RL Environment (Command-Conditioned)
===============================================================
Wraps the V6 MuJoCo longworm2 model as a standard Gymnasium environment
for reinforcement learning training.

Action space:  11-dim continuous [-1, 1] → scaled to joint ranges
               [slide0..5, yaw0..4]

Observation:   35-dim: command(2) + joint_pos(11) + joint_vel(11)
               + projected_gravity(3) + base_angvel(3) + base_linvel(3)
               + phase_clock(2)

Command:       [v_forward_cmd, yaw_rate_cmd]
               - v_forward_cmd ∈ [0, 0.025] m/s   (forward speed target)
               - yaw_rate_cmd  ∈ [-0.3, 0.3] rad/s (turning rate target)

Reward:        velocity tracking (exp kernel) - energy - action smoothness

Forward direction: -X (chain extends in -X from base_link)
"""

import os
import sys
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# Import model builder and constants from V6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worm_v6 import (
    build_xml, inject_strips,
    NUM_SLIDES, NUM_YAWS, NUM_ACTUATORS,
    SLIDE_RANGE_VAL, YAW_RANGE_VAL,
    BODY_Z,
)

# ─── Environment constants ────────────────────────────────────────────────────
OBS_DIM     = 35                            # 2+11+11+3+3+3+2
CTRL_DT     = 0.02                          # 50 Hz control frequency
PHASE_FREQ  = 0.5                           # Hz — locomotion rhythm clock
PHYSICS_DT  = 0.002                         # 500 Hz physics (from XML)
N_FRAMES    = int(CTRL_DT / PHYSICS_DT)     # 10 physics steps per control step
MAX_EP_TIME = 20.0                          # seconds per episode
MAX_EP_STEPS = int(MAX_EP_TIME / CTRL_DT)   # 1000 steps
SETTLE_STEPS = 250                          # 0.5s settle after reset

# Command ranges (sampled randomly each episode)
CMD_VEL_RANGE   = (0.0, 0.025)   # m/s forward speed target (open-loop worm ~23 mm/s)
CMD_YAW_RANGE   = (-0.3, 0.3)    # rad/s yaw rate target (worm turns slowly)
CMD_RESAMPLE_P  = 0.005          # probability of resampling command each step

# Reward weights — velocity tracking with exponential kernel
W_VEL_TRACK = 2.0       # forward speed tracking: exp(-err²/σ²)
W_YAW_TRACK = 1.0       # yaw rate tracking: exp(-err²/σ²)
SIGMA_VEL   = 0.010     # m/s — ~40% of CMD_VEL range for good gradient
SIGMA_YAW   = 0.15      # rad/s
W_VEL_LIN   = 8.0       # capped+normalized forward bonus [0,1] — MAIN exploration driver
W_OVERSPEED = 5.0       # quadratic overspeed penalty (was 200, caused divergence)
W_LATERAL   = 2.0       # lateral drift penalty
W_BACKWARD  = 3.0       # backward motion penalty
W_ENERGY    = 0.002     # energy cost
W_SMOOTH    = 0.02      # low smoothness penalty (worm gait = fast alternating actions)
ACTION_EMA  = 0.3       # EMA filter coefficient


class WormEnvV6(gym.Env):
    """MuJoCo longworm2 robot environment for RL training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── Build model ──
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        mesh_dir = os.path.join(project_root, "meshes")
        urdf_path = os.path.join(mesh_dir, "longworm2", "longworm2.SLDASM.urdf")

        # Copy URDF from CAD export if not present locally
        if not os.path.exists(urdf_path):
            import shutil
            src_urdf = os.path.join(
                "D:/inovxio/3d/longworm2/longworm2.SLDASM/urdf",
                "longworm2.SLDASM.urdf")
            if os.path.exists(src_urdf):
                os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
                shutil.copy2(src_urdf, urdf_path)
            else:
                raise FileNotFoundError(
                    f"URDF not found at {src_urdf}. "
                    "Run worm_v6.py first to set up meshes.")

        xml_str = build_xml(mesh_dir, urdf_path)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # ── Locate actuated joint indices ──
        self._act_qpos_idx = np.zeros(NUM_ACTUATORS, dtype=int)
        self._act_qvel_idx = np.zeros(NUM_ACTUATORS, dtype=int)
        self._slide_act_ids = []
        self._yaw_act_ids = []

        for i in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            self._act_qpos_idx[i] = self.model.jnt_qposadr[jnt_id]
            self._act_qvel_idx[i] = self.model.jnt_dofadr[jnt_id]
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name.startswith('act_back'):
                self._slide_act_ids.append(i)
            elif name.startswith('act_front'):
                self._yaw_act_ids.append(i)

        # ── Body IDs ──
        def get_bid(name):
            return mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        self._root_body_id = get_bid('base_link')
        self._seg_ids = [get_bid('base_link')] + \
            [get_bid(f'back{i}_Link') for i in range(1, 7)]

        # ── Slide pairs and spacings for strip rendering ──
        self._slide_pairs = [
            (get_bid('base_link'),   get_bid('back1_Link')),
            (get_bid('front2_Link'), get_bid('back2_Link')),
            (get_bid('front3_Link'), get_bid('back3_Link')),
            (get_bid('front4_Link'), get_bid('back4_Link')),
            (get_bid('front5_Link'), get_bid('back5_Link')),
            (get_bid('front6_Link'), get_bid('back6_Link')),
        ]
        self._strip_spacings = [0.151] + [0.1175] * 5

        # ── Gym spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_ACTUATORS,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # ── State tracking ──
        self._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        self._step_count = 0

        # ── Command (sampled at reset) ──
        self._cmd_vel = 0.0       # forward speed target (m/s)
        self._cmd_yaw = 0.0       # yaw rate target (rad/s)

        # ── Renderer (lazy init) ──
        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Sample velocity command for this episode
        self._cmd_vel = self.np_random.uniform(*CMD_VEL_RANGE)
        self._cmd_yaw = self.np_random.uniform(*CMD_YAW_RANGE)

        # Small random noise on actuated joints
        if self.np_random is not None:
            for i in range(NUM_ACTUATORS):
                idx = self._act_qpos_idx[i]
                if i < NUM_SLIDES:
                    self.data.qpos[idx] += self.np_random.uniform(
                        -0.003, 0.003)   # ~3mm on slides
                else:
                    self.data.qpos[idx] += self.np_random.uniform(
                        -0.05, 0.05)     # ~3° on yaws

        # Settle to stabilize on ground
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self._last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Occasionally resample command mid-episode (curriculum diversity)
        if self.np_random.random() < CMD_RESAMPLE_P:
            self._cmd_vel = self.np_random.uniform(*CMD_VEL_RANGE)
            self._cmd_yaw = self.np_random.uniform(*CMD_YAW_RANGE)

        # EMA filter — anti-vibration
        action = ACTION_EMA * action + (1.0 - ACTION_EMA) * self._last_action

        # Scale action to joint ranges
        # action[0:6] → slide targets, action[6:11] → yaw targets
        ctrl = np.zeros(NUM_ACTUATORS)
        ctrl[:NUM_SLIDES] = action[:NUM_SLIDES] * SLIDE_RANGE_VAL
        ctrl[NUM_SLIDES:] = action[NUM_SLIDES:] * YAW_RANGE_VAL
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
        cam.distance = 2.0
        cam.elevation = -25
        cam.azimuth = 135
        mid = np.mean(
            [self.data.xpos[sid] for sid in self._seg_ids], axis=0)
        if self._cam_lookat_smooth is None:
            self._cam_lookat_smooth = mid.copy()
        else:
            self._cam_lookat_smooth += 0.05 * (
                mid - self._cam_lookat_smooth)
        cam.lookat[:] = self._cam_lookat_smooth

        self._renderer.update_scene(self.data, cam)
        inject_strips(
            self._renderer.scene, self.data,
            self._slide_pairs, self._strip_spacings)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        # Command vector (2) — at front so policy sees the goal first
        # Normalize: vel/max_vel, yaw/max_yaw → roughly [-1, 1]
        cmd = np.array([
            self._cmd_vel / max(CMD_VEL_RANGE[1], 1e-6),  # [0, 1]
            self._cmd_yaw / max(abs(CMD_YAW_RANGE[1]), 1e-6),  # [-1, 1]
        ], dtype=np.float32)

        # Actuated joint positions (11)
        joint_pos = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        for i in range(NUM_ACTUATORS):
            joint_pos[i] = self.data.qpos[self._act_qpos_idx[i]]

        # Actuated joint velocities (11)
        joint_vel = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        for i in range(NUM_ACTUATORS):
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
        phase_clock = np.array(
            [math.sin(phase), math.cos(phase)], dtype=np.float32)

        obs = np.concatenate([
            cmd,                                # 2  (command)
            joint_pos,                          # 11
            joint_vel,                          # 11
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
        # ── Actual velocities ──
        # Forward speed: -X direction in world frame
        forward_speed = -self.data.qvel[0]

        # Yaw rate: rotation around world Z axis
        yaw_rate = self.data.qvel[5]

        # Lateral speed: Y direction (for penalty)
        lateral_speed = abs(self.data.qvel[1])

        # ── Velocity tracking (exp kernel) ──
        vel_err = forward_speed - self._cmd_vel
        yaw_err = yaw_rate - self._cmd_yaw
        r_vel_track = math.exp(-(vel_err ** 2) / (SIGMA_VEL ** 2))
        r_yaw_track = math.exp(-(yaw_err ** 2) / (SIGMA_YAW ** 2))

        # ── Linear forward velocity bonus (capped + normalized) ──
        # Rewards forward movement UP TO target speed, no bonus beyond.
        # Normalized to [0,1] so weight W_VEL_LIN directly controls magnitude.
        r_vel_lin = (min(max(0.0, forward_speed), self._cmd_vel)
                     / max(CMD_VEL_RANGE[1], 1e-6))

        # ── Overspeed penalty (quadratic — strongly penalizes exceeding command) ──
        overspeed_sq = max(0.0, forward_speed - self._cmd_vel) ** 2

        # ── Other penalties ──
        backward_speed = max(0.0, -forward_speed)  # only penalize backward

        energy = 0.0
        for i in range(self.model.nu):
            energy += abs(
                self.data.ctrl[i] * self.data.qvel[self._act_qvel_idx[i]])

        action_rate = float(np.sum(np.square(action - self._last_action)))

        reward = (
            + W_VEL_TRACK * r_vel_track    # exp tracking (precision)
            + W_YAW_TRACK * r_yaw_track    # exp tracking (turning)
            + W_VEL_LIN   * r_vel_lin      # linear bonus (exploration gradient)
            - W_OVERSPEED * overspeed_sq   # penalize going faster than commanded
            - W_LATERAL   * lateral_speed   # penalize sideways drift
            - W_BACKWARD  * backward_speed  # penalize going backward
            - W_ENERGY    * energy
            - W_SMOOTH    * action_rate
        )
        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # Termination
    # ──────────────────────────────────────────────────────────────────────

    def _check_termination(self):
        # Root body too low
        z = self.data.xpos[self._root_body_id][2]
        if z < 0.03:
            return True

        # Body flipped
        root_xmat = self.data.xmat[self._root_body_id].reshape(3, 3)
        up_component = root_xmat[2, 2]
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
    print("=== WormEnvV6 self-test ===")
    env = WormEnvV6()
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

    assert np.all(np.isfinite(obs)), "Non-finite obs!"
    print("  All checks passed.")
    env.close()

"""
Worm Robot V6 — Gymnasium RL Environment (Command-Conditioned)
===============================================================
Wraps the V6 MuJoCo longworm2 model as a standard Gymnasium environment
for reinforcement learning training.

Action space:  11-dim continuous [-1, 1] → scaled to joint ranges
               [slide0..5, yaw0..4]

Observation:   80-dim deployable state:
               command(3) + joint_pos(11) + joint_vel(11)
               + previous_action(11) + segment_gravity(7*3)
               + segment_gyro(7*3) + phase_clock(2)

Command:       [v_forward_cmd, yaw_rate_cmd, gait_blend]
               - v_forward_cmd ∈ [0, 0.025] m/s   (forward speed target)
               - yaw_rate_cmd  ∈ [-0.3, 0.3] rad/s (turning rate target)

               - gait_blend    in [0, 1] (0=worm, 1=snake)

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
    build_xml, inject_strips, setup_terrain, TERRAIN_PRESETS,
    NUM_SLIDES, NUM_YAWS, NUM_ACTUATORS,
    SLIDE_RANGE_VAL, YAW_RANGE_VAL,
    BODY_Z, PERISTALTIC_ACTUATION_PERIOD_S,
)
from motor_contract_v6 import (
    neutral_normalized_action,
    normalized_action_to_ctrl,
)
from action_adapter_v6 import (
    DEFAULT_GAIT_PRIOR_SCALE,
    DEFAULT_POLICY_RESIDUAL_SCALE,
    action_adapter_contract,
    compose_deployable_action,
)

# ─── Environment constants ────────────────────────────────────────────────────
NUM_IMUS    = 7
OBS_DIM     = 80                            # 3+11+11+11+21+21+2
CTRL_DT     = 0.02                          # 50 Hz control frequency
PHASE_FREQ  = 1.0 / PERISTALTIC_ACTUATION_PERIOD_S
PHYSICS_DT  = 0.002                         # 500 Hz physics (from XML)
N_FRAMES    = int(CTRL_DT / PHYSICS_DT)     # 10 physics steps per control step
MAX_EP_TIME = 20.0                          # seconds per episode
MAX_EP_STEPS = int(MAX_EP_TIME / CTRL_DT)   # 1000 steps
SETTLE_STEPS = 250                          # 0.5s settle after reset

# Command ranges (sampled randomly each episode)
CMD_VEL_RANGE   = (0.0, 0.25)    # m/s forward speed target (CMA-ES full ~248 mm/s)
CMD_YAW_RANGE   = (-1.0, 1.0)    # rad/s yaw rate target for directional control
CMD_RESAMPLE_P  = 0.005          # probability of resampling command each step

GAIT_BLENDS = {
    "worm": 0.0,
    "peristaltic": 0.0,
    "mixed": 0.5,
    "combined": 0.5,
    "snake": 1.0,
    "serpentine": 1.0,
}
GAIT_MODES = tuple(list(GAIT_BLENDS.keys()) + ["random"])

SLIDE_VEL_SCALE = 0.10
YAW_VEL_SCALE = math.pi
IMU_GYRO_SCALE = 2.0 * math.pi

OBS_LAYOUT = {
    "command": slice(0, 3),
    "joint_pos": slice(3, 14),
    "joint_vel": slice(14, 25),
    "previous_action": slice(25, 36),
    "segment_gravity": slice(36, 57),
    "segment_gyro": slice(57, 78),
    "phase_clock": slice(78, 80),
}

# Reward weights — velocity tracking with exponential kernel
W_VEL_TRACK = 2.0       # forward speed tracking: exp(-err²/σ²)
W_YAW_TRACK = 1.0       # yaw rate tracking: exp(-err²/σ²)
SIGMA_VEL   = 0.010     # m/s — ~40% of CMD_VEL range for good gradient
SIGMA_YAW   = 0.15      # rad/s
W_VEL_LIN   = 8.0       # capped+normalized forward bonus [0,1] — MAIN exploration driver
W_OVERSPEED = 5.0       # quadratic overspeed penalty (was 200, caused divergence)
W_FORWARD_DEFICIT = 0.5 # soft penalty; cyclic gaits naturally back-slip
W_COMMAND_COST = 0.5    # cost for nonzero forward commands without progress
W_LATERAL   = 0.5       # normalized lateral drift penalty
W_BACKWARD  = 0.5       # normalized backward motion penalty
W_ENERGY    = 0.002     # energy cost
W_SMOOTH    = 0.02      # low smoothness penalty (worm gait = fast alternating actions)
ACTION_EMA  = 0.3       # EMA filter coefficient
REWARD_CONTRACT_VERSION = "forward_progress_v3"

# Override the older slow-tracking reward with the formal high-speed contract.
# Keeping the assignment block local makes old checkpoints incompatible through
# the reward contract without changing the deployable observation layout.
SIGMA_VEL = 0.050
SIGMA_YAW = 0.30
W_OVERSPEED = 2.0
W_FORWARD_DEFICIT = 0.5
W_COMMAND_COST = 1.0
W_LATERAL = 0.3
W_BACKWARD = 0.5
W_ENERGY = 0.001
W_SMOOTH = 0.01
REWARD_CONTRACT_VERSION = "high_speed_directional_v1"


def reward_contract():
    return {
        "version": REWARD_CONTRACT_VERSION,
        "forward_direction": "-world_x",
        "weights": {
            "vel_track": W_VEL_TRACK,
            "yaw_track": W_YAW_TRACK,
            "vel_lin": W_VEL_LIN,
            "overspeed": W_OVERSPEED,
            "forward_deficit": W_FORWARD_DEFICIT,
            "command_cost": W_COMMAND_COST,
            "lateral": W_LATERAL,
            "backward": W_BACKWARD,
            "energy": W_ENERGY,
            "smooth": W_SMOOTH,
        },
        "normalization": {
            "speed_scale_m_s": CMD_VEL_RANGE[1],
            "cmaes_full_combined_target_m_s": 0.24797,
            "positive_forward_required_for_vel_track": True,
            "yaw_tracking_gated_by_forward_progress": True,
            "cyclic_backslip_is_soft_penalized": True,
        },
    }


class WormEnvV6(gym.Env):
    """MuJoCo longworm2 robot environment for RL training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, terrain='flat',
                 gait_mode='random', gait_blend=None,
                 encoder_pos_noise_std=0.0, encoder_vel_noise_std=0.0,
                 imu_gravity_noise_std=0.0, imu_gyro_noise_std=0.0,
                 action_delay_steps=0, action_saturation=1.0,
                 fixed_cmd_vel=None, fixed_cmd_yaw=None,
                 command_resample_prob=CMD_RESAMPLE_P,
                 gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
                 policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
        super().__init__()
        self.render_mode = render_mode
        self.terrain = terrain
        self.gait_mode = gait_mode
        self._fixed_gait_blend = gait_blend
        self._fixed_cmd_vel = fixed_cmd_vel
        self._fixed_cmd_yaw = fixed_cmd_yaw
        self.command_resample_prob = float(command_resample_prob)
        self.gait_prior_scale = float(gait_prior_scale)
        self.policy_residual_scale = float(policy_residual_scale)
        self.encoder_pos_noise_std = float(encoder_pos_noise_std)
        self.encoder_vel_noise_std = float(encoder_vel_noise_std)
        self.imu_gravity_noise_std = float(imu_gravity_noise_std)
        self.imu_gyro_noise_std = float(imu_gyro_noise_std)
        self.action_delay_steps = int(action_delay_steps)
        self.action_saturation = float(action_saturation)

        if terrain not in TERRAIN_PRESETS:
            raise ValueError(f"Unknown terrain: {terrain}")
        if gait_mode not in GAIT_MODES:
            raise ValueError(
                f"Unknown gait_mode: {gait_mode}. Expected one of {GAIT_MODES}")
        if gait_blend is not None and not 0.0 <= gait_blend <= 1.0:
            raise ValueError("gait_blend must be in [0, 1]")
        if self.action_delay_steps < 0:
            raise ValueError("action_delay_steps must be >= 0")
        if not 0.0 <= self.action_saturation <= 1.0:
            raise ValueError("action_saturation must be in [0, 1]")

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

        xml_str = build_xml(mesh_dir, urdf_path, terrain=terrain)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        setup_terrain(self.model, terrain)
        self.data = mujoco.MjData(self.model)

        # Terrain-specific z termination threshold (rough/steps need lower floor)
        t_cfg = TERRAIN_PRESETS[terrain]
        self._z_term_lo = t_cfg.get('z_lo', 0.03) - 0.01

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
        self._imu_body_ids = self._seg_ids
        if len(self._imu_body_ids) != NUM_IMUS:
            raise RuntimeError(
                f"Expected {NUM_IMUS} segment IMUs, got {len(self._imu_body_ids)}")

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
        self._last_action = neutral_normalized_action()
        self._last_residual_action = neutral_normalized_action()
        self._action_delay_buffer = [
            neutral_normalized_action()
            for _ in range(self.action_delay_steps)
        ]
        self._step_count = 0

        # ── Command (sampled at reset) ──
        self._cmd_vel = 0.0       # forward speed target (m/s)
        self._cmd_yaw = 0.0       # yaw rate target (rad/s)
        self._gait_blend = 0.5    # 0=worm/peristaltic, 1=snake/serpentine

        # ── Renderer (lazy init) ──
        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Sample velocity command for this episode, unless fixed for eval.
        self._cmd_vel = self._sample_cmd_vel()
        self._cmd_yaw = self._sample_cmd_yaw()
        self._gait_blend = self._sample_gait_blend()

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

        self._last_action = neutral_normalized_action()
        self._last_residual_action = neutral_normalized_action()
        self._action_delay_buffer = [
            neutral_normalized_action()
            for _ in range(self.action_delay_steps)
        ]
        self._last_root_pos = self.data.xpos[self._root_body_id].copy()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Occasionally resample command mid-episode (curriculum diversity)
        if self.np_random.random() < self.command_resample_prob:
            self._cmd_vel = self._sample_cmd_vel()
            self._cmd_yaw = self._sample_cmd_yaw()

        # EMA filter — anti-vibration
        residual_action = (
            ACTION_EMA * action
            + (1.0 - ACTION_EMA) * self._last_residual_action)
        phase = 2.0 * math.pi * PHASE_FREQ * self._step_count * CTRL_DT
        applied_action = compose_deployable_action(
            residual_action,
            phase=phase,
            gait_blend=self._gait_blend,
            gait_prior_scale=self.gait_prior_scale,
            policy_residual_scale=self.policy_residual_scale,
        )
        if self.action_delay_steps > 0:
            self._action_delay_buffer.append(applied_action.copy())
            applied_action = self._action_delay_buffer.pop(0)
        applied_action = np.clip(
            applied_action,
            -self.action_saturation,
            self.action_saturation).astype(np.float32)

        # Scale action to deployable joint targets; slide targets are clipped.
        # action[0:6] → slide targets, action[6:11] → yaw targets
        ctrl = normalized_action_to_ctrl(applied_action)
        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(N_FRAMES):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(
            applied_action, residual_action=residual_action)
        terminated = self._check_termination()
        truncated = self._step_count >= MAX_EP_STEPS

        self._last_action = applied_action.copy()
        self._last_residual_action = residual_action.copy()
        self._last_root_pos = self.data.xpos[self._root_body_id].copy()
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

    def set_command(self, velocity=None, yaw_rate=None, gait_blend=None):
        """Override command values for deterministic evaluation."""
        if velocity is not None:
            self._cmd_vel = float(np.clip(
                velocity, CMD_VEL_RANGE[0], CMD_VEL_RANGE[1]))
        if yaw_rate is not None:
            self._cmd_yaw = float(np.clip(
                yaw_rate, CMD_YAW_RANGE[0], CMD_YAW_RANGE[1]))
        if gait_blend is not None:
            self._gait_blend = float(np.clip(gait_blend, 0.0, 1.0))

    def _add_noise(self, values, std):
        if std <= 0.0:
            return values
        rng = self.np_random if self.np_random is not None else np.random.default_rng()
        return values + rng.normal(
            0.0, std, size=values.shape).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        # Command vector (2) — at front so policy sees the goal first
        # Normalize: vel/max_vel, yaw/max_yaw → roughly [-1, 1]
        cmd = np.array([
            self._cmd_vel / max(CMD_VEL_RANGE[1], 1e-6),  # [0, 1]
            self._cmd_yaw / max(abs(CMD_YAW_RANGE[1]), 1e-6),  # [-1, 1]
            self._gait_blend,  # [0, 1]
        ], dtype=np.float32)

        # Actuated joint positions (11)
        joint_pos = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        for i in range(NUM_ACTUATORS):
            raw = self.data.qpos[self._act_qpos_idx[i]]
            scale = SLIDE_RANGE_VAL if i < NUM_SLIDES else YAW_RANGE_VAL
            joint_pos[i] = raw / max(scale, 1e-6)
        joint_pos = self._add_noise(joint_pos, self.encoder_pos_noise_std)

        # Actuated joint velocities (11)
        joint_vel = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        for i in range(NUM_ACTUATORS):
            raw = self.data.qvel[self._act_qvel_idx[i]]
            scale = SLIDE_VEL_SCALE if i < NUM_SLIDES else YAW_VEL_SCALE
            joint_vel[i] = raw / max(scale, 1e-6)
        joint_vel = self._add_noise(joint_vel, self.encoder_vel_noise_std)

        # Root body rotation matrix (world → body)
        gravity_world = np.array([0.0, 0.0, -1.0])
        segment_gravity = np.zeros((NUM_IMUS, 3), dtype=np.float32)
        segment_gyro = np.zeros((NUM_IMUS, 3), dtype=np.float32)
        for i, body_id in enumerate(self._imu_body_ids):
            xmat = self.data.xmat[body_id].reshape(3, 3)
            segment_gravity[i] = (xmat.T @ gravity_world).astype(np.float32)

            local_vel = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                body_id, local_vel, 1)
            segment_gyro[i] = (local_vel[:3] / IMU_GYRO_SCALE).astype(np.float32)
        segment_gravity = self._add_noise(
            segment_gravity, self.imu_gravity_noise_std)
        segment_gyro = self._add_noise(segment_gyro, self.imu_gyro_noise_std)

        # Phase clock — periodic signal for locomotion rhythm
        t = self._step_count * CTRL_DT
        phase = 2.0 * math.pi * PHASE_FREQ * t
        phase_clock = np.array(
            [math.sin(phase), math.cos(phase)], dtype=np.float32)

        obs = np.concatenate([
            cmd,                                # 3
            joint_pos,                          # 11
            joint_vel,                          # 11
            self._last_action.astype(np.float32),  # 11
            segment_gravity.reshape(-1),        # 21
            segment_gyro.reshape(-1),           # 21
            phase_clock,                        # 2
        ])
        return obs.astype(np.float32)

    def _sample_gait_blend(self):
        if self._fixed_gait_blend is not None:
            return float(self._fixed_gait_blend)
        if self.gait_mode == "random":
            return float(self.np_random.uniform(0.0, 1.0))
        return GAIT_BLENDS[self.gait_mode]

    def _sample_cmd_vel(self):
        if self._fixed_cmd_vel is not None:
            return float(np.clip(
                self._fixed_cmd_vel, CMD_VEL_RANGE[0], CMD_VEL_RANGE[1]))
        return float(self.np_random.uniform(*CMD_VEL_RANGE))

    def _sample_cmd_yaw(self):
        if self._fixed_cmd_yaw is not None:
            return float(np.clip(
                self._fixed_cmd_yaw, CMD_YAW_RANGE[0], CMD_YAW_RANGE[1]))
        return float(self.np_random.uniform(*CMD_YAW_RANGE))

    # ──────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action, residual_action=None):
        # ── Actual velocities ──
        # Forward speed: -X direction in world frame. Prefer per-control-step
        # root displacement for locomotion reward because worm gaits have large
        # cyclic instantaneous qvel spikes and back-slip.
        if hasattr(self, "_last_root_pos") and hasattr(self, "_root_body_id"):
            root_pos = self.data.xpos[self._root_body_id]
            delta_pos = root_pos - self._last_root_pos
            forward_speed = -float(delta_pos[0]) / CTRL_DT
            lateral_speed = abs(float(delta_pos[1])) / CTRL_DT
        else:
            forward_speed = -self.data.qvel[0]
            lateral_speed = abs(self.data.qvel[1])

        # Yaw rate: rotation around world Z axis
        yaw_rate = self.data.qvel[5]

        # ── Velocity tracking (exp kernel) ──
        vel_err = forward_speed - self._cmd_vel
        yaw_err = yaw_rate - self._cmd_yaw
        progress_ratio = 1.0
        if self._cmd_vel > 1e-6:
            progress_ratio = np.clip(
                forward_speed / max(self._cmd_vel, 1e-6), 0.0, 1.0)
        r_vel_track = (
            math.exp(-(vel_err ** 2) / (SIGMA_VEL ** 2))
            if forward_speed > 0.0 else 0.0)
        r_yaw_track = math.exp(-(yaw_err ** 2) / (SIGMA_YAW ** 2))
        if self._cmd_vel > 1e-6:
            r_yaw_track *= 0.25 + 0.75 * progress_ratio

        # ── Linear forward velocity bonus (capped + normalized) ──
        # Rewards forward movement UP TO target speed, no bonus beyond.
        # Normalized to [0,1] so weight W_VEL_LIN directly controls magnitude.
        r_vel_lin = (min(max(0.0, forward_speed), self._cmd_vel)
                     / max(CMD_VEL_RANGE[1], 1e-6))

        # ── Overspeed penalty (quadratic — strongly penalizes exceeding command) ──
        overspeed_sq = max(0.0, forward_speed - self._cmd_vel) ** 2

        # ── Other penalties ──
        speed_scale = max(CMD_VEL_RANGE[1], 1e-6)
        forward_deficit = max(0.0, self._cmd_vel - forward_speed) / speed_scale
        backward_speed = max(0.0, -forward_speed) / speed_scale
        lateral_speed = lateral_speed / speed_scale

        energy = 0.0
        for i in range(self.model.nu):
            energy += abs(
                self.data.ctrl[i] * self.data.qvel[self._act_qvel_idx[i]])

        if residual_action is None:
            rate_source = action
            last_rate_source = self._last_action
        else:
            rate_source = residual_action
            last_rate_source = self._last_residual_action
        action_rate = float(np.sum(np.square(rate_source - last_rate_source)))

        reward = (
            + W_VEL_TRACK * r_vel_track    # exp tracking (precision)
            + W_YAW_TRACK * r_yaw_track    # exp tracking (turning)
            + W_VEL_LIN   * r_vel_lin      # linear bonus (exploration gradient)
            - W_OVERSPEED * overspeed_sq
            - W_FORWARD_DEFICIT * forward_deficit
            - W_COMMAND_COST * float(self._cmd_vel > 1e-6)
            - W_LATERAL   * lateral_speed
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
        if z < self._z_term_lo:
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="flat",
                    choices=list(TERRAIN_PRESETS.keys()))
    ap.add_argument("--gait-mode", default="random", choices=GAIT_MODES)
    ap.add_argument("--gait-blend", type=float, default=None)
    args = ap.parse_args()

    print(f"=== WormEnvV6 self-test [terrain={args.terrain}] ===")
    env = WormEnvV6(
        terrain=args.terrain, gait_mode=args.gait_mode,
        gait_blend=args.gait_blend)
    print(f"  obs_space:    {env.observation_space.shape}")
    print(f"  action_space: {env.action_space.shape}")
    print(f"  model: bodies={env.model.nbody}, nv={env.model.nv}, nu={env.model.nu}")

    obs, info = env.reset(seed=42)
    print(f"  reset obs shape: {obs.shape}")
    print(f"  reset obs range: [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"  gait_blend: {obs[OBS_LAYOUT['command']][2]:.3f}")
    assert obs.shape == (OBS_DIM,), f"Expected obs dim {OBS_DIM}, got {obs.shape}"
    assert OBS_LAYOUT["phase_clock"].stop == OBS_DIM
    assert np.all(np.isfinite(obs)), "Non-finite reset obs!"

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

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

Command:       [vx_cmd, vy_cmd, yaw_rate_cmd]
               - vx_cmd        in [-0.25, 0.25] m/s (body forward speed target)
               - vy_cmd        in [-0.15, 0.15] m/s (body lateral speed target)
               - yaw_rate_cmd  in [-0.5, 0.5] rad/s (turning rate target)

Policy action: residual joint action(11) + learned gait gate(1)
               - high-sensitivity gait gate maps [-1, 1] to gait_blend [0, 1]
                 (0=worm, 0.5=mixed, 1=snake)

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
    COMMAND_GATE_LATERAL_CENTER,
    COMMAND_GATE_MIXED_CENTER,
    COMMAND_GATE_WORM_CENTER_FAST,
    COMMAND_GATE_WORM_CENTER_SLOW,
    COMMAND_GATE_WORM_FAST_THRESHOLD,
    COMMAND_GATE_YAW_CENTER,
    DEFAULT_GAIT_PRIOR_SCALE,
    DEFAULT_POLICY_RESIDUAL_SCALE,
    action_adapter_contract,
    command_conditioned_gate_center,
    command_conditioned_gait_blend,
    compose_deployable_action,
    gait_blend_from_policy_gate,
)

# ─── Environment constants ────────────────────────────────────────────────────
NUM_IMUS    = 7
NUM_POLICY_ACTIONS = NUM_ACTUATORS + 1      # 11 residual joints + 1 gait gate
OBS_DIM     = 80                            # 3+11+11+11+21+21+2
CTRL_DT     = 0.02                          # 50 Hz control frequency
PHASE_FREQ  = 1.0 / PERISTALTIC_ACTUATION_PERIOD_S
PHYSICS_DT  = 0.002                         # 500 Hz physics (from XML)
N_FRAMES    = int(CTRL_DT / PHYSICS_DT)     # 10 physics steps per control step
MAX_EP_TIME = 20.0                          # seconds per episode
MAX_EP_STEPS = int(MAX_EP_TIME / CTRL_DT)   # 1000 steps
SETTLE_STEPS = 250                          # 0.5s settle after reset

# Command ranges (sampled randomly each episode)
CMD_VX_RANGE    = (-0.25, 0.25)  # m/s body-forward target (+ forward, - reverse)
CMD_VY_RANGE    = (-0.15, 0.15)  # m/s body-lateral target
CMD_YAW_RANGE   = (-0.5, 0.5)    # rad/s yaw target; matched to observed authority
CMD_VEL_RANGE   = (0.0, CMD_VX_RANGE[1])  # legacy forward-speed alias
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
COMMAND_CURRICULA = (
    "straight",
    "planar",
    "heading_hold",
    "heading_omni",
    "lateral",
    "lateral_right",
    "yaw",
    "yaw_right",
    "right_recovery",
    "omni",
    "continuous_omni",
    "axis_separation",
)

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
W_YAW_ALIGN = 0.0       # signed yaw-command alignment
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

# Override the older slow-tracking reward with the formal omni-directional
# auto-gated contract. Keeping the assignment block local makes old checkpoints
# incompatible through the reward contract without changing the deployable
# observation layout.
SIGMA_VEL = 0.050
SIGMA_YAW = 0.20
W_VEL_TRACK = 3.0
W_YAW_TRACK = 3.0
W_YAW_ALIGN = 4.0
W_VEL_LIN = 6.0
W_OVERSPEED = 3.0
W_FORWARD_DEFICIT = 0.5
W_COMMAND_COST = 1.0
W_LATERAL = 4.5
W_BACKWARD = 0.5
W_ENERGY = 0.001
W_SMOOTH = 0.01
W_YAW_ERROR = 6.0
W_YAW_DRIFT = 6.0
W_YAW_STATIONARY = 8.0
W_LATERAL_ONLY_FORWARD_DRIFT = 5.0
W_GAIT_GATE_TARGET = 3.0
YAW_DRIFT_TOLERANCE_RAD = 0.20
YAW_STATIONARY_TOLERANCE_M_S = 0.02
LATERAL_ONLY_FORWARD_TOLERANCE_M_S = 0.04
GAIT_GATE_WORM_TARGET_SLOW = COMMAND_GATE_WORM_CENTER_SLOW
GAIT_GATE_WORM_TARGET_FAST = COMMAND_GATE_WORM_CENTER_FAST
GAIT_GATE_WORM_FAST_THRESHOLD = COMMAND_GATE_WORM_FAST_THRESHOLD
GAIT_GATE_WORM_TARGET = GAIT_GATE_WORM_TARGET_SLOW
GAIT_GATE_MIXED_TARGET = COMMAND_GATE_MIXED_CENTER
GAIT_GATE_LATERAL_TARGET = COMMAND_GATE_LATERAL_CENTER
GAIT_GATE_YAW_TARGET = COMMAND_GATE_YAW_CENTER
REWARD_CONTRACT_VERSION = "omni_directional_offaxis_yaw_v19"


def reward_contract():
    return {
        "version": REWARD_CONTRACT_VERSION,
        "forward_direction": "-world_x",
        "weights": {
            "vel_track": W_VEL_TRACK,
            "yaw_track": W_YAW_TRACK,
            "yaw_align": W_YAW_ALIGN,
            "vel_lin": W_VEL_LIN,
            "overspeed": W_OVERSPEED,
            "forward_deficit": W_FORWARD_DEFICIT,
            "command_cost": W_COMMAND_COST,
            "lateral": W_LATERAL,
            "backward": W_BACKWARD,
            "yaw_error": W_YAW_ERROR,
            "yaw_drift": W_YAW_DRIFT,
            "yaw_stationary": W_YAW_STATIONARY,
            "lateral_only_forward_drift": W_LATERAL_ONLY_FORWARD_DRIFT,
            "gait_gate_target": W_GAIT_GATE_TARGET,
            "energy": W_ENERGY,
            "smooth": W_SMOOTH,
        },
        "normalization": {
            "speed_scale_m_s": CMD_VX_RANGE[1],
            "lateral_speed_scale_m_s": CMD_VY_RANGE[1],
            "cmaes_full_combined_target_m_s": 0.24797,
            "body_frame_vx_vy_command_tracking": True,
            "yaw_tracking_gated_by_forward_progress": True,
            "cyclic_backslip_is_soft_penalized": True,
            "yaw_range_m_s_is_authority_matched": True,
            "off_axis_penalty_tapers_with_planar_command": True,
            "strong_off_axis_suppression": True,
            "signed_yaw_alignment_reward": True,
            "zero_yaw_integrated_drift_penalty": True,
            "zero_yaw_translation_uses_reset_body_axes": True,
            "zero_yaw_heading_hold_weight_boost": True,
            "yaw_drift_tolerance_rad": YAW_DRIFT_TOLERANCE_RAD,
            "yaw_only_stationary_speed_penalty": True,
            "yaw_stationary_tolerance_m_s": YAW_STATIONARY_TOLERANCE_M_S,
            "pure_lateral_forward_drift_penalty": True,
            "pure_lateral_forward_tolerance_m_s": (
                LATERAL_ONLY_FORWARD_TOLERANCE_M_S),
            "continuous_omni_repair_oversampling": True,
            "continuous_omni_mixed_yaw_repair_sampling": True,
            "axis_separation_curriculum": True,
            "yaw_only_prior_scaling_applied": True,
            "gait_blend_is_policy_gate": True,
                "command_conditioned_gait_gate_regularizer": {
                    "enabled": True,
                    "worm_target_for_axial_translation": GAIT_GATE_WORM_TARGET,
                    "worm_target_for_slow_axial_translation": (
                        GAIT_GATE_WORM_TARGET_SLOW),
                    "mixed_target_for_fast_axial_translation": (
                        GAIT_GATE_WORM_TARGET_FAST),
                    "fast_axial_threshold_norm": (
                        GAIT_GATE_WORM_FAST_THRESHOLD),
                    "mixed_target_for_mixed_commands": GAIT_GATE_MIXED_TARGET,
                    "snake_target_for_lateral_translation": (
                        GAIT_GATE_LATERAL_TARGET),
                    "snake_target_for_yaw": GAIT_GATE_YAW_TARGET,
                    "reason": (
                        "A deployable-command-conditioned regularizer keeps "
                        "the learned latent gate from collapsing to the combined "
                    "anchor while still allowing the policy to override it "
                    "when tracking reward requires another gait. V16 makes "
                    "the axial target speed-dependent: low-speed axial "
                    "commands remain visibly worm-like, while full-speed "
                    "axial commands return to the mixed target to avoid "
                    "capping forward tracking speed."),
            },
            "command_curriculum_supported": list(COMMAND_CURRICULA),
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
                 fixed_cmd_vx=None, fixed_cmd_vy=None,
                 command_curriculum="omni",
                 command_resample_prob=CMD_RESAMPLE_P,
                 gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
                 policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
        super().__init__()
        self.render_mode = render_mode
        self.terrain = terrain
        self.gait_mode = gait_mode
        if gait_blend is not None:
            self._fixed_gait_blend = gait_blend
        elif gait_mode in GAIT_BLENDS:
            self._fixed_gait_blend = GAIT_BLENDS[gait_mode]
        else:
            self._fixed_gait_blend = None
        self._fixed_cmd_vx = fixed_cmd_vx
        if self._fixed_cmd_vx is None and fixed_cmd_vel is not None:
            self._fixed_cmd_vx = fixed_cmd_vel
        self._fixed_cmd_vy = fixed_cmd_vy
        self._fixed_cmd_yaw = fixed_cmd_yaw
        self.command_curriculum = command_curriculum
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
        if command_curriculum not in COMMAND_CURRICULA:
            raise ValueError(
                f"Unknown command_curriculum: {command_curriculum}. "
                f"Expected one of {COMMAND_CURRICULA}")
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
            low=-1.0, high=1.0, shape=(NUM_POLICY_ACTIONS,), dtype=np.float32)
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
        self._cmd_vx = 0.0        # body-forward speed target (m/s)
        self._cmd_vy = 0.0        # body-lateral speed target (m/s)
        self._cmd_yaw = 0.0       # yaw rate target (rad/s)
        self._gait_blend = 0.5    # learned gate: 0=worm, 1=snake

        # ── Renderer (lazy init) ──
        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Sample velocity command for this episode, unless fixed for eval.
        self._cmd_vx, self._cmd_vy, self._cmd_yaw = self._sample_command()
        self._gait_blend = self._initial_gait_blend()

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
        self._start_root_yaw = self._root_yaw_rad()
        start_xmat = self.data.xmat[self._root_body_id].reshape(3, 3)
        self._start_forward_axis = -start_xmat[:2, 0].copy()
        self._start_lateral_axis = start_xmat[:2, 1].copy()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (NUM_POLICY_ACTIONS,):
            raise ValueError(
                f"policy action shape {action.shape} != {(NUM_POLICY_ACTIONS,)}")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        residual_command = action[:NUM_ACTUATORS]
        learned_gait_blend = gait_blend_from_policy_gate(action[-1])

        # Occasionally resample command mid-episode (curriculum diversity)
        if self.np_random.random() < self.command_resample_prob:
            self._cmd_vx, self._cmd_vy, self._cmd_yaw = self._sample_command()
        command_norm = (
            self._cmd_vx / max(abs(CMD_VX_RANGE[1]), 1e-6),
            self._cmd_vy / max(abs(CMD_VY_RANGE[1]), 1e-6),
            self._cmd_yaw / max(abs(CMD_YAW_RANGE[1]), 1e-6),
        )
        self._gait_blend = (
            float(self._fixed_gait_blend)
            if self._fixed_gait_blend is not None
            else command_conditioned_gait_blend(
                learned_gait_blend, command_norm))

        # EMA filter — anti-vibration
        residual_action = (
            ACTION_EMA * residual_command
            + (1.0 - ACTION_EMA) * self._last_residual_action)
        phase = 2.0 * math.pi * PHASE_FREQ * self._step_count * CTRL_DT
        applied_action = compose_deployable_action(
            residual_action,
            phase=phase,
            gait_blend=self._gait_blend,
            command=command_norm,
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
        info = {
            "cmd_vx_m_s": float(self._cmd_vx),
            "cmd_vy_m_s": float(self._cmd_vy),
            "cmd_vel_m_s": float(self._cmd_vx),
            "cmd_yaw_rad_s": float(self._cmd_yaw),
            "gait_blend": float(self._gait_blend),
            "learned_gait_blend": float(learned_gait_blend),
            "gait_blend_source": (
                "fixed" if self._fixed_gait_blend is not None
                else "policy_action_gate"),
            "root_x_m": float(self.data.xpos[self._root_body_id, 0]),
            "root_y_m": float(self.data.xpos[self._root_body_id, 1]),
            "root_yaw_rad": float(self._root_yaw_rad()),
            "elapsed_s": float(self._step_count * CTRL_DT),
        }
        info.update(getattr(self, "_last_reward_terms", {}))

        self._last_action = applied_action.copy()
        self._last_residual_action = residual_action.copy()
        self._last_root_pos = self.data.xpos[self._root_body_id].copy()
        return obs, reward, terminated, truncated, info

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

    def set_command(self, velocity=None, yaw_rate=None, gait_blend=None,
                    vx=None, vy=None):
        """Override command values for deterministic evaluation."""
        if vx is None and velocity is not None:
            vx = velocity
        if vx is not None:
            self._cmd_vx = float(np.clip(
                vx, CMD_VX_RANGE[0], CMD_VX_RANGE[1]))
        if vy is not None:
            self._cmd_vy = float(np.clip(
                vy, CMD_VY_RANGE[0], CMD_VY_RANGE[1]))
        if yaw_rate is not None:
            self._cmd_yaw = float(np.clip(
                yaw_rate, CMD_YAW_RANGE[0], CMD_YAW_RANGE[1]))
        if gait_blend is not None:
            self._gait_blend = float(np.clip(gait_blend, 0.0, 1.0))

    def _root_yaw_rad(self):
        mat = self.data.xmat[self._root_body_id].reshape(3, 3)
        return math.atan2(mat[1, 0], mat[0, 0])

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
        # Command vector (3) at front so policy sees the goal first.
        # Normalize: vx/vy/yaw max ranges -> roughly [-1, 1].
        cmd = np.array([
            self._cmd_vx / max(abs(CMD_VX_RANGE[1]), 1e-6),  # [-1, 1]
            self._cmd_vy / max(abs(CMD_VY_RANGE[1]), 1e-6),  # [-1, 1]
            self._cmd_yaw / max(abs(CMD_YAW_RANGE[1]), 1e-6),  # [-1, 1]
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

    def _initial_gait_blend(self):
        if self._fixed_gait_blend is not None:
            return float(self._fixed_gait_blend)
        if self.gait_mode == "random":
            return float(self.np_random.uniform(0.0, 1.0))
        return GAIT_BLENDS[self.gait_mode]

    def _sample_cmd_vx(self):
        if self._fixed_cmd_vx is not None:
            return float(np.clip(
                self._fixed_cmd_vx, CMD_VX_RANGE[0], CMD_VX_RANGE[1]))
        return float(self.np_random.uniform(*CMD_VX_RANGE))

    def _sample_cmd_vy(self):
        if self._fixed_cmd_vy is not None:
            return float(np.clip(
                self._fixed_cmd_vy, CMD_VY_RANGE[0], CMD_VY_RANGE[1]))
        return float(self.np_random.uniform(*CMD_VY_RANGE))

    def _sample_cmd_vel(self):
        return self._sample_cmd_vx()

    def _sample_cmd_yaw(self):
        if self._fixed_cmd_yaw is not None:
            return float(np.clip(
                self._fixed_cmd_yaw, CMD_YAW_RANGE[0], CMD_YAW_RANGE[1]))
        return float(self.np_random.uniform(*CMD_YAW_RANGE))

    def _sample_signed_range(self, bounds, min_abs_fraction=0.30):
        lo, hi = bounds
        max_abs = max(abs(lo), abs(hi))
        min_abs = max_abs * min_abs_fraction
        if self.np_random.random() < 0.5 and lo < -min_abs:
            return float(self.np_random.uniform(lo, -min_abs))
        if hi > min_abs:
            return float(self.np_random.uniform(min_abs, hi))
        return float(self.np_random.uniform(lo, hi))

    def _sample_low_axis_command(self, bounds, max_fraction=0.25):
        lo, hi = bounds
        max_abs = max(abs(lo), abs(hi))
        if max_abs <= 0.0:
            return 0.0
        value = self.np_random.uniform(-max_abs * max_fraction,
                                       max_abs * max_fraction)
        return float(np.clip(value, lo, hi))

    def _with_fixed_command_overrides(self, vx, vy, yaw):
        if self._fixed_cmd_vx is not None:
            vx = float(np.clip(
                self._fixed_cmd_vx, CMD_VX_RANGE[0], CMD_VX_RANGE[1]))
        if self._fixed_cmd_vy is not None:
            vy = float(np.clip(
                self._fixed_cmd_vy, CMD_VY_RANGE[0], CMD_VY_RANGE[1]))
        if self._fixed_cmd_yaw is not None:
            yaw = float(np.clip(
                self._fixed_cmd_yaw, CMD_YAW_RANGE[0], CMD_YAW_RANGE[1]))
        return float(vx), float(vy), float(yaw)

    def _sample_command(self):
        if self.command_curriculum == "straight":
            return self._with_fixed_command_overrides(
                self._sample_signed_range(CMD_VX_RANGE), 0.0, 0.0)

        if self.command_curriculum == "planar":
            if self.np_random.random() < 0.5:
                vx = self._sample_signed_range(CMD_VX_RANGE)
                vy = 0.0
            else:
                vx = 0.0
                vy = self._sample_signed_range(CMD_VY_RANGE)
            return self._with_fixed_command_overrides(vx, vy, 0.0)

        if self.command_curriculum == "heading_hold":
            primitive = int(self.np_random.integers(0, 4))
            if primitive == 0:
                vx, vy = self._sample_signed_range(
                    (0.0, CMD_VX_RANGE[1]), min_abs_fraction=0.50), 0.0
            elif primitive == 1:
                vx, vy = self._sample_signed_range(
                    (CMD_VX_RANGE[0], 0.0), min_abs_fraction=0.50), 0.0
            elif primitive == 2:
                vx, vy = 0.0, self._sample_signed_range(
                    (0.0, CMD_VY_RANGE[1]), min_abs_fraction=0.50)
            else:
                vx, vy = 0.0, self._sample_signed_range(
                    (CMD_VY_RANGE[0], 0.0), min_abs_fraction=0.50)
            return self._with_fixed_command_overrides(vx, vy, 0.0)

        if self.command_curriculum == "heading_omni":
            primitive = int(self.np_random.integers(0, 8))
            if primitive == 0:
                vx, vy, yaw = self._sample_signed_range(
                    (0.0, CMD_VX_RANGE[1]), min_abs_fraction=0.50), 0.0, 0.0
            elif primitive == 1:
                vx, vy, yaw = self._sample_signed_range(
                    (CMD_VX_RANGE[0], 0.0), min_abs_fraction=0.50), 0.0, 0.0
            elif primitive == 2:
                vx, vy, yaw = 0.0, self._sample_signed_range(
                    (0.0, CMD_VY_RANGE[1]), min_abs_fraction=0.50), 0.0
            elif primitive == 3:
                vx, vy, yaw = 0.0, self._sample_signed_range(
                    (CMD_VY_RANGE[0], 0.0), min_abs_fraction=0.50), 0.0
            elif primitive == 4:
                vx, vy, yaw = 0.0, 0.0, self._sample_signed_range(
                    (0.0, CMD_YAW_RANGE[1]), min_abs_fraction=0.50)
            elif primitive == 5:
                vx, vy, yaw = 0.0, 0.0, self._sample_signed_range(
                    (CMD_YAW_RANGE[0], 0.0), min_abs_fraction=0.50)
            elif primitive == 6:
                vx, vy, yaw = self._sample_signed_range(
                    (0.0, CMD_VX_RANGE[1]), min_abs_fraction=0.50), 0.0, (
                    self._sample_signed_range(
                        (0.0, CMD_YAW_RANGE[1]), min_abs_fraction=0.50))
            else:
                vx, vy, yaw = self._sample_signed_range(
                    (0.0, CMD_VX_RANGE[1]), min_abs_fraction=0.50), 0.0, (
                    self._sample_signed_range(
                        (CMD_YAW_RANGE[0], 0.0), min_abs_fraction=0.50))
            return self._with_fixed_command_overrides(vx, vy, yaw)

        if self.command_curriculum == "lateral":
            return self._with_fixed_command_overrides(
                0.0, self._sample_signed_range(CMD_VY_RANGE), 0.0)

        if self.command_curriculum == "lateral_right":
            return self._with_fixed_command_overrides(
                0.0, self._sample_signed_range((CMD_VY_RANGE[0], 0.0)), 0.0)

        if self.command_curriculum == "yaw":
            return self._with_fixed_command_overrides(
                0.0, 0.0, self._sample_signed_range(CMD_YAW_RANGE))

        if self.command_curriculum == "yaw_right":
            return self._with_fixed_command_overrides(
                0.0, 0.0, self._sample_signed_range((CMD_YAW_RANGE[0], 0.0)))

        if self.command_curriculum == "right_recovery":
            primitive = int(self.np_random.integers(0, 6))
            if primitive in (0, 1):
                vx, vy, yaw = (
                    0.0,
                    self._sample_signed_range((CMD_VY_RANGE[0], 0.0)),
                    0.0,
                )
            elif primitive == 2:
                vx, vy, yaw = (
                    0.0,
                    0.0,
                    self._sample_signed_range((CMD_YAW_RANGE[0], 0.0)),
                )
            elif primitive == 3:
                vx = self._sample_signed_range((0.0, CMD_VX_RANGE[1]))
                vy = 0.0
                yaw = self._sample_signed_range((CMD_YAW_RANGE[0], 0.0))
            elif primitive == 4:
                vx, vy, yaw = (
                    0.0,
                    self._sample_signed_range((0.0, CMD_VY_RANGE[1])),
                    0.0,
                )
            else:
                vx, vy, yaw = (
                    0.0,
                    0.0,
                    self._sample_signed_range((0.0, CMD_YAW_RANGE[1])),
                )
            return self._with_fixed_command_overrides(vx, vy, yaw)

        if self.command_curriculum == "continuous_omni":
            primitive = int(self.np_random.integers(0, 32))
            if primitive in (0, 1):
                vx, vy, yaw = 0.0, 0.0, 0.0
            elif primitive == 2:
                vx = self._sample_low_axis_command(CMD_VX_RANGE)
                vy, yaw = 0.0, 0.0
            elif primitive == 3:
                vx = 0.0
                vy = self._sample_low_axis_command(CMD_VY_RANGE)
                yaw = 0.0
            elif primitive == 4:
                vx, vy = 0.0, 0.0
                yaw = self._sample_low_axis_command(CMD_YAW_RANGE)
            elif primitive == 5:
                vx = self._sample_signed_range(CMD_VX_RANGE,
                                               min_abs_fraction=0.05)
                vy, yaw = 0.0, 0.0
            elif primitive == 6:
                vx = 0.0
                vy = self._sample_signed_range(CMD_VY_RANGE,
                                               min_abs_fraction=0.05)
                yaw = 0.0
            elif primitive == 7:
                vx, vy = 0.0, 0.0
                yaw = self._sample_signed_range(CMD_YAW_RANGE,
                                                min_abs_fraction=0.05)
            elif primitive == 8:
                vx = self.np_random.uniform(*CMD_VX_RANGE)
                vy = self.np_random.uniform(*CMD_VY_RANGE)
                yaw = 0.0
            elif primitive == 9:
                vx = self.np_random.uniform(*CMD_VX_RANGE)
                vy = 0.0
                yaw = self.np_random.uniform(*CMD_YAW_RANGE)
            elif primitive == 10:
                vx = 0.0
                vy = self.np_random.uniform(*CMD_VY_RANGE)
                yaw = self.np_random.uniform(*CMD_YAW_RANGE)
            elif primitive == 11:
                vx = self.np_random.uniform(*CMD_VX_RANGE)
                vy = self.np_random.uniform(*CMD_VY_RANGE)
                yaw = self.np_random.uniform(*CMD_YAW_RANGE)
            elif primitive == 12:
                vx, vy = 0.0, 0.0
                yaw = self._sample_signed_range(
                    (0.0, CMD_YAW_RANGE[1]), min_abs_fraction=0.50)
            elif primitive == 13:
                vx, vy = 0.0, 0.0
                yaw = self._sample_signed_range(
                    (CMD_YAW_RANGE[0], 0.0), min_abs_fraction=0.50)
            elif primitive == 14:
                vx = self._sample_signed_range(
                    (CMD_VX_RANGE[0], 0.0), min_abs_fraction=0.50)
                vy = self.np_random.uniform(*CMD_VY_RANGE)
                yaw = 0.0
            elif primitive == 15:
                vx = self.np_random.uniform(*CMD_VX_RANGE)
                vy = self.np_random.uniform(*CMD_VY_RANGE)
                yaw = 0.0
            elif primitive == 16:
                vx, vy, yaw = (
                    0.0,
                    self._sample_signed_range(
                        (0.0, CMD_VY_RANGE[1]), min_abs_fraction=0.50),
                    0.0,
                )
            elif primitive == 17:
                vx, vy, yaw = (
                    0.0,
                    self._sample_signed_range(
                        (CMD_VY_RANGE[0], 0.0), min_abs_fraction=0.50),
                    0.0,
                )
            elif primitive == 18:
                vx, vy, yaw = (
                    0.0,
                    self._sample_signed_range(
                        (0.0, CMD_VY_RANGE[1]), min_abs_fraction=0.75),
                    0.0,
                )
            elif primitive == 19:
                vx, vy, yaw = (
                    0.0,
                    self._sample_signed_range(
                        (CMD_VY_RANGE[0], 0.0), min_abs_fraction=0.75),
                    0.0,
                )
            elif primitive == 20:
                vx, vy, yaw = (
                    0.0,
                    0.0,
                    self._sample_signed_range(
                        (0.0, CMD_YAW_RANGE[1]), min_abs_fraction=0.50),
                )
            elif primitive == 21:
                vx, vy, yaw = (
                    0.0,
                    0.0,
                    self._sample_signed_range(
                        (CMD_YAW_RANGE[0], 0.0), min_abs_fraction=0.50),
                )
            elif primitive == 22:
                vx, vy, yaw = (
                    0.0,
                    0.0,
                    self._sample_signed_range(
                        (0.0, CMD_YAW_RANGE[1]), min_abs_fraction=0.75),
                )
            elif primitive == 23:
                vx, vy, yaw = (
                    0.0,
                    0.0,
                    self._sample_signed_range(
                        (CMD_YAW_RANGE[0], 0.0), min_abs_fraction=0.75),
                )
            elif primitive in (24, 26):
                vx, vy, yaw = (
                    self._sample_signed_range(
                        (0.0, CMD_VX_RANGE[1]),
                        min_abs_fraction=0.50 if primitive == 24 else 0.75),
                    0.0,
                    self._sample_signed_range(
                        (0.0, CMD_YAW_RANGE[1]),
                        min_abs_fraction=0.50 if primitive == 24 else 0.75),
                )
            elif primitive in (25, 27):
                vx, vy, yaw = (
                    self._sample_signed_range(
                        (0.0, CMD_VX_RANGE[1]),
                        min_abs_fraction=0.50 if primitive == 25 else 0.75),
                    0.0,
                    self._sample_signed_range(
                        (CMD_YAW_RANGE[0], 0.0),
                        min_abs_fraction=0.50 if primitive == 25 else 0.75),
                )
            elif primitive in (28, 30):
                vx, vy, yaw = (
                    self._sample_signed_range(
                        (CMD_VX_RANGE[0], 0.0),
                        min_abs_fraction=0.50 if primitive == 28 else 0.75),
                    0.0,
                    self._sample_signed_range(
                        (0.0, CMD_YAW_RANGE[1]),
                        min_abs_fraction=0.50 if primitive == 28 else 0.75),
                )
            else:
                vx, vy, yaw = (
                    self._sample_signed_range(
                        (CMD_VX_RANGE[0], 0.0),
                        min_abs_fraction=0.50 if primitive == 29 else 0.75),
                    0.0,
                    self._sample_signed_range(
                        (CMD_YAW_RANGE[0], 0.0),
                        min_abs_fraction=0.50 if primitive == 29 else 0.75),
                )
            return self._with_fixed_command_overrides(vx, vy, yaw)

        if self.command_curriculum == "axis_separation":
            primitive = int(self.np_random.integers(0, 18))
            if primitive in (0, 1):
                vx, vy, yaw = 0.0, 0.0, 0.0
            elif primitive == 2:
                vx = self._sample_low_axis_command(CMD_VX_RANGE)
                vy, yaw = 0.0, 0.0
            elif primitive in (3, 4):
                vx = self._sample_signed_range(
                    (0.0, CMD_VX_RANGE[1]),
                    min_abs_fraction=0.50 if primitive == 3 else 0.75)
                vy, yaw = 0.0, 0.0
            elif primitive in (5, 6):
                vx = self._sample_signed_range(
                    (CMD_VX_RANGE[0], 0.0),
                    min_abs_fraction=0.50 if primitive == 5 else 0.75)
                vy, yaw = 0.0, 0.0
            elif primitive == 7:
                vx = 0.0
                vy = self._sample_low_axis_command(CMD_VY_RANGE)
                yaw = 0.0
            elif primitive in (8, 9):
                vx, yaw = 0.0, 0.0
                vy = self._sample_signed_range(
                    (0.0, CMD_VY_RANGE[1]),
                    min_abs_fraction=0.50 if primitive == 8 else 0.75)
            elif primitive in (10, 11):
                vx, yaw = 0.0, 0.0
                vy = self._sample_signed_range(
                    (CMD_VY_RANGE[0], 0.0),
                    min_abs_fraction=0.50 if primitive == 10 else 0.75)
            elif primitive == 12:
                vx, vy = 0.0, 0.0
                yaw = self._sample_low_axis_command(CMD_YAW_RANGE)
            elif primitive in (13, 14):
                vx, vy = 0.0, 0.0
                yaw = self._sample_signed_range(
                    (0.0, CMD_YAW_RANGE[1]),
                    min_abs_fraction=0.50 if primitive == 13 else 0.75)
            elif primitive in (15, 16):
                vx, vy = 0.0, 0.0
                yaw = self._sample_signed_range(
                    (CMD_YAW_RANGE[0], 0.0),
                    min_abs_fraction=0.50 if primitive == 15 else 0.75)
            else:
                axis = int(self.np_random.integers(0, 3))
                if axis == 0:
                    vx = self._sample_signed_range(
                        CMD_VX_RANGE, min_abs_fraction=0.20)
                    vy, yaw = 0.0, 0.0
                elif axis == 1:
                    vx, yaw = 0.0, 0.0
                    vy = self._sample_signed_range(
                        CMD_VY_RANGE, min_abs_fraction=0.20)
                else:
                    vx, vy = 0.0, 0.0
                    yaw = self._sample_signed_range(
                        CMD_YAW_RANGE, min_abs_fraction=0.20)
            return self._with_fixed_command_overrides(vx, vy, yaw)

        primitive = int(self.np_random.integers(0, 10))
        if primitive == 0:
            vx, vy, yaw = self._sample_signed_range((0.0, CMD_VX_RANGE[1])), 0.0, 0.0
        elif primitive == 1:
            vx, vy, yaw = self._sample_signed_range((CMD_VX_RANGE[0], 0.0)), 0.0, 0.0
        elif primitive == 2:
            vx, vy, yaw = 0.0, self._sample_signed_range((0.0, CMD_VY_RANGE[1])), 0.0
        elif primitive == 3:
            vx, vy, yaw = 0.0, self._sample_signed_range((CMD_VY_RANGE[0], 0.0)), 0.0
        elif primitive == 4:
            vx, vy, yaw = 0.0, 0.0, self._sample_signed_range((0.0, CMD_YAW_RANGE[1]))
        elif primitive == 5:
            vx, vy, yaw = 0.0, 0.0, self._sample_signed_range((CMD_YAW_RANGE[0], 0.0))
        elif primitive == 6:
            vx = self._sample_signed_range((0.0, CMD_VX_RANGE[1]))
            vy = 0.0
            yaw = self._sample_signed_range((0.0, CMD_YAW_RANGE[1]))
        elif primitive == 7:
            vx = self._sample_signed_range((0.0, CMD_VX_RANGE[1]))
            vy = 0.0
            yaw = self._sample_signed_range((CMD_YAW_RANGE[0], 0.0))
        elif primitive == 8:
            vx = self._sample_signed_range(CMD_VX_RANGE, min_abs_fraction=0.20)
            vy = self._sample_signed_range(CMD_VY_RANGE, min_abs_fraction=0.20)
            yaw = 0.0
        else:
            vx = self.np_random.uniform(*CMD_VX_RANGE)
            vy = self.np_random.uniform(*CMD_VY_RANGE)
            yaw = self.np_random.uniform(*CMD_YAW_RANGE)
        return self._with_fixed_command_overrides(vx, vy, yaw)

    # ──────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action, residual_action=None):
        cmd_vx = float(getattr(self, "_cmd_vx", getattr(self, "_cmd_vel", 0.0)))
        cmd_vy = float(getattr(self, "_cmd_vy", 0.0))
        cmd_yaw = float(getattr(self, "_cmd_yaw", 0.0))
        # ── Actual velocities ──
        # Forward speed: -X direction in world frame. Prefer per-control-step
        # root displacement for locomotion reward because worm gaits have large
        # cyclic instantaneous qvel spikes and back-slip.
        if hasattr(self, "_last_root_pos") and hasattr(self, "_root_body_id"):
            root_pos = self.data.xpos[self._root_body_id]
            delta_pos = root_pos - self._last_root_pos
            world_vel_xy = delta_pos[:2] / CTRL_DT
            root_xmat = self.data.xmat[self._root_body_id].reshape(3, 3)
            if (abs(cmd_yaw) <= 1e-6
                    and hasattr(self, "_start_forward_axis")
                    and hasattr(self, "_start_lateral_axis")):
                forward_axis = self._start_forward_axis
                lateral_axis = self._start_lateral_axis
            else:
                forward_axis = -root_xmat[:2, 0]
                lateral_axis = root_xmat[:2, 1]
            forward_speed = float(np.dot(world_vel_xy, forward_axis))
            lateral_speed = float(np.dot(world_vel_xy, lateral_axis))
        else:
            forward_speed = -float(self.data.qvel[0])
            lateral_speed = float(self.data.qvel[1])

        # Yaw rate: rotation around world Z axis
        yaw_rate = self.data.qvel[5]

        # ── Velocity tracking (exp kernel) ──
        cmd_vec = np.array([cmd_vx, cmd_vy], dtype=np.float64)
        vel_vec = np.array([forward_speed, lateral_speed], dtype=np.float64)
        cmd_speed = float(np.linalg.norm(cmd_vec))
        speed_scale = max(CMD_VX_RANGE[1], 1e-6)
        if cmd_speed > 1e-6:
            along_cmd = float(np.dot(vel_vec, cmd_vec) / cmd_speed)
            perp_vec = vel_vec - (along_cmd / cmd_speed) * cmd_vec
            off_axis_speed = float(np.linalg.norm(perp_vec))
            progress_ratio = np.clip(along_cmd / cmd_speed, 0.0, 1.0)
        else:
            along_cmd = 0.0
            off_axis_speed = float(np.linalg.norm(vel_vec))
            progress_ratio = 1.0
        vel_err = float(np.linalg.norm(vel_vec - cmd_vec))
        yaw_err = yaw_rate - cmd_yaw
        yaw_error_norm = min(
            abs(yaw_err) / max(abs(CMD_YAW_RANGE[1]), 1e-6),
            2.0,
        )
        if hasattr(self, "_root_body_id"):
            current_yaw = self._root_yaw_rad()
        else:
            current_yaw = 0.0
        start_yaw = float(getattr(self, "_start_root_yaw", current_yaw))
        yaw_drift = (
            (current_yaw - start_yaw + math.pi) % (2.0 * math.pi)
            - math.pi)
        yaw_drift_norm = min(
            abs(yaw_drift) / max(YAW_DRIFT_TOLERANCE_RAD, 1e-6),
            3.0,
        )
        yaw_hold_gate = 1.0 if abs(cmd_yaw) <= 1e-6 else 0.0
        yaw_only_gate = (
            1.0 if abs(cmd_yaw) > 1e-6 and cmd_speed <= 1e-6 else 0.0)
        lateral_only_gate = (
            1.0
            if (abs(cmd_vy) > 1e-6
                and abs(cmd_vx) <= 1e-6
                and abs(cmd_yaw) <= 1e-6)
            else 0.0)
        cmd_vx_norm = abs(cmd_vx) / max(abs(CMD_VX_RANGE[1]), 1e-6)
        cmd_vy_norm = abs(cmd_vy) / max(abs(CMD_VY_RANGE[1]), 1e-6)
        cmd_yaw_norm = abs(cmd_yaw) / max(abs(CMD_YAW_RANGE[1]), 1e-6)
        cmd_mag_norm = max(cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm)
        gate_target_active = 1.0 if cmd_mag_norm > 1e-6 else 0.0
        desired_gait_blend = command_conditioned_gate_center(
            (cmd_vx_norm, cmd_vy_norm, cmd_yaw_norm))
        gait_gate_error = abs(
            float(getattr(self, "_gait_blend", GAIT_GATE_MIXED_TARGET))
            - desired_gait_blend)
        r_vel_track = math.exp(-(vel_err ** 2) / (SIGMA_VEL ** 2))
        if cmd_speed > 1e-6 and along_cmd <= 0.0:
            r_vel_track *= 0.25
        r_yaw_track = math.exp(-(yaw_err ** 2) / (SIGMA_YAW ** 2))
        if cmd_speed > 1e-6:
            r_yaw_track *= 0.25 + 0.75 * progress_ratio
        r_yaw_align = 0.0
        if abs(cmd_yaw) > 1e-6:
            yaw_scale = max(abs(CMD_YAW_RANGE[1]), 1e-6)
            r_yaw_align = np.clip(
                (yaw_rate * cmd_yaw) / (yaw_scale ** 2),
                -1.0,
                1.0,
            )
            r_yaw_align *= 0.25 + 0.75 * progress_ratio

        # ── Linear forward velocity bonus (capped + normalized) ──
        # Rewards forward movement UP TO target speed, no bonus beyond.
        # Normalized to [0,1] so weight W_VEL_LIN directly controls magnitude.
        r_vel_lin = (
            min(max(0.0, along_cmd), cmd_speed) / speed_scale
            if cmd_speed > 1e-6 else 0.0)

        # ── Overspeed penalty (quadratic — strongly penalizes exceeding command) ──
        overspeed_sq = max(
            0.0, float(np.linalg.norm(vel_vec)) - max(cmd_speed, 1e-6)) ** 2

        # ── Other penalties ──
        forward_deficit = max(0.0, cmd_speed - along_cmd) / speed_scale
        backward_speed = max(0.0, -along_cmd) / speed_scale
        lateral_speed = off_axis_speed / speed_scale
        yaw_stationary_speed_norm = min(
            float(np.linalg.norm(vel_vec))
            / max(YAW_STATIONARY_TOLERANCE_M_S, 1e-6),
            3.0,
        )
        lateral_only_forward_norm = min(
            abs(forward_speed)
            / max(LATERAL_ONLY_FORWARD_TOLERANCE_M_S, 1e-6),
            3.0,
        )

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
        self._last_reward_terms = {
            "body_vx_m_s": float(forward_speed),
            "body_vy_m_s": float(vel_vec[1]),
            "body_yaw_rate_rad_s": float(yaw_rate),
            "planar_velocity_error_m_s": float(vel_err),
            "yaw_rate_error_rad_s": float(yaw_err),
            "yaw_drift_rad": float(yaw_drift),
            "command_aligned_speed_m_s": float(along_cmd),
            "off_axis_speed_m_s": float(off_axis_speed),
            "reward_vel_track": float(r_vel_track),
            "reward_yaw_track": float(r_yaw_track),
            "reward_yaw_align": float(r_yaw_align),
            "reward_vel_lin": float(r_vel_lin),
            "reward_yaw_error_penalty": float(yaw_error_norm),
            "reward_yaw_drift_penalty": float(yaw_hold_gate * yaw_drift_norm),
            "reward_yaw_stationary_penalty": float(
                yaw_only_gate * yaw_stationary_speed_norm),
            "reward_lateral_only_forward_penalty": float(
                lateral_only_gate * lateral_only_forward_norm),
            "desired_gait_blend": float(desired_gait_blend),
            "gait_gate_error": float(gate_target_active * gait_gate_error),
        }

        reward = (
            + W_VEL_TRACK * r_vel_track    # exp tracking (precision)
            + W_YAW_TRACK * r_yaw_track    # exp tracking (turning)
            + W_YAW_ALIGN * r_yaw_align    # signed yaw command response
            + W_VEL_LIN   * r_vel_lin      # linear bonus (exploration gradient)
            - W_OVERSPEED * overspeed_sq
            - W_FORWARD_DEFICIT * forward_deficit
            - W_COMMAND_COST * float(cmd_speed > 1e-6)
            - W_LATERAL   * lateral_speed
            - W_BACKWARD  * backward_speed  # penalize going backward
            - W_YAW_ERROR * yaw_error_norm
            - W_YAW_DRIFT * yaw_hold_gate * yaw_drift_norm
            - W_YAW_STATIONARY * yaw_only_gate * yaw_stationary_speed_norm
            - W_LATERAL_ONLY_FORWARD_DRIFT * lateral_only_gate * (
                lateral_only_forward_norm)
            - W_GAIT_GATE_TARGET * gate_target_active * gait_gate_error
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
    print(f"  command[vx,vy,yaw]: {obs[OBS_LAYOUT['command']]}")
    print(f"  initial_gait_blend: {env._gait_blend:.3f}")
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

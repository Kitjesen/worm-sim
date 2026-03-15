"""
GPU-accelerated Worm V6 RL Environment using MJX (JAX backend for MuJoCo).
==========================================================================
Runs thousands of parallel environments on a single GPU via JAX vectorization.
Compatible with brax's PPO training pipeline.

Same reward structure as the CPU version (worm_env_v6.py):
  - Velocity tracking (exp kernel)
  - Overspeed penalty (quadratic)
  - Lateral drift / backward penalties
  - Energy / smoothness costs
"""

import os
import sys
import functools

import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx

from brax.envs.base import PipelineEnv, State

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worm_v6 import build_xml, NUM_SLIDES, NUM_YAWS, NUM_ACTUATORS

# ─── Constants (matching CPU worm_env_v6.py) ─────────────────────────────────
OBS_DIM      = 35
CTRL_DT      = 0.02
PHASE_FREQ   = 0.5
PHYSICS_DT   = 0.002
N_FRAMES     = int(CTRL_DT / PHYSICS_DT)   # 10
MAX_EP_STEPS = 1000
SETTLE_STEPS = 250

CMD_VEL_RANGE  = (0.0, 0.025)
CMD_YAW_RANGE  = (-0.3, 0.3)
CMD_RESAMPLE_P = 0.005

W_VEL_TRACK  = 2.0
W_YAW_TRACK  = 1.0
SIGMA_VEL    = 0.010
SIGMA_YAW    = 0.15
W_VEL_LIN    = 8.0
W_OVERSPEED  = 5.0
W_LATERAL    = 2.0
W_BACKWARD   = 3.0
W_ENERGY     = 0.002
W_SMOOTH     = 0.02
ACTION_EMA   = 0.3


class WormMJXEnv(PipelineEnv):
    """GPU-parallel worm environment using MJX + brax PipelineEnv."""

    def __init__(self, **kwargs):
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        mesh_dir = os.path.join(project_root, "meshes")
        urdf_path = os.path.join(mesh_dir, "longworm2", "longworm2.SLDASM.urdf")

        # Build XML (no inject_strips — that's a rendering-time callback)
        xml_str = build_xml(mesh_dir, urdf_path)
        mj_model = mujoco.MjModel.from_xml_string(xml_str)

        # Precompute actuator → qvel index mapping
        act_qvel = []
        for i in range(mj_model.nu):
            jnt_id = mj_model.actuator_trnid[i, 0]
            act_qvel.append(mj_model.jnt_dofadr[jnt_id])
        self._act_qvel_idx = jnp.array(act_qvel)

        # Root body index
        self._root_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Store dims
        self._nu = mj_model.nu
        self._nv = mj_model.nv
        self._init_q = jnp.array(mj_model.qpos0.copy())

        # Convert to MJX model (JAX arrays on GPU)
        mjx_model = mjx.put_model(mj_model)

        # Initialize PipelineEnv with MJX backend
        super().__init__(mjx_model, backend='mjx', n_frames=N_FRAMES, **kwargs)

        # Pre-settle using raw MJX steps. Uses a Python loop with a JIT-compiled
        # single-step function (avoids slow jax.lax.scan compilation for 2500 steps).
        print("  MJX: pre-settling robot (JIT compile ~60s first time)...")
        _data = mjx.make_data(mjx_model)
        _data = _data.replace(qpos=self._init_q, qvel=jnp.zeros(self._nv))
        _data = mjx.forward(mjx_model, _data)

        @jax.jit
        def _one_step(data):
            return mjx.step(mjx_model, data)

        # Python loop: first call compiles (~60s), rest are fast (~16ms each)
        n_settle = SETTLE_STEPS * N_FRAMES  # 2500 physics steps
        for i in range(n_settle):
            _data = _one_step(_data)
            if i == 0:
                jax.block_until_ready(_data.qpos)
                print(f"  MJX: JIT compiled. Running {n_settle - 1} more settle steps...")

        jax.block_until_ready(_data.qpos)
        self._settled_q = _data.qpos
        self._settled_qd = _data.qvel
        print("  MJX: settle done.")

    @property
    def observation_size(self) -> int:
        return OBS_DIM

    @property
    def action_size(self) -> int:
        return self._nu

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> State:
        rng, rng_vel, rng_yaw = random.split(rng, 3)

        cmd_vel = random.uniform(
            rng_vel, minval=CMD_VEL_RANGE[0], maxval=CMD_VEL_RANGE[1])
        cmd_yaw = random.uniform(
            rng_yaw, minval=CMD_YAW_RANGE[0], maxval=CMD_YAW_RANGE[1])

        # Use pre-settled qpos/qvel (computed once during __init__)
        pipeline_state = self.pipeline_init(self._settled_q, self._settled_qd)
        phase = jnp.array(0.0)
        obs = self._get_obs(pipeline_state, cmd_vel, cmd_yaw, phase)

        info = {
            'cmd_vel': cmd_vel,
            'cmd_yaw': cmd_yaw,
            'last_action': jnp.zeros(self._nu),
            'step_count': jnp.array(0, dtype=jnp.int32),
            'phase': phase,
            'rng': rng,
        }

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics={'forward_speed': jnp.array(0.0)},
            info=info,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, state: State, action: jax.Array) -> State:
        # Clip + EMA filter
        action = jnp.clip(action, -1.0, 1.0)
        last_action = state.info['last_action']
        filtered = ACTION_EMA * action + (1.0 - ACTION_EMA) * last_action

        # Step physics (N_FRAMES substeps)
        pipeline_state = self.pipeline_step(state.pipeline_state, filtered)

        # Phase clock
        phase = state.info['phase'] + CTRL_DT * PHASE_FREQ * 2.0 * jnp.pi
        phase = phase % (2.0 * jnp.pi)

        # ── Velocities ──
        fwd_speed = -pipeline_state.qvel[0]       # forward = -X
        yaw_rate = pipeline_state.qvel[5]          # yaw = rz
        lat_speed = jnp.abs(pipeline_state.qvel[1])

        cmd_vel = state.info['cmd_vel']
        cmd_yaw = state.info['cmd_yaw']

        # ── Reward ──
        r_vel_track = jnp.exp(-((fwd_speed - cmd_vel) ** 2) / (SIGMA_VEL ** 2))
        r_yaw_track = jnp.exp(-((yaw_rate - cmd_yaw) ** 2) / (SIGMA_YAW ** 2))

        r_vel_lin = (jnp.clip(fwd_speed, 0.0, cmd_vel)
                     / jnp.maximum(CMD_VEL_RANGE[1], 1e-6))

        overspeed_sq = jnp.maximum(0.0, fwd_speed - cmd_vel) ** 2
        backward = jnp.maximum(0.0, -fwd_speed)

        energy = jnp.sum(jnp.abs(
            pipeline_state.ctrl * pipeline_state.qvel[self._act_qvel_idx]))

        act_rate = jnp.sum(jnp.square(action - last_action))

        reward = (
            + W_VEL_TRACK * r_vel_track
            + W_YAW_TRACK * r_yaw_track
            + W_VEL_LIN   * r_vel_lin
            - W_OVERSPEED * overspeed_sq
            - W_LATERAL   * lat_speed
            - W_BACKWARD  * backward
            - W_ENERGY    * energy
            - W_SMOOTH    * act_rate
        )

        # ── Termination ──
        step_count = state.info['step_count'] + 1
        root_z = pipeline_state.xpos[self._root_body_id, 2]
        done = (root_z < 0.005) | (step_count >= MAX_EP_STEPS)
        done = done.astype(jnp.float32)

        # ── Command resampling (on episode end or random) ──
        rng = state.info['rng']
        rng, rk1, rk2, rk3 = random.split(rng, 4)
        resample = (random.uniform(rk1) < CMD_RESAMPLE_P) | (done > 0.5)
        cmd_vel = jnp.where(
            resample,
            random.uniform(rk2, minval=CMD_VEL_RANGE[0], maxval=CMD_VEL_RANGE[1]),
            cmd_vel)
        cmd_yaw = jnp.where(
            resample,
            random.uniform(rk3, minval=CMD_YAW_RANGE[0], maxval=CMD_YAW_RANGE[1]),
            cmd_yaw)

        obs = self._get_obs(pipeline_state, cmd_vel, cmd_yaw, phase)

        # Preserve wrapper-added fields in info and metrics
        info = {**state.info}
        info.update({
            'cmd_vel': cmd_vel,
            'cmd_yaw': cmd_yaw,
            'last_action': filtered,
            'step_count': step_count,
            'phase': phase,
            'rng': rng,
        })

        metrics = {**state.metrics}
        metrics['forward_speed'] = fwd_speed

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Observation (35-dim)
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self, data, cmd_vel, cmd_yaw, phase):
        # Normalized commands → [-1, 1]
        cmd_vel_norm = (2.0 * (cmd_vel - CMD_VEL_RANGE[0])
                        / (CMD_VEL_RANGE[1] - CMD_VEL_RANGE[0]) - 1.0)
        cmd_yaw_norm = cmd_yaw / max(abs(CMD_YAW_RANGE[0]), abs(CMD_YAW_RANGE[1]))

        # Joint positions (skip 7-DOF free joint) → 11 dims
        joint_pos = data.qpos[7:7 + self._nu]
        # Joint velocities (skip 6-DOF free joint) → 11 dims
        joint_vel = data.qvel[6:6 + self._nu]

        # Projected gravity in body frame → 3 dims
        rot = data.xmat[self._root_body_id].reshape(3, 3)
        proj_grav = rot.T @ jnp.array([0.0, 0.0, -1.0])

        # Base angular velocity → 3 dims
        base_angvel = data.qvel[3:6]
        # Base linear velocity → 3 dims
        base_linvel = data.qvel[0:3]

        # Phase clock → 2 dims
        return jnp.concatenate([
            jnp.array([cmd_vel_norm, cmd_yaw_norm]),    # 2
            joint_pos,                                   # 11
            joint_vel,                                   # 11
            proj_grav,                                   # 3
            base_angvel,                                 # 3
            base_linvel,                                 # 3
            jnp.array([jnp.sin(phase), jnp.cos(phase)]),# 2
        ])  # Total: 35


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing WormMJXEnv...")
    env = WormMJXEnv()
    print(f"  obs_size: {env.observation_size}")
    print(f"  act_size: {env.action_size}")

    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    print(f"  obs shape: {state.obs.shape}")
    print(f"  cmd_vel: {float(state.info['cmd_vel'])*1000:.1f} mm/s")
    print(f"  cmd_yaw: {float(state.info['cmd_yaw']):.3f} rad/s")

    # Step test
    action = jnp.zeros(env.action_size)
    state = env.step(state, action)
    print(f"  reward after 1 step: {float(state.reward):.3f}")
    print(f"  done: {float(state.done)}")

    # JIT compilation test
    jit_step = jax.jit(env.step)
    state = jit_step(state, action)
    print(f"  JIT step OK, reward: {float(state.reward):.3f}")

    # Vectorization test
    batch_size = 8
    batch_rng = jax.random.split(rng, batch_size)
    batch_reset = jax.vmap(env.reset)
    batch_states = batch_reset(batch_rng)
    print(f"  Batched reset OK: obs shape = {batch_states.obs.shape}")

    batch_step = jax.vmap(env.step)
    batch_actions = jnp.zeros((batch_size, env.action_size))
    batch_states = batch_step(batch_states, batch_actions)
    print(f"  Batched step OK: rewards = {batch_states.reward}")

    print("OK")

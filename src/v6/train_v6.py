"""
Worm Robot V6 — RL Training Script
====================================
Train the longworm2 robot with PPO using Stable-Baselines3.

Usage:
    # Quick self-test (CPU, 10k steps)
    python train_v6.py --test

    # Full local training (CPU is usually faster for SB3 MLP-PPO)
    python train_v6.py --timesteps 1000000 --device cpu

    # Resume from checkpoint
    python train_v6.py --timesteps 1000000 --device cpu --resume runs/worm_v6_ppo/best_model.zip
"""

import os
import argparse
import json
import shutil
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from training_contract_v6 import (
    DEFAULT_ENT_COEF,
    DEFAULT_LOG_STD_INIT,
    residual_exploration_contract,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
PAPER_TERRAINS = ("flat", "sand", "slope")
GAIT_MODES = ("worm", "snake", "mixed", "random")
FIXED_GAIT_BLEND_BY_MODE = {
    "worm": 0.0,
    "mixed": 0.5,
    "snake": 1.0,
}
RANDOM_POLICY_EVAL_BLENDS = (0.0, 0.5, 1.0)


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError(
                "--device cuda requested, but PyTorch is not importable"
            ) from exc
        return "cpu"
    cuda_ok = torch.cuda.is_available()
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError(
            "--device cuda requested, but torch.cuda.is_available() is false"
        )
    if requested == "cuda":
        return "cuda"
    return "cuda" if cuda_ok else "cpu"


def make_run_dirs(terrain='flat', gait_mode='random'):
    run_dir  = os.path.join(PROJECT_ROOT, "runs",
                            f"worm_v6_ppo_{terrain}_{gait_mode}")
    log_dir  = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    return run_dir, log_dir, ckpt_dir


def infer_vecnormalize_path(model_path):
    """Return the VecNormalize file paired with a final/best/checkpoint model."""
    stem, ext = os.path.splitext(model_path)
    if ext.lower() != ".zip":
        return None

    candidates = [f"{stem}_vecnormalize.pkl"]
    base = os.path.basename(model_path)
    directory = os.path.dirname(model_path)
    marker = "_steps.zip"
    if base.endswith(marker):
        step_part = base[:-len(marker)].split("_")[-1]
        prefix = base[:-len(step_part + marker)]
        candidates.append(os.path.join(
            directory, f"{prefix}vecnormalize_{step_part}_steps.pkl"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def close_float(value, expected, tol=1e-6):
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def dict_float_match(actual, expected):
    if not isinstance(actual, dict):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if not close_float(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def best_eval_schedule(gait_mode, gait_blend=None):
    """Return deterministic command cases used for best-model selection."""
    from worm_env_v6 import CMD_YAW_RANGE

    if gait_blend is not None:
        blends = (float(gait_blend),)
    elif gait_mode == "random":
        blends = RANDOM_POLICY_EVAL_BLENDS
    else:
        blends = (FIXED_GAIT_BLEND_BY_MODE[gait_mode],)

    yaw_cases = (float(CMD_YAW_RANGE[0]), 0.0, float(CMD_YAW_RANGE[1]))
    schedule = []
    for blend in blends:
        for yaw in yaw_cases:
            schedule.append({
                "gait_blend": float(blend),
                "cmd_yaw_rad_s": float(yaw),
            })
    return schedule


def best_eval_schedule_fingerprint(schedule):
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def comparable_training_fields(config):
    return {
        "terrain": config.get("terrain"),
        "gait_mode": config.get("gait_mode"),
        "gait_blend": config.get("gait_blend"),
        "obs_dim": config.get("obs_dim"),
        "num_actuators": config.get("num_actuators"),
        "num_imus": config.get("num_imus"),
    }


def training_config_compatible(existing, expected):
    if not isinstance(existing, dict):
        return False, ["missing training_config.json"]
    reasons = []
    if comparable_training_fields(existing) != comparable_training_fields(expected):
        reasons.append("training fields")
    if not dict_float_match(
            existing.get("sensor_robustness"),
            expected.get("sensor_robustness", {})):
        reasons.append("sensor_robustness")
    if not dict_float_match(
            existing.get("control_timing"),
            expected.get("control_timing", {})):
        reasons.append("control_timing")
    if existing.get("reward_contract") != expected.get("reward_contract"):
        reasons.append("reward_contract")
    if existing.get("eval_command") != expected.get("eval_command"):
        reasons.append("eval_command")
    if existing.get("action_adapter") != expected.get("action_adapter"):
        reasons.append("action_adapter")
    if (existing.get("residual_exploration")
            != expected.get("residual_exploration")):
        reasons.append("residual_exploration")
    expected_actuator = expected.get("actuator_contract_fingerprint")
    if (expected_actuator
            and existing.get("actuator_contract_fingerprint")
            != expected_actuator):
        reasons.append("actuator_contract")
    return not reasons, reasons


def run_dir_for_model(model_path):
    directory = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(directory) == "checkpoints":
        return os.path.dirname(directory)
    return directory


def resume_model_compatible(model_path, expected_config):
    config = read_json(os.path.join(
        run_dir_for_model(model_path), "training_config.json"))
    return training_config_compatible(config, expected_config)


def archive_existing_run_artifacts(run_dir, reason):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(
        run_dir, "incompatible_timing_archive", timestamp)
    names = [
        "best_model.zip",
        "best_model_vecnormalize.pkl",
        "final_model.zip",
        "final_model_vecnormalize.pkl",
        "training_config.json",
        "training_result.json",
        "best_eval_summary.json",
        "eval_metrics.json",
        "eval_metrics_robust.json",
        "checkpoints",
    ]
    moved = []
    for name in names:
        src = os.path.join(run_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(archive_dir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved.append(dst)
    if moved:
        marker = {
            "created_unix_time": time.time(),
            "reason": reason,
            "moved": moved,
        }
        os.makedirs(archive_dir, exist_ok=True)
        write_json(os.path.join(archive_dir, "archive_reason.json"), marker)
        print(f"  Archived incompatible previous artifacts: {archive_dir}")
    return moved


def make_env(terrain='flat', gait_mode='random', gait_blend=None,
             encoder_pos_noise_std=0.0, encoder_vel_noise_std=0.0,
             imu_gravity_noise_std=0.0, imu_gyro_noise_std=0.0,
             action_delay_steps=0, action_saturation=1.0, seed=0,
             fixed_cmd_vel=None, fixed_cmd_yaw=None,
             command_resample_prob=None, gait_prior_scale=None,
             policy_residual_scale=None):
    """Factory for creating a monitored WormEnvV6 instance."""
    def _init():
        from worm_env_v6 import WormEnvV6
        env = WormEnvV6(
            terrain=terrain, gait_mode=gait_mode, gait_blend=gait_blend,
            encoder_pos_noise_std=encoder_pos_noise_std,
            encoder_vel_noise_std=encoder_vel_noise_std,
            imu_gravity_noise_std=imu_gravity_noise_std,
            imu_gyro_noise_std=imu_gyro_noise_std,
            action_delay_steps=action_delay_steps,
            action_saturation=action_saturation,
            fixed_cmd_vel=fixed_cmd_vel,
            fixed_cmd_yaw=fixed_cmd_yaw,
            command_resample_prob=(
                command_resample_prob
                if command_resample_prob is not None else 0.005),
            **({} if gait_prior_scale is None else {
                "gait_prior_scale": gait_prior_scale}),
            **({} if policy_residual_scale is None else {
                "policy_residual_scale": policy_residual_scale}))
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def obs_layout_json(layout):
    return {key: [value.start, value.stop] for key, value in layout.items()}


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def best_eval_summary_path(run_dir):
    return os.path.join(run_dir, "best_eval_summary.json")


def has_best_model_pair(run_dir):
    return (
        os.path.exists(os.path.join(run_dir, "best_model.zip"))
        and os.path.exists(os.path.join(run_dir, "best_model_vecnormalize.pkl"))
    )


def load_persistent_best_eval(run_dir, log_dir, eval_schedule_fingerprint=None):
    if not has_best_model_pair(run_dir):
        return -np.inf

    summary = read_json(best_eval_summary_path(run_dir))
    if eval_schedule_fingerprint is not None:
        if ((summary or {}).get("eval_schedule_fingerprint")
                != eval_schedule_fingerprint):
            return -np.inf
    try:
        value = float((summary or {}).get("best_mean_reward"))
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    if eval_schedule_fingerprint is not None:
        return -np.inf

    eval_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(eval_path):
        return -np.inf
    try:
        data = np.load(eval_path)
        results = data["results"]
        if len(results) == 0:
            return -np.inf
        return float(np.max(np.mean(results, axis=1)))
    except (KeyError, OSError, ValueError):
        return -np.inf


def write_best_eval_summary(path, mean_reward, timestep,
                            eval_schedule=None,
                            eval_schedule_fingerprint=None):
    payload = {
        "format_version": 1,
        "updated_unix_time": time.time(),
        "best_mean_reward": float(mean_reward),
        "best_timestep": int(timestep),
    }
    if eval_schedule_fingerprint is not None:
        payload["eval_schedule_fingerprint"] = eval_schedule_fingerprint
    if eval_schedule is not None:
        payload["eval_schedule"] = eval_schedule
    write_json(path, payload)


def build_training_config(args, run_gait_label, sensor_kwargs, device):
    from motor_contract_v6 import motor_contract
    from worm_env_v6 import (
        CMD_VEL_RANGE,
        CMD_YAW_RANGE,
        CTRL_DT,
        NUM_ACTUATORS,
        NUM_IMUS,
        OBS_DIM,
        OBS_LAYOUT,
        PERISTALTIC_ACTUATION_PERIOD_S,
        PHASE_FREQ,
        action_adapter_contract,
        reward_contract,
    )

    return {
        "format_version": 1,
        "created_unix_time": time.time(),
        "algorithm": "PPO",
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": args.gait_blend,
        "run_gait_label": run_gait_label,
        "obs_dim": OBS_DIM,
        "obs_layout": obs_layout_json(OBS_LAYOUT),
        "num_actuators": NUM_ACTUATORS,
        "num_imus": NUM_IMUS,
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "action_adapter": action_adapter_contract(
            args.gait_prior_scale, args.policy_residual_scale),
        "residual_exploration": residual_exploration_contract(
            args.ent_coef, args.log_std_init),
        "reward_contract": reward_contract(),
        "deployable_observation_sources": [
            "velocity command",
            "yaw-rate command",
            "continuous gait_blend command",
            "actuated joint encoder positions",
            "actuated joint encoder velocities",
            "previous applied action",
            "per-segment IMU projected gravity",
            "per-segment IMU angular velocity",
            "phase clock",
        ],
        "forbidden_policy_sources": [
            "base linear velocity",
            "global position",
            "global yaw",
            "MuJoCo freejoint state as policy input",
            "external localization as policy input",
        ],
        "command_ranges": {
            "cmd_vel_m_s": list(CMD_VEL_RANGE),
            "cmd_yaw_rad_s": list(CMD_YAW_RANGE),
            "gait_blend": [0.0, 1.0],
        },
        "eval_command": {
            "cmd_vel_m_s": CMD_VEL_RANGE[1],
            "cmd_yaw_rad_s": 0.0,
            "command_resample_prob": 0.0,
        },
        "best_eval_schedule": {
            "cmd_vel_m_s": CMD_VEL_RANGE[1],
            "command_resample_prob": 0.0,
            "cases": best_eval_schedule(args.gait_mode, args.gait_blend),
        },
        "control_timing": {
            "control_dt_s": CTRL_DT,
            "control_rate_hz": 1.0 / CTRL_DT,
            "peristaltic_actuation_period_s": (
                PERISTALTIC_ACTUATION_PERIOD_S),
            "phase_freq_hz": PHASE_FREQ,
        },
        "sensor_robustness": sensor_kwargs,
        "training": {
            "timesteps": args.timesteps,
            "train_chunk_timesteps": args.train_chunk_timesteps,
            "n_envs": args.n_envs,
            "requested_device": getattr(args, "device", "cpu"),
            "resolved_device": device,
            "seed": 42,
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 1024,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": args.ent_coef,
            "log_std_init": args.log_std_init,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0,
        },
        "resume": args.resume,
    }


class NormSyncCallback(BaseCallback):
    """Sync VecNormalize stats from train → eval, and save alongside best model."""
    def __init__(self, train_env, eval_env, save_path, print_freq=5000):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.save_path = save_path
        self.print_freq = print_freq
        best_path = os.path.join(self.save_path, "best_model.zip")
        self._last_best = (
            os.path.getmtime(best_path) if os.path.exists(best_path) else None)

    def _on_step(self):
        # Sync obs normalization
        self.eval_env.obs_rms = self.train_env.obs_rms

        # Save VecNormalize when EvalCallback finds new best
        best_path = os.path.join(self.save_path, "best_model.zip")
        if os.path.exists(best_path):
            mtime = os.path.getmtime(best_path)
            if self._last_best is None or mtime > self._last_best:
                self._last_best = mtime
                norm_path = os.path.join(
                    self.save_path, "best_model_vecnormalize.pkl")
                self.train_env.save(norm_path)

        # Print progress
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mean_r = np.mean(ep_rewards)
                mean_l = np.mean(ep_lengths)
                print(f"  step {self.num_timesteps:>8d}  "
                      f"ep_reward={mean_r:>8.2f}  ep_len={mean_l:>6.0f}")
        return True


class PersistentBestEvalCallback(EvalCallback):
    def __init__(self, *args, persistent_best_mean=-np.inf,
                 persistent_path=None, eval_schedule=None,
                 eval_schedule_fingerprint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = float(persistent_best_mean)
        self.persistent_path = persistent_path
        self.eval_schedule = eval_schedule
        self.eval_schedule_fingerprint = eval_schedule_fingerprint

    def _on_step(self):
        previous_best = self.best_mean_reward
        keep_training = super()._on_step()
        if (self.persistent_path
                and self.best_mean_reward > previous_best
                and np.isfinite(self.best_mean_reward)):
            write_best_eval_summary(
                self.persistent_path,
                self.best_mean_reward,
                self.num_timesteps,
                eval_schedule=self.eval_schedule,
                eval_schedule_fingerprint=self.eval_schedule_fingerprint,
            )
        return keep_training


def train(args):
    from worm_env_v6 import CMD_VEL_RANGE

    terrain = args.terrain
    gait_mode = args.gait_mode
    gait_blend = args.gait_blend
    sensor_kwargs = dict(
        encoder_pos_noise_std=args.encoder_pos_noise,
        encoder_vel_noise_std=args.encoder_vel_noise,
        imu_gravity_noise_std=args.imu_gravity_noise,
        imu_gyro_noise_std=args.imu_gyro_noise,
        action_delay_steps=args.action_delay_steps,
        action_saturation=args.action_saturation,
    )
    run_gait_label = gait_mode
    if gait_blend is not None:
        run_gait_label = f"blend_{gait_blend:.2f}".replace(".", "p")
    RUN_DIR, LOG_DIR, CKPT_DIR = make_run_dirs(terrain, run_gait_label)
    device = resolve_device(args.device)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    training_config = build_training_config(
        args, run_gait_label, sensor_kwargs, device)
    config_path = os.path.join(RUN_DIR, "training_config.json")
    requested_resume = args.resume
    if requested_resume:
        resume_ok, resume_reasons = resume_model_compatible(
            requested_resume, training_config)
        if not resume_ok:
            print(
                "  WARNING: ignoring incompatible resume checkpoint "
                f"({', '.join(resume_reasons)}): {requested_resume}")
            args.resume = None

    existing_config = read_json(config_path)
    existing_ok, existing_reasons = training_config_compatible(
        existing_config, training_config)
    if not existing_ok:
        archive_existing_run_artifacts(
            RUN_DIR, ", ".join(existing_reasons) or "configuration mismatch")
        os.makedirs(CKPT_DIR, exist_ok=True)

    training_config = build_training_config(
        args, run_gait_label, sensor_kwargs, device)
    write_json(config_path, training_config)

    n_envs = args.n_envs
    print(f"Worm V6 RL Training — PPO (Longworm2) [{terrain}]")
    print(f"  terrain:    {terrain}")
    print(f"  gait_mode:  {gait_mode}")
    print(f"  gait_blend: {gait_blend if gait_blend is not None else 'mode/default'}")
    print(f"  sensor:     {sensor_kwargs}")
    print(f"  device:     {device} (requested: {args.device})")
    print(f"  envs:       {n_envs}")
    print(f"  timesteps:  {args.timesteps:,}")
    print(f"  run_dir:    {RUN_DIR}")
    print(f"  metadata:   {config_path}")

    # ── Create vectorized environments ──
    if n_envs == 1:
        raw_vec_env = DummyVecEnv([make_env(
            terrain=terrain, gait_mode=gait_mode,
            gait_blend=gait_blend, seed=42,
            gait_prior_scale=args.gait_prior_scale,
            policy_residual_scale=args.policy_residual_scale,
            **sensor_kwargs)])
    else:
        raw_vec_env = SubprocVecEnv(
            [make_env(terrain=terrain, gait_mode=gait_mode,
                      gait_blend=gait_blend, seed=42 + i,
                      gait_prior_scale=args.gait_prior_scale,
                      policy_residual_scale=args.policy_residual_scale,
                      **sensor_kwargs)
             for i in range(n_envs)])

    # Observation normalization — reward normalization DISABLED
    # ── Tensorboard ──
    try:
        import tensorboard  # noqa: F401
        tb_log = LOG_DIR
        print(f"  tensorboard: {LOG_DIR}")
    except ImportError:
        tb_log = None
        print(f"  tensorboard: not installed (logging disabled)")

    # ── Create or load model ──
    start_timesteps = 0
    learn_timesteps = args.timesteps
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        norm_path = infer_vecnormalize_path(args.resume)
        if norm_path:
            vec_env = VecNormalize.load(norm_path, raw_vec_env)
            vec_env.training = True
            vec_env.norm_reward = False
            print(f"  Loaded VecNormalize from: {norm_path}")
        else:
            print("  WARNING: no paired VecNormalize file found; "
                  "observation normalization will start from defaults")
            vec_env = VecNormalize(
                raw_vec_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )
        model = PPO.load(args.resume, env=vec_env, device=device)
        start_timesteps = int(model.num_timesteps)
        learn_timesteps = max(args.timesteps - start_timesteps, 0)
        print(f"  resume_start_timesteps: {start_timesteps:,}")
        print(f"  remaining_to_target:     {learn_timesteps:,}")
    else:
        vec_env = VecNormalize(
            raw_vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=args.log_std_init,
            ),
            tensorboard_log=tb_log,
            verbose=0,
            device=device,
            seed=42,
        )
    if args.train_chunk_timesteps is not None:
        learn_timesteps = min(learn_timesteps, args.train_chunk_timesteps)
        print(f"  chunk_timesteps:         {learn_timesteps:,}")

    print(f"  Policy network: {model.policy}")

    # ── Callbacks ──
    eval_schedule = best_eval_schedule(gait_mode, gait_blend)
    eval_schedule_fp = best_eval_schedule_fingerprint(eval_schedule)
    print("  best_eval_schedule:")
    for case in eval_schedule:
        print("    "
              f"blend={case['gait_blend']:.2f} "
              f"cmd_yaw={case['cmd_yaw_rad_s']:+.1f}")

    eval_env = DummyVecEnv([
        make_env(
            terrain=terrain, gait_mode=gait_mode,
            gait_blend=case["gait_blend"], seed=999 + idx,
            fixed_cmd_vel=CMD_VEL_RANGE[1],
            fixed_cmd_yaw=case["cmd_yaw_rad_s"],
            command_resample_prob=0.0,
            gait_prior_scale=args.gait_prior_scale,
            policy_residual_scale=args.policy_residual_scale,
            **sensor_kwargs)
        for idx, case in enumerate(eval_schedule)
    ])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False)
    eval_env.obs_rms = vec_env.obs_rms

    persistent_best = load_persistent_best_eval(
        RUN_DIR, LOG_DIR, eval_schedule_fingerprint=eval_schedule_fp)
    if np.isfinite(persistent_best):
        print(f"  persistent_best_eval_reward: {persistent_best:.2f}")
    else:
        print("  persistent_best_eval_reward: reset for eval schedule")

    eval_callback = PersistentBestEvalCallback(
        eval_env,
        best_model_save_path=RUN_DIR,
        log_path=LOG_DIR,
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=len(eval_schedule),
        deterministic=True,
        persistent_best_mean=persistent_best,
        persistent_path=best_eval_summary_path(RUN_DIR),
        eval_schedule=eval_schedule,
        eval_schedule_fingerprint=eval_schedule_fp,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // n_envs, 1),
        save_path=CKPT_DIR,
        name_prefix="worm_v6_ppo",
        save_vecnormalize=True,
    )

    norm_sync = NormSyncCallback(
        train_env=vec_env, eval_env=eval_env,
        save_path=RUN_DIR,
        print_freq=max(5000 // n_envs, 1),
    )

    # ── Train ──
    if learn_timesteps > 0:
        print(f"\n  Training started...")
        model.learn(
            total_timesteps=learn_timesteps,
            callback=[eval_callback, checkpoint_callback, norm_sync],
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )
    else:
        print("\n  Target timesteps already reached; saving current artifacts.")

    # ── Save ──
    final_path = os.path.join(RUN_DIR, "final_model")
    model.save(final_path)
    vec_env.save(f"{final_path}_vecnormalize.pkl")
    result_path = os.path.join(RUN_DIR, "training_result.json")
    result = dict(training_config)
    result["completed_unix_time"] = time.time()
    result["resume_start_timesteps"] = start_timesteps
    result["target_timesteps"] = int(args.timesteps)
    result["train_chunk_timesteps"] = (
        int(args.train_chunk_timesteps)
        if args.train_chunk_timesteps is not None else None)
    result["completed_timesteps"] = int(model.num_timesteps)
    result["artifacts"] = {
        "final_model": f"{final_path}.zip",
        "final_vecnormalize": f"{final_path}_vecnormalize.pkl",
        "best_model": os.path.join(RUN_DIR, "best_model.zip"),
        "best_vecnormalize": os.path.join(
            RUN_DIR, "best_model_vecnormalize.pkl"),
        "best_eval_summary": best_eval_summary_path(RUN_DIR),
        "training_config": config_path,
    }
    best_eval = read_json(best_eval_summary_path(RUN_DIR))
    if best_eval:
        result["best_eval"] = best_eval
    write_json(result_path, result)
    print(f"\n  Saved final model: {final_path}.zip")
    print(f"  Saved VecNormalize: {final_path}_vecnormalize.pkl")
    print(f"  Saved training result: {result_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train worm V6 robot with PPO")
    ap.add_argument("--terrain", type=str, default="flat",
                    choices=["flat", "sand", "slope", "rough", "steps", "channel"],
                    help="Terrain type (default: flat)")
    ap.add_argument("--gait-mode", type=str, default="random",
                    choices=GAIT_MODES,
                    help="Mode command: worm=0, mixed=0.5, snake=1, random=sample per episode")
    ap.add_argument("--gait-blend", type=float, default=None,
                    help="Override gait blend in [0, 1]")
    ap.add_argument("--encoder-pos-noise", type=float, default=0.0,
                    help="Normalized encoder position noise std")
    ap.add_argument("--encoder-vel-noise", type=float, default=0.0,
                    help="Normalized encoder velocity noise std")
    ap.add_argument("--imu-gravity-noise", type=float, default=0.0,
                    help="Projected-gravity IMU noise std")
    ap.add_argument("--imu-gyro-noise", type=float, default=0.0,
                    help="Normalized gyro noise std")
    ap.add_argument("--action-delay-steps", type=int, default=0,
                    help="Integer control-step delay before action is applied")
    ap.add_argument("--action-saturation", type=float, default=1.0,
                    help="Applied action limit in [0, 1] before actuator scaling")
    ap.add_argument("--gait-prior-scale", type=float, default=1.0,
                    help="Scale for deterministic phase/gait_blend action prior")
    ap.add_argument("--policy-residual-scale", type=float, default=0.35,
                    help="Scale applied to PPO residual before adding gait prior")
    ap.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF,
                    help="PPO entropy coefficient for residual exploration")
    ap.add_argument("--log-std-init", type=float, default=DEFAULT_LOG_STD_INIT,
                    help="Initial Gaussian log std for residual policy actions")
    ap.add_argument("--timesteps", type=int, default=1_000_000,
                    help="Total training timesteps")
    ap.add_argument("--train-chunk-timesteps", type=int, default=None,
                    help="Train at most this many additional timesteps "
                         "while keeping --timesteps as the formal target")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="Number of parallel environments")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["auto", "cpu", "cuda"],
                    help="PPO network device; MuJoCo env stepping remains CPU-bound")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to model checkpoint to resume from")
    ap.add_argument("--test", action="store_true",
                    help="Quick test run (10k steps, 1 env)")
    args = ap.parse_args()

    if args.test:
        args.timesteps = 10_000
        args.n_envs = 1
        print("=== TEST MODE (10k steps, 1 env) ===")

    import sys
    sys.path.insert(0, SCRIPT_DIR)

    train(args)

"""
Export and replay a deployable Worm V6 policy.

The export path freezes the deterministic PPO actor plus VecNormalize
observation statistics into a TorchScript module. The replay path consumes a
hardware CSV with the 80-D deployable observation columns and writes normalized
11-D actions plus physical joint targets, so the hardware log schema can be
tested without MuJoCo or SB3.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from motor_contract_v6 import (  # noqa: E402
    NUM_SLIDES,
    NUM_YAWS,
    action_mapping_config,
    motor_contract,
)
from action_adapter_v6 import (  # noqa: E402
    DEFAULT_GAIT_PRIOR_SCALE,
    DEFAULT_POLICY_RESIDUAL_SCALE,
    SNAKE_AMP_NORMALIZED,
    action_adapter_contract,
    compose_deployable_action,
    phase_from_clock,
)


class DeployablePPOActor(torch.nn.Module):
    def __init__(
            self, policy, obs_mean, obs_var, epsilon, clip_obs,
            gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
            policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.register_buffer(
            "obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer(
            "obs_var", torch.as_tensor(obs_var, dtype=torch.float32))
        self.epsilon = float(epsilon)
        self.clip_obs = float(clip_obs)
        self.gait_prior_scale = float(gait_prior_scale)
        self.policy_residual_scale = float(policy_residual_scale)
        self.register_buffer(
            "slide_offsets",
            torch.arange(NUM_SLIDES, dtype=torch.float32)
            * (2.0 * torch.pi / float(NUM_SLIDES)))
        self.register_buffer(
            "yaw_offsets",
            torch.arange(NUM_YAWS, dtype=torch.float32)
            * (2.0 * torch.pi * 1.5 / float(NUM_YAWS)))

    def forward(self, raw_obs):
        if raw_obs.dim() == 1:
            raw_obs = raw_obs.unsqueeze(0)
        raw_obs = raw_obs.to(dtype=torch.float32)
        obs = (raw_obs - self.obs_mean) * torch.rsqrt(
            self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        residual = torch.clamp(self.action_net(latent_pi), -1.0, 1.0)
        phase = torch.atan2(raw_obs[:, 78:79], raw_obs[:, 79:80])
        gait_blend = torch.clamp(raw_obs[:, 2:3], 0.0, 1.0)
        slide_prior = (
            -0.5
            * (1.0 + torch.sin(phase + self.slide_offsets.unsqueeze(0)))
            * (1.0 - gait_blend))
        yaw_prior = (
            float(SNAKE_AMP_NORMALIZED)
            * torch.sin(phase + self.yaw_offsets.unsqueeze(0))
            * gait_blend)
        prior = torch.cat([slide_prior, yaw_prior], dim=1)
        actions = (
            self.gait_prior_scale * prior
            + self.policy_residual_scale * residual)
        return torch.clamp(actions, -1.0, 1.0)


def default_run_dir(terrain, gait_mode):
    return os.path.join(PROJECT_ROOT, "runs",
                        f"worm_v6_ppo_{terrain}_{gait_mode}")


def default_model_path(run_dir):
    for name in ("best_model.zip", "final_model.zip"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(run_dir, "best_model.zip")


def find_vecnormalize(model_path):
    candidates = [
        model_path.replace(".zip", "_vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_path), "best_model_vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_path), "final_model_vecnormalize.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def gait_blend_for(mode, override):
    if override is not None:
        return float(np.clip(override, 0.0, 1.0))
    if mode == "worm":
        return 0.0
    if mode == "snake":
        return 1.0
    if mode == "mixed":
        return 0.5
    return 0.5


def layout_to_json(layout):
    return {key: [value.start, value.stop] for key, value in layout.items()}


def load_vecnormalize_stats(vec_path, terrain, gait_mode, gait_blend, obs_dim):
    if vec_path is None:
        return {
            "mean": np.zeros(obs_dim, dtype=np.float32),
            "var": np.ones(obs_dim, dtype=np.float32),
            "epsilon": 1e-8,
            "clip_obs": 10.0,
            "path": None,
        }

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from worm_env_v6 import WormEnvV6

    vec_env = DummyVecEnv([lambda: WormEnvV6(
        terrain=terrain, gait_mode=gait_mode, gait_blend=gait_blend)])
    vec_norm = VecNormalize.load(vec_path, vec_env)
    try:
        return {
            "mean": vec_norm.obs_rms.mean.astype(np.float32),
            "var": vec_norm.obs_rms.var.astype(np.float32),
            "epsilon": float(vec_norm.epsilon),
            "clip_obs": float(vec_norm.clip_obs),
            "path": vec_path,
        }
    finally:
        vec_norm.close()


def write_config(path, config):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_policy(args):
    sys.path.insert(0, SCRIPT_DIR)
    from stable_baselines3 import PPO
    from observation_contract_v6 import attach_contract_to_config
    from validate_hardware_log_v6 import observation_columns
    from worm_env_v6 import (
        CMD_VEL_RANGE,
        CMD_YAW_RANGE,
        CTRL_DT,
        NUM_ACTUATORS,
        NUM_SLIDES,
        OBS_DIM,
        OBS_LAYOUT,
        PERISTALTIC_ACTUATION_PERIOD_S,
        PHASE_FREQ,
    )

    run_dir = args.run_dir or default_run_dir(args.terrain, args.gait_mode)
    model_path = args.model or default_model_path(run_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "record", "v6", "deploy_bundles",
        f"{args.terrain}_{args.gait_mode}")
    gait_blend = gait_blend_for(args.gait_mode, args.gait_blend)
    vec_path = args.vecnormalize or find_vecnormalize(model_path)
    norm = load_vecnormalize_stats(
        vec_path, args.terrain, args.gait_mode, gait_blend, OBS_DIM)

    model = PPO.load(model_path, device="cpu")
    model.policy.eval()

    actor = DeployablePPOActor(
        model.policy,
        norm["mean"],
        norm["var"],
        norm["epsilon"],
        norm["clip_obs"],
    )
    actor.eval()

    dummy_raw = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        actor_action = actor(dummy_raw).cpu().numpy()
        dummy_norm = (
            (dummy_raw.cpu().numpy() - norm["mean"]) /
            np.sqrt(norm["var"] + norm["epsilon"]))
        dummy_norm = np.clip(dummy_norm, -norm["clip_obs"], norm["clip_obs"])
        sb3_action, _ = model.predict(dummy_norm, deterministic=True)
        expected_action = compose_deployable_action(
            sb3_action[0],
            phase=phase_from_clock(dummy_raw[0, 78], dummy_raw[0, 79]),
            gait_blend=dummy_raw[0, 2],
        )[None, :]
        max_diff = float(np.max(np.abs(actor_action - expected_action)))
        if max_diff > args.max_export_diff:
            raise RuntimeError(
                f"Exported deployable actor mismatch: max_diff={max_diff:.3g}")

    os.makedirs(out_dir, exist_ok=True)
    actor_path = os.path.join(out_dir, "policy_actor.pt")
    traced = torch.jit.trace(actor, dummy_raw, check_trace=True)
    traced.save(actor_path)

    action_columns = [f"action_{i:02d}" for i in range(NUM_ACTUATORS)]
    config = {
        "format_version": 1,
        "model_type": "sb3_ppo_deterministic_actor_torchscript",
        "terrain": args.terrain,
        "gait_mode": args.gait_mode,
        "gait_blend": gait_blend,
        "obs_dim": OBS_DIM,
        "obs_layout": layout_to_json(OBS_LAYOUT),
        "observation_columns": observation_columns(),
        "action_dim": NUM_ACTUATORS,
        "action_columns": action_columns,
        "action_range": [-1.0, 1.0],
        "action_adapter": action_adapter_contract(),
        "action_mapping": action_mapping_config(),
        "actuator_contract_fingerprint": (
            motor_contract()["contract_fingerprint"]),
        "actuator_contract": motor_contract(),
        "command_ranges": {
            "cmd_vel_m_s": list(CMD_VEL_RANGE),
            "cmd_yaw_rad_s": list(CMD_YAW_RANGE),
            "gait_blend": [0.0, 1.0],
        },
        "control_timing": {
            "control_dt_s": CTRL_DT,
            "control_rate_hz": 1.0 / CTRL_DT,
            "peristaltic_actuation_period_s": (
                PERISTALTIC_ACTUATION_PERIOD_S),
            "phase_freq_hz": PHASE_FREQ,
        },
        "normalization": {
            "mean": norm["mean"].tolist(),
            "var": norm["var"].tolist(),
            "epsilon": norm["epsilon"],
            "clip_obs": norm["clip_obs"],
            "source": norm["path"],
        },
        "source_model": model_path,
        "torchscript_actor": actor_path,
        "export_validation": {
            "dummy_obs_max_abs_diff_vs_sb3_predict": max_diff,
        },
    }
    config = attach_contract_to_config(config)
    config_path = os.path.join(out_dir, "deploy_config.json")
    write_config(config_path, config)

    result = {
        "bundle_dir": out_dir,
        "actor": actor_path,
        "config": config_path,
        "source_model": model_path,
        "vecnormalize": norm["path"],
        "max_export_diff": max_diff,
    }
    print(json.dumps(result, indent=2))
    return result


def default_actions_path(input_csv):
    root, ext = os.path.splitext(input_csv)
    return f"{root}_actions{ext or '.csv'}"


def action_mapping_fields(config):
    action_dim = int(config["action_dim"])
    defaults = action_mapping_config()
    action_mapping = {**defaults, **config.get("action_mapping", {})}
    slide_start, slide_stop = action_mapping.get("slide_indices", [0, 0])
    yaw_start, yaw_stop = action_mapping.get("yaw_indices", [slide_stop, action_dim])
    slide_range_m = float(action_mapping.get("slide_range_m", 1.0))
    yaw_range_rad = float(action_mapping.get("yaw_range_rad", 1.0))
    slide_min_m = float(action_mapping.get("slide_min_m", -slide_range_m))
    slide_max_m = float(action_mapping.get("slide_max_m", slide_range_m))
    yaw_min_rad = float(action_mapping.get("yaw_min_rad", -yaw_range_rad))
    yaw_max_rad = float(action_mapping.get("yaw_max_rad", yaw_range_rad))
    slide_target_cols = [
        f"slide_target_m_{i - slide_start:02d}"
        for i in range(slide_start, slide_stop)
    ]
    yaw_target_cols = [
        f"yaw_target_rad_{i - yaw_start:02d}"
        for i in range(yaw_start, yaw_stop)
    ]
    return {
        "slide_start": slide_start,
        "slide_stop": slide_stop,
        "yaw_start": yaw_start,
        "yaw_stop": yaw_stop,
        "slide_range_m": slide_range_m,
        "slide_min_m": slide_min_m,
        "slide_max_m": slide_max_m,
        "yaw_range_rad": yaw_range_rad,
        "yaw_min_rad": yaw_min_rad,
        "yaw_max_rad": yaw_max_rad,
        "slide_target_cols": slide_target_cols,
        "yaw_target_cols": yaw_target_cols,
    }


def action_targets(config, action):
    mapping = action_mapping_fields(config)
    action = np.asarray(action, dtype=np.float32)
    return {
        "slide_targets_m": np.clip(
            action[mapping["slide_start"]:mapping["slide_stop"]] *
            mapping["slide_range_m"],
            mapping["slide_min_m"],
            mapping["slide_max_m"],
        ).astype(np.float32),
        "yaw_targets_rad": np.clip(
            action[mapping["yaw_start"]:mapping["yaw_stop"]] *
            mapping["yaw_range_rad"],
            mapping["yaw_min_rad"],
            mapping["yaw_max_rad"],
        ).astype(np.float32),
        "mapping": mapping,
    }


def replay_csv(args):
    config_path = args.config or os.path.join(args.bundle_dir, "deploy_config.json")
    actor_path = args.actor or os.path.join(args.bundle_dir, "policy_actor.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"TorchScript actor not found: {actor_path}")

    config = read_config(config_path)
    obs_cols = config["observation_columns"]
    action_cols = config["action_columns"]
    obs_dim = int(config["obs_dim"])
    action_dim = int(config["action_dim"])
    output_csv = args.output_csv or default_actions_path(args.input_csv)
    mapping = action_mapping_fields(config)
    slide_target_cols = mapping["slide_target_cols"]
    yaw_target_cols = mapping["yaw_target_cols"]

    actor = torch.jit.load(actor_path, map_location="cpu")
    actor.eval()

    rows = []
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        missing = [col for col in obs_cols if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing observation columns: {missing}")
        for row_index, row in enumerate(reader):
            obs = np.array([float(row[col]) for col in obs_cols], dtype=np.float32)
            if obs.shape != (obs_dim,) or not np.all(np.isfinite(obs)):
                raise ValueError(f"Invalid observation at row {row_index + 1}")
            rows.append((row_index, row, obs))

    if not rows:
        raise ValueError("Input CSV contains no data rows")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    metadata_cols = [
        col for col in ("time_s", "terrain", "mode", "video_file")
        if col in rows[0][1]
    ]
    fieldnames = [
        "row_index", *metadata_cols, *action_cols,
        *slide_target_cols, *yaw_target_cols,
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start:start + args.batch_size]
                obs_batch = np.stack([item[2] for item in chunk], axis=0)
                actions = actor(torch.from_numpy(obs_batch)).cpu().numpy()
                if actions.shape != (len(chunk), action_dim):
                    raise RuntimeError(
                        f"Actor returned shape {actions.shape}, "
                        f"expected {(len(chunk), action_dim)}")
                if not np.all(np.isfinite(actions)):
                    raise RuntimeError("Actor returned non-finite actions")
                for (row_index, row, _), action in zip(chunk, actions):
                    targets = action_targets(config, action)
                    out = {"row_index": row_index}
                    for col in metadata_cols:
                        out[col] = row[col]
                    for col, value in zip(action_cols, action):
                        out[col] = f"{float(value):.8f}"
                    for col, value in zip(
                            slide_target_cols, targets["slide_targets_m"]):
                        out[col] = f"{float(value):.8f}"
                    for col, value in zip(
                            yaw_target_cols, targets["yaw_targets_rad"]):
                        out[col] = f"{float(value):.8f}"
                    writer.writerow(out)

    metrics = {
        "input_csv": args.input_csv,
        "output_csv": output_csv,
        "rows": len(rows),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "slide_target_columns": slide_target_cols,
        "yaw_target_columns": yaw_target_cols,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export or replay a deployable Worm V6 policy")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Export PPO policy to a deploy bundle")
    export.add_argument("--model", default=None, help="Path to PPO .zip model")
    export.add_argument("--vecnormalize", default=None,
                        help="Path to VecNormalize .pkl stats")
    export.add_argument("--run-dir", default=None, help="Run directory override")
    export.add_argument("--terrain", default="flat",
                        choices=["flat", "sand", "slope", "rough", "steps",
                                 "channel"])
    export.add_argument("--gait-mode", default="random",
                        choices=["worm", "snake", "mixed", "random"])
    export.add_argument("--gait-blend", type=float, default=None)
    export.add_argument("--out-dir", default=None)
    export.add_argument("--max-export-diff", type=float, default=1e-5)

    replay = sub.add_parser(
        "replay", help="Replay hardware observation CSV through an export")
    replay.add_argument("--bundle-dir", default=os.path.join(
        PROJECT_ROOT, "record", "v6", "deploy_bundles", "flat_random"))
    replay.add_argument("--config", default=None,
                        help="deploy_config.json override")
    replay.add_argument("--actor", default=None,
                        help="policy_actor.pt override")
    replay.add_argument("--input-csv", required=True,
                        help="Hardware CSV with 80-D observation columns")
    replay.add_argument("--output-csv", default=None,
                        help="Output action CSV")
    replay.add_argument("--batch-size", type=int, default=256)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "export":
        export_policy(args)
    elif args.command == "replay":
        replay_csv(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

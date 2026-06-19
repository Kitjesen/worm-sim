"""
Create a PPO checkpoint with a copied actor and a freshly initialized critic.

This is used when changing only the value-function architecture. SB3 cannot
load an old checkpoint into a different critic shape, but the actor can still be
reused if the policy network architecture is unchanged.
"""

import argparse
import json
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_v6 import (
    DEFAULT_POLICY_NET_ARCH,
    DEFAULT_VALUE_NET_ARCH,
    build_training_config,
    infer_vecnormalize_path,
    make_env,
    parse_net_arch_arg,
)
from training_contract_v6 import DEFAULT_ENT_COEF, DEFAULT_LOG_STD_INIT
from action_adapter_v6 import (
    DEFAULT_GAIT_PRIOR_SCALE,
    DEFAULT_POLICY_RESIDUAL_SCALE,
)


def copy_module_state(target_module, source_module):
    source_state = source_module.state_dict()
    target_module.load_state_dict(source_state, strict=True)
    max_diff = 0.0
    target_state = target_module.state_dict()
    for name, source_tensor in source_state.items():
        diff = torch.max(torch.abs(
            target_state[name].detach().cpu()
            - source_tensor.detach().cpu())).item()
        max_diff = max(max_diff, float(diff))
    return max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", required=True)
    ap.add_argument("--source-vecnormalize", default=None)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--gait-mode", default="random")
    ap.add_argument("--command-curriculum", default="continuous_omni")
    ap.add_argument("--policy-net-arch", default="512,256,128")
    ap.add_argument("--value-net-arch", default="512,256,128")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    policy_net_arch = parse_net_arch_arg(
        args.policy_net_arch, DEFAULT_POLICY_NET_ARCH)
    value_net_arch = parse_net_arch_arg(
        args.value_net_arch, DEFAULT_VALUE_NET_ARCH)

    raw_env = DummyVecEnv([make_env(
        terrain=args.terrain,
        gait_mode=args.gait_mode,
        seed=args.seed,
        command_curriculum=args.command_curriculum,
    )])
    source_vecnormalize = (
        args.source_vecnormalize
        or infer_vecnormalize_path(args.source_model)
    )
    if source_vecnormalize:
        vec_env = VecNormalize.load(source_vecnormalize, raw_env)
        vec_env.training = True
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(
            raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    source = PPO.load(args.source_model, device=args.device)
    target = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=DEFAULT_ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=policy_net_arch, vf=value_net_arch),
            log_std_init=DEFAULT_LOG_STD_INIT,
        ),
        verbose=0,
        device=args.device,
        seed=args.seed,
    )

    policy_net_diff = copy_module_state(
        target.policy.mlp_extractor.policy_net,
        source.policy.mlp_extractor.policy_net,
    )
    action_net_diff = copy_module_state(
        target.policy.action_net,
        source.policy.action_net,
    )
    target.policy.log_std.data.copy_(source.policy.log_std.data)
    log_std_diff = torch.max(torch.abs(
        target.policy.log_std.detach().cpu()
        - source.policy.log_std.detach().cpu())).item()
    target.num_timesteps = int(source.num_timesteps)

    out_model = os.path.abspath(args.out_model)
    if out_model.lower().endswith(".zip"):
        out_model = out_model[:-4]
    out_dir = os.path.dirname(out_model)
    os.makedirs(out_dir, exist_ok=True)
    target.save(out_model)
    vec_path = f"{out_model}_vecnormalize.pkl"
    vec_env.save(vec_path)

    config_args = argparse.Namespace(
        terrain=args.terrain,
        gait_mode=args.gait_mode,
        gait_blend=None,
        run_label=os.path.basename(out_dir),
        command_curriculum=args.command_curriculum,
        command_resample_prob=None,
        gait_prior_scale=DEFAULT_GAIT_PRIOR_SCALE,
        policy_residual_scale=DEFAULT_POLICY_RESIDUAL_SCALE,
        ent_coef=DEFAULT_ENT_COEF,
        log_std_init=DEFAULT_LOG_STD_INIT,
        policy_net_arch=args.policy_net_arch,
        value_net_arch=args.value_net_arch,
        timesteps=int(target.num_timesteps),
        train_chunk_timesteps=None,
        n_envs=1,
        device=args.device,
        directional_eval_freq_steps=10000,
        directional_eval_seconds=6.0,
        resume=args.source_model,
    )
    config = build_training_config(
        config_args,
        run_gait_label=args.gait_mode,
        sensor_kwargs={
            "encoder_pos_noise_std": 0.0,
            "encoder_vel_noise_std": 0.0,
            "imu_gravity_noise_std": 0.0,
            "imu_gyro_noise_std": 0.0,
            "action_delay_steps": 0,
            "action_saturation": 1.0,
        },
        device=args.device,
    )
    config_path = os.path.join(out_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    summary = {
        "format_version": 1,
        "source_model": os.path.abspath(args.source_model),
        "source_vecnormalize": (
            os.path.abspath(source_vecnormalize)
            if source_vecnormalize else None),
        "out_model": f"{out_model}.zip",
        "out_vecnormalize": vec_path,
        "training_config": config_path,
        "copied_actor": True,
        "copied_critic": False,
        "source_num_timesteps": int(source.num_timesteps),
        "target_num_timesteps": int(target.num_timesteps),
        "policy_net_arch": {
            "pi": policy_net_arch,
            "vf": value_net_arch,
        },
        "max_policy_net_diff": float(policy_net_diff),
        "max_action_net_diff": float(action_net_diff),
        "max_log_std_diff": float(log_std_diff),
    }
    summary_path = f"{out_model}_transfer_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    vec_env.close()


if __name__ == "__main__":
    main()

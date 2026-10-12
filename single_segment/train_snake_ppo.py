"""Legacy gait-prior SOFA PPO comparisons; navigation uses move_control.py --task goals.

Worm/mixed modes tie the two tendon servos of each segment together. These old
paired-action experiments and their frozen checkpoints are not the current
14-independent-action navigation route.
"""
from snake_rl_env import SnakeResidualEnv  # Initializes SOFA before Torch/NumPy.
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack

HERE=Path(__file__).resolve().parent
SOURCES=['sofa_worm_env.py','parameters_sofa_candidate.json','sensor_observation.py',
         'snake_rl_env.py','train_snake_ppo.py']


def write_json(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False))
    os.replace(temporary,path)


def evaluate(checkpoint, learned, physics_settings):
    """One nominal and four randomized seeds, identical conditions for both controllers."""
    raw=VecMonitor(SubprocVecEnv([partial(SnakeResidualEnv,randomize=i>0,**physics_settings) for i in range(5)],start_method='spawn'))
    norm=VecNormalize.load(str(checkpoint/'normalization.pkl'),raw)
    norm.training=False; norm.norm_reward=False
    env=VecFrameStack(norm,n_stack=4)
    try:
        model=PPO.load(checkpoint/'model.zip',device='cpu') if learned else None
        env.seed(1000); obs=env.reset(); results={}
        for _ in range(120):
            action=model.predict(obs,deterministic=True)[0] if learned else np.zeros((5,*env.action_space.shape))
            obs,_,dones,infos=env.step(action)
            for i,done in enumerate(dones):
                if done and i not in results:
                    keys=['physical_s','mean_plate_forward_m','lateral_m','mean_sampled_slip_m_s',
                          'joint_abs_work_proxy_j','servo_abs_work_proxy_j','total_abs_work_proxy_j',
                          'peak_joint_deg','peak_tension_n','task_return','success']
                    results[i]=dict(seed=1000+i,randomized=i>0,**{k:infos[i][k] for k in keys})
            if len(results)==5:
                break
        assert len(results)==5
        return [results[i] for i in range(5)]
    finally:
        env.close()


def train(args):
    if args.mode in ('worm', 'mixed') and not getattr(args, 'allow_paired_legacy', False):
        raise ValueError('Legacy worm/mixed modes bind left and right tendons. Use '
            'move_control.py --task goals for 10 independent tendon + 4 yaw targets. '
            'Use --allow-paired-legacy only for an intentional paired-gait comparison; '
            'its old checkpoint/action dimensions are incompatible with GoalEnv.')
    if args.mode in ('worm', 'mixed'):
        print('LEGACY COMPARISON: paired left/right tendons; not independent navigation control.', flush=True)
    gate=json.loads(args.gate.read_text())
    if not gate.get('passed'):
        raise ValueError('A passing residual-environment gate is required')
    for name,digest in gate['source_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Gate source mismatch: '+name)
    physics_settings=dict(mode=args.mode,beam_model='saint_venant',rolling=args.rolling,elements=args.elements)
    if gate.get('physics_settings') != physics_settings:
        raise ValueError('Gate settings do not match training')
    plugin=Path(os.environ['STRIP_SECTION_PLUGIN'])
    if hashlib.sha256(plugin.read_bytes()).hexdigest()!=gate['plugin_sha256']:
        raise ValueError('Gate plugin mismatch')
    n={'worm':5,'snake':4,'mixed':9}[args.mode]
    args.out.mkdir(parents=True,exist_ok=False)
    source=args.out/'source'; source.mkdir()
    for name in SOURCES:
        (source/name).write_bytes((HERE/name).read_bytes())
    config=dict(started_utc=datetime.now(timezone.utc).isoformat(),requested_policy_steps=args.steps,
        workers=args.workers,max_training_hours=args.hours,seed=7,algorithm='SB3 PPO MlpPolicy',
        legacy_paired_tendon_comparison=args.mode in ('worm', 'mixed'),
        sb3_version=stable_baselines3.__version__,torch_version=torch.__version__,device='cpu',physics='SOFA CPU',
        policy_dt_s=.1,controller_dt_s=.02,physics_dt_s=.0005,episode_s=12,
        physics_settings=physics_settings,plugin_sha256=gate['plugin_sha256'],
        raw_observation_dim=50+n,history_frames=4,policy_observation_dim=4*(50+n),action_dim=n,
        observation='50 encoder/IMU/controller inputs + filtered residual commands; no privileged physical state',
        action='worm: five paired-servo amplitude residuals +/-20%, 1.5 rad peak prior, 4 s period, backward phase onset delay .65 s per segment; snake: four yaw residuals +/-5 deg on 20 deg/4 s/90 deg sine prior, clipped +/-20 deg; mixed: both',
        reward='sum at 50 Hz: 20*forward_dx -10*abs(lateral_dy) -dt*(50*sampled_slip +0.2*sampled_abs_yaw_power +0.05*mean(filtered_action^2)) -0.2*abs_servo_work_proxy; minus 0.01*mean(action_delta^2) per decision and 2 on failure; no commanded speed',
        normalization='VecNormalize observations only, saved with each checkpoint; frozen in evaluation',
        randomization='Per episode: E x U(.85,1.15), stiffness Rayleigh damping x U(.8,1.2), plate mass x U(.9,1.1), mu x U(.8,1.2)',
        sensor_assumptions=dict(angle_noise_rad=.002,gyro_noise_rad_s=.01,accel_noise_m_s2=.1,delay_s=.02),
        ppo=dict(n_steps=32,batch_size=128,n_epochs=5,gamma=.995,gae_lambda=.95,learning_rate=.0003,
                 clip_range=.2,ent_coef=.001,target_kl=.03,net_arch=[64,64],log_std_init=-1.5),
        evaluation='Zero residual baseline versus deterministic learned policy, nominal seed1000 and UDR seeds1001..1004; success=12 s survival, >0.02 m mean-plate advance and |lateral|<0.1 m',
        limitations=['Uncalibrated physical and sensor parameters','No self/body-ground collision response',
                     'Mechanical-work proxies sampled at 50 Hz, not electrical energy','Single training seed pilot',
                     'Backward phase prior does not enforce measured contraction peak ordering','Sano not integrated into whole-body training'],
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in SOURCES},gate=gate)
    write_json(args.out/'config.json',config)
    raw=VecMonitor(SubprocVecEnv([partial(SnakeResidualEnv,randomize=True,**physics_settings) for _ in range(args.workers)],start_method='spawn'))
    norm=VecNormalize(raw,norm_obs=True,norm_reward=False,clip_obs=10.)
    env=VecFrameStack(norm,n_stack=4)
    torch.set_num_threads(1)
    model=PPO('MlpPolicy',env,seed=7,device='cpu',verbose=1,
        policy_kwargs=dict(net_arch=[64,64],log_std_init=-1.5),
        **{k:v for k,v in config['ppo'].items() if k not in ('net_arch','log_std_init')})
    # Start deterministic mean at the known zero-residual controller, not old privileged weights.
    with torch.no_grad():
        model.policy.action_net.weight.zero_(); model.policy.action_net.bias.zero_()
    assert model.observation_space.shape==(4*(50+n),) and model.action_space.shape==(n,)
    model.set_logger(configure(str(args.out),['stdout','csv']))
    started=time.monotonic(); stop=[False]; latest=[None]; last_saved=[-1]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))

    def status(state,**extra):
        elapsed=time.monotonic()-started
        write_json(args.out/'status.json',dict(status=state,policy_steps=model.num_timesteps,
            requested_policy_steps=args.steps,optimizer_epochs=model._n_updates,
            elapsed_s=elapsed,policy_steps_per_s=model.num_timesteps/max(elapsed,1e-9),
            latest_checkpoint=latest[0],updated_utc=datetime.now(timezone.utc).isoformat(),**extra))

    def save():
        folder=args.out/f'checkpoint_{model.num_timesteps:08d}'; folder.mkdir(exist_ok=True)
        model.save(folder/'model.tmp.zip'); norm.save(str(folder/'normalization.tmp.pkl'))
        os.replace(folder/'model.tmp.zip',folder/'model.zip')
        os.replace(folder/'normalization.tmp.pkl',folder/'normalization.pkl')
        write_json(folder/'metadata.json',dict(policy_steps=model.num_timesteps,optimizer_epochs=model._n_updates,
            sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['model.zip','normalization.pkl']}))
        latest[0]=folder.name; last_saved[0]=model.num_timesteps

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__(); self.last_status=time.monotonic()
        def _on_rollout_start(self):
            # This hook runs AFTER the preceding PPO gradient update.
            if model.num_timesteps and (last_saved[0]==0 or model.num_timesteps-last_saved[0]>=2048):
                save()
            status('training')
        def _on_step(self):
            now=time.monotonic()
            if now-self.last_status>30:
                status('training',recent_mean_return=float(np.mean([i['task_return'] for i in self.locals['infos']])))
                self.last_status=now
            return not stop[0] and now-started<args.hours*3600

    save(); status('training')
    try:
        model.learn(total_timesteps=args.steps,callback=Progress())
        save()
        training_seconds=time.monotonic()-started
        status('interrupted' if stop[0] else 'evaluating',training_elapsed_s=training_seconds)
    except BaseException as exc:
        save(); status('failed',error=repr(exc)); raise
    finally:
        env.close()
    if stop[0]:
        return
    try:
        checkpoint=args.out/latest[0]
        baseline=evaluate(checkpoint,False,physics_settings)
        write_json(args.out/'baseline_evaluation.json',baseline)
        learned=evaluate(checkpoint,True,physics_settings)
        result=dict(baseline=baseline,learned=learned,
            baseline_success_rate=float(np.mean([r['success'] for r in baseline])),
            learned_success_rate=float(np.mean([r['success'] for r in learned])),
            mean_return_gain=float(np.mean([b['task_return']-a['task_return'] for a,b in zip(baseline,learned)])))
        write_json(args.out/'evaluation.json',result)
        status('completed' if model.num_timesteps>=args.steps else 'time_budget_reached',training_elapsed_s=training_seconds)
    except BaseException as exc:
        status('failed_evaluation',error=repr(exc)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--steps',type=int,default=32768)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--hours',type=float,default=3.)
    p.add_argument('--mode',choices=['worm','snake','mixed'],default='snake')
    p.add_argument('--allow-paired-legacy',action='store_true',
        help='Explicitly opt into the old paired-tendon worm/mixed comparison, not 14-action navigation')
    p.add_argument('--rolling',type=float,default=.06)
    p.add_argument('--elements',type=int,default=16)
    a=p.parse_args()
    if a.workers<4 or a.workers>32 or a.steps<128 or not 0<a.hours<=4 or not 0<=a.rolling<=.2 or not 4<=a.elements<=64:
        p.error('Use 4..32 workers, >=128 steps and a positive time budget <=4 h')
    train(a)

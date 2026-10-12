"""Audit the six snake conditions and optionally compose three real-time demos."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'snake_conditions_20261002'
CASES = [
    ('10 / 4 / 90', OUT/'a10_t4_p90', 10, 4, 90),
    ('15 / 4 / 90', HERE/'gait_compare_20261002/snake', 15, 4, 90),
    ('20 / 4 / 90', HERE/'sensor_gait_revision_20261002/snake20', 20, 4, 90),
    ('20 / 2 / 90', OUT/'a20_t2_p90', 20, 2, 90),
    ('20 / 6 / 90', OUT/'a20_t6_p90', 20, 6, 90),
    ('20 / 4 / 60', OUT/'a20_t4_p60', 20, 4, 60),
]


def main(videos=False, render=False):
    baseline = np.load(CASES[2][1]/'whole_rollout.npz')
    smoke = np.load(OUT/'baseline_check/whole_rollout.npz')
    assert np.array_equal(smoke['poses'], baseline['poses'][:len(smoke['poses'])])
    assert np.array_equal(smoke['actions'], baseline['actions'][:len(smoke['actions'])])
    summaries = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    initial_hash = None
    for label, folder, amplitude, period, phase in CASES:
        report = json.loads((folder/'comparison.json').read_text())
        rows = json.loads((folder/'replay_metrics.json').read_text())
        saved = np.load(folder/'whole_rollout.npz')
        assert not report['failed'] and report['physical_s'] == 12
        initial_hash = initial_hash or report['initial_state_sha256']
        assert report['initial_state_sha256'] == initial_hash
        for name, value in report['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == value, name
        assert saved['poses'].shape == (600, 290, 7) and np.isfinite(saved['poses']).all()
        assert np.max(np.abs(np.linalg.norm(saved['poses'][:,:,3:],axis=-1)-1)) < 1e-6
        assert np.all(saved['actions'][:,:10] == 0) and np.all(saved['servo'] == 0)
        time = np.arange(600)*.02
        expected = np.radians(amplitude)*(.5*(1-np.cos(np.pi*np.minimum(time,1))))[:,None]*np.sin(
            2*np.pi*time[:,None]/period-(3-np.arange(4))*np.radians(phase))
        assert np.allclose(expected, np.array([r['yaw_target_rad'] for r in rows]), atol=1e-12)
        forward = saved['initial_poses'][:10,0].mean()-saved['poses'][:,:10,0].mean(axis=1)
        assert np.allclose(forward, [r['mean_plate_forward_m'] for r in rows])
        summary = dict(label=label, amplitude_deg=amplitude, period_s=period, phase_deg=phase,
            **{k:report[k] for k in ['mean_plate_forward_m','mean_plate_lateral_m','max_joint_deg',
                'loaded_wheel_slip_mean_m_s','loaded_friction_limit_fraction','max_connector_error_m',
                'min_clearance_m','max_tension_n','wall_s']})
        summaries.append(summary)
        axes[0].plot(time+.02, forward, label=label)
    axes[0].set(xlabel='Time (s)', ylabel='Mean plate advance (m)')
    axes[0].legend(title='Amplitude / period / phase', fontsize=7, title_fontsize=8)
    labels = [r['label'] for r in summaries]
    axes[1].bar(labels, [r['max_joint_deg'] for r in summaries], color='#507f98')
    axes[1].set(ylabel='Actual peak joint angle (deg)')
    axes[2].bar(labels, [1000*r['loaded_wheel_slip_mean_m_s'] for r in summaries], color='#bd7955')
    axes[2].set(ylabel='Mean loaded-wheel slip (mm/s)')
    for ax in axes[1:]:
        ax.tick_params(axis='x', labelrotation=60, labelsize=8)
    fig.tight_layout(); fig.savefig(OUT/'conditions.png',dpi=180); plt.close(fig)
    audit = dict(passed=True, baseline_first_20_steps_bitwise_equal=True,
        initial_state_sha256=initial_hash, cases=summaries,
        scope='All six runs share the frozen nominal plant; input conditions vary, no PPO. Same 12 s, different cycle counts. No mesh/timestep convergence or hardware calibration.')
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2),encoding='utf-8')
    print(json.dumps(audit),flush=True)
    if render:
        from record_sofa_policy import render_whole
        for index in (0,2,3,4,5):
            label, source, *_ = CASES[index]
            folder = OUT/'rendered'/source.name
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder/'whole_rollout.npz'
            if temporary.exists():
                raise FileExistsError(temporary)
            with np.load(source/'whole_rollout.npz') as original:
                data = {k:original[k] for k in original.files}
            # Same fixed view for all cases. Only the decorative ground guide changes.
            assert data['poses'][:,:10,0].min() > -2.3
            data['reference_xy'] = np.column_stack([np.linspace(-2.3,.15,200),np.zeros(200)])
            np.savez_compressed(temporary, **data)
            try:
                render_whole(folder, clean=True)
            finally:
                temporary.unlink()
            (folder/'rendering.json').write_text(json.dumps(dict(source=str(source.relative_to(HERE)),
                source_sha256=hashlib.sha256((source/'whole_rollout.npz').read_bytes()).hexdigest(),
                ground_guide_x_bounds_m=[-2.3,.15], physical_states_modified=False),indent=2))
    if videos:
        from make_gait_demo import main as compose
        for name, indices in [('amplitude',(0,2)),('period',(3,4)),('phase',(2,5))]:
            folder = OUT/name; folder.mkdir(exist_ok=True)
            selected = [CASES[i] for i in indices]
            compose(folder,[OUT/'rendered'/c[1].name for c in selected],
                [f'A = +/-{a} deg | T = {t} s | phase = {p} deg | 1x' for _,_,a,t,p in selected])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', action='store_true')
    p.add_argument('--render', action='store_true')
    args=p.parse_args()
    main(args.videos,args.render)

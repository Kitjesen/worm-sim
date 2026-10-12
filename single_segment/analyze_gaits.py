"""Audit and plot the matched SOFA experiment, including per-wheel substeps."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'gait_compare_20261002'


def main():
    reports={};states={};records={};substeps={}
    for mode in ['worm','snake','idle']:
        folder=OUT/mode
        report=json.loads((folder/'comparison.json').read_text())
        assert not report['failed'],(mode,'failed physical rollout')
        for name,sha in report['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
        d=np.load(folder/'whole_rollout.npz')
        contact=np.load(folder/'wheel_substeps.npz')['samples']
        n=len(d['poses']);assert n==round(report['physical_s']/.02)
        assert contact.shape==(round(report['physical_s']/report['dt_s']),20,12)
        assert np.isfinite(d['poses']).all() and np.isfinite(contact).all()
        assert np.allclose(np.linalg.norm(d['poses'][:,:,3:],axis=-1),1.,atol=1e-6)
        assert np.all(np.linalg.norm(contact[:,:,1:3],axis=-1)<=.8*contact[:,:,0]+2e-6)
        if mode=='snake': assert np.all(d['actions'][:,:10]==0), 'Snake has commanded contraction'
        if mode=='worm': assert np.all(d['actions'][:,10:]==0), 'Worm has commanded yaw'
        np.testing.assert_allclose(d['segment_gap_m'],np.linalg.norm(d['poses'][:,1:10:2,:3]-d['poses'][:,:10:2,:3],axis=-1))
        reports[mode]=report;states[mode]=d;substeps[mode]=contact
        records[mode]=json.loads((folder/'replay_metrics.json').read_text())
    assert len({r['initial_state_sha256'] for r in reports.values()})==1
    np.testing.assert_array_equal(states['worm']['initial_poses'],states['snake']['initial_poses'])
    assert reports['worm']['physical_s']==reports['snake']['physical_s']==12.
    assert abs(reports['idle']['mean_plate_forward_m'])<.001
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'worm':'#236b8e','snake':'#cf7043'}
    names={'worm':'Pure peristalsis','snake':'Pure snake (15 deg)'}
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for mode in ['worm','snake']:
        d=states[mode];r=records[mode];c=substeps[mode];color=colors[mode]
        t=np.arange(1,len(r)+1)*.02
        v=lambda key:np.array([row[key] for row in r])
        axes[0,0].plot(t,v('mean_plate_forward_m'),color=color,label=names[mode])
        axes[0,1].plot(t,v('mean_plate_lateral_m')*1000,color=color)
        loaded=c[:,:,0]>.05;slip=np.linalg.norm(c[:,:,3:5],axis=-1)
        count=round(.02/reports[mode]['dt_s'])
        numerator=(slip*loaded).reshape(len(r),count,20).sum(axis=(1,2))
        denominator=loaded.reshape(len(r),count,20).sum(axis=(1,2))
        mean=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
        axes[1,0].plot(t,mean*1000,color=color)
        gap=d['segment_gap_m']*1000
        axes[1,1].fill_between(t,gap.min(1),gap.max(1),alpha=.2,color=color)
        axes[1,1].plot(t,gap.mean(1),color=color)
        axes[2,0].plot(t,v('max_tension_n'),color=color)
        axes[2,1].plot(t,np.max(np.abs(np.degrees(d['joint_angle_rad'])),axis=1),color=color)
    labels=['Mean plate forward displacement (m)','Mean plate lateral displacement (mm)',
            'Loaded-wheel mean slip (mm/s)','Plate spacing: mean and min/max (mm)',
            'Maximum cable tension (N)','Maximum absolute joint yaw (deg)']
    for ax,label in zip(axes.ravel(),labels):ax.set(xlabel='Physical time (s)',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.tight_layout();fig.savefig(OUT/'comparison.png');fig.savefig(OUT/'comparison.svg');plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    cmap=ListedColormap(['#e8ecef','#3b6984','#43a294','#d67743'])
    for ax,mode in zip(axes,['worm','snake']):
        c=substeps[mode];slip=np.linalg.norm(c[:,:,3:5],axis=-1)
        status=np.where(c[:,:,0]<=.05,0,np.where(slip>=.001,3,np.where(c[:,:,5]>.05,2,1)))
        # Show states at 50 Hz; statistics above use every 0.5 ms substep.
        ax.imshow(status[::40].T,origin='lower',aspect='auto',extent=[0,12,-.5,19.5],
                        cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5),cmap.N),interpolation='nearest')
        ax.set(ylabel='Wheel index',title=names[mode])
        ax.set_yticks([0,4,8,12,16,19])
    axes[-1].set_xlabel('Physical time (s)')
    fig.tight_layout(rect=[0,0,1,.9])
    labels=['Load <= 0.05 N','Low slip; nearly locked','Low slip; rolling','Slip >= 1 mm/s']
    fig.legend(handles=[Patch(facecolor=color,label=label) for color,label in zip(cmap.colors,labels)],
               loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.99))
    fig.savefig(OUT/'wheel_contact_states.png');fig.savefig(OUT/'wheel_contact_states.svg');plt.close(fig)
    for svg in OUT.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    peaks={}
    for mode in ['worm','snake']:
        a=substeps[mode];slip=np.linalg.norm(a[:,:,3:5],axis=-1);loaded=a[:,:,0]>.05
        index=np.unravel_index(np.argmax(np.where(loaded,slip,-1)),slip.shape)
        peaks[mode]=dict(percentiles_50_95_99_mm_s=(np.percentile(slip[loaded],[50,95,99])*1000).tolist(),
                        time_s=float((index[0]+1)*reports[mode]['dt_s']),wheel=int(index[1]),
                        normal_n=float(a[index][0]),tangent_force_n=a[index][1:3].tolist(),
                        slip_m_s=a[index][3:5].tolist(),friction_limited=bool(a[index][6]))
    summary=dict(passed=True,identical_initial_state=True,unchanged_physics_source=True,
                 slip_peaks=peaks,
                 modes=reports,contact_display='50 Hz samples; low slip threshold 1 mm/s, nearly locked threshold 0.05 rad/s; statistics use all substeps.',
                 conclusion_scope='One nominal setting, chosen non-RL gaits; not equal power, not gait optimization, not calibrated hardware.')
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:{field:r[field] for field in ['physical_s','wall_s','mean_plate_forward_m','head_forward_m',
        'last_cycle_advance_m','last_cycle_fitted_speed_m_s','loaded_wheel_slip_mean_m_s','loaded_wheel_slip_max_m_s',
        'loaded_friction_limit_fraction','max_joint_deg','max_tension_n','gap_min_mm','gap_max_mm']} for k,r in reports.items()},indent=2))


if __name__=='__main__':main()

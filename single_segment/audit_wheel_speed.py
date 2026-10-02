"""Audit wheel RPM from frozen substep logs and 50 Hz animation angles."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'sensor_gait_revision_20261002'


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8))
    for row, mode in enumerate(('worm', 'snake')):
        folder = HERE/'gait_compare_20261002'/mode
        data = np.load(folder/'wheel_substeps.npz')
        s = data['samples']; dt = float(data['dt_s'])
        mask = s[:, :, 0] > .05
        rpm = s[:, :, 5]*60/(2*np.pi)
        omega = s[:, :, 5]
        rolling = .015*s[:, :, 0]*.018
        viscous = 2e-6*omega
        speed = s[:, :, 7]
        slip = s[:, :, 3]
        assert np.max(np.abs(speed-.018*omega-slip)) < 1e-6
        angles = np.load(folder/'whole_rollout.npz')['wheel_angle']
        frame_rpm = np.diff(angles, axis=0)/.02*60/(2*np.pi)
        report[mode] = dict(loaded_rpm_p50_p95_p99_max=np.percentile(rpm[mask], [50,95,99,100]).tolist(),
            frame_rpm_p50_p95_p99_max=np.percentile(frame_rpm, [50,95,99,100]).tolist(),
            loaded_nominal_rolling_torque_median_nm=float(np.median(rolling[mask])),
            loaded_nominal_viscous_torque_median_nm=float(np.median(viscous[mask])))
        time = np.arange(1, len(s)+1)*dt
        axes[row, 0].plot(time[::20], np.percentile(rpm, 95, axis=1)[::20], label='95th percentile across wheels')
        axes[row, 0].plot(time[::20], np.median(rpm, axis=1)[::20], label='Median across wheels')
        axes[row, 0].set(ylabel=f'{mode.capitalize()} wheel RPM', xlabel='Time (s)')
        axes[row, 0].legend(fontsize=8)
        # Time traces include zero placeholders for airborne wheels; masked statistics do not.
        axes[row, 1].hist(rpm[mask], bins=60, density=True, color='#467b96')
        axes[row, 1].set(xlabel='Loaded wheel RPM (N > 0.05 N)', ylabel='Density')
    fig.tight_layout(); fig.savefig(OUT/'wheel_rpm.png', dpi=180); plt.close(fig)
    report['scope'] = 'Airborne substep rows are zero placeholders. Histogram/stats use N>0.05 N. Torque summaries precede the no-reversal cap.'
    report['parameters'] = dict(radius_m=.018, wheel_mass_kg=.009, inertia_kg_m2=.5*.009*.018**2,
        axle_viscous_nm_s=2e-6, rolling_coefficient=.015, tangential_damping_ns_m=8,
        normal_damping_ns_m=8, mu=.8)
    (OUT/'wheel_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

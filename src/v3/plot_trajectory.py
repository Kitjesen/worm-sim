"""
Plot circular locomotion trajectory from saved .npz trajectory data.
Usage: python plot_trajectory.py <traj_file.npz> [--output <output.png>]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import sys
import os


def compute_circularity(com_c, heading_uw, times):
    """Compute circularity metrics for the trajectory.

    Returns dict with:
      - path_length: total COM path length (mm)
      - per_rot_closure: list of distances from start at each full rotation
      - per_rot_path_len: path length per rotation
      - mean_radius: mean distance from centroid of path
      - radius_cv: coefficient of variation of radius (lower = more circular)
      - aspect_ratio: ratio of X-range to Y-range (1.0 = circular)
    """
    # Total path length
    diffs = np.diff(com_c, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    path_length = np.sum(seg_lens)

    total_rot = heading_uw[-1] / 360.0
    sign = -1 if heading_uw[-1] < 0 else 1

    # Per-rotation closure distance and path length
    per_rot_closure = []
    per_rot_path_len = []
    prev_idx = 0
    for n_rot in range(1, int(abs(total_rot)) + 1):
        deg = sign * n_rot * 360
        if sign < 0:
            cross_idx = np.argmax(heading_uw <= deg)
        else:
            cross_idx = np.argmax(heading_uw >= deg)
        if cross_idx > 0:
            closure_dist = np.sqrt(com_c[cross_idx, 0]**2 + com_c[cross_idx, 1]**2)
            per_rot_closure.append(closure_dist)
            rot_path = np.sum(seg_lens[prev_idx:cross_idx])
            per_rot_path_len.append(rot_path)
            prev_idx = cross_idx

    # Centroid and radius analysis
    centroid = np.mean(com_c, axis=0)
    radii = np.sqrt((com_c[:, 0] - centroid[0])**2 + (com_c[:, 1] - centroid[1])**2)
    mean_radius = np.mean(radii)
    radius_cv = np.std(radii) / mean_radius if mean_radius > 0 else float('inf')

    # Aspect ratio of bounding box
    x_range = np.max(com_c[:, 0]) - np.min(com_c[:, 0])
    y_range = np.max(com_c[:, 1]) - np.min(com_c[:, 1])
    if x_range > 0 and y_range > 0:
        aspect_ratio = min(x_range, y_range) / max(x_range, y_range)
    else:
        aspect_ratio = 0.0

    return dict(
        path_length=path_length,
        per_rot_closure=per_rot_closure,
        per_rot_path_len=per_rot_path_len,
        mean_radius=mean_radius,
        radius_cv=radius_cv,
        aspect_ratio=aspect_ratio,
        centroid=centroid,
    )


def plot_trajectory(traj_path, output_path=None):
    data = np.load(traj_path, allow_pickle=True)
    times = data['times']
    plates_xy = data['plates_xy']      # (T, num_plates, 2) in meters
    com_xy = data['com_xy']            # (T, 2) in meters
    ht_dist = data['ht_dist']          # (T,) in mm
    max_bend_raw = data['max_bend']    # (T,)
    heading_uw = data['heading_unwrapped']  # (T,) in degrees
    num_plates = int(data['num_plates'])
    seg_length = float(data['seg_length'])
    body_length = seg_length * (num_plates - 1) * 1000  # mm

    # Derive experiment name from filename
    exp_name = os.path.basename(traj_path).replace('_traj.npz', '').replace('_plot', '')

    # Fix max_bend wrapping artifacts: values > 180° are wrapping errors
    max_bend = np.copy(max_bend_raw)
    max_bend[max_bend > 180] = 360 - max_bend[max_bend > 180]

    # Convert to mm
    com_mm = com_xy * 1000
    plates_mm = plates_xy * 1000

    # Initial COM position
    com0 = com_mm[0]
    com_c = com_mm - com0
    plates_c = plates_mm - com0[np.newaxis, :]

    # Stats
    total_rot = heading_uw[-1] / 360.0
    sim_time = times[-1]
    heading_rate = heading_uw[-1] / sim_time if sim_time > 0 else 0
    rot_period = abs(360 / heading_rate) if heading_rate != 0 else sim_time
    skip = max(1, int(len(times) * 0.02))

    # COM displacement from start
    com_disp = np.sqrt(com_c[:, 0]**2 + com_c[:, 1]**2)
    max_com_disp = np.max(com_disp[skip:])
    final_com_disp = com_disp[-1]

    # Circularity analysis
    circ = compute_circularity(com_c, heading_uw, times)

    # Find rotation completion times
    rot_times = []
    rot_positions = []
    sign = -1 if heading_uw[-1] < 0 else 1
    for n_rot in range(1, int(abs(total_rot)) + 1):
        deg = sign * n_rot * 360
        if sign < 0:
            cross_idx = np.argmax(heading_uw <= deg)
        else:
            cross_idx = np.argmax(heading_uw >= deg)
        if cross_idx > 0:
            rot_times.append(times[cross_idx])
            rot_positions.append(com_c[cross_idx])

    # --- Create figure ---
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], height_ratios=[1.2, 0.8, 0.8],
                          hspace=0.35, wspace=0.30)

    # ========== Panel 1: XY Trajectory (bird's eye) ==========
    ax1 = fig.add_subplot(gs[0:2, 0])

    # Plot COM trajectory with time-colored gradient
    points = com_c.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=0, vmax=sim_time)
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidths=2.0, alpha=0.85)
    lc.set_array(times[:-1])
    ax1.add_collection(lc)

    # Draw worm body snapshots at regular intervals
    n_snapshots = min(20, max(6, int(abs(total_rot) * 3)))
    snap_indices = np.linspace(int(len(times) * 0.03), len(times) - 1,
                               n_snapshots, dtype=int)
    for si, idx in enumerate(snap_indices):
        px = plates_c[idx, :, 0]
        py = plates_c[idx, :, 1]
        t_frac = idx / len(times)
        alpha = 0.10 + 0.25 * t_frac
        ax1.plot(px, py, '-', color='gray', linewidth=1.0, alpha=alpha)

    # Draw worm body at start (green) and end (red)
    ax1.plot(plates_c[0, :, 0], plates_c[0, :, 1], '-o', color='#27ae60',
             linewidth=3, markersize=5, label='Start', zorder=10)
    ax1.plot(plates_c[-1, :, 0], plates_c[-1, :, 1], '-o', color='#c0392b',
             linewidth=3, markersize=5, label='End', zorder=10)
    # Head triangles
    ax1.plot(plates_c[0, -1, 0], plates_c[0, -1, 1], '^', color='#27ae60',
             markersize=11, zorder=11, markeredgecolor='white', markeredgewidth=0.5)
    ax1.plot(plates_c[-1, -1, 0], plates_c[-1, -1, 1], '^', color='#c0392b',
             markersize=11, zorder=11, markeredgecolor='white', markeredgewidth=0.5)

    # Mark centroid
    ax1.plot(circ['centroid'][0], circ['centroid'][1], '+', color='#9b59b6',
             markersize=15, markeredgewidth=2, zorder=12, label='Centroid')

    # Mark rotation completion points on trajectory
    for i, (rt, rp) in enumerate(zip(rot_times, rot_positions)):
        ax1.plot(rp[0], rp[1], 'D', color='#e74c3c', markersize=6,
                 markeredgecolor='white', markeredgewidth=0.8, zorder=9)
        ax1.annotate(f'{i+1}x360\nt={rt:.0f}s', (rp[0], rp[1]),
                     textcoords="offset points", xytext=(10, 8),
                     fontsize=7, color='#e74c3c', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8))

    # Direction arrows along the trajectory
    n_arrows = max(4, min(12, int(abs(total_rot) * 2)))
    arrow_indices = np.linspace(int(len(times) * 0.1), len(times) - 2,
                                n_arrows, dtype=int)
    for idx in arrow_indices:
        dx = com_c[idx+1, 0] - com_c[idx, 0]
        dy = com_c[idx+1, 1] - com_c[idx, 1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            scale = max(max_com_disp * 0.04, body_length * 0.3)
            ax1.annotate('', xy=(com_c[idx, 0] + dx/length*scale,
                                 com_c[idx, 1] + dy/length*scale),
                         xytext=(com_c[idx, 0], com_c[idx, 1]),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50',
                                         lw=1.5, alpha=0.5))

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax1, shrink=0.55, pad=0.02, aspect=30)
    cbar.set_label('Time (s)', fontsize=10)

    # Scale bar (1 body length)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    sb_x = xlim[0] + span_x * 0.05
    sb_y = ylim[0] + span_y * 0.06
    ax1.plot([sb_x, sb_x + body_length], [sb_y, sb_y], '-', color='black', linewidth=3)
    ax1.text(sb_x + body_length / 2, sb_y - span_y * 0.025,
             f'{body_length:.0f}mm (1 body)', ha='center', fontsize=8, fontweight='bold')

    ax1.set_xlabel('X (mm)', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle='-')
    ax1.set_title(f'COM Trajectory — {abs(total_rot):.1f} rotations in {sim_time:.0f}s\n'
                  f'Path: {circ["path_length"]:.0f}mm, '
                  f'Aspect: {circ["aspect_ratio"]:.2f}, '
                  f'Radius CV: {circ["radius_cv"]:.2f}',
                  fontsize=12, fontweight='bold')

    # ========== Panel 2: Heading vs Time ==========
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(times, heading_uw, color='#2c3e50', linewidth=1.3)
    # Mark full rotations
    for n_rot in range(1, int(abs(total_rot)) + 1):
        deg = sign * n_rot * 360
        ax2.axhline(y=deg, color='#e74c3c', linestyle=':', alpha=0.3, linewidth=0.7)
        if sign < 0:
            cross_idx = np.argmax(heading_uw <= deg)
        else:
            cross_idx = np.argmax(heading_uw >= deg)
        if cross_idx > 0:
            ax2.plot(times[cross_idx], deg, 'o', color='#e74c3c', markersize=5, alpha=0.7)
            ax2.text(times[cross_idx] + sim_time * 0.015, deg + sign * 40,
                     f'{n_rot}x', fontsize=9, color='#e74c3c', alpha=0.8,
                     fontweight='bold')

    # Add linear fit line
    t_fit = times[skip:]
    h_fit = heading_uw[skip:]
    coeffs = np.polyfit(t_fit, h_fit, 1)
    ax2.plot(t_fit, np.polyval(coeffs, t_fit), '--', color='#3498db',
             linewidth=1.0, alpha=0.5, label=f'Linear fit: {coeffs[0]:.2f} deg/s')
    ax2.legend(fontsize=9, loc='lower left' if sign < 0 else 'upper left')

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Cumulative Heading (deg)', fontsize=11)
    ax2.set_title(f'Heading Accumulation\n'
                  f'Rate: {heading_rate:.2f} deg/s, '
                  f'Period: {abs(rot_period):.0f}s/rot',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ========== Panel 3: Circularity Metrics ==========
    ax4 = fig.add_subplot(gs[1, 1])

    if circ['per_rot_closure']:
        rot_nums = list(range(1, len(circ['per_rot_closure']) + 1))
        bars = ax4.bar(rot_nums, circ['per_rot_closure'], color='#3498db', alpha=0.7,
                       label='Closure dist (mm)')
        ax4.set_xlabel('Rotation #', fontsize=11)
        ax4.set_ylabel('Dist from start (mm)', fontsize=11, color='#3498db')
        ax4.tick_params(axis='y', labelcolor='#3498db')

        # Add path length per rotation on twin axis
        if circ['per_rot_path_len']:
            ax4b = ax4.twinx()
            ax4b.plot(rot_nums, circ['per_rot_path_len'], 's-', color='#e67e22',
                      linewidth=1.5, markersize=6, label='Path len/rot (mm)')
            ax4b.set_ylabel('Path length per rot (mm)', fontsize=10, color='#e67e22')
            ax4b.tick_params(axis='y', labelcolor='#e67e22')

            # Ideal circle circumference from mean radius
            ideal_circ = 2 * np.pi * circ['mean_radius']
            ax4b.axhline(y=ideal_circ, color='#e67e22', linestyle=':', alpha=0.4)
            ax4b.text(rot_nums[-1] + 0.3, ideal_circ,
                      f'2piR={ideal_circ:.0f}', fontsize=8, color='#e67e22', va='bottom')

        mean_closure = np.mean(circ['per_rot_closure'])
        ax4.set_title(f'Per-Rotation Circularity\n'
                      f'Mean closure: {mean_closure:.0f}mm, '
                      f'Mean R: {circ["mean_radius"]:.0f}mm',
                      fontsize=11, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Less than 1 full rotation\nNo closure data',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Per-Rotation Circularity', fontsize=11, fontweight='bold')

    ax4.grid(True, alpha=0.3)

    # ========== Panel 4: Body Integrity ==========
    ax3 = fig.add_subplot(gs[2, :])

    color_ht = '#2980b9'
    color_bend = '#e67e22'

    ln1 = ax3.plot(times, ht_dist, color=color_ht, linewidth=0.6, alpha=0.7,
                   label='Head-tail dist')
    ax3.set_ylabel('Head-tail distance (mm)', fontsize=10, color=color_ht)
    ax3.tick_params(axis='y', labelcolor=color_ht)
    ht_mean = np.median(ht_dist[skip:])
    ht_std = np.std(ht_dist[skip:])
    ax3.axhline(y=ht_mean, color=color_ht, linestyle='--', alpha=0.4)
    ax3.fill_between(times, ht_mean - ht_std, ht_mean + ht_std,
                     color=color_ht, alpha=0.08)
    ax3.set_ylim(max(200, ht_mean - 50), min(380, ht_mean + 50))

    ax3b = ax3.twinx()
    # Smoothed max_bend
    kernel_size = max(1, min(21, len(max_bend) // 50))
    if kernel_size > 2:
        kernel = np.ones(kernel_size) / kernel_size
        max_bend_smooth = np.convolve(max_bend, kernel, mode='same')
    else:
        max_bend_smooth = max_bend
    ln2 = ax3b.plot(times, max_bend_smooth, color=color_bend, linewidth=0.8, alpha=0.8,
                    label='Max bend (smoothed)')
    ax3b.set_ylabel('Max inter-segment bend (deg)', fontsize=10, color=color_bend)
    ax3b.tick_params(axis='y', labelcolor=color_bend)
    bend_p95 = np.percentile(max_bend[skip:], 95)
    bend_max = np.max(max_bend[skip:])
    ax3b.set_ylim(0, max(15, bend_p95 * 1.8))

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='upper right', fontsize=9)

    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_title(f'Body Integrity — ht={ht_mean:.1f}+/-{ht_std:.1f}mm, '
                  f'bend p95={bend_p95:.1f} deg, max={bend_max:.1f} deg',
                  fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ========== Super title ==========
    plt.suptitle(f'{exp_name} — 5-Segment Worm Trajectory ({sim_time:.0f}s)\n'
                 f'{abs(total_rot):.1f} rotations, '
                 f'path={circ["path_length"]:.0f}mm, '
                 f'aspect={circ["aspect_ratio"]:.2f}',
                 fontsize=13, fontweight='bold', y=1.0)

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(traj_path), '..', 'plots')
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(traj_path).replace('_traj.npz', '')
        output_path = os.path.join(output_dir, f'{base}_trajectory.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved trajectory plot to: {output_path}")
    plt.close()
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trajectory.py <traj_file.npz> [--output <output.png>]")
        sys.exit(1)

    traj_path = sys.argv[1]
    output_path = None
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    plot_trajectory(traj_path, output_path)

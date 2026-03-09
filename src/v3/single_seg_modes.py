"""
Single-Segment Mode Verification
=================================
Zhan et al. (IJRR 2019) 定义了 4 种节段状态：
  State 0: Axially Relaxed  — 自然长度
  State 1: Axially Contracted (Anchor)  — 全部纵肌收缩，变短
  State 2: Left-Contracted  — 左侧纵肌收缩，向左弯曲
  State 3: Right-Contracted — 右侧纵肌收缩，向右弯曲

本脚本用单体节 (num_segments=1) 验证每种模式的变形量。
两种模型对比：cable (钢片) vs no-cable (纯板+tendon)。

测试序列 (每模式 3s + 2s 恢复):
  0-2s:   Settle (重力沉降)
  2-5s:   State 1 — 全收缩 (anchor)
  5-7s:   Release
  7-10s:  State 2 — 左弯 (differential: left ON)
  10-12s: Release
  12-15s: State 3 — 右弯 (differential: right ON)
  15-17s: Release
  17-20s: Ring 收缩 (仅 cable 模型)
  20-22s: Release

量化指标：
  - 轴向缩短 ΔY (mm)
  - 偏航角 Δyaw (°)
  - 横向偏移 ΔX (mm)
  - 环径变化 ΔR (mm, 仅 cable)
  - 板间角度 (°)

Usage:
  python single_seg_modes.py [--video] [--cable]
"""
import mujoco
import numpy as np
import math
import time
import os
import sys
import argparse

# Import build_model_xml from exp_runner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_runner import build_model_xml

# ======================== Helpers ========================

def measure_state(m, d, plate_ids, num_strips=8, has_cables=False):
    """Measure segment state: Y-distance, X-offset, yaw, ring radius."""
    p0, p1 = plate_ids
    pos0, pos1 = d.xpos[p0], d.xpos[p1]

    dy = (pos1[1] - pos0[1]) * 1000  # axial distance (mm)
    dx = (pos1[0] - pos0[0]) * 1000  # lateral offset (mm)
    dz = (pos1[2] - pos0[2]) * 1000  # vertical offset (mm)

    # Yaw angle from rotation matrices
    R0 = d.xmat[p0].reshape(3, 3)
    R1 = d.xmat[p1].reshape(3, 3)
    # Y-axis of each plate (axial direction)
    y0 = R0[:, 1]
    y1 = R1[:, 1]
    # Project onto XY plane for yaw
    yaw0 = math.degrees(math.atan2(y0[0], y0[1]))
    yaw1 = math.degrees(math.atan2(y1[0], y1[1]))
    yaw_rel = yaw1 - yaw0  # relative yaw (positive = left bend)

    # Pitch from Z component
    cos_a = np.clip(np.dot(y0, y1), -1, 1)
    inter_angle = math.degrees(math.acos(cos_a))

    # Ring radius (only for cable model)
    ring_r = 0.0
    if has_cables:
        rb_pos = []
        for si in range(num_strips):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"rb0_{si}")
            if bid >= 0:
                rb_pos.append(d.xpos[bid].copy())
        if rb_pos:
            c = np.mean(rb_pos, axis=0)
            radii = [math.sqrt((p[0]-c[0])**2 + (p[2]-c[2])**2) for p in rb_pos]
            ring_r = np.mean(radii) * 1000  # mm

    return {
        'dy': dy,
        'dx': dx,
        'dz': dz,
        'yaw': yaw_rel,
        'angle': inter_angle,
        'ring_r': ring_r,
    }


def run_single_seg_test(use_cables=False, record_video=False):
    """Run single-segment mode tests."""
    # Build model
    params = {
        'num_segments': 1,
        'no_cables': 0 if use_cables else 1,
        'plate_stiff_x': 0.0,  # free lateral motion
        'plate_stiff_y': 0.0,
        'plate_stiff_yaw': 0.0,
        'constraint_type': 'connect',
        'tendon_stiffness': 10000 if not use_cables else 0,
        'tendon_damping': 15 if not use_cables else 0,
        'axial_muscle_force': 50,
        'ring_muscle_force': 10,
        'steer_in_state2': 0,  # no diagonal muscles for this test
        'gait_s0': '0',  # single segment, controlled manually
        'sim_time': 22.0,
        'settle_time': 2.0,
    }

    model_tag = "cable" if use_cables else "nocable"
    xml_str, P = build_model_xml(f"single_{model_tag}", params)

    # Insert <visual> block for offscreen rendering (before <worldbody>)
    visual_block = '  <visual><global offwidth="1280" offheight="720"/></visual>\n'
    xml_str = xml_str.replace('<worldbody>', visual_block + '  <worldbody>')

    # Save XML
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    bin_dir = os.path.join(project_root, "bin", "v3", "experiments")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, f"single_seg_{model_tag}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"\n{'='*60}")
    print(f"Single-Segment Mode Test — {'Cable' if use_cables else 'No-Cable'} Model")
    print(f"{'='*60}")
    print(f"Bodies: {m.nbody}, DOF: {m.nv}, Actuators: {m.nu}")

    # Find plate body IDs
    p0_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate0")
    p1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate1")
    plate_ids = (p0_id, p1_id)

    # Find actuator IDs
    # Axial muscles: am0_0 (right/0°), am0_1 (top/90°), am0_2 (left/180°), am0_3 (bottom/270°)
    am_ids = []
    for mi in range(4):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"am0_{mi}")
        am_ids.append(aid)
    print(f"Axial muscle IDs: {am_ids}")

    # Ring muscle (cable model only)
    rm_id = -1
    if use_cables:
        rm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "rm0")
        print(f"Ring muscle ID: {rm_id}")

    # Timeline phases
    phases = [
        (0,  2,  "SETTLE",    "settle"),
        (2,  5,  "STATE_1",   "contract"),   # all axial ON
        (5,  7,  "RELEASE_1", "release"),
        (7,  10, "STATE_2",   "left"),        # left axial ON (differential)
        (10, 12, "RELEASE_2", "release"),
        (12, 15, "STATE_3",   "right"),       # right axial ON (differential)
        (15, 17, "RELEASE_3", "release"),
        (17, 20, "RING",      "ring"),        # ring muscle ON
        (20, 22, "RELEASE_4", "release"),
    ]

    def get_phase(t):
        for t0, t1, name, action in phases:
            if t0 <= t < t1:
                return name, action, t0
        return "END", "release", 22

    # Video setup
    frames = []
    renderer = None
    camera = None
    if record_video:
        renderer = mujoco.Renderer(m, 720, 1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.15
        camera.elevation = -20
        camera.azimuth = 45

    # Measurement storage
    results = {}  # phase_name -> {baseline, peak, delta}
    baseline = None

    dt = m.opt.timestep
    total_steps = int(22.0 / dt)
    fps = 30
    t0_wall = time.time()

    current_phase = "SETTLE"
    phase_peak = None  # track extreme deformation within current phase

    for step in range(total_steps):
        t = step * dt
        phase_name, action, phase_start = get_phase(t)

        # Measure at phase transitions
        if phase_name != current_phase:
            # Record peak measurement for the just-ended phase
            if current_phase == "SETTLE":
                baseline = measure_state(m, d, plate_ids, has_cables=use_cables)
                print(f"\nBaseline (after settle):")
                print(f"  ΔY={baseline['dy']:.2f}mm  ΔX={baseline['dx']:.2f}mm  yaw={baseline['yaw']:.2f}°  angle={baseline['angle']:.2f}°", end="")
                if use_cables:
                    print(f"  ring_R={baseline['ring_r']:.2f}mm")
                else:
                    print()
            elif phase_peak is not None:
                results[current_phase] = phase_peak
            current_phase = phase_name

        # Control
        d.ctrl[:] = 0
        ramp = min((t - phase_start) / 0.3, 1.0) if action != "settle" else 0

        if action == "contract":
            for ai in am_ids:
                d.ctrl[ai] = ramp
        elif action == "left":
            d.ctrl[am_ids[2]] = ramp  # am0_2 = left (180°)
        elif action == "right":
            d.ctrl[am_ids[0]] = ramp  # am0_0 = right (0°)
        elif action == "ring" and rm_id >= 0:
            d.ctrl[rm_id] = ramp

        mujoco.mj_step(m, d)

        if np.any(np.isnan(d.qpos)):
            print(f"  NaN at t={t:.2f}s — stopping.")
            break

        # Measure near end of active phase (last 0.5s)
        for pt0, pt1, pname, paction in phases:
            if pname == current_phase and paction not in ("settle", "release"):
                if t >= pt1 - 0.5:
                    s = measure_state(m, d, plate_ids, has_cables=use_cables)
                    if phase_peak is None or abs(s['dy'] - baseline['dy']) + abs(s['dx']) >= abs(phase_peak['dy'] - baseline['dy']) + abs(phase_peak['dx']):
                        phase_peak = s
                break

        # Reset phase_peak on new active phase
        if action not in ("settle", "release") and (phase_peak is None or current_phase not in results):
            if t < phase_start + 0.5:
                phase_peak = None

        # Capture video frame
        if record_video and renderer is not None:
            if len(frames) < t * fps:
                center = (d.xpos[p0_id] + d.xpos[p1_id]) / 2
                camera.lookat[:] = center
                renderer.update_scene(d, camera)
                frames.append(renderer.render().copy())

        # Progress report
        if step > 0 and step % 4000 == 0:
            s = measure_state(m, d, plate_ids, has_cables=use_cables)
            print(f"  t={t:.1f}s [{current_phase:>10s}] ΔY={s['dy']:.1f}mm  ΔX={s['dx']:.1f}mm  yaw={s['yaw']:.1f}°", end="")
            if use_cables:
                print(f"  R={s['ring_r']:.1f}mm", end="")
            print()

    # Last phase
    if phase_peak is not None and current_phase not in results:
        results[current_phase] = phase_peak

    elapsed = time.time() - t0_wall
    print(f"\nSim done in {elapsed:.1f}s")

    # ======================== Summary ========================
    print(f"\n{'='*60}")
    print(f"Results Summary — {'Cable' if use_cables else 'No-Cable'} Model")
    print(f"{'='*60}")
    print(f"{'Mode':<12s} {'ΔY(mm)':>8s} {'ΔX(mm)':>8s} {'yaw(°)':>8s} {'angle(°)':>9s}", end="")
    if use_cables:
        print(f" {'ring_R':>8s}", end="")
    print(f" {'描述'}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9}", end="")
    if use_cables:
        print(f" {'-'*8}", end="")
    print(f" {'-'*20}")

    ref_dy = baseline['dy']
    ref_ring = baseline['ring_r'] if use_cables else 0

    mode_labels = {
        'STATE_1': ('收缩/锚固', '全4纵肌ON'),
        'STATE_2': ('左弯', '左纵肌ON'),
        'STATE_3': ('右弯', '右纵肌ON'),
        'RING':    ('环肌收缩', '环肌ON'),
    }

    summary = {}
    for mode_name in ['STATE_1', 'STATE_2', 'STATE_3', 'RING']:
        if mode_name in results:
            s = results[mode_name]
            delta_y = s['dy'] - ref_dy
            delta_ring = s['ring_r'] - ref_ring if use_cables else 0
            label, desc = mode_labels[mode_name]
            print(f"{label:<12s} {delta_y:>+8.2f} {s['dx']:>+8.2f} {s['yaw']:>+8.2f} {s['angle']:>9.2f}", end="")
            if use_cables:
                print(f" {delta_ring:>+8.2f}", end="")
            print(f" {desc}")
            summary[mode_name] = {
                'delta_y': delta_y,
                'dx': s['dx'],
                'yaw': s['yaw'],
                'angle': s['angle'],
                'delta_ring': delta_ring,
            }

    print(f"\nBaseline: Y={ref_dy:.2f}mm", end="")
    if use_cables:
        print(f", ring_R={ref_ring:.2f}mm", end="")
    print()

    # Validation checks
    print(f"\n--- Validation ---")
    if 'STATE_1' in summary:
        dy1 = summary['STATE_1']['delta_y']
        if dy1 < -1.0:
            print(f"  [PASS] State 1 contraction: dY = {dy1:.2f}mm (axial shortening)")
        else:
            print(f"  [FAIL] State 1 insufficient: dY = {dy1:.2f}mm (expect < -1mm)")

    if 'STATE_2' in summary:
        yaw2 = summary['STATE_2']['yaw']
        if abs(yaw2) > 1.0:
            print(f"  [PASS] State 2 left-bend: yaw = {yaw2:+.2f} deg")
        else:
            print(f"  [FAIL] State 2 insufficient yaw: {yaw2:.2f} deg (expect > 1)")

    if 'STATE_3' in summary:
        yaw3 = summary['STATE_3']['yaw']
        if abs(yaw3) > 1.0:
            sign_ok = (yaw3 * summary.get('STATE_2', {}).get('yaw', 1)) < 0
            sym = "opposite to State2" if sign_ok else "same as State2?"
            print(f"  [PASS] State 3 right-bend: yaw = {yaw3:+.2f} deg ({sym})")
        else:
            print(f"  [FAIL] State 3 insufficient yaw: {yaw3:.2f} deg (expect > 1)")

    if 'STATE_2' in summary and 'STATE_3' in summary:
        yaw2 = abs(summary['STATE_2']['yaw'])
        yaw3 = abs(summary['STATE_3']['yaw'])
        sym_ratio = min(yaw2, yaw3) / max(yaw2, yaw3) if max(yaw2, yaw3) > 0 else 0
        if sym_ratio > 0.8:
            print(f"  [PASS] L/R symmetry: ratio = {sym_ratio:.3f} (|yaw2|={yaw2:.2f}, |yaw3|={yaw3:.2f})")
        else:
            print(f"  [WARN] L/R asymmetry: ratio = {sym_ratio:.3f}")

    if 'RING' in summary and use_cables:
        dr = summary['RING']['delta_ring']
        if dr < -0.5:
            print(f"  [PASS] Ring contraction: dR = {dr:.2f}mm")
        else:
            print(f"  [FAIL] Ring insufficient: dR = {dr:.2f}mm (expect < -0.5mm)")
    elif not use_cables:
        print(f"  [SKIP] Ring muscle: no cables in this model")

    # Save video
    if record_video and frames:
        vid_dir = os.path.join(project_root, "record", "v3", "videos")
        os.makedirs(vid_dir, exist_ok=True)
        vid_path = os.path.join(vid_dir, f"single_seg_{model_tag}.mp4")

        import mediapy
        mediapy.write_video(vid_path, frames, fps=fps)
        renderer.close()
        print(f"\nVideo: {vid_path} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")

    return summary, baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-segment mode verification")
    parser.add_argument("--video", action="store_true", help="Record video")
    parser.add_argument("--cable", action="store_true", help="Use cable model (default: no-cable)")
    parser.add_argument("--both", action="store_true", help="Run both cable and no-cable")
    args = parser.parse_args()

    if args.both:
        print("Running both models...\n")
        summary_nc, base_nc = run_single_seg_test(use_cables=False, record_video=args.video)
        summary_c, base_c = run_single_seg_test(use_cables=True, record_video=args.video)

        # Side-by-side comparison
        print(f"\n{'='*70}")
        print(f"Comparison: No-Cable vs Cable")
        print(f"{'='*70}")
        print(f"{'Mode':<12s} │ {'No-Cable ΔY':>12s} {'yaw':>8s} │ {'Cable ΔY':>12s} {'yaw':>8s} {'ΔR':>8s}")
        print(f"{'-'*12} │ {'-'*12} {'-'*8} │ {'-'*12} {'-'*8} {'-'*8}")
        for mode in ['STATE_1', 'STATE_2', 'STATE_3', 'RING']:
            nc = summary_nc.get(mode, {})
            c = summary_c.get(mode, {})
            label = {'STATE_1': '收缩/锚固', 'STATE_2': '左弯', 'STATE_3': '右弯', 'RING': '环肌'}[mode]
            nc_dy = f"{nc.get('delta_y', 0):+.2f}mm" if nc else "—"
            nc_yaw = f"{nc.get('yaw', 0):+.2f}°" if nc else "—"
            c_dy = f"{c.get('delta_y', 0):+.2f}mm" if c else "—"
            c_yaw = f"{c.get('yaw', 0):+.2f}°" if c else "—"
            c_dr = f"{c.get('delta_ring', 0):+.2f}mm" if c else "—"
            print(f"{label:<12s} │ {nc_dy:>12s} {nc_yaw:>8s} │ {c_dy:>12s} {c_yaw:>8s} {c_dr:>8s}")
    else:
        run_single_seg_test(use_cables=args.cable, record_video=args.video)

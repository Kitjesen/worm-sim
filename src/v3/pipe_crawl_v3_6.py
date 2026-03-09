"""
Pipe Crawl V3.6 — Hybrid: Real Cable Physics + Dynamic Visual Strips
=====================================================================
Worm crawls through a channel with a 90-degree bend.  The box walls
redirect heading passively — pure rectilinear gait, no active steering.

HYBRID approach combining V3.4 physics with V3.5 visuals:
  - Physics: real cable model (no_cables=0, weld, bend=2e7) for 87° turning
  - Rendering: hide blue cable geoms + overlay dynamic visual strips
  - Gait: axial + ring muscles (full rectilinear gait)

Changes from V3.5:
  - Restored real cable physics (no_cables=0) for proper turning
  - Restored ring muscles in gait
  - Cable geoms hidden (alpha=0) instead of recolored
  - Dynamic visual strips overlaid for clean rendering

Usage:
    python pipe_crawl_v3_6.py --video             # record video
    python pipe_crawl_v3_6.py --radius 200        # bend radius (mm)
    python pipe_crawl_v3_6.py --quick             # short 10s test run
"""
import mujoco
import numpy as np
import math
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_runner import build_model_xml


# =====================================================================
#  Dynamic visual strip injection into MjvScene (from V3.5)
# =====================================================================

# Strip geometry parameters (must match exp_runner.py DEFAULTS)
STRIP_CIRCLE_R = 0.017
STRIP_R        = 0.002      # capsule radius
BOW_AMOUNT     = 0.007
NUM_STRIPS     = 8
NUM_VERTS      = 7          # 7 vertices per strip per segment = 6 capsules
Z_CENTER       = 0.022 + 0.001   # plate_radius + 0.001
SEG_LENGTH     = 0.065
STRIP_ANGLES   = [2 * math.pi * i / NUM_STRIPS for i in range(NUM_STRIPS)]

# Colors
CABLE_RGBA = np.array([0.15, 0.15, 0.18, 1.0], dtype=np.float32)   # black
RING_RGBA  = np.array([0.20, 0.20, 0.22, 1.0], dtype=np.float32)   # dark gray


def capsule_geom_params(p0, p1, radius):
    """Compute MjvGeom parameters for a capsule between two 3D points."""
    mid = (p0 + p1) * 0.5
    diff = p1 - p0
    length = np.linalg.norm(diff)
    if length < 1e-10:
        length = 1e-10
    half_len = length * 0.5

    z_ax = diff / length
    if abs(z_ax[2]) < 0.99:
        x_ax = np.cross(np.array([0.0, 0.0, 1.0]), z_ax)
    else:
        x_ax = np.cross(np.array([0.0, 1.0, 0.0]), z_ax)
    x_ax /= (np.linalg.norm(x_ax) + 1e-12)
    y_ax = np.cross(z_ax, x_ax)

    mat = np.zeros(9, dtype=np.float64)
    mat[0], mat[1], mat[2] = x_ax
    mat[3], mat[4], mat[5] = y_ax
    mat[6], mat[7], mat[8] = z_ax

    size = np.array([radius, half_len, 0.0], dtype=np.float64)
    pos = mid.astype(np.float64)
    return pos, mat, size


def hide_cable_geoms(scene):
    """Hide real cable composite geoms by setting alpha=0.

    MuJoCo's update_scene() hardcodes cable composite geoms to pure blue
    [0,0,1,1].  We detect these and make them invisible so that the
    dynamic visual strips can be overlaid instead.

    Must be called AFTER renderer.update_scene() and BEFORE inject_visual_strips().
    """
    for i in range(scene.ngeom):
        sg = scene.geoms[i]
        r, g, b, a = float(sg.rgba[0]), float(sg.rgba[1]), float(sg.rgba[2]), float(sg.rgba[3])
        # Cable composites are set to pure blue [0,0,1,1] by MuJoCo
        if b > 0.9 and r < 0.1 and g < 0.1:
            sg.rgba = [0.0, 0.0, 0.0, 0.0]  # fully transparent


def inject_visual_strips(scene, d, plate_ids, num_segments):
    """Inject dynamic visual steel strip capsules + ring balls into the scene.

    Computes strip positions from actual plate body poses at render time.
    Must be called AFTER hide_cable_geoms() and BEFORE renderer.render().
    """
    CAPSULE_TYPE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    SPHERE_TYPE  = int(mujoco.mjtGeom.mjGEOM_SPHERE)

    for seg in range(num_segments):
        pi = plate_ids[seg]
        pj = plate_ids[seg + 1]
        pos_i = d.xpos[pi].copy()
        pos_j = d.xpos[pj].copy()
        mat_i = d.xmat[pi].reshape(3, 3).copy()
        mat_j = d.xmat[pj].reshape(3, 3).copy()

        for si, angle in enumerate(STRIP_ANGLES):
            ca = math.cos(angle)
            sa = math.sin(angle)

            verts = []
            for k in range(NUM_VERTS):
                t = k / (NUM_VERTS - 1)
                center = (1.0 - t) * pos_i + t * pos_j
                R = (1.0 - t) * mat_i + t * mat_j
                # Gram-Schmidt orthonormalize
                R[:, 0] /= (np.linalg.norm(R[:, 0]) + 1e-12)
                R[:, 1] -= np.dot(R[:, 1], R[:, 0]) * R[:, 0]
                R[:, 1] /= (np.linalg.norm(R[:, 1]) + 1e-12)
                R[:, 2] = np.cross(R[:, 0], R[:, 1])

                bow_r = BOW_AMOUNT * 4.0 * t * (1.0 - t)
                r = STRIP_CIRCLE_R + bow_r
                local = np.array([r * ca, 0.0, Z_CENTER + r * sa])
                world_pos = center + R @ local
                verts.append(world_pos)

            # Add capsule geoms between consecutive vertices
            for k in range(NUM_VERTS - 1):
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                pos, mat, size = capsule_geom_params(verts[k], verts[k + 1], STRIP_R)
                mujoco.mjv_initGeom(geom, CAPSULE_TYPE, size, pos, mat, CABLE_RGBA)
                scene.ngeom += 1

            # Add ring ball sphere at midpoint vertex
            if scene.ngeom >= scene.maxgeom:
                return
            mid_k = NUM_VERTS // 2
            geom = scene.geoms[scene.ngeom]
            ring_size = np.array([0.002, 0.002, 0.002], dtype=np.float64)
            ring_pos  = verts[mid_k].astype(np.float64)
            ring_mat  = np.eye(3, dtype=np.float64).flatten()
            mujoco.mjv_initGeom(geom, SPHERE_TYPE, ring_size, ring_pos, ring_mat, RING_RGBA)
            scene.ngeom += 1


# =====================================================================
#  Channel geometry (box walls) — same as V3.4/V3.5
# =====================================================================

def generate_pipe_xml(channel_width=0.050, wall_height=0.055,
                      wall_thickness=0.005, straight_length=0.40,
                      bend_radius=0.20, n_bend_segments=16,
                      extra=0.05, ceiling_z=0.050):
    """Generate MuJoCo XML for an enclosed channel with a 90-degree right bend."""
    CW  = channel_width
    WH  = wall_height
    WT  = wall_thickness
    BR  = bend_radius
    SL  = straight_length
    HCW = CW / 2.0
    HWT = WT / 2.0
    HWH = WH / 2.0
    wall_cz = HWH
    wall_rgba = "0.6 0.6 0.7 0.3"
    ceil_rgba = "0.6 0.6 0.7 0.15"
    wall_attrs = f'contype="0" conaffinity="3" friction="1.0" rgba="{wall_rgba}"'
    ceil_attrs = f'contype="0" conaffinity="3" friction="0.01 0 0" rgba="{ceil_rgba}"'
    ceil_hz = WT / 2.0
    ceil_cz = ceiling_z + ceil_hz

    geoms = []

    # ---- Straight 1 (entry, along +Y) ----
    s1_len = SL + extra
    s1_cy  = (SL - extra) / 2.0
    s1_hy  = s1_len / 2.0
    geoms.append(
        f'    <geom name="wall_s1_L" type="box" '
        f'size="{HWT:.5f} {s1_hy:.5f} {HWH:.5f}" '
        f'pos="{-HCW - HWT:.5f} {s1_cy:.5f} {wall_cz:.5f}" {wall_attrs}/>')
    geoms.append(
        f'    <geom name="wall_s1_R" type="box" '
        f'size="{HWT:.5f} {s1_hy:.5f} {HWH:.5f}" '
        f'pos="{HCW + HWT:.5f} {s1_cy:.5f} {wall_cz:.5f}" {wall_attrs}/>')
    geoms.append(
        f'    <geom name="ceil_s1" type="box" '
        f'size="{HCW + WT:.5f} {s1_hy:.5f} {ceil_hz:.5f}" '
        f'pos="0.00000 {s1_cy:.5f} {ceil_cz:.5f}" {ceil_attrs}/>')

    # ---- 90-degree bend ----
    dphi = (math.pi / 2.0) / n_bend_segments
    seg_len = BR * dphi
    for j in range(n_bend_segments):
        phi_mid = math.pi - (j + 0.5) * dphi
        cx = BR + BR * math.cos(phi_mid)
        cy = SL + BR * math.sin(phi_mid)
        nx = math.cos(phi_mid)
        ny = math.sin(phi_mid)
        tx = math.sin(phi_mid)
        ty = -math.cos(phi_mid)
        yaw_deg = math.degrees(math.atan2(-tx, ty))
        ix = cx - (HCW + HWT) * nx
        iy = cy - (HCW + HWT) * ny
        geoms.append(
            f'    <geom name="wall_b{j}_I" type="box" '
            f'size="{HWT:.5f} {seg_len/2:.5f} {HWH:.5f}" '
            f'pos="{ix:.5f} {iy:.5f} {wall_cz:.5f}" '
            f'euler="0 0 {yaw_deg:.3f}" {wall_attrs}/>')
        ox = cx + (HCW + HWT) * nx
        oy = cy + (HCW + HWT) * ny
        geoms.append(
            f'    <geom name="wall_b{j}_O" type="box" '
            f'size="{HWT:.5f} {seg_len/2:.5f} {HWH:.5f}" '
            f'pos="{ox:.5f} {oy:.5f} {wall_cz:.5f}" '
            f'euler="0 0 {yaw_deg:.3f}" {wall_attrs}/>')
        geoms.append(
            f'    <geom name="ceil_b{j}" type="box" '
            f'size="{HCW + WT:.5f} {seg_len/2:.5f} {ceil_hz:.5f}" '
            f'pos="{cx:.5f} {cy:.5f} {ceil_cz:.5f}" '
            f'euler="0 0 {yaw_deg:.3f}" {ceil_attrs}/>')

    # ---- Straight 2 (exit, along +X) ----
    s2_cy = SL + BR
    s2_cx = BR + SL / 2.0
    s2_hx = SL / 2.0
    geoms.append(
        f'    <geom name="wall_s2_L" type="box" '
        f'size="{s2_hx:.5f} {HWT:.5f} {HWH:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy - HCW - HWT:.5f} {wall_cz:.5f}" {wall_attrs}/>')
    geoms.append(
        f'    <geom name="wall_s2_U" type="box" '
        f'size="{s2_hx:.5f} {HWT:.5f} {HWH:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy + HCW + HWT:.5f} {wall_cz:.5f}" {wall_attrs}/>')
    geoms.append(
        f'    <geom name="ceil_s2" type="box" '
        f'size="{s2_hx:.5f} {HCW + WT:.5f} {ceil_hz:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy:.5f} {ceil_cz:.5f}" {ceil_attrs}/>')

    xml_str = "\n".join(geoms)
    pipe_info = {
        "channel_width": CW, "wall_height": WH,
        "bend_radius": BR, "straight_length": SL,
        "n_pipe_geoms": len(geoms),
        "entry_start_y": -extra, "bend_entry_y": SL,
        "bend_center": (BR, SL),
        "bend_exit": (BR, SL + BR),
        "exit_end_x": BR + SL, "exit_y": SL + BR,
    }
    return xml_str, pipe_info


# =====================================================================
#  Simulation
# =====================================================================

def run_pipe_crawl(bend_radius=0.20, channel_width=0.056,
                   record_video=False, sim_time=80.0, straight_length=0.40):
    """Run worm through a channel with 90-degree bend.

    V3.6: Real cable physics (V3.4) + dynamic visual strips (V3.5).
    """
    params = {
        'num_segments': 5,
        'no_cables': 0,              # REAL cable model — for proper turning
        'cable_constraint': 'weld',  # weld keeps cables aligned with plates
        'cable_weld_solref': '0.005 1',
        'bend_stiff': 2e7,           # softer for passive turning
        'twist_stiff': 1e6,
        'plate_stiff_x': 0.0,       # free lateral — for passive turning
        'plate_stiff_y': 0.0,
        'plate_stiff_yaw': 0.0,
        # No extra plate_stiff_z/damp — real cables provide structural support
        'tendon_stiffness': 10000,
        'tendon_damping': 15,
        'axial_muscle_force': 50,
        'gait_s0': '0,0,0,1,1',     # rectilinear {0,0,2|1}
        'step_duration': 0.5,
        'sim_time': sim_time,
        'settle_time': 2.0,
        'steer_in_state2': 0,
        'state2_mode': 'symmetric',
        'state2_axial': 0.0,
    }

    xml_str, P = build_model_xml("pipe_crawl", params)
    num_segments = P['num_segments']
    num_plates = num_segments + 1
    num_axial = num_segments * 4

    # ---- Generate enclosed channel ----
    pipe_xml, pipe_info = generate_pipe_xml(
        channel_width=channel_width,
        bend_radius=bend_radius,
        straight_length=straight_length,
        ceiling_z=0.055,
    )

    # ---- Visual block + channel injection ----
    visual_block = """  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <map znear="0.001" zfar="5.0"/>
  </visual>
"""
    xml_str = xml_str.replace('<worldbody>', visual_block + '  <worldbody>')
    xml_str = xml_str.replace('  </worldbody>', pipe_xml + '\n  </worldbody>')

    # ---- Save XML & load model ----
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    bin_dir = os.path.join(project_root, "bin", "v3", "experiments")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, "exp_pipe_crawl.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}")
            for p in range(num_plates)]
    head_id = pids[-1]
    tail_id = pids[0]

    # ---- Color plates: head=red, tail=blue ----
    for gi in range(m.ngeom):
        body_id = m.geom_bodyid[gi]
        geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if geom_name.startswith(("wall_", "ceil")):
            continue
        if body_id == head_id:
            m.geom_rgba[gi] = [0.9, 0.25, 0.2, 0.95]
        elif body_id == tail_id:
            m.geom_rgba[gi] = [0.2, 0.25, 0.9, 0.95]

    print(f"Model loaded: bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}, geoms={m.ngeom}")
    print(f"Channel: width={channel_width*1000:.0f}mm, bend_R={bend_radius*1000:.0f}mm")
    print(f"Mode: HYBRID — real cables (weld, bend=2e7) + dynamic visual strips")

    # ---- Video setup ----
    frames = []
    renderer = None
    fps = 30
    if record_video:
        renderer = mujoco.Renderer(m, 720, 1280)

    # ---- Gait state machine ----
    gait_s0  = [int(x) for x in str(P['gait_s0']).split(',')]
    TN       = num_segments
    step_dur = P['step_duration']
    settle_time = P['settle_time']

    def get_states(t_act):
        j = int(t_act / step_dur) % TN
        return [gait_s0[(k + j) % TN] for k in range(TN)]

    # ---- Simulate ----
    dt = m.opt.timestep
    total_steps    = int(sim_time / dt)
    frame_interval = max(1, int(1.0 / (fps * dt)))

    head_traj       = []
    heading_history = []

    # Settle
    print(f"Settling {settle_time:.1f}s ...")
    for _ in range(int(settle_time / dt)):
        mujoco.mj_step(m, d)
    if np.any(np.isnan(d.qpos)):
        print("NaN after settling!")
        return {"error": "NaN after settling"}

    hx0 = d.xpos[head_id, 0] * 1000
    hy0 = d.xpos[head_id, 1] * 1000
    hz0 = d.xpos[head_id, 2] * 1000
    print(f"After settle: head=({hx0:.1f}, {hy0:.1f}, {hz0:.1f})mm")

    t0_wall = time.time()
    nan_flag = False
    print(f"Simulating {sim_time:.0f}s ({total_steps} steps) ...")

    for step in range(total_steps):
        t = step * dt

        # NaN check
        if step % 5000 == 0 and step > 0:
            if np.any(np.isnan(d.qpos)):
                nan_flag = True
                print(f"  NaN at t={t:.2f}s!")
                break

        # Track head
        if step % 500 == 0:
            hx = d.xpos[head_id, 0]
            hy = d.xpos[head_id, 1]
            tx = d.xpos[tail_id, 0]
            ty = d.xpos[tail_id, 1]
            hdg = math.degrees(math.atan2(hx - tx, hy - ty))
            head_traj.append((t + settle_time, hx * 1000, hy * 1000))
            heading_history.append((t + settle_time, hdg))

        # Progress
        if step % 20000 == 0 and step > 0:
            hx_mm = d.xpos[head_id, 0] * 1000
            hy_mm = d.xpos[head_id, 1] * 1000
            hz_mm = d.xpos[head_id, 2] * 1000
            hdg_now = heading_history[-1][1] if heading_history else 0
            elapsed_now = time.time() - t0_wall
            rate = step / elapsed_now if elapsed_now > 0 else 0
            print(f"  t={t:.1f}s  head=({hx_mm:.1f}, {hy_mm:.1f}, {hz_mm:.1f})mm  "
                  f"hdg={hdg_now:.1f} deg  ncon={d.ncon}  [{rate:.0f} steps/s]")

        # ---- Gait control: axial + ring muscles ----
        d.ctrl[:] = 0
        states = get_states(t)
        for seg in range(num_segments):
            s = states[seg]
            if s == 1:  # anchor: all 4 axial ON
                for mi in range(4):
                    d.ctrl[seg * 4 + mi] = 1.0
            # Ring muscle: contract in state 0 (extension), relax otherwise
            if num_axial + seg < m.nu:
                d.ctrl[num_axial + seg] = 1.0 if s == 0 else 0.0

        # Capture video frame
        if record_video and step % frame_interval == 0:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [bend_radius * 0.5,
                             straight_length + bend_radius * 0.3,
                             0.02]
            cam.distance = 0.9
            cam.azimuth  = 90
            cam.elevation = -75
            renderer.update_scene(d, cam)
            # HYBRID rendering: hide blue cables, then overlay visual strips
            hide_cable_geoms(renderer.scene)
            inject_visual_strips(renderer.scene, d, pids, num_segments)
            frames.append(renderer.render().copy())

        mujoco.mj_step(m, d)

    elapsed = time.time() - t0_wall

    # ---- Results ----
    hx_mm = d.xpos[head_id, 0] * 1000
    hy_mm = d.xpos[head_id, 1] * 1000

    heading_total = 0.0
    if len(heading_history) >= 2:
        for i in range(1, len(heading_history)):
            dh = heading_history[i][1] - heading_history[i - 1][1]
            if dh > 180:  dh -= 360
            if dh < -180: dh += 360
            heading_total += dh

    bend_entry_y = pipe_info['bend_entry_y'] * 1000
    bend_exit_x  = pipe_info['bend_exit'][0] * 1000
    reached_bend = any(pt[2] >= bend_entry_y * 0.9 for pt in head_traj)
    exited_bend  = any(pt[1] >= bend_exit_x * 0.8 for pt in head_traj)

    print(f"\n{'='*60}")
    print(f"  Pipe Crawl Results (V3.6 — HYBRID)")
    print(f"  bend_R={bend_radius*1000:.0f}mm  channel_W={channel_width*1000:.0f}mm")
    print(f"{'='*60}")
    print(f"  Head final:     ({hx_mm:.1f}, {hy_mm:.1f}) mm")
    print(f"  Heading change: {heading_total:.1f} deg  (target ~90 deg)")
    print(f"  Reached bend:   {'YES' if reached_bend else 'NO'}")
    print(f"  Exited bend:    {'YES' if exited_bend else 'NO'}")
    print(f"  NaN:            {nan_flag}")
    print(f"  Wall time:      {elapsed:.1f}s  ({total_steps/elapsed:.0f} steps/s)")
    print(f"{'='*60}")

    # ---- Save video ----
    if record_video and frames:
        import mediapy
        vid_dir = os.path.join(project_root, "record", "v3", "videos")
        os.makedirs(vid_dir, exist_ok=True)
        vid_path = os.path.join(vid_dir, f"pipe_crawl_R{int(bend_radius*1000)}_v3_6.mp4")
        mediapy.write_video(vid_path, frames, fps=fps)
        print(f"\nVideo saved: {vid_path} ({len(frames)} frames)")

        # Extract key frames
        n_frames = len(frames)
        for i, pct in enumerate([0.10, 0.35, 0.65, 0.90]):
            idx = min(int(n_frames * pct), n_frames - 1)
            frame_path = os.path.join(vid_dir, f"pipe_frame_v3_6_{i}.png")
            import cv2
            cv2.imwrite(frame_path, cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
            print(f"  Frame {idx}/{n_frames} -> {frame_path}")

        # t=32s frame specifically (user requested)
        t32_idx = min(int(32 * fps), n_frames - 1)
        t32_path = os.path.join(vid_dir, "pipe_v3_6_t32s.png")
        import cv2
        cv2.imwrite(t32_path, cv2.cvtColor(frames[t32_idx], cv2.COLOR_RGB2BGR))
        print(f"  t=32s frame -> {t32_path}")

    if renderer:
        renderer.close()

    return {
        "bend_radius_mm": bend_radius * 1000,
        "channel_width_mm": channel_width * 1000,
        "head_final_mm": (round(hx_mm, 1), round(hy_mm, 1)),
        "heading_total_deg": round(heading_total, 1),
        "reached_bend": reached_bend,
        "exited_bend": exited_bend,
        "nan": nan_flag,
        "wall_time_s": round(elapsed, 1),
    }


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipe crawl V3.6 — hybrid: real cables + visual strips")
    parser.add_argument("--video",    action="store_true",  help="Record video")
    parser.add_argument("--radius",   type=float, default=200,  help="Bend radius (mm)")
    parser.add_argument("--width",    type=float, default=56,   help="Channel width (mm)")
    parser.add_argument("--time",     type=float, default=80,   help="Simulation time (s)")
    parser.add_argument("--straight", type=float, default=400,  help="Straight section length (mm)")
    parser.add_argument("--quick",    action="store_true",      help="Quick 10s test run")
    args = parser.parse_args()

    st = 10.0 if args.quick else args.time

    run_pipe_crawl(
        bend_radius=args.radius / 1000.0,
        channel_width=args.width / 1000.0,
        record_video=args.video,
        sim_time=st,
        straight_length=args.straight / 1000.0,
    )

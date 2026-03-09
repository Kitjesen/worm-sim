"""
Pipe Crawl — Passive Turning via Wall Constraints
==================================================
Worm crawls through a channel with a 90-degree bend.  The box walls
redirect heading passively — pure rectilinear gait, no active steering.

Physics uses the no-cable plate model (fast, proven 90° turns).
Visual-only cable capsules + ring ball spheres are injected so the
steel-strip lattice is visible in rendered output.

Usage:
    python pipe_crawl.py                    # run simulation (headless)
    python pipe_crawl.py --video            # record video
    python pipe_crawl.py --radius 200       # bend radius (mm)
    python pipe_crawl.py --width 50         # channel width (mm)
    python pipe_crawl.py --quick            # short 10s test run
"""
import mujoco
import numpy as np
import math
import os
import sys
import re
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_runner import build_model_xml


# =====================================================================
#  Visual cable generation (matches cable model appearance)
# =====================================================================

def generate_visual_cables(P):
    """Generate short visual-only cable stubs attached to each plate.

    Each plate gets 8 short cable stubs (30% of seg_length) that stay
    close to the plate body.  During bending the short stubs don't
    protrude past the channel walls and stay visually connected to
    their plate.

    All geoms are visual only (contype=0, conaffinity=0, mass=0).

    Returns:
        dict mapping plate index -> string of geom XML lines
    """
    num_segments   = P['num_segments']
    seg_length     = P['seg_length']
    plate_radius   = P['plate_radius']
    strip_circle_r = P['strip_circle_r'] * 0.82  # tighter radius (less wall clip)
    num_strips     = P['num_strips']
    strip_r        = P['strip_r']
    z_center       = plate_radius + 0.001

    # Short stubs: 30% of seg_length, centered on plate midpoint
    # y spans from 0.35*seg to 0.65*seg in plate body frame
    stub_frac = 0.30
    y_start   = seg_length * (0.5 - stub_frac / 2)
    y_end     = seg_length * (0.5 + stub_frac / 2)
    num_verts = 4  # 3 capsule sub-segments per stub

    strip_angles = [2 * math.pi * i / num_strips for i in range(num_strips)]
    cable_rgba = "0.15 0.15 0.18 1.0"  # black steel strip (matches V3 original)
    ring_rgba  = "0.20 0.20 0.22 1.0"  # dark ring balls
    vis_attrs  = 'mass="0" contype="0" conaffinity="0"'

    plate_geoms = {}
    for seg in range(num_segments):
        lines = []
        for si, angle in enumerate(strip_angles):
            ca, sa = math.cos(angle), math.sin(angle)

            # Generate vertices for short stub
            verts = []
            for k in range(num_verts):
                t = k / (num_verts - 1)
                y = y_start + t * (y_end - y_start)
                r = strip_circle_r
                x = r * ca
                z = z_center + r * sa
                verts.append((x, y, z))

            # Cable capsule segments
            for k in range(num_verts - 1):
                x0, y0, z0 = verts[k]
                x1, y1, z1 = verts[k + 1]
                lines.append(
                    f'      <geom name="vc{seg}s{si}v{k}" type="capsule" '
                    f'size="{strip_r}" '
                    f'fromto="{x0:.5f} {y0:.5f} {z0:.5f} '
                    f'{x1:.5f} {y1:.5f} {z1:.5f}" '
                    f'rgba="{cable_rgba}" {vis_attrs}/>'
                )

            # Ring ball sphere at stub midpoint
            mid_k = num_verts // 2
            mx, my, mz = verts[mid_k]
            lines.append(
                f'      <geom name="rb{seg}s{si}" type="sphere" '
                f'size="0.002" pos="{mx:.5f} {my:.5f} {mz:.5f}" '
                f'rgba="{ring_rgba}" {vis_attrs}/>'
            )

        plate_geoms[seg] = "\n".join(lines)
    return plate_geoms


def inject_visual_cables(xml_str, P):
    """Inject visual cable capsules into the no-cable model XML."""
    vis_cables = generate_visual_cables(P)
    for plate_idx, geom_lines in vis_cables.items():
        marker = f'<site name="p{plate_idx}_center"'
        pattern = rf'({re.escape(marker)}[^/]*/>\n)'
        xml_str = re.sub(pattern, r'\1' + geom_lines + '\n', xml_str)
    return xml_str


def hide_foot_geoms(xml_str, num_plates):
    """Make foot geoms invisible (keep physics contact, hide visual)."""
    for p in range(num_plates):
        xml_str = xml_str.replace(
            f'name="foot{p}" type="capsule" size="0.003" fromto="-0.015 0 0.003 0.015 0 0.003"\n'
            f'            rgba="0.3 0.3 0.3 0.5"',
            f'name="foot{p}" type="capsule" size="0.003" fromto="-0.015 0 0.003 0.015 0 0.003"\n'
            f'            rgba="0.3 0.3 0.3 0.0"',
        )
    return xml_str


# =====================================================================
#  Channel geometry (box walls)
# =====================================================================

def generate_pipe_xml(channel_width=0.050, wall_height=0.055,
                      wall_thickness=0.005, straight_length=0.40,
                      bend_radius=0.20, n_bend_segments=16,
                      extra=0.05, ceiling_z=0.050):
    """Generate MuJoCo XML for an enclosed channel with a 90-degree right bend.

    Box walls (left/right) + ceiling prevent the worm from escaping.
    The worm walks on the default ground plane.
    ceiling_z: height of ceiling bottom surface (worm top at ~45mm).
    """
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
    ceil_rgba = "0.6 0.6 0.7 0.15"  # more transparent for visibility
    wall_attrs = f'contype="0" conaffinity="3" friction="1.0" rgba="{wall_rgba}"'
    ceil_attrs = f'contype="0" conaffinity="3" friction="0.01 0 0" rgba="{ceil_rgba}"'
    ceil_hz = WT / 2.0  # ceiling half-thickness
    ceil_cz = ceiling_z + ceil_hz  # ceiling center z

    geoms = []

    # ---- Straight 1 (entry, along +Y) ----
    s1_len = SL + extra
    s1_cy  = (SL - extra) / 2.0
    s1_hy  = s1_len / 2.0

    geoms.append(
        f'    <geom name="wall_s1_L" type="box" '
        f'size="{HWT:.5f} {s1_hy:.5f} {HWH:.5f}" '
        f'pos="{-HCW - HWT:.5f} {s1_cy:.5f} {wall_cz:.5f}" {wall_attrs}/>'
    )
    geoms.append(
        f'    <geom name="wall_s1_R" type="box" '
        f'size="{HWT:.5f} {s1_hy:.5f} {HWH:.5f}" '
        f'pos="{HCW + HWT:.5f} {s1_cy:.5f} {wall_cz:.5f}" {wall_attrs}/>'
    )
    # Ceiling straight 1
    geoms.append(
        f'    <geom name="ceil_s1" type="box" '
        f'size="{HCW + WT:.5f} {s1_hy:.5f} {ceil_hz:.5f}" '
        f'pos="0.00000 {s1_cy:.5f} {ceil_cz:.5f}" {ceil_attrs}/>'
    )

    # ---- 90-degree bend (arc wall segments) ----
    dphi = (math.pi / 2.0) / n_bend_segments
    seg_len = BR * dphi
    for j in range(n_bend_segments):
        phi_mid = math.pi - (j + 0.5) * dphi
        cx = BR + BR * math.cos(phi_mid)
        cy = SL + BR * math.sin(phi_mid)
        tx = math.sin(phi_mid)
        ty = -math.cos(phi_mid)
        nx = math.cos(phi_mid)
        ny = math.sin(phi_mid)
        yaw_deg = math.degrees(math.atan2(-tx, ty))

        ix = cx - (HCW + HWT) * nx
        iy = cy - (HCW + HWT) * ny
        geoms.append(
            f'    <geom name="wall_b{j}_I" type="box" '
            f'size="{HWT:.5f} {seg_len/2:.5f} {HWH:.5f}" '
            f'pos="{ix:.5f} {iy:.5f} {wall_cz:.5f}" '
            f'euler="0 0 {yaw_deg:.3f}" {wall_attrs}/>'
        )
        ox = cx + (HCW + HWT) * nx
        oy = cy + (HCW + HWT) * ny
        geoms.append(
            f'    <geom name="wall_b{j}_O" type="box" '
            f'size="{HWT:.5f} {seg_len/2:.5f} {HWH:.5f}" '
            f'pos="{ox:.5f} {oy:.5f} {wall_cz:.5f}" '
            f'euler="0 0 {yaw_deg:.3f}" {wall_attrs}/>'
        )
        # Ceiling over bend segment
        geoms.append(
            f'    <geom name="ceil_b{j}" type="box" '
            f'size="{HCW + WT:.5f} {seg_len/2:.5f} {ceil_hz:.5f}" '
            f'pos="{cx:.5f} {cy:.5f} {ceil_cz:.5f}" '
            f'euler="0 0 {yaw_deg:.3f}" {ceil_attrs}/>'
        )

    # ---- Straight 2 (exit, along +X) ----
    s2_cy = SL + BR
    s2_cx = BR + SL / 2.0
    s2_hx = SL / 2.0

    geoms.append(
        f'    <geom name="wall_s2_L" type="box" '
        f'size="{s2_hx:.5f} {HWT:.5f} {HWH:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy - HCW - HWT:.5f} {wall_cz:.5f}" {wall_attrs}/>'
    )
    geoms.append(
        f'    <geom name="wall_s2_U" type="box" '
        f'size="{s2_hx:.5f} {HWT:.5f} {HWH:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy + HCW + HWT:.5f} {wall_cz:.5f}" {wall_attrs}/>'
    )
    # Ceiling straight 2
    geoms.append(
        f'    <geom name="ceil_s2" type="box" '
        f'size="{s2_hx:.5f} {HCW + WT:.5f} {ceil_hz:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy:.5f} {ceil_cz:.5f}" {ceil_attrs}/>'
    )

    n_wall_geoms = 2 + 2 * n_bend_segments + 2 + n_bend_segments + 2  # walls + ceilings
    xml_str = "\n".join(geoms)
    pipe_info = {
        "channel_width": CW, "wall_height": WH,
        "bend_radius": BR, "straight_length": SL,
        "n_pipe_geoms": len(geoms), "n_wall_geoms": n_wall_geoms,
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
                   record_video=False, sim_time=60.0, straight_length=0.40):
    """Run worm through a channel with 90-degree bend.

    Uses cable model with spherical joints (<connect>) at cable-plate
    junctions instead of rigid welds.  This allows the body to bend
    passively when pushed by pipe walls — inspired by Riddle et al. 2025.

    Collision groups prevent cable-wall jamming while maintaining
    cable-ground friction for locomotion.
    Enclosed channel (walls + ceiling) prevents escape.
    """
    # ---- V3.1: Revert to <weld> constraint for correct cable visuals ----
    # V3.0 used <connect> (3DOF position-only) which caused cable geoms to
    # scatter during bending.  V3.1 tests whether passive turning works
    # with <weld> (6DOF) + softer bend_stiff + free plate lateral motion.
    # Hypothesis: turning comes from plate_stiff_x=0, not cable endpoint freedom.
    params = {
        'num_segments': 5,
        'no_cables': 0,              # REAL cable model
        'cable_constraint': 'weld',  # V3.1: back to weld (cables stay aligned)
        'cable_weld_solref': '0.005 1',
        'bend_stiff': 5e7,           # softer than V1 (1e8) for bending
        'twist_stiff': 1e6,
        'plate_stiff_x': 0.0,       # free lateral — for passive turning
        'plate_stiff_y': 0.0,
        'plate_stiff_yaw': 0.0,
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

    # V3.1: No collision group override — cables use default contype=1/conaffinity=1
    # (contact ground + plates).  No visual cable injection — use real cables.
    num_plates = P['num_segments'] + 1

    # ---- Generate enclosed channel (walls + ceiling) ----
    pipe_xml, pipe_info = generate_pipe_xml(
        channel_width=channel_width,
        bend_radius=bend_radius,
        straight_length=straight_length,
        ceiling_z=0.055,   # 55mm: 10mm above worm top (45mm)
    )

    # ---- Visual block for offscreen rendering ----
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

    num_segments = P['num_segments']
    num_axial    = num_segments * 4
    num_ring     = num_segments       # ring muscles (cable model)

    pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}")
            for p in range(num_plates)]
    head_id = pids[-1]
    tail_id = pids[0]

    # V3.1: No cable hiding — real cables are visible (weld keeps them aligned)

    # ---- Color plates: head=red, tail=blue ----
    for gi in range(m.ngeom):
        body_id = m.geom_bodyid[gi]
        geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        # Skip wall/ceiling geoms
        if geom_name.startswith(("wall_", "ceil")):
            continue
        if body_id == head_id:
            m.geom_rgba[gi] = [0.9, 0.25, 0.2, 0.95]
        elif body_id == tail_id:
            m.geom_rgba[gi] = [0.2, 0.25, 0.9, 0.95]

    print(f"Model loaded: bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}, geoms={m.ngeom}")
    print(f"Channel: width={channel_width*1000:.0f}mm, bend_R={bend_radius*1000:.0f}mm, "
          f"{pipe_info['n_wall_geoms']} wall geoms")

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

        # Progress report
        if step % 20000 == 0 and step > 0:
            hx_mm = d.xpos[head_id, 0] * 1000
            hy_mm = d.xpos[head_id, 1] * 1000
            hz_mm = d.xpos[head_id, 2] * 1000
            hdg_now = heading_history[-1][1] if heading_history else 0
            elapsed_now = time.time() - t0_wall
            rate = step / elapsed_now if elapsed_now > 0 else 0
            ncon = d.ncon
            print(f"  t={t:.1f}s  head=({hx_mm:.1f}, {hy_mm:.1f}, {hz_mm:.1f})mm  "
                  f"hdg={hdg_now:.1f} deg  ncon={ncon}  [{rate:.0f} steps/s]")

        # ---- Gait control (cable model: axial + ring muscles) ----
        d.ctrl[:] = 0
        states = get_states(t)
        for seg in range(num_segments):
            s = states[seg]
            if s == 1:  # anchor: all 4 axial ON
                for mi in range(4):
                    d.ctrl[seg * 4 + mi] = 1.0
            # Ring muscle: contract in state 0 (extension), relax otherwise
            d.ctrl[num_axial + seg] = 1.0 if s == 0 else 0.0

        # Capture video frame — fixed overhead camera showing full channel
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
    print(f"  Pipe Crawl Results")
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
        vid_path = os.path.join(vid_dir, f"pipe_crawl_R{int(bend_radius*1000)}_v3_1.mp4")
        mediapy.write_video(vid_path, frames, fps=fps)
        print(f"\nVideo saved: {vid_path} ({len(frames)} frames)")

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
        "head_traj": head_traj,
    }


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipe crawl — worm in channel with bend")
    parser.add_argument("--video",    action="store_true",  help="Record video")
    parser.add_argument("--radius",   type=float, default=200,  help="Bend radius (mm)")
    parser.add_argument("--width",    type=float, default=56,   help="Channel width (mm)")
    parser.add_argument("--time",     type=float, default=60,   help="Simulation time (s)")
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

"""
Worm Robot V4 — Rectilinear Locomotion + Pipe Crawling
=======================================================
Steel strips rendered as flat BOX geoms (3mm wide × 0.3mm thick),
driven by actual plate body poses — correct and realistic.

Usage:
    python worm_v4.py                    # open-field straight crawl (headless)
    python worm_v4.py --video            # record video (open field)
    python worm_v4.py --pipe             # pipe crawl, 90° bend (headless)
    python worm_v4.py --pipe --video     # pipe crawl with video
    python worm_v4.py --time 30         # set sim duration (seconds)

Coordinate frame: Y = forward, Z = up, X = lateral
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

# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants — real robot dimensions (Fang/Zhan prototype)
# ─────────────────────────────────────────────────────────────────────────────
NUM_SEGMENTS   = 5
SEG_LENGTH     = 0.1012        # 板间距 101.2mm
PLATE_RADIUS   = 0.055         # 板直径 110mm / 2
STRIP_CIRCLE_R = 0.042         # strip circle ≈ 76% plate radius
BOW_AMOUNT     = 0.023         # 钢片长度115.2mm → bow 23mm
NUM_STRIPS     = 8
NUM_VERTS      = 40
Z_CENTER       = PLATE_RADIUS + 0.001   # 0.056 m
STRIP_ANGLES   = [2 * math.pi * i / NUM_STRIPS for i in range(NUM_STRIPS)]

# Visual strip dimensions — 钢片宽度 21mm, 0.3mm厚弹簧钢
STRIP_W      = 0.021     # 21 mm wide — real robot strip width
STRIP_T      = 0.004     # 4 mm visual — thick solid bands matching reference
VIS_BOW      = 0.028     # 28 mm bow — prominent barrel bulge
VIS_BOW_PIPE = 0.015     # 15 mm bow — pipe mode
STRIP_RGBA   = np.array([0.10, 0.10, 0.12, 1.0], dtype=np.float32)  # all-black, same tone as plates
PLATE_RGBA   = np.array([0.08, 0.08, 0.10, 0.97], dtype=np.float32)  # near-black disc


# ─────────────────────────────────────────────────────────────────────────────
# Flat steel strip injection
# ─────────────────────────────────────────────────────────────────────────────

def hide_cable_geoms(scene, m, plate_id_set, n_orig):
    """Hide all original scene geoms except plates and ground plane.

    n_orig: scene.ngeom before inject_flat_strips — only process original geoms,
    leave injected strip geoms (indices n_orig..scene.ngeom-1) untouched.
    """
    OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
    for i in range(n_orig):
        g = scene.geoms[i]
        keep = False
        if int(g.objtype) == OBJ_GEOM and 0 <= g.objid < m.ngeom:
            bid = int(m.geom_bodyid[g.objid])
            if bid == 0 or bid in plate_id_set:
                keep = True
        elif int(g.objtype) != OBJ_GEOM:
            # Non-geom scene objects (lights, etc.) — keep
            keep = True
        if not keep:
            g.rgba[3] = 0.0


def fix_plate_orientations(scene, m, d, plate_ids, plate_id_set):
    """Override plate geom rotation in scene → always perpendicular to body axis.

    Identifies plate geoms using scene.geoms[i].objtype / objid (no fragile
    position matching).  Only the rotation matrix is overridden; color, size,
    and all shading properties stay exactly as MuJoCo rendered them.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    n        = len(plate_ids)

    # Build target rotation matrix (row-major) for each plate body
    target_mats = {}
    for i, pid in enumerate(plate_ids):
        if i == 0:
            axis = d.xpos[plate_ids[1]] - d.xpos[plate_ids[0]]
        elif i == n - 1:
            axis = d.xpos[plate_ids[n - 1]] - d.xpos[plate_ids[n - 2]]
        else:
            axis = d.xpos[plate_ids[i + 1]] - d.xpos[plate_ids[i - 1]]

        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            continue
        e_fwd = axis / axis_len

        e_lat = np.cross(e_fwd, world_up)
        lat_n = np.linalg.norm(e_lat)
        e_lat = e_lat / lat_n if lat_n > 1e-6 else np.array([1.0, 0.0, 0.0])
        e_up  = np.cross(e_lat, e_fwd)
        e_up /= (np.linalg.norm(e_up) + 1e-12)

        # Row-major: local X→e_lat, Y→e_up, Z→e_fwd (disc face ⊥ body axis)
        target_mats[pid] = np.array([
            e_lat[0], e_lat[1], e_lat[2],
            e_up[0],  e_up[1],  e_up[2],
            e_fwd[0], e_fwd[1], e_fwd[2],
        ], dtype=np.float32)

    # Find plate geoms via objtype/objid and override only their mat
    OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if int(g.objtype) == OBJ_GEOM and 0 <= g.objid < m.ngeom:
            bid = int(m.geom_bodyid[g.objid])
            if bid in target_mats:
                g.mat[:] = target_mats[bid].reshape(3, 3)


def inject_flat_strips(scene, d, plate_ids, pipe_mode=False):
    """Inject flat BOX geoms for spring steel strips into the render scene.

    Uses each plate's actual orientation (d.xmat) to anchor strip endpoints,
    then interpolates along the segment — strips follow plate rotation exactly,
    eliminating twisting artefacts under high contraction force.
    """
    BOX      = int(mujoco.mjtGeom.mjGEOM_BOX)
    num_segs = len(plate_ids) - 1
    world_up = np.array([0.0, 0.0, 1.0])

    for seg in range(num_segs):
        pi = plate_ids[seg]
        pj = plate_ids[seg + 1]
        pos_i = d.xpos[pi].copy()
        pos_j = d.xpos[pj].copy()

        body_axis = pos_j - pos_i
        body_len  = np.linalg.norm(body_axis)
        if body_len < 1e-6:
            continue
        e_fwd = body_axis / body_len

        # Get plate orientations → extract lateral reference from each plate's xmat
        mat_i = d.xmat[pi].reshape(3, 3)   # columns = local axes in world frame
        mat_j = d.xmat[pj].reshape(3, 3)

        # Use plate's local X axis projected onto plane ⊥ body_axis as ref
        def make_ref(mat_col):
            ref = mat_col - np.dot(mat_col, e_fwd) * e_fwd
            n = np.linalg.norm(ref)
            if n < 1e-6:
                ref = np.cross(e_fwd, world_up)
                n = np.linalg.norm(ref)
            return ref / (n + 1e-12)

        ref_i = make_ref(mat_i[:, 0])
        ref_j = make_ref(mat_j[:, 0])

        # Dynamic bow: contracted segments bulge more
        if pipe_mode:
            vis_bow = VIS_BOW_PIPE
        else:
            bow_scale = min(3.0, max(0.2, SEG_LENGTH / max(body_len, 0.012)))
            vis_bow = VIS_BOW * bow_scale

        for si in range(NUM_STRIPS):
            ca = math.cos(STRIP_ANGLES[si])
            sa = math.sin(STRIP_ANGLES[si])

            # Compute vertex positions with interpolated circumferential ref
            verts  = []
            e_lats = []
            e_ups  = []
            for k in range(NUM_VERTS):
                t = k / (NUM_VERTS - 1)

                # Interpolate circumferential reference between plates
                e_lat_k = (1.0 - t) * ref_i + t * ref_j
                e_lat_k -= np.dot(e_lat_k, e_fwd) * e_fwd  # keep ⊥ body axis
                n = np.linalg.norm(e_lat_k)
                e_lat_k = e_lat_k / (n + 1e-12)

                e_up_k = np.cross(e_lat_k, e_fwd)
                e_up_k /= (np.linalg.norm(e_up_k) + 1e-12)

                center = (1.0 - t) * pos_i + t * pos_j
                bow_r  = vis_bow * 4.0 * t * (1.0 - t)
                r      = STRIP_CIRCLE_R + bow_r

                verts.append(center + r * ca * e_lat_k + (Z_CENTER + r * sa) * e_up_k)
                e_lats.append(e_lat_k)
                e_ups.append(e_up_k)

            # Insert one flat BOX per consecutive vertex pair
            for k in range(NUM_VERTS - 1):
                if scene.ngeom >= scene.maxgeom:
                    return

                p0, p1 = verts[k], verts[k + 1]
                mid = (p0 + p1) * 0.5
                dv  = p1 - p0
                L   = np.linalg.norm(dv)
                if L < 1e-10:
                    continue
                half_len = L * 0.5 * 1.25   # 25% overlap — continuous solid bands

                z_ax = dv / L

                # Use midpoint circumferential ref for BOX orientation
                e_lat_mid = (e_lats[k] + e_lats[k + 1]) * 0.5
                e_up_mid  = (e_ups[k]  + e_ups[k + 1])  * 0.5
                e_lat_mid /= (np.linalg.norm(e_lat_mid) + 1e-12)
                e_up_mid  /= (np.linalg.norm(e_up_mid)  + 1e-12)

                radial = ca * e_lat_mid + sa * e_up_mid
                rn = np.linalg.norm(radial)
                radial = radial / rn if rn > 1e-6 else e_up_mid

                tang = np.cross(z_ax, radial)
                tn = np.linalg.norm(tang)
                tang = tang / tn if tn > 1e-6 else np.cross(z_ax, e_up_mid)

                radial = np.cross(tang, z_ax)

                mat = np.array([
                    tang[0],   tang[1],   tang[2],
                    radial[0], radial[1], radial[2],
                    z_ax[0],   z_ax[1],   z_ax[2],
                ], dtype=np.float64)

                size = np.array([STRIP_W / 2, STRIP_T / 2, half_len], dtype=np.float64)
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, BOX, size, mid.astype(np.float64), mat, STRIP_RGBA)
                geom.emission  = 0.12
                geom.specular  = 0.5
                geom.shininess = 0.4
                scene.ngeom += 1


# ─────────────────────────────────────────────────────────────────────────────
# Pipe geometry (90° bend)
# ─────────────────────────────────────────────────────────────────────────────

def build_pipe_xml(channel_width=0.130, wall_height=0.120,
                   wall_thickness=0.005, straight_length=0.40,
                   bend_radius=0.20, n_bend=16,
                   entry_extra=0.05, ceiling_z=0.055):
    """Generate MuJoCo XML fragments for an enclosed 90° channel.

    Entry section along +Y; exit section along +X.
    Returns (xml_str, info_dict).
    """
    CW  = channel_width
    WH  = wall_height
    WT  = wall_thickness
    BR  = bend_radius
    SL  = straight_length
    HCW = CW / 2.0
    HWT = WT / 2.0
    HWH = WH / 2.0
    wall_cz  = HWH
    ceil_hz  = WT / 2.0
    ceil_cz  = ceiling_z + ceil_hz

    w_attr = f'contype="0" conaffinity="3" friction="1.0" rgba="0.55 0.60 0.70 0.30"'
    c_attr = f'contype="0" conaffinity="3" friction="0.01 0 0" rgba="0.55 0.60 0.70 0.10"'

    geoms = []

    # ── Entry straight (along +Y) ──
    s1_len = SL + entry_extra
    s1_cy  = (SL - entry_extra) / 2.0
    s1_hy  = s1_len / 2.0
    geoms += [
        f'    <geom name="w_s1L" type="box" size="{HWT:.5f} {s1_hy:.5f} {HWH:.5f}" '
        f'pos="{-HCW-HWT:.5f} {s1_cy:.5f} {wall_cz:.5f}" {w_attr}/>',
        f'    <geom name="w_s1R" type="box" size="{HWT:.5f} {s1_hy:.5f} {HWH:.5f}" '
        f'pos="{HCW+HWT:.5f} {s1_cy:.5f} {wall_cz:.5f}" {w_attr}/>',
        f'    <geom name="c_s1"  type="box" size="{HCW+WT:.5f} {s1_hy:.5f} {ceil_hz:.5f}" '
        f'pos="0.0 {s1_cy:.5f} {ceil_cz:.5f}" {c_attr}/>',
    ]

    # ── 90° bend ──
    dphi    = (math.pi / 2.0) / n_bend
    seg_len = BR * dphi
    for j in range(n_bend):
        phi  = math.pi - (j + 0.5) * dphi
        cx   = BR + BR * math.cos(phi)
        cy   = SL + BR * math.sin(phi)
        nx, ny = math.cos(phi), math.sin(phi)
        yaw  = math.degrees(math.atan2(-math.sin(phi), -math.cos(phi)))
        ix   = cx - (HCW + HWT) * nx
        iy   = cy - (HCW + HWT) * ny
        ox   = cx + (HCW + HWT) * nx
        oy   = cy + (HCW + HWT) * ny
        geoms += [
            f'    <geom name="w_b{j}I" type="box" size="{HWT:.5f} {seg_len/2:.5f} {HWH:.5f}" '
            f'pos="{ix:.5f} {iy:.5f} {wall_cz:.5f}" euler="0 0 {yaw:.3f}" {w_attr}/>',
            f'    <geom name="w_b{j}O" type="box" size="{HWT:.5f} {seg_len/2:.5f} {HWH:.5f}" '
            f'pos="{ox:.5f} {oy:.5f} {wall_cz:.5f}" euler="0 0 {yaw:.3f}" {w_attr}/>',
            f'    <geom name="c_b{j}"  type="box" size="{HCW+WT:.5f} {seg_len/2:.5f} {ceil_hz:.5f}" '
            f'pos="{cx:.5f} {cy:.5f} {ceil_cz:.5f}" euler="0 0 {yaw:.3f}" {c_attr}/>',
        ]

    # ── Exit straight (along +X) ──
    s2_cy = SL + BR
    s2_cx = BR + SL / 2.0
    s2_hx = SL / 2.0
    geoms += [
        f'    <geom name="w_s2A" type="box" size="{s2_hx:.5f} {HWT:.5f} {HWH:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy-HCW-HWT:.5f} {wall_cz:.5f}" {w_attr}/>',
        f'    <geom name="w_s2B" type="box" size="{s2_hx:.5f} {HWT:.5f} {HWH:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy+HCW+HWT:.5f} {wall_cz:.5f}" {w_attr}/>',
        f'    <geom name="c_s2"  type="box" size="{s2_hx:.5f} {HCW+WT:.5f} {ceil_hz:.5f}" '
        f'pos="{s2_cx:.5f} {s2_cy:.5f} {ceil_cz:.5f}" {c_attr}/>',
    ]

    info = dict(bend_entry_y=SL, bend_exit_x=BR, exit_y=SL + BR)
    return "\n".join(geoms), info


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run(pipe_mode=False, turn_mode=None, fast_mode=False,
        record_video=False, sim_time=None):
    # ── Physics parameters ──────────────────────────────────────────────────
    # fast_mode: no_cables=1 → plates + tendons only (DOF ~36 vs ~1116)
    #            Visual strips still rendered from plate positions.
    #            Needs weld constraints + tendon springs for structural integrity.
    no_cables_flag = 1 if fast_mode else 0

    # Structural params for no-cable mode (replace cable stiffness)
    fast_extras = dict(
        plate_constraint    = 'connect',        # 3DOF position constraint (keeps inter-plate distance)
        plate_weld_solref   = '0.002 1',        # tight constraint
        plate_stiff_y       = 300.0,            # axial restoring stiffness
        plate_stiff_pitch   = 30.0,             # resist tumbling
        plate_stiff_roll    = 30.0,
        plate_damp_x        = 2.0,              # lateral damping
        plate_damp_y        = 3.0,              # axial damping
        plate_damp_yaw      = 2.0,
    ) if fast_mode else {}

    if turn_mode:
        # Circular locomotion — State 2/3 with diagonal steer tendons
        turn_gait = '2,0,0,1,1' if turn_mode == 'left' else '3,0,0,1,1'
        params = dict(
            num_segments      = NUM_SEGMENTS,
            no_cables         = no_cables_flag,
            cable_constraint  = 'weld',
            cable_weld_solref = '0.002 1',
            bend_stiff        = 1e8,
            twist_stiff       = 2e6,
            plate_stiff_x     = 0.0,        # FREE lateral — critical for turning
            plate_stiff_y     = 0.0,
            plate_stiff_yaw   = 0.0,        # free yaw for heading change
            axial_muscle_force= 50,
            steer_muscle_force= 25,         # gentle turning (axial=50)
            gait_s0           = turn_gait,
            step_duration     = 0.5,
            settle_time       = 1.0 if not fast_mode else 0.3,
            sim_time          = sim_time or 40.0,
        )
    elif pipe_mode:
        params = dict(
            num_segments      = NUM_SEGMENTS,
            no_cables         = no_cables_flag,
            cable_constraint  = 'weld',
            cable_weld_solref = '0.005 1',
            bend_stiff        = 2e7,        # softer → passive turning in pipe
            twist_stiff       = 1e6,
            plate_stiff_x     = 0.0,        # free lateral → guided by walls
            plate_stiff_y     = 0.0,
            plate_stiff_yaw   = 0.0,        # free yaw → pipe passive turning
            axial_muscle_force= 50,
            gait_s0           = '0,0,0,1,1',   # rectilinear {0,0,2|1}
            step_duration     = 0.5,
            settle_time       = 2.0 if not fast_mode else 0.5,
            sim_time          = sim_time or 80.0,
        )
    else:
        params = dict(
            num_segments      = NUM_SEGMENTS,
            no_cables         = no_cables_flag,
            cable_constraint  = 'weld',
            cable_weld_solref = '0.002 1',
            bend_stiff        = 1e8,        # stiff → stable upright posture
            twist_stiff       = 2e6,
            plate_stiff_x     = 500.0,      # anti-lateral drift
            plate_stiff_y     = 0.0,
            plate_stiff_yaw   = 0.0,
            axial_muscle_force= 100,
            gait_s0           = '0,0,0,1,1',   # rectilinear {0,0,2|1}
            step_duration     = 0.5,
            settle_time       = 1.0 if not fast_mode else 0.3,
            sim_time          = sim_time or 25.0,
        )

    # Merge fast-mode structural params
    params.update(fast_extras)

    # ── Build XML ───────────────────────────────────────────────────────────
    if turn_mode:
        exp_id = f"turn_{turn_mode}"
    elif pipe_mode:
        exp_id = "pipe"
    else:
        exp_id = "straight"
    if fast_mode:
        exp_id += "_fast"
    xml_str, P = build_model_xml(exp_id, params)

    num_plates = NUM_SEGMENTS + 1
    num_axial  = NUM_SEGMENTS * 4

    # Inject pipe geometry into worldbody
    if pipe_mode:
        pipe_xml, pipe_info = build_pipe_xml()
        xml_str = xml_str.replace('  </worldbody>', pipe_xml + '\n  </worldbody>')

    # Inject offscreen framebuffer large enough for 1280-wide rendering
    if record_video and '<visual>' not in xml_str:
        xml_str = xml_str.replace(
            '<size memory=',
            '<visual><global offwidth="1280" offheight="720"/></visual>\n  <size memory='
        )

    # Save XML
    src_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    bin_dir      = os.path.join(project_root, "bin", "v3")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, f"worm_v4_{exp_id}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    # ── Load model ──────────────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    plate_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}")
                 for p in range(num_plates)]
    head_id = plate_ids[-1]
    tail_id = plate_ids[0]

    # Steer muscle indices: axial(20) + ring(0 or 5) + steer(10)
    num_ring = 0 if no_cables_flag else NUM_SEGMENTS
    num_steer_base = num_axial + num_ring

    fast_tag = " [FAST]" if fast_mode else ""
    if turn_mode:
        mode_str = f"TURN {turn_mode.upper()}{fast_tag}"
    elif pipe_mode:
        mode_str = f"PIPE CRAWL{fast_tag}"
    else:
        mode_str = f"OPEN FIELD{fast_tag}"
    print(f"Model: bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}")
    print(f"Mode: {mode_str}")

    # ── Hide cable/ring geoms at model level, colour plates dark
    plate_id_set = set(plate_ids)
    cable_body_ids = set()
    for bid in range(m.nbody):
        bname = m.body(bid).name
        if bname and ((bname.startswith('c') and 's' in bname and 'B' in bname) or
                       bname.startswith('rb')):
            cable_body_ids.add(bid)
    for gi in range(m.ngeom):
        bid = int(m.geom_bodyid[gi])
        if bid in cable_body_ids:
            m.geom_rgba[gi] = [0, 0, 0, 0]
            m.geom_group[gi] = 4
        elif bid in plate_id_set:
            m.geom_rgba[gi] = [0.10, 0.10, 0.12, 0.95]

    # ── Settle ──────────────────────────────────────────────────────────────
    settle_time = P['settle_time']
    settle_steps = int(settle_time / m.opt.timestep)
    print(f"Settling {settle_time:.1f}s ...")
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    if np.any(np.isnan(d.qpos)):
        print("FATAL: NaN after settling"); return

    hx0 = d.xpos[head_id, 0] * 1000
    hy0 = d.xpos[head_id, 1] * 1000
    print(f"Post-settle: head=({hx0:.1f}, {hy0:.1f}) mm")

    # ── Gait: Zhan/Fang 2019 IJRR — discrete retrograde peristalsis {0,0,2|1}
    gait_s0   = [int(x) for x in str(P['gait_s0']).split(',')]
    step_dur  = P['step_duration']

    def get_states(t):
        j = int(t / step_dur) % NUM_SEGMENTS
        return [gait_s0[(k + j) % NUM_SEGMENTS] for k in range(NUM_SEGMENTS)]

    def apply_control(d, t):
        d.ctrl[:] = 0
        states = get_states(t)
        for seg, s in enumerate(states):
            if s == 1:
                # State 1 — Anchor: all 4 axial ON → contracted, grips ground
                for mi in range(4):
                    d.ctrl[seg * 4 + mi] = 1.0
            elif s == 2:
                # State 2 — Left bend: diagonal steer tendon (stL) ON
                steer_idx = num_steer_base + seg * 2
                if steer_idx < m.nu:
                    d.ctrl[steer_idx] = 1.0
            elif s == 3:
                # State 3 — Right bend: diagonal steer tendon (stR) ON
                steer_idx = num_steer_base + seg * 2 + 1
                if steer_idx < m.nu:
                    d.ctrl[steer_idx] = 1.0
            # Ring muscle: slim when extending/bending (0,2,3), expanded when anchored (1)
            ring_idx = num_axial + seg
            if ring_idx < m.nu:
                d.ctrl[ring_idx] = 1.0 if s != 1 else 0.0

    # ── Video renderer ──────────────────────────────────────────────────────
    renderer = None
    frames   = []
    fps      = 30
    if record_video:
        renderer = mujoco.Renderer(m, 720, 1280, max_geom=10000)
        vopt = mujoco.MjvOption()
        vopt.geomgroup[4] = 0   # hide cable geoms
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
        print("Renderer created (720×1280, max_geom=5000)")

    # ── Main simulation loop ─────────────────────────────────────────────────
    sim_total   = P['sim_time']
    total_steps = int(sim_total / m.opt.timestep)
    frame_iv    = max(1, int(1.0 / (fps * m.opt.timestep)))
    dt          = m.opt.timestep

    head_traj    = []
    # Smoothed camera lookat — EMA over frames to kill gait-cycle vibration
    lookat_smooth = ((d.xpos[head_id] + d.xpos[tail_id]) / 2.0).copy()
    t0_wall      = time.time()
    print(f"Simulating {sim_total:.0f}s ({total_steps} steps) ...")

    for step in range(total_steps):
        t = step * dt
        apply_control(d, t)
        mujoco.mj_step(m, d)

        # NaN check every 5000 steps
        if step % 5000 == 0 and step > 0:
            if np.any(np.isnan(d.qpos)):
                print(f"NaN at t={t:.2f}s!"); break

        # Track head XY
        if step % 500 == 0:
            head_traj.append((t + settle_time,
                              d.xpos[head_id, 0] * 1000,
                              d.xpos[head_id, 1] * 1000))

        # Progress log every 10 000 steps
        if step % 10000 == 0 and step > 0:
            hx = d.xpos[head_id, 0] * 1000
            hy = d.xpos[head_id, 1] * 1000
            tx = d.xpos[tail_id, 0] * 1000
            ty = d.xpos[tail_id, 1] * 1000
            hdg = math.degrees(math.atan2(hx - tx, hy - ty))
            rate = step / (time.time() - t0_wall)
            print(f"  t={t:.1f}s  head=({hx:.1f},{hy:.1f})mm  hdg={hdg:.1f}°  "
                  f"ncon={d.ncon}  [{rate:.0f} steps/s]")

        # Capture frame
        if record_video and step % frame_iv == 0 and renderer is not None:
            cam          = mujoco.MjvCamera()
            cam.type     = mujoco.mjtCamera.mjCAMERA_FREE

            if pipe_mode:
                cam.distance  = 2.0
                cam.elevation = -60
                cam.azimuth   = 90
                cam.lookat[:] = [0.25, 0.80, 0.05]
            elif turn_mode:
                cam.distance  = 1.2
                cam.elevation = -50
                cam.azimuth   = 90
                mid = (d.xpos[head_id] + d.xpos[tail_id]) / 2.0
                lookat_smooth += 0.02 * (mid - lookat_smooth)
                cam.lookat[:] = lookat_smooth
            else:
                cam.distance  = 1.0
                cam.elevation = -25
                cam.azimuth   = 45
                mid = (d.xpos[head_id] + d.xpos[tail_id]) / 2.0
                lookat_smooth += 0.03 * (mid - lookat_smooth)
                cam.lookat[:] = lookat_smooth

            renderer.update_scene(d, cam, scene_option=vopt)
            n_orig = renderer.scene.ngeom
            hide_cable_geoms(renderer.scene, m, plate_id_set, n_orig)
            fix_plate_orientations(renderer.scene, m, d, plate_ids, plate_id_set)
            inject_flat_strips(renderer.scene, d, plate_ids, pipe_mode=pipe_mode)
            frames.append(renderer.render().copy())

    elapsed = time.time() - t0_wall

    # ── Results ──────────────────────────────────────────────────────────────
    hx_f = d.xpos[head_id, 0] * 1000
    hy_f = d.xpos[head_id, 1] * 1000

    heading = math.degrees(math.atan2(
        d.xpos[head_id, 0] - d.xpos[tail_id, 0],
        d.xpos[head_id, 1] - d.xpos[tail_id, 1]))

    displacement = math.hypot(hx_f - hx0, hy_f - hy0)

    print(f"\n{'─'*55}")
    print(f"  Worm V4 — {mode_str}")
    print(f"  Head final:    ({hx_f:.1f}, {hy_f:.1f}) mm")
    print(f"  Displacement:  {displacement:.1f} mm in {sim_total:.0f}s")
    print(f"  Speed:         {displacement/sim_total:.2f} mm/s")
    print(f"  Heading:       {heading:.1f}°  (0°=+Y forward)")
    print(f"  Wall time:     {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")
    print(f"{'─'*55}")

    # ── Save video ───────────────────────────────────────────────────────────
    if record_video and frames:
        try:
            import mediapy
            vid_dir  = os.path.join(project_root, "record", "v4", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            tag      = f"turn_{turn_mode}" if turn_mode else ("pipe" if pipe_mode else "straight")
            vid_path = os.path.join(vid_dir, f"worm_v4_{tag}.mp4")
            mediapy.write_video(vid_path, frames, fps=fps)
            print(f"Video: {vid_path}  ({len(frames)} frames)")
        except ImportError:
            print("mediapy not found — video not saved")

    if renderer:
        renderer.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Worm V4 — rectilinear + pipe + circular locomotion")
    ap.add_argument("--pipe",  action="store_true", help="Pipe crawling mode (90° bend)")
    ap.add_argument("--turn",  choices=["left", "right"], default=None,
                    help="Circular locomotion (State 2/3 diagonal steer)")
    ap.add_argument("--fast",  action="store_true",
                    help="Fast mode: no cables (DOF ~36 vs ~1116), ~30× faster")
    ap.add_argument("--video", action="store_true", help="Record video")
    ap.add_argument("--time",  type=float, default=None, help="Sim duration (seconds)")
    args = ap.parse_args()

    run(pipe_mode=args.pipe, turn_mode=args.turn, fast_mode=args.fast,
        record_video=args.video, sim_time=args.time)

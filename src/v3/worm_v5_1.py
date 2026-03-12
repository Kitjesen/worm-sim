"""
Worm Robot V5.1 — Rigid Body + Passive Wheel + Real Mesh
=========================================================
Based on real hardware URDF (v6.urdf) with SolidWorks STL meshes.

Each segment is a single rigid body with passive (free-rolling) wheels,
connected by slide (peristaltic) + hinge (yaw) joints.

Architecture:
  16 bodies, 24 DOF, 8 actuators (4 slide + 4 yaw), NO wheel motors
  Real STL meshes for visual, primitive geoms for collision

Passive wheels provide anisotropic friction:
  Roll freely along Y (low resistance) / resist lateral X slip (high friction)

Modes:
  snake:    sinusoidal yaw wave — primary flat-ground locomotion
  worm:     slide-only peristalsis — visible contraction wave
  combined: both yaw wave + slide wave

Usage:
    python worm_v5_1.py --mode snake --video
    python worm_v5_1.py --mode worm --video
    python worm_v5_1.py --mode combined --video

Coordinate frame: Y = forward, Z = up, X = lateral
"""

import mujoco
import numpy as np
import math
import os
import time
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants — from v6.urdf / SolidWorks meshes (real hardware scale)
# ─────────────────────────────────────────────────────────────────────────────
NUM_SEGMENTS  = 5

# Body: mesh ~141mm diam × 189mm long; collision capsule smaller (hidden inside mesh)
SEG_HALF_LEN  = 0.080       # 80mm collision half-length
COL_RADIUS    = 0.030       # 30mm collision capsule radius (inside mesh)
# Per-segment spacing from v5 URDF (Rz(-90°) mapped: URDF -X → sim +Y)
SEG_SPACING_01   = 0.201    # base_link → body2 (URDF X=-0.201)
SEG_SPACING_REST = 0.1615   # body2→body3, body3→body4, body4→body5 (URDF X=-0.1615)
BODY_MASS     = 0.45        # 450g per segment (from URDF)

# Wheel: mesh 13×25×25mm → radius 12.5mm
WHEEL_RADIUS  = 0.0125      # 12.5mm
WHEEL_THICK   = 0.0065      # 6.5mm half-thickness
WHEEL_MASS    = 0.048       # 48g per wheel

# Per-segment wheel positions in SIM frame (v5 URDF Rz(-90°) mapped)
# URDF body0:  (0.005, ±0.067, -0.068) → sim (±0.067, -0.005, -0.068)
# URDF body1:  (-0.151, ±0.061, -0.068) → sim (±0.061, +0.151, -0.068)  (rear of base_link)
# URDF body2+: (-0.112, ±0.061, -0.096) → sim (±0.061, +0.112, -0.096)
_W0_FRONT = ((0.067262, -0.005, -0.06759), (-0.067262, -0.005, -0.06759))
_W0_REAR  = ((0.060762,  0.151, -0.06759), (-0.060762, 0.151, -0.06759))
_W1 = ((0.060762,  0.1115, -0.09559), (-0.060762, 0.1115, -0.09559))
WHEEL_POS = [_W0_FRONT, _W1, _W1, _W1, _W1]

# Body center height: seg0 wheel bottom at ground
BODY_Z        = 0.06759 + WHEEL_RADIUS  # 80.1mm

# seg1 (body2) is 28mm above seg0 (from URDF body2→base_link Z offset)
SEG1_DZ       = 0.028

SLIDE_RANGE   = 0.015       # ±15mm slide joint range (larger robot)
SLIDE_STIFF   = 150.0       # N/m spring stiffness (steel strip elastic return)
YAW_RANGE     = 0.524       # ±30° yaw joint range

# ── Steel strip deformation visual ──────────────────────────────────────────
# Strips connect end plates (隔板) of adjacent segments across the joint.
# V-shape: 2 flat panels per strip (parent→apex, apex→child). Clean, no staircase.
NUM_STRIPS      = 3          # 3 wide bands per joint (120° apart, matches reference)
STRIP_CIRCLE_R  = 0.046      # 46mm base radius at end plates
STRIP_W         = 0.020      # 20mm wide band (real sheet metal width)
STRIP_T         = 0.0006     # 0.6mm thickness
STRIP_HALF_SPAN = 0.050      # 50mm each side of joint center
VIS_BOW_MIN     = 0.0005     # 0.5mm resting bow (barely visible hint)
VIS_BOW_MAX     = 0.016      # 16mm max bow
STRIP_RGBA      = np.array([0.82, 0.80, 0.78, 1.0], dtype=np.float32)  # warm metallic
STRIP_ANGLES    = [2.0 * math.pi * k / NUM_STRIPS for k in range(NUM_STRIPS)]

# Peristaltic gait vector [tail→head]: 0=advancing, 1=anchored
GAIT = [0, 0, 0, 1, 1]
STEP_DURATION = 0.6          # seconds per gait phase

# Mesh names: seg_index → (body_mesh, wL_mesh, wR_mesh)
# seg1-3: custom open-frame mesh (from gen_segment_mesh.py) for strip visibility
# seg0/seg4: original SolidWorks STL
MESH_MAP = [
    ("base_link",      "body0_left_wheel",  "body0_right_wheel"),
    ("seg_mid_frame",  "body2_left_wheel",  "body2_right_wheel"),
    ("seg_mid_frame",  "body3_left_wheel",  "body3_right_wheel"),
    ("seg_mid_frame",  "body4_left_wheel",  "body4_right_wheel"),
    ("body4",          "body5_left_wheel",  "body5_right_wheel"),  # seg4: original head
]

# seg0 (base_link) has an extra REAR wheel pair (body1 wheels) from the URDF
SEG0_REAR_WHEELS = ("body1_left_wheel", "body1_right_wheel")

# URDF colors
BODY_COLORS = [
    "0.75 0.75 0.75 1",     # base_link — light grey
    "0.95 0.95 0.97 1",     # body2 — white
    "0.50 0.50 0.50 1",     # body3 — dark grey
    "0.95 0.95 0.97 1",     # body4 — white
    "0.65 0.62 0.59 1",     # body5 — warm grey
]
WHEEL_COLOR = "0.88 0.86 0.84 1"
FLOOR_COLOR = "0.93 0.93 0.90 1"

# ── Body mesh transforms (from v5.urdf) ──────────────────────────────────────
# v5 URDF: chain along -X, all bodies identity quat (no alternating rotation).
# Simulation: chain along +Y.  Rz(-90°) maps URDF -X → sim +Y.
#
# Rz(-90°) quat = (0.707107, 0, 0, -0.707107)  — same for ALL segments.
# Pos = negated Rz(-90°)-rotated mesh AABB center.
# Rz(-90°): (x,y,z) → (y, -x, z)
#
# Format: (pos, quat_wxyz)
BODY_MESH_XFORM = [
    # seg0: base_link — original SolidWorks mesh, Rz(-90°)
    ("-0.00115 0.00007 0.02178",  "0.707107 0 0 -0.707107"),
    # seg1-3: custom open-frame mesh, generated in sim frame (identity transform)
    ("0 0 0",  "1 0 0 0"),
    ("0 0 0",  "1 0 0 0"),
    ("0 0 0",  "1 0 0 0"),
    # seg4: original body4 mesh, Rz(-90°)
    ("-0.00086 0.00002 0.01412",  "0.707107 0 0 -0.707107"),
]

# Wheel mesh: centered at origin, Rz(-90°) to match body frame mapping.
# Format: (left_pos, left_quat, right_pos, right_quat)
_WQ = "0.707107 0 0 -0.707107"   # Rz(-90°) same as body meshes
WHEEL_MESH_XFORM = [
    ("0 0 0", _WQ, "0 0 0", _WQ),  # seg0
    ("0 0 0", _WQ, "0 0 0", _WQ),  # seg1
    ("0 0 0", _WQ, "0 0 0", _WQ),  # seg2
    ("0 0 0", _WQ, "0 0 0", _WQ),  # seg3
    ("0 0 0", _WQ, "0 0 0", _WQ),  # seg4
]


# ─────────────────────────────────────────────────────────────────────────────
# Steel strip deformation rendering
# ─────────────────────────────────────────────────────────────────────────────

def inject_strips(scene, d, seg_ids, natural_spacings):
    """Inject deformed steel strip visuals at joints between adjacent segments.

    V-shape approach: 2 flat BOX panels per strip (end→apex, apex→end).
    Clean flat surfaces, no staircase artifacts from curve approximation.
    """
    BOX = mujoco.mjtGeom.mjGEOM_BOX

    for pair_idx in range(len(seg_ids) - 1):
        p0 = d.xpos[seg_ids[pair_idx]].copy()
        p1 = d.xpos[seg_ids[pair_idx + 1]].copy()

        axis = p1 - p0
        body_len = np.linalg.norm(axis)
        if body_len < 0.001:
            continue
        e_y = axis / body_len

        # Dynamic bow: more compression → more outward bulge
        nat_len = natural_spacings[pair_idx]
        comp_ratio = 1.0 - body_len / nat_len
        bow_t = max(0.0, min(1.0, comp_ratio * 10.0))
        vis_bow = VIS_BOW_MIN + (VIS_BOW_MAX - VIS_BOW_MIN) * bow_t

        # Joint center = child body origin
        joint_pos = p1

        # Perpendicular frame
        up = np.array([0.0, 0.0, 1.0])
        e_x = np.cross(e_y, up)
        e_x_n = np.linalg.norm(e_x)
        if e_x_n < 1e-6:
            e_x = np.array([1.0, 0.0, 0.0])
        else:
            e_x /= e_x_n
        e_z = np.cross(e_x, e_y)

        # Yaw bending detection
        r0 = d.xmat[seg_ids[pair_idx]].reshape(3, 3)
        r1 = d.xmat[seg_ids[pair_idx + 1]].reshape(3, 3)
        fwd0, fwd1 = r0[:, 1], r1[:, 1]
        bend_cross = np.cross(fwd0, fwd1)
        bend_amount = np.linalg.norm(bend_cross)

        for angle in STRIP_ANGLES:
            ca, sa = math.cos(angle), math.sin(angle)
            e_r = ca * e_x + sa * e_z

            # Per-strip yaw modulation
            yaw_mod = 1.0
            if bend_amount > 0.001:
                bend_dir = bend_cross / bend_amount
                yaw_mod = 1.0 + 3.0 * np.dot(e_r, bend_dir)
                yaw_mod = max(0.2, min(2.5, yaw_mod))

            strip_bow = vis_bow * yaw_mod

            # 3 key points: parent end plate, apex (peak bow), child end plate
            pt_parent = joint_pos - STRIP_HALF_SPAN * e_y + STRIP_CIRCLE_R * e_r
            pt_apex   = joint_pos + (STRIP_CIRCLE_R + strip_bow) * e_r
            pt_child  = joint_pos + STRIP_HALF_SPAN * e_y + STRIP_CIRCLE_R * e_r

            # Panel 1: parent → apex
            # Panel 2: apex → child
            for (va, vb) in [(pt_parent, pt_apex), (pt_apex, pt_child)]:
                if scene.ngeom >= scene.maxgeom:
                    return
                mid = (va + vb) * 0.5
                dv = vb - va
                seg_len = np.linalg.norm(dv)
                if seg_len < 1e-6:
                    continue
                z_ax = dv / seg_len

                # Width direction = tangent to body surface
                tang = np.cross(z_ax, e_r)
                tang_n = np.linalg.norm(tang)
                if tang_n < 1e-6:
                    continue
                tang /= tang_n
                radial = np.cross(tang, z_ax)

                mat = np.zeros(9, dtype=np.float64)
                mat[0:3] = tang
                mat[3:6] = radial
                mat[6:9] = z_ax

                size = np.array([STRIP_W / 2, STRIP_T / 2, seg_len / 2])

                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, BOX, size, mid, mat, STRIP_RGBA)
                geom.emission = 0.25
                geom.specular = 0.5
                geom.shininess = 0.4
                scene.ngeom += 1


# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo XML generation
# ─────────────────────────────────────────────────────────────────────────────

def build_xml(mesh_dir):
    """Generate MuJoCo XML with real STL meshes."""
    L = []
    a = L.append

    a('<?xml version="1.0"?>')
    a('<mujoco model="worm_v5_1">')
    a(f'  <compiler meshdir="{mesh_dir}" angle="radian"/>')
    a('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    a('  <size memory="200M"/>')
    a('')
    a('  <visual>')
    a('    <global offwidth="1280" offheight="720"/>')
    a('    <quality shadowsize="2048"/>')
    a('    <map znear="0.001"/>')
    a('  </visual>')
    a('')
    a('  <default>')
    a('    <geom condim="4"/>')
    a('  </default>')
    a('')

    # ── Assets ──
    a('  <asset>')
    # Grid floor texture
    a('    <texture name="grid" type="2d" builtin="checker" width="512" height="512"'
      ' rgb1="0.93 0.93 0.90" rgb2="0.82 0.82 0.80"/>')
    a('    <material name="grid_mat" texture="grid" texrepeat="10 10"'
      ' reflectance="0.1"/>')
    # Mesh assets (deduplicated)
    seen = set()
    for body_m, wl_m, wr_m in MESH_MAP:
        for name in (body_m, wl_m, wr_m):
            if name not in seen:
                a(f'    <mesh name="{name}" file="{name}.STL"/>')
                seen.add(name)
    # seg0 rear wheels (body1)
    for name in SEG0_REAR_WHEELS:
        if name not in seen:
            a(f'    <mesh name="{name}" file="{name}.STL"/>')
            seen.add(name)
    a('  </asset>')
    a('')

    a('  <worldbody>')
    a('    <light pos="0 0.5 2.5" dir="0 0 -1" diffuse="0.9 0.9 0.9"'
      ' ambient="0.25 0.25 0.25"/>')
    a('    <light pos="0.8 -0.3 2.0" dir="-0.3 0.2 -1" diffuse="0.5 0.5 0.5"/>')
    a('    <light pos="-0.6 0.3 1.5" dir="0.2 0 -1" diffuse="0.3 0.3 0.3"/>')
    a('    <geom name="floor" type="plane" size="5 5 0.1" material="grid_mat"'
      ' friction="1.0 0.005 0.001"/>')
    a('')

    # ── Nested body chain: seg0 (tail) → seg4 (head) ──
    for i in range(NUM_SEGMENTS):
        depth = i + 2
        ind = "  " * depth
        body_mesh, wl_mesh, wr_mesh = MESH_MAP[i]
        body_rgba = BODY_COLORS[i]

        if i == 0:
            a(f'{ind}<body name="seg{i}" pos="0 0 {BODY_Z:.4f}">')
            a(f'{ind}  <freejoint name="root"/>')
        else:
            sp = SEG_SPACING_01 if i == 1 else SEG_SPACING_REST
            dz = SEG1_DZ if i == 1 else 0
            a(f'{ind}<body name="seg{i}" pos="0 {sp} {dz}">')
            a(f'{ind}  <joint name="slide{i-1}_{i}" type="slide" axis="0 1 0"'
              f' range="{-SLIDE_RANGE} {SLIDE_RANGE}" stiffness="{SLIDE_STIFF}"'
              f' damping="5"/>')
            a(f'{ind}  <joint name="yaw{i-1}_{i}" type="hinge" axis="0 0 1"'
              f' range="{-YAW_RANGE} {YAW_RANGE}" damping="3"/>')

        # Visual mesh — quat from URDF compiled model
        mesh_pos, mesh_quat = BODY_MESH_XFORM[i]
        a(f'{ind}  <geom name="vis{i}" type="mesh" mesh="{body_mesh}"'
          f' pos="{mesh_pos}" quat="{mesh_quat}" rgba="{body_rgba}"'
          f' contype="0" conaffinity="0" mass="0.001"/>')

        # Collision capsule (hidden inside mesh)
        a(f'{ind}  <geom name="col{i}" type="capsule"'
          f' fromto="0 {-SEG_HALF_LEN} 0  0 {SEG_HALF_LEN} 0"'
          f' size="{COL_RADIUS}" mass="{BODY_MASS}" rgba="1 0 0 0"'
          f' friction="0.3 0.005 0.001"/>')

        # Wheel mesh transforms (URDF global quats)
        wl_pos, wl_quat, wr_pos, wr_quat = WHEEL_MESH_XFORM[i]

        # Left wheel — PASSIVE (free hinge, axle along X = lateral, rolls along Y)
        wl_bpos = WHEEL_POS[i][0]
        a(f'{ind}  <body name="seg{i}_wL" pos="{wl_bpos[0]} {wl_bpos[1]} {wl_bpos[2]}">')
        a(f'{ind}    <joint name="wL{i}" type="hinge" axis="1 0 0"'
          f' damping="0.001" armature="0.0001"/>')
        # Wheel mesh visual
        a(f'{ind}    <geom name="vis_wL{i}" type="mesh" mesh="{wl_mesh}"'
          f' pos="{wl_pos}" quat="{wl_quat}"'
          f' rgba="{WHEEL_COLOR}" contype="0" conaffinity="0" mass="0.001"/>')
        # Wheel collision cylinder (axis along X via Ry(90°))
        a(f'{ind}    <geom name="col_wL{i}" type="cylinder"'
          f' size="{WHEEL_RADIUS} {WHEEL_THICK}" euler="0 1.5708 0"'
          f' mass="{WHEEL_MASS}" rgba="0 0 0 0"'
          f' friction="1.5 0.01 0.001"/>')
        a(f'{ind}  </body>')

        # Right wheel — PASSIVE
        wr_bpos = WHEEL_POS[i][1]
        a(f'{ind}  <body name="seg{i}_wR" pos="{wr_bpos[0]} {wr_bpos[1]} {wr_bpos[2]}">')
        a(f'{ind}    <joint name="wR{i}" type="hinge" axis="1 0 0"'
          f' damping="0.001" armature="0.0001"/>')
        a(f'{ind}    <geom name="vis_wR{i}" type="mesh" mesh="{wr_mesh}"'
          f' pos="{wr_pos}" quat="{wr_quat}"'
          f' rgba="{WHEEL_COLOR}" contype="0" conaffinity="0" mass="0.001"/>')
        a(f'{ind}    <geom name="col_wR{i}" type="cylinder"'
          f' size="{WHEEL_RADIUS} {WHEEL_THICK}" euler="0 1.5708 0"'
          f' mass="{WHEEL_MASS}" rgba="0 0 0 0"'
          f' friction="1.5 0.01 0.001"/>')
        a(f'{ind}  </body>')

        # seg0: add body1 REAR wheel pair (URDF base_link has 4 wheels total)
        if i == 0:
            rear_wl_mesh, rear_wr_mesh = SEG0_REAR_WHEELS
            rear_wl_bpos = _W0_REAR[0]
            rear_wr_bpos = _W0_REAR[1]
            rear_wl_pos, rear_wl_quat, rear_wr_pos, rear_wr_quat = WHEEL_MESH_XFORM[0]

            a(f'{ind}  <body name="seg0_wL_rear" pos="{rear_wl_bpos[0]} {rear_wl_bpos[1]} {rear_wl_bpos[2]}">')
            a(f'{ind}    <joint name="wL0_rear" type="hinge" axis="1 0 0"'
              f' damping="0.001" armature="0.0001"/>')
            a(f'{ind}    <geom name="vis_wL0_rear" type="mesh" mesh="{rear_wl_mesh}"'
              f' pos="{rear_wl_pos}" quat="{rear_wl_quat}"'
              f' rgba="{WHEEL_COLOR}" contype="0" conaffinity="0" mass="0.001"/>')
            a(f'{ind}    <geom name="col_wL0_rear" type="cylinder"'
              f' size="{WHEEL_RADIUS} {WHEEL_THICK}" euler="0 1.5708 0"'
              f' mass="{WHEEL_MASS}" rgba="0 0 0 0"'
              f' friction="1.5 0.01 0.001"/>')
            a(f'{ind}  </body>')

            a(f'{ind}  <body name="seg0_wR_rear" pos="{rear_wr_bpos[0]} {rear_wr_bpos[1]} {rear_wr_bpos[2]}">')
            a(f'{ind}    <joint name="wR0_rear" type="hinge" axis="1 0 0"'
              f' damping="0.001" armature="0.0001"/>')
            a(f'{ind}    <geom name="vis_wR0_rear" type="mesh" mesh="{rear_wr_mesh}"'
              f' pos="{rear_wr_pos}" quat="{rear_wr_quat}"'
              f' rgba="{WHEEL_COLOR}" contype="0" conaffinity="0" mass="0.001"/>')
            a(f'{ind}    <geom name="col_wR0_rear" type="cylinder"'
              f' size="{WHEEL_RADIUS} {WHEEL_THICK}" euler="0 1.5708 0"'
              f' mass="{WHEEL_MASS}" rgba="0 0 0 0"'
              f' friction="1.5 0.01 0.001"/>')
            a(f'{ind}  </body>')

    # Close nested body tags (innermost first)
    for i in range(NUM_SEGMENTS - 1, -1, -1):
        ind = "  " * (i + 2)
        a(f'{ind}</body>')

    a('  </worldbody>')
    a('')

    # ── Actuators (NO wheel motors — passive wheels) ──
    a('  <actuator>')
    for i in range(NUM_SEGMENTS - 1):
        a(f'    <position name="act_slide{i}_{i+1}" joint="slide{i}_{i+1}"'
          f' kp="200" forcerange="-10 10"/>')
    for i in range(NUM_SEGMENTS - 1):
        a(f'    <position name="act_yaw{i}_{i+1}" joint="yaw{i}_{i+1}"'
          f' kp="80" forcerange="-10 10"/>')
    a('  </actuator>')
    a('</mujoco>')

    return '\n'.join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run(mode='snake', record_video=False, sim_time=None,
        snake_amp=0.30, snake_freq=0.5, snake_waves=1.0,
        step_dur=None):

    peri_amp = SLIDE_RANGE
    if step_dur is None:
        step_dur = STEP_DURATION

    src_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    mesh_dir     = os.path.join(project_root, "meshes")
    bin_dir      = os.path.join(project_root, "bin", "v3")
    os.makedirs(bin_dir, exist_ok=True)

    xml_str  = build_xml(mesh_dir)
    xml_path = os.path.join(bin_dir, f"worm_v5_1_{mode}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # ── Actuator index mapping ──
    n_slides = NUM_SEGMENTS - 1   # 4
    n_yaws   = NUM_SEGMENTS - 1   # 4
    slide_ids = list(range(n_slides))
    yaw_ids   = list(range(n_slides, n_slides + n_yaws))

    # Body IDs
    seg_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"seg{i}")
               for i in range(NUM_SEGMENTS)]
    head_id = seg_ids[-1]
    tail_id = seg_ids[0]

    # Natural spacings for strip deformation (per adjacent pair)
    natural_spacings = [SEG_SPACING_01] + [SEG_SPACING_REST] * (NUM_SEGMENTS - 2)

    print(f"Worm V5.1 — Real Mesh + Passive Wheels")
    print(f"  bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}")
    print(f"  Mode: {mode.upper()}")
    if mode in ('snake', 'combined'):
        print(f"  snake: amp={snake_amp:.2f} rad ({math.degrees(snake_amp):.1f}°),"
              f" freq={snake_freq:.2f} Hz, waves={snake_waves:.1f}")
    if mode in ('worm', 'combined'):
        print(f"  peri:  gait={GAIT}, step={step_dur:.2f}s, slide=±{peri_amp*1000:.1f}mm")
    print(f"  wheels: PASSIVE (no motors)")

    # ── Settle ──
    settle_time  = 1.0
    settle_steps = int(settle_time / m.opt.timestep)
    print(f"  Settling {settle_time:.1f}s ...")
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    if np.any(np.isnan(d.qpos)):
        print("FATAL: NaN after settling")
        return

    hx0 = d.xpos[head_id, 0] * 1000
    hy0 = d.xpos[head_id, 1] * 1000
    hz0 = d.xpos[head_id, 2] * 1000
    print(f"  Post-settle: head=({hx0:.1f}, {hy0:.1f}, z={hz0:.1f}) mm")

    # ── Control ──
    def apply_control(d, t):
        d.ctrl[:] = 0

        # ── Peristaltic: slide-only wave ──
        if mode in ('worm', 'combined'):
            step_idx = int(t / step_dur) % NUM_SEGMENTS
            states = [GAIT[(seg + step_idx) % NUM_SEGMENTS]
                      for seg in range(NUM_SEGMENTS)]
            for j in range(n_slides):
                if states[j + 1] == 0:
                    d.ctrl[slide_ids[j]] = peri_amp
                else:
                    d.ctrl[slide_ids[j]] = -peri_amp

        # ── Snake: sinusoidal yaw wave (retrograde, head→tail) ──
        if mode in ('snake', 'combined'):
            for j in range(n_yaws):
                phase = (2.0 * math.pi * snake_freq * t
                         + 2.0 * math.pi * snake_waves * j / n_yaws)
                d.ctrl[yaw_ids[j]] = snake_amp * math.sin(phase)

    # ── Video setup ──
    renderer = None
    frames   = []
    fps      = 30
    if record_video:
        renderer = mujoco.Renderer(m, 720, 1280)

    # ── Main loop ──
    sim_total   = sim_time or 25.0
    total_steps = int(sim_total / m.opt.timestep)
    frame_iv    = max(1, int(1.0 / (fps * m.opt.timestep)))
    dt          = m.opt.timestep
    lookat_smooth = ((d.xpos[head_id] + d.xpos[tail_id]) / 2.0).copy()
    t0_wall     = time.time()
    print(f"  Simulating {sim_total:.0f}s ({total_steps} steps) ...")

    for step in range(total_steps):
        t = step * dt
        apply_control(d, t)
        mujoco.mj_step(m, d)

        if step % 5000 == 0 and step > 0:
            if np.any(np.isnan(d.qpos)):
                print(f"  NaN at t={t:.2f}s!")
                break

        if step % 5000 == 0 and step > 0:
            hx = d.xpos[head_id, 0] * 1000
            hy = d.xpos[head_id, 1] * 1000
            hz = d.xpos[head_id, 2] * 1000
            tx = d.xpos[tail_id, 0] * 1000
            ty = d.xpos[tail_id, 1] * 1000
            hdg = math.degrees(math.atan2(hx - tx, hy - ty))
            rate = step / (time.time() - t0_wall)
            print(f"    t={t:.1f}s  head=({hx:.1f},{hy:.1f},z={hz:.1f})mm  "
                  f"hdg={hdg:.1f}°  ncon={d.ncon}  [{rate:.0f} steps/s]")

        if record_video and step % frame_iv == 0 and renderer is not None:
            cam            = mujoco.MjvCamera()
            cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance   = 1.2
            cam.elevation  = -25
            cam.azimuth    = 45
            mid = (d.xpos[head_id] + d.xpos[tail_id]) / 2.0
            lookat_smooth += 0.03 * (mid - lookat_smooth)
            cam.lookat[:]  = lookat_smooth
            renderer.update_scene(d, cam)
            inject_strips(renderer.scene, d, seg_ids, natural_spacings)
            frames.append(renderer.render().copy())

    # ── Results ──
    elapsed = time.time() - t0_wall
    hx_f = d.xpos[head_id, 0] * 1000
    hy_f = d.xpos[head_id, 1] * 1000
    hz_f = d.xpos[head_id, 2] * 1000
    disp_y  = hy_f - hy0
    disp_xy = math.hypot(hx_f - hx0, hy_f - hy0)
    heading = math.degrees(math.atan2(
        d.xpos[head_id, 0] - d.xpos[tail_id, 0],
        d.xpos[head_id, 1] - d.xpos[tail_id, 1]))

    print(f"\n{'─'*60}")
    print(f"  Worm V5.1 — {mode.upper()} (real mesh, passive wheels)")
    print(f"  Head final:     ({hx_f:.1f}, {hy_f:.1f}, z={hz_f:.1f}) mm")
    print(f"  Forward (Y):    {disp_y:.1f} mm  ({disp_y/sim_total:.2f} mm/s)")
    print(f"  Displacement:   {disp_xy:.1f} mm  ({disp_xy/sim_total:.2f} mm/s)")
    print(f"  Heading:        {heading:.1f}°")
    print(f"  Wall time:      {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")
    print(f"{'─'*60}")

    # ── Save video ──
    if record_video and frames:
        try:
            import mediapy
            vid_dir  = os.path.join(project_root, "record", "v5_1", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"worm_v5_1_{mode}.mp4")
            mediapy.write_video(vid_path, frames, fps=fps)
            print(f"  Video: {vid_path}  ({len(frames)} frames)")
        except ImportError:
            print("  mediapy not found — video not saved")

    if renderer:
        renderer.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Worm V5.1 — real mesh + passive wheel")
    ap.add_argument("--mode", choices=["worm", "snake", "combined"], default="snake")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--snake-amp", type=float, default=0.30,
                    help="Snake yaw amplitude in rad (default 0.30 = 17°)")
    ap.add_argument("--snake-freq", type=float, default=0.5,
                    help="Snake yaw frequency in Hz")
    ap.add_argument("--snake-waves", type=float, default=1.0,
                    help="Snake spatial wavelength in body lengths")
    ap.add_argument("--step-dur", type=float, default=None,
                    help="Peristaltic gait step duration in seconds")
    args = ap.parse_args()

    run(mode=args.mode, record_video=args.video, sim_time=args.time,
        snake_amp=args.snake_amp, snake_freq=args.snake_freq,
        snake_waves=args.snake_waves, step_dur=args.step_dur)

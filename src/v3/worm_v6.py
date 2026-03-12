"""
Worm Robot V6 — Longworm2 CAD Mesh Simulation
===============================================
Based on SolidWorks longworm2.SLDASM URDF export.

7 body segments, 6 slide + 5 yaw = 11 actuated joints,
24 passive free-rolling wheels, full STL mesh visuals.

Architecture:
  base_link → [back1 → front2 → back2 → ... → front6 → back6]
  36 bodies, 35 joints (11 actuated + 24 passive wheels)
  Chain extends in -X, lateral ±Y, up +Z

Usage:
    python worm_v6.py --mode snake --video
    python worm_v6.py --mode worm --video
    python worm_v6.py --mode combined --video
"""

import mujoco
import numpy as np
import math
import os
import time
import argparse
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────────────────────────────────────
# Constants from URDF analysis
# ─────────────────────────────────────────────────────────────────────────────
NUM_SLIDES = 6          # back1-back6
NUM_YAWS   = 5          # front2-front6
NUM_ACTUATORS = NUM_SLIDES + NUM_YAWS  # 11

# Collision shape dimensions (from STL mesh analysis)
WHEEL_RADIUS = 0.016    # 16mm
WHEEL_THICK  = 0.0085   # 8.5mm half-thickness

# Body collision capsule (hidden inside mesh)
SEG_COL_RADIUS = 0.032  # 32mm
BASE_COL_HLEN  = 0.020  # base_link half-length along X
BACK_COL_HLEN  = 0.040  # back segments half-length
FRONT_COL_HLEN = 0.015  # front connector half-length
TAIL_COL_HLEN  = 0.015  # back6 half-length

# Initial body height: wheel z-offset (~70mm) + wheel radius (16mm)
BODY_Z = 0.086

# Actuator parameters (scaled for heavier robot)
SLIDE_KP = 800.0        # N/m position gain (heavier robot needs more force)
SLIDE_FORCE = 50.0      # N max force
YAW_KP   = 200.0        # Nm/rad position gain
YAW_FORCE = 20.0        # Nm max torque

# Wheel parameters
WHEEL_DAMPING = 0.002
WHEEL_ARMATURE = 0.0002
WHEEL_FRICTION = "1.5 0.01 0.001"

# Open-loop gait parameters
SLIDE_RANGE_VAL = 0.05  # 50mm slide range (from URDF limits)
YAW_RANGE_VAL   = 1.57  # ±90° yaw range
SNAKE_AMP   = 0.40      # yaw amplitude (rad)
SNAKE_FREQ  = 0.4       # Hz (slower for longer body)
SNAKE_WAVES = 1.5       # wavelengths across body
STEP_DURATION = 0.8     # seconds per peristaltic phase

# Body colors (from URDF)
COLOR_BASE  = "0.79 0.82 0.93 1"    # light blue-grey
COLOR_BACK  = "0.78 0.76 0.74 1"    # warm grey
COLOR_FRONT = "1.0 1.0 1.0 1"       # white
COLOR_WHEEL = "0.79 0.82 0.93 1"    # matching base
COLOR_FLOOR = "0.93 0.93 0.90 1"

# Steel strip visual — 8 strips per slide joint, 45° spacing
NUM_STRIPS      = 8
STRIP_CIRCLE_R  = 0.068    # 68mm — flush with body edge (mesh radius ~70mm)
STRIP_W         = 0.018    # 18mm wide band (narrower, more refined)
STRIP_T         = 0.002    # 2mm visual thickness (thinner, more realistic)
VIS_BOW_MIN     = 0.001    # 1mm rest bow (subtle at rest)
VIS_BOW_MAX     = 0.035    # 35mm max outward bow at full compression
ARC_SEGS        = 24       # segments per strip for very smooth parabolic arc
ARC_OVERLAP     = 1.12     # 12% overlap to hide seams without excess bulk
STRIP_RGBA      = np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32)  # black
STRIP_ANGLES    = [2.0 * math.pi * k / NUM_STRIPS for k in range(NUM_STRIPS)]


# ─────────────────────────────────────────────────────────────────────────────
# Parse URDF → chain structure
# ─────────────────────────────────────────────────────────────────────────────

def parse_urdf(urdf_path):
    """Parse longworm2 URDF and extract chain structure."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link in root.findall('link'):
        name = link.get('name')
        mass_el = link.find('.//mass')
        mass = float(mass_el.get('value')) if mass_el is not None else 0
        links[name] = {'mass': mass}

    joints = []
    for joint in root.findall('joint'):
        name = joint.get('name')
        jtype = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz').split()]
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        axis_el = joint.find('axis')
        axis = [float(v) for v in axis_el.get('xyz').split()] if axis_el is not None else [0, 0, 0]
        limit_el = joint.find('limit')
        lo = float(limit_el.get('lower')) if limit_el is not None else 0
        hi = float(limit_el.get('upper')) if limit_el is not None else 0

        joints.append({
            'name': name, 'type': jtype,
            'parent': parent, 'child': child,
            'origin': xyz, 'rpy': rpy,
            'axis': axis, 'lower': lo, 'upper': hi,
        })

    return links, joints


# ─────────────────────────────────────────────────────────────────────────────
# MJCF XML generation
# ─────────────────────────────────────────────────────────────────────────────

def build_xml(mesh_dir, urdf_path):
    """Generate MuJoCo XML from longworm2 URDF with STL meshes."""
    links, joints = parse_urdf(urdf_path)

    # Build parent→children map
    children = {}
    joint_map = {}
    for j in joints:
        parent = j['parent']
        if parent not in children:
            children[parent] = []
        children[parent].append(j)
        joint_map[j['child']] = j

    L = []
    a = L.append

    a('<?xml version="1.0"?>')
    a('<mujoco model="worm_v6_longworm2">')
    a(f'  <compiler meshdir="{mesh_dir}" angle="radian"/>')
    a('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    a('  <size memory="500M"/>')
    a('')
    a('  <visual>')
    a('    <global offwidth="1920" offheight="1080"/>')
    a('    <quality shadowsize="4096"/>')
    a('    <map znear="0.001"/>')
    a('  </visual>')
    a('')
    a('  <default>')
    a('    <geom condim="4"/>')
    a('  </default>')
    a('')

    # ── Assets ──
    a('  <asset>')
    a('    <texture name="grid" type="2d" builtin="checker" width="512" height="512"'
      ' rgb1="0.93 0.93 0.90" rgb2="0.82 0.82 0.80"/>')
    a('    <material name="grid_mat" texture="grid" texrepeat="20 20"'
      ' reflectance="0.1"/>')
    # STL meshes
    mesh_names = set()
    for link_name in links:
        if link_name != 'world':
            fname = f"longworm2/{link_name}.STL"
            if link_name not in mesh_names:
                a(f'    <mesh name="{link_name}" file="{fname}" '
                  f'smoothnormal="true"/>')
                mesh_names.add(link_name)
    a('  </asset>')
    a('')

    # ── Worldbody ──
    a('  <worldbody>')
    a('    <light pos="0 0 3.0" dir="0 0 -1" diffuse="0.9 0.9 0.9"'
      ' ambient="0.3 0.3 0.3"/>')
    a('    <light pos="1.0 -0.5 2.5" dir="-0.3 0.2 -1" diffuse="0.5 0.5 0.5"/>')
    a('    <light pos="-1.0 0.5 2.0" dir="0.3 -0.1 -1" diffuse="0.3 0.3 0.3"/>')
    a('    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat"'
      ' friction="1.0 0.005 0.001"/>')
    a('')

    # ── Recursive body tree generation ──
    def get_color(link_name):
        if link_name == 'base_link':
            return COLOR_BASE
        elif link_name.startswith('back'):
            return COLOR_BACK
        elif link_name.startswith('front'):
            return COLOR_FRONT
        else:
            return COLOR_WHEEL

    def get_collision(link_name, indent):
        """Generate collision geom for a link."""
        lines = []
        if link_name == 'base_link':
            lines.append(f'{indent}<geom name="col_{link_name}" type="capsule"'
                        f' fromto="{-BASE_COL_HLEN} 0 0  {BASE_COL_HLEN} 0 0"'
                        f' size="{SEG_COL_RADIUS}" mass="{links[link_name]["mass"]:.4f}"'
                        f' rgba="1 0 0 0" friction="0.3 0.005 0.001"/>')
        elif link_name.startswith('back'):
            hlen = TAIL_COL_HLEN if link_name == 'back6_Link' else BACK_COL_HLEN
            lines.append(f'{indent}<geom name="col_{link_name}" type="capsule"'
                        f' fromto="{-hlen} 0 0  {hlen} 0 0"'
                        f' size="{SEG_COL_RADIUS}" mass="{links[link_name]["mass"]:.4f}"'
                        f' rgba="1 0 0 0" friction="0.3 0.005 0.001"/>')
        elif link_name.startswith('front'):
            lines.append(f'{indent}<geom name="col_{link_name}" type="capsule"'
                        f' fromto="{-FRONT_COL_HLEN} 0 0  {FRONT_COL_HLEN} 0 0"'
                        f' size="{SEG_COL_RADIUS * 0.8}" mass="{links[link_name]["mass"]:.4f}"'
                        f' rgba="1 0 0 0" friction="0.3 0.005 0.001"/>')
        elif link_name.startswith('w'):
            # Wheel: cylinder with axis along Y (axle direction)
            lines.append(f'{indent}<geom name="col_{link_name}" type="cylinder"'
                        f' size="{WHEEL_RADIUS} {WHEEL_THICK}"'
                        f' euler="1.5708 0 0"'
                        f' mass="{links[link_name]["mass"]:.4f}"'
                        f' rgba="0 0 0 0"'
                        f' friction="{WHEEL_FRICTION}"/>')
        return '\n'.join(lines)

    def emit_body(link_name, joint_info, depth):
        """Recursively emit a body and its children."""
        indent = '  ' * depth
        xyz = joint_info['origin'] if joint_info else [0, 0, BODY_Z]
        pos_str = f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"

        a(f'{indent}<body name="{link_name}" pos="{pos_str}">')

        # Joint (if not root)
        if joint_info is None:
            # Root body
            a(f'{indent}  <freejoint name="root"/>')
        elif joint_info['type'] == 'prismatic':
            ax = joint_info['axis']
            lo, hi = joint_info['lower'], joint_info['upper']
            # Normalize: ensure all slides have consistent direction
            if ax[0] > 0:
                ax = [-ax[0], -ax[1], -ax[2]]
                lo, hi = -hi, -lo
            ax_str = f"{ax[0]:.5f} {ax[1]:.5f} {ax[2]:.5f}"
            jname = joint_info['name']
            a(f'{indent}  <joint name="{jname}" type="slide"'
              f' axis="{ax_str}" range="{lo:.4f} {hi:.4f}"'
              f' damping="10" stiffness="300"/>')
        elif joint_info['type'] == 'revolute':
            ax = joint_info['axis']
            ax_str = f"{ax[0]:.5f} {ax[1]:.5f} {ax[2]:.5f}"
            lo, hi = joint_info['lower'], joint_info['upper']
            jname = joint_info['name']
            if link_name.startswith('w'):
                # Passive wheel joint
                a(f'{indent}  <joint name="{jname}" type="hinge"'
                  f' axis="{ax_str}"'
                  f' damping="{WHEEL_DAMPING}" armature="{WHEEL_ARMATURE}"/>')
            else:
                # Actuated yaw joint
                a(f'{indent}  <joint name="{jname}" type="hinge"'
                  f' axis="{ax_str}" range="{lo:.4f} {hi:.4f}"'
                  f' damping="5"/>')

        # Visual mesh
        color = get_color(link_name)
        if link_name.startswith('w'):
            # Wheels: visual mesh only (collision cylinder above)
            a(f'{indent}  <geom name="vis_{link_name}" type="mesh"'
              f' mesh="{link_name}" rgba="{color}"'
              f' contype="0" conaffinity="0" mass="0.001"/>')
        else:
            # Body: visual mesh (no collision)
            a(f'{indent}  <geom name="vis_{link_name}" type="mesh"'
              f' mesh="{link_name}" rgba="{color}"'
              f' contype="0" conaffinity="0" mass="0.001"/>')

        # Collision shape
        col = get_collision(link_name, indent + '  ')
        if col:
            a(col)

        # Recurse into children
        if link_name in children:
            for child_joint in children[link_name]:
                child_link = child_joint['child']
                emit_body(child_link, child_joint, depth + 1)

        a(f'{indent}</body>')

    # Start from base_link
    emit_body('base_link', None, 2)

    a('  </worldbody>')
    a('')

    # ── Actuators ──
    a('  <actuator>')
    # Slide actuators (back1-back6)
    for i in range(1, NUM_SLIDES + 1):
        jname = f'back{i}'
        a(f'    <position name="act_{jname}" joint="{jname}"'
          f' kp="{SLIDE_KP}" forcerange="-{SLIDE_FORCE} {SLIDE_FORCE}"/>')
    # Yaw actuators (front2-front6)
    for i in range(2, NUM_YAWS + 2):
        jname = f'front{i}'
        a(f'    <position name="act_{jname}" joint="{jname}"'
          f' kp="{YAW_KP}" forcerange="-{YAW_FORCE} {YAW_FORCE}"/>')
    a('  </actuator>')
    a('</mujoco>')

    return '\n'.join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Steel strip deformation rendering (same V-shape as V5.1)
# ─────────────────────────────────────────────────────────────────────────────

def inject_strips(scene, d, slide_pairs, natural_spacings):
    """Inject steel strip visuals at each slide joint.

    Smooth parabolic arc: ARC_SEGS segments per strip, each following
    the local tangent of  r(t) = R + bow * 4t(1-t).
    Adjacent segments share endpoints → seamless curve.
    8 strips × 6 joints × 5 segs = 240 geoms.
    """
    BOX = mujoco.mjtGeom.mjGEOM_BOX

    for pair_idx, (bid_parent, bid_child) in enumerate(slide_pairs):
        p_par = d.xpos[bid_parent].copy()
        p_chi = d.xpos[bid_child].copy()
        r_par = d.xmat[bid_parent].reshape(3, 3)

        link = p_chi - p_par
        dist = np.linalg.norm(link)
        if dist < 0.005:
            continue
        e_ax = link / dist

        # Dynamic bow based on compression ratio
        nat_len = natural_spacings[pair_idx]
        comp = max(0.0, 1.0 - dist / nat_len)
        vis_bow = VIS_BOW_MIN + (VIS_BOW_MAX - VIS_BOW_MIN) * min(1.0, comp * 8.0)

        # Axial span slightly shorter than full gap (leave plate edges visible)
        span = dist * 0.92
        half_span = span * 0.5

        # Body-frame radial directions
        body_y = r_par[:, 1]
        body_z = r_par[:, 2]
        mid = (p_par + p_chi) * 0.5

        # Precompute arc sample points along parabola: r(t) = R + bow*4t(1-t)
        # t goes from 0 (parent plate) to 1 (child plate)
        arc_t = [s / ARC_SEGS for s in range(ARC_SEGS + 1)]
        arc_r = [STRIP_CIRCLE_R + vis_bow * 4.0 * t * (1.0 - t) for t in arc_t]
        # Axial positions relative to midpoint
        arc_ax = [-half_span + span * t for t in arc_t]

        for angle in STRIP_ANGLES:
            ca, sa = math.cos(angle), math.sin(angle)
            e_r = ca * body_y + sa * body_z

            # Fixed tangent direction for all segments (no face twisting)
            e_tang = np.cross(e_ax, e_r)
            tang_n = np.linalg.norm(e_tang)
            if tang_n < 1e-6:
                continue
            e_tang /= tang_n

            # 3D sample points on the parabolic arc
            pts = [mid + arc_ax[i] * e_ax + arc_r[i] * e_r
                   for i in range(ARC_SEGS + 1)]

            for s in range(ARC_SEGS):
                if scene.ngeom >= scene.maxgeom:
                    return

                pa, pb = pts[s], pts[s + 1]
                seg_mid = (pa + pb) * 0.5
                dv = pb - pa
                seg_len = np.linalg.norm(dv)
                if seg_len < 1e-6:
                    continue
                z_dir = dv / seg_len

                # Derive radial from fixed tangent + local z_dir
                radial = np.cross(e_tang, z_dir)
                rn = np.linalg.norm(radial)
                if rn < 1e-6:
                    continue
                radial /= rn

                # Rotation: columns = local axes in world frame
                R = np.column_stack([e_tang, radial, z_dir])
                mat = R.flatten()

                # Overlap: extend segment length to cover seams
                size = np.array([STRIP_W / 2, STRIP_T / 2,
                                 seg_len * ARC_OVERLAP / 2])

                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, BOX, size, seg_mid, mat, STRIP_RGBA)
                geom.emission = 0.05
                geom.specular = 0.3
                geom.shininess = 0.2
                scene.ngeom += 1


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run(mode='snake', record_video=False, sim_time=None):
    src_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    mesh_dir     = os.path.join(project_root, "meshes")
    bin_dir      = os.path.join(project_root, "bin", "v3")
    os.makedirs(bin_dir, exist_ok=True)

    # URDF path
    urdf_path = os.path.join(project_root, "meshes", "longworm2", "longworm2.SLDASM.urdf")
    # Copy URDF if not present
    if not os.path.exists(urdf_path):
        import shutil
        src_urdf = os.path.join("D:/inovxio/3d/longworm2/longworm2.SLDASM/urdf",
                                "longworm2.SLDASM.urdf")
        if os.path.exists(src_urdf):
            shutil.copy2(src_urdf, urdf_path)
        else:
            print(f"ERROR: URDF not found at {src_urdf}")
            return

    xml_str  = build_xml(mesh_dir, urdf_path)
    xml_path = os.path.join(bin_dir, f"worm_v6_{mode}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # ── Identify actuated joints ──
    slide_act_ids = []
    yaw_act_ids = []
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith('act_back'):
            slide_act_ids.append(i)
        elif name.startswith('act_front'):
            yaw_act_ids.append(i)

    # ── Body IDs ──
    def get_bid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            print(f"WARNING: body '{name}' not found")
        return i

    head_id = get_bid('base_link')
    tail_id = get_bid('back6_Link')

    # All main segment IDs (for camera tracking)
    seg_ids = [get_bid('base_link')] + [get_bid(f'back{i}_Link') for i in range(1, 7)]

    # ── Slide joint pairs: (parent_plate, child_plate) for strip rendering ──
    # 8 steel strips per pair, connecting end plates at each prismatic joint
    slide_pairs = [
        (get_bid('base_link'),   get_bid('back1_Link')),    # slide 1
        (get_bid('front2_Link'), get_bid('back2_Link')),    # slide 2
        (get_bid('front3_Link'), get_bid('back3_Link')),    # slide 3
        (get_bid('front4_Link'), get_bid('back4_Link')),    # slide 4
        (get_bid('front5_Link'), get_bid('back5_Link')),    # slide 5
        (get_bid('front6_Link'), get_bid('back6_Link')),    # slide 6
    ]
    # Rest distances from URDF: base→back1=0.151m, frontN→backN=0.1175m
    strip_spacings = [0.151] + [0.1175] * 5

    print(f"Worm V6 — Longworm2 CAD Mesh")
    print(f"  bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}")
    print(f"  Mode: {mode.upper()}")
    print(f"  slides: {len(slide_act_ids)}, yaws: {len(yaw_act_ids)}")

    # ── Settle ──
    settle_time  = 2.0
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
    n_slides = len(slide_act_ids)
    n_yaws = len(yaw_act_ids)

    def apply_control(d, t):
        d.ctrl[:] = 0

        # ── Peristaltic: slide wave ──
        if mode in ('worm', 'combined'):
            wave_len = n_slides
            for j in range(n_slides):
                phase = 2.0 * math.pi * (t / STEP_DURATION - j / wave_len)
                # Sinusoidal slide: negative = contract (pull segments together)
                target = -SLIDE_RANGE_VAL * 0.5 * (1.0 + math.sin(phase))
                d.ctrl[slide_act_ids[j]] = target

        # ── Snake: sinusoidal yaw wave (head→tail) ──
        if mode in ('snake', 'combined'):
            for j in range(n_yaws):
                phase = (2.0 * math.pi * SNAKE_FREQ * t
                         + 2.0 * math.pi * SNAKE_WAVES * j / n_yaws)
                d.ctrl[yaw_act_ids[j]] = SNAKE_AMP * math.sin(phase)

    # ── Video setup ──
    renderer = None
    frames   = []
    fps      = 30
    if record_video:
        renderer = mujoco.Renderer(m, 1080, 1920)

    # ── Main loop ──
    sim_total   = sim_time or 30.0
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
            hx = d.xpos[head_id, 0] * 1000
            hy = d.xpos[head_id, 1] * 1000
            hz = d.xpos[head_id, 2] * 1000
            rate = step / (time.time() - t0_wall)
            print(f"    t={t:.1f}s  head=({hx:.1f},{hy:.1f},z={hz:.1f})mm"
                  f"  ncon={d.ncon}  [{rate:.0f} steps/s]")

        if record_video and step % frame_iv == 0 and renderer is not None:
            cam            = mujoco.MjvCamera()
            cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance   = 2.0
            cam.elevation  = -25
            cam.azimuth    = 135
            mid = (d.xpos[head_id] + d.xpos[tail_id]) / 2.0
            lookat_smooth += 0.03 * (mid - lookat_smooth)
            cam.lookat[:]  = lookat_smooth
            renderer.update_scene(d, cam)
            inject_strips(renderer.scene, d, slide_pairs, strip_spacings)
            frames.append(renderer.render().copy())

    # ── Results ──
    elapsed = time.time() - t0_wall
    hx_f = d.xpos[head_id, 0] * 1000
    hy_f = d.xpos[head_id, 1] * 1000
    hz_f = d.xpos[head_id, 2] * 1000
    # Forward is -X direction (chain extends in -X)
    disp_x = -(hx_f - hx0)  # positive = moved forward (-X)
    disp_total = math.hypot(hx_f - hx0, hy_f - hy0)

    print(f"\n{'─'*60}")
    print(f"  Worm V6 — {mode.upper()} (longworm2 CAD mesh)")
    print(f"  Head final:     ({hx_f:.1f}, {hy_f:.1f}, z={hz_f:.1f}) mm")
    print(f"  Forward (-X):   {disp_x:.1f} mm  ({disp_x/sim_total:.2f} mm/s)")
    print(f"  Displacement:   {disp_total:.1f} mm  ({disp_total/sim_total:.2f} mm/s)")
    print(f"  Wall time:      {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")
    print(f"{'─'*60}")

    # ── Save video ──
    if record_video and frames:
        try:
            import mediapy
            vid_dir  = os.path.join(project_root, "record", "v6", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"worm_v6_{mode}.mp4")
            mediapy.write_video(vid_path, frames, fps=fps)
            print(f"  Video: {vid_path}  ({len(frames)} frames)")
        except ImportError:
            print("  mediapy not found — video not saved")

    if renderer:
        renderer.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Worm V6 — Longworm2 simulation")
    ap.add_argument("--mode", choices=["snake", "worm", "combined"],
                    default="snake", help="Locomotion mode")
    ap.add_argument("--video", action="store_true", help="Record video")
    ap.add_argument("--time", type=float, default=None, help="Simulation time (s)")
    args = ap.parse_args()

    run(mode=args.mode, record_video=args.video, sim_time=args.time)

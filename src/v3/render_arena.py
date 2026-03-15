"""
Training Arena — 4K visualization of parallel RL training
===========================================================
Renders 20 worm robots in a cinematic arena, each at a different
"training stage" (random → early → mid → converged), with orbiting camera
and steel strip deformation visuals.

Usage:
    python render_arena.py                  # 4K (3840x2160), 25s
    python render_arena.py --preview        # 720p quick test
    python render_arena.py --duration 40    # longer video
"""

import mujoco
import numpy as np
import math
import os
import sys
import time
import copy
import subprocess
import shutil
import argparse
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from worm_v6 import (
    build_xml, NUM_SLIDES, NUM_YAWS,
    SLIDE_RANGE_VAL, SNAKE_AMP, SNAKE_FREQ, SNAKE_WAVES, STEP_DURATION,
    BODY_Z,
    STRIP_CIRCLE_R, VIS_BOW_MIN, VIS_BOW_MAX,
    ARC_OVERLAP, STRIP_W, STRIP_T, STRIP_RGBA,
)

# ─── Arena Layout ──────────────────────────────────────────────────
N_ROWS    = 4       # training stages: random, early, mid, converged
N_COLS    = 5       # robots per stage
SPACING_X = 1.5     # meters along X (forward axis)
SPACING_Y = 0.55    # meters lateral

# ─── Video Defaults ────────────────────────────────────────────────
WIDTH    = 3840
HEIGHT   = 2160
FPS      = 30
DURATION = 25.0

# ─── Strip Rendering ──────────────────────────────────────────────
ARENA_N_STRIPS  = 8
ARENA_ARC_SEGS  = 12
ARENA_STRIP_ANGLES = [2.0 * math.pi * k / ARENA_N_STRIPS
                      for k in range(ARENA_N_STRIPS)]


# ═══════════════════════════════════════════════════════════════════
# XML Generation
# ═══════════════════════════════════════════════════════════════════

def prefix_names(elem, prefix):
    """Recursively prefix 'name' attributes in a body subtree."""
    if elem.get('name') is not None:
        elem.set('name', prefix + elem.get('name'))
    for child in elem:
        prefix_names(child, prefix)


def create_arena_xml(mesh_dir, urdf_path, width, height):
    """Clone single-robot XML into a multi-robot training arena."""
    single_xml = build_xml(mesh_dir, urdf_path)
    root = ET.fromstring(single_xml)

    # ── Memory ──
    size_el = root.find('size')
    if size_el is not None:
        size_el.set('memory', '2G')

    # ── Visual quality ──
    visual = root.find('visual')
    if visual is not None:
        g = visual.find('global')
        if g is not None:
            g.set('offwidth', str(width))
            g.set('offheight', str(height))
        q = visual.find('quality')
        if q is not None:
            q.set('shadowsize', '8192')

    # ── Dark floor texture ──
    asset = root.find('asset')
    for tex in asset.findall('texture'):
        if tex.get('name') == 'grid':
            tex.set('rgb1', '0.22 0.23 0.26')
            tex.set('rgb2', '0.17 0.18 0.20')
    for mat in asset.findall('material'):
        if mat.get('name') == 'grid_mat':
            mat.set('reflectance', '0.12')

    # ── Worldbody modifications ──
    worldbody = root.find('worldbody')

    # Larger floor
    floor = worldbody.find('.//geom[@name="floor"]')
    if floor is not None:
        floor.set('size', '20 20 0.1')
        floor.set('contype', '1')
        floor.set('conaffinity', '2')

    # Replace lighting with cinematic 3-point setup
    for light in worldbody.findall('light'):
        worldbody.remove(light)

    def add_light(pos, direc, diffuse, ambient=None, shadow=False):
        el = ET.SubElement(worldbody, 'light')
        el.set('pos', pos)
        el.set('dir', direc)
        el.set('diffuse', diffuse)
        if ambient:
            el.set('ambient', ambient)
        if shadow:
            el.set('castshadow', 'true')

    add_light('6 -6 10',  '-0.3 0.3 -1', '0.85 0.82 0.78',
              ambient='0.22 0.22 0.25', shadow=True)
    add_light('-5 5 7',   '0.3 -0.3 -1', '0.3 0.33 0.38')
    add_light('0 -4 5',   '0 0.3 -1',    '0.35 0.35 0.4')

    # ── Clone robot N times ──
    robot_body = worldbody.find('.//body[@name="base_link"]')
    actuator_el = root.find('actuator')
    orig_acts = list(actuator_el)

    worldbody.remove(robot_body)
    for act in orig_acts:
        actuator_el.remove(act)

    # Scatter robots randomly (with minimum separation)
    n_robots = N_ROWS * N_COLS
    rng_layout = np.random.RandomState(123)
    arena_rx, arena_ry = 4.0, 1.8  # half-extents of scatter area
    min_sep = 0.55  # minimum distance between robots
    positions = []
    for _ in range(n_robots):
        for _try in range(200):
            x = rng_layout.uniform(-arena_rx, arena_rx)
            y = rng_layout.uniform(-arena_ry, arena_ry)
            if all(math.hypot(x - px, y - py) >= min_sep for px, py in positions):
                positions.append((x, y))
                break

    for idx in range(n_robots):
        row, col = idx // N_COLS, idx % N_COLS
        prefix = f"r{idx}_"
        x, y = positions[idx]

        body = copy.deepcopy(robot_body)
        prefix_names(body, prefix)
        body.set('pos', f'{x:.4f} {y:.4f} {BODY_Z:.4f}')

        # Collision groups: robots collide with floor only
        for geom in body.iter('geom'):
            gname = geom.get('name', '')
            if gname.startswith(prefix + 'col_'):
                geom.set('contype', '2')
                geom.set('conaffinity', '1')

        worldbody.append(body)

        for act in orig_acts:
            a = copy.deepcopy(act)
            a.set('name', prefix + a.get('name'))
            a.set('joint', prefix + a.get('joint'))
            actuator_el.append(a)

    return ET.tostring(root, encoding='unicode', xml_declaration=True)


# ═══════════════════════════════════════════════════════════════════
# Robot Info & Control
# ═══════════════════════════════════════════════════════════════════

def get_robot_info(model, n_robots):
    """Look up actuator IDs, body IDs, and pre-generate gait params."""
    rng = np.random.RandomState(42)
    robots = []

    for i in range(n_robots):
        prefix = f"r{i}_"
        row, col = i // N_COLS, i % N_COLS

        # Actuator IDs
        slide_ids, yaw_ids = [], []
        for j in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
            if name and name.startswith(prefix + 'act_back'):
                slide_ids.append(j)
            elif name and name.startswith(prefix + 'act_front'):
                yaw_ids.append(j)

        # Body IDs for strip rendering
        def bid(name):
            return mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + name)

        slide_pairs = [
            (bid('base_link'),   bid('back1_Link')),
            (bid('front2_Link'), bid('back2_Link')),
            (bid('front3_Link'), bid('back3_Link')),
            (bid('front4_Link'), bid('back4_Link')),
            (bid('front5_Link'), bid('back5_Link')),
            (bid('front6_Link'), bid('back6_Link')),
        ]

        # Pre-generate random gait params for "random" stage
        n_act = len(slide_ids) + len(yaw_ids)
        rand_freqs  = rng.uniform(0.3, 2.0, n_act)
        rand_phases = rng.uniform(0, 2 * math.pi, n_act)

        robots.append({
            'slide_ids': slide_ids,
            'yaw_ids': yaw_ids,
            'slide_pairs': slide_pairs,
            'spacings': [0.151] + [0.1175] * 5,
            'row': row,
            'col': col,
            'rand_freqs': rand_freqs,
            'rand_phases': rand_phases,
        })

    return robots


def apply_control(data, robots, t):
    """Apply diverse gaits — each row is a training stage."""
    for r in robots:
        row, col = r['row'], r['col']
        phase_off = col * 0.5
        n_s = len(r['slide_ids'])
        n_y = len(r['yaw_ids'])

        if row == 0:
            # ── RANDOM: uncorrelated oscillations ──
            for j, aid in enumerate(r['slide_ids']):
                data.ctrl[aid] = SLIDE_RANGE_VAL * 0.25 * math.sin(
                    2 * math.pi * r['rand_freqs'][j] * t + r['rand_phases'][j])
            for j, aid in enumerate(r['yaw_ids']):
                k = n_s + j
                data.ctrl[aid] = 0.15 * math.sin(
                    2 * math.pi * r['rand_freqs'][k] * t + r['rand_phases'][k])

        elif row == 1:
            # ── EARLY: emerging wave, sloppy ──
            for j, aid in enumerate(r['slide_ids']):
                phase = 2 * math.pi * (t / (STEP_DURATION * 2) - j / n_s) + phase_off
                noise = 0.15 * math.sin(7.3 * t + j * 4.1 + col)
                data.ctrl[aid] = (-SLIDE_RANGE_VAL * 0.3
                                  * (1 + math.sin(phase))
                                  + noise * SLIDE_RANGE_VAL)
            for j, aid in enumerate(r['yaw_ids']):
                phase = (2 * math.pi * SNAKE_FREQ * 0.5 * t
                         + j * 1.0 + phase_off)
                data.ctrl[aid] = (SNAKE_AMP * 0.3 * math.sin(phase)
                                  + 0.08 * math.sin(5 * t + col * 2))

        elif row == 2:
            # ── MID: decent combined gait ──
            for j, aid in enumerate(r['slide_ids']):
                phase = (2 * math.pi
                         * (t / (STEP_DURATION * 1.2) - j / n_s)
                         + phase_off)
                data.ctrl[aid] = -SLIDE_RANGE_VAL * 0.4 * (1 + math.sin(phase))
            for j, aid in enumerate(r['yaw_ids']):
                phase = (2 * math.pi * SNAKE_FREQ * 0.8 * t
                         + 2 * math.pi * SNAKE_WAVES * j / n_y
                         + phase_off)
                data.ctrl[aid] = SNAKE_AMP * 0.5 * math.sin(phase)

        elif row == 3:
            # ── CONVERGED: clean combined gait ──
            for j, aid in enumerate(r['slide_ids']):
                phase = (2 * math.pi
                         * (t / STEP_DURATION - j / n_s)
                         + phase_off)
                data.ctrl[aid] = -SLIDE_RANGE_VAL * 0.5 * (1 + math.sin(phase))
            for j, aid in enumerate(r['yaw_ids']):
                phase = (2 * math.pi * SNAKE_FREQ * t
                         + 2 * math.pi * SNAKE_WAVES * j / n_y
                         + phase_off)
                data.ctrl[aid] = SNAKE_AMP * math.sin(phase)


# ═══════════════════════════════════════════════════════════════════
# Strip Rendering (multi-robot)
# ═══════════════════════════════════════════════════════════════════

def inject_arena_strips(scene, data, robots):
    """Render steel strip deformation for all robots (reduced fidelity)."""
    BOX = mujoco.mjtGeom.mjGEOM_BOX

    for robot in robots:
        for pair_idx, (bid_p, bid_c) in enumerate(robot['slide_pairs']):
            if bid_p < 0 or bid_c < 0:
                continue

            p_par = data.xpos[bid_p].copy()
            p_chi = data.xpos[bid_c].copy()
            r_par = data.xmat[bid_p].reshape(3, 3)

            link = p_chi - p_par
            dist = np.linalg.norm(link)
            if dist < 0.005:
                continue
            e_ax = link / dist

            nat_len = robot['spacings'][pair_idx]
            comp = max(0.0, 1.0 - dist / nat_len)
            vis_bow = VIS_BOW_MIN + (VIS_BOW_MAX - VIS_BOW_MIN) * min(1.0, comp * 8.0)

            span = dist * 0.92
            half_span = span * 0.5
            body_y = r_par[:, 1]
            body_z = r_par[:, 2]
            mid = (p_par + p_chi) * 0.5

            arc_t  = [s / ARENA_ARC_SEGS for s in range(ARENA_ARC_SEGS + 1)]
            arc_r  = [STRIP_CIRCLE_R + vis_bow * 4.0 * t * (1.0 - t) for t in arc_t]
            arc_ax = [-half_span + span * t for t in arc_t]

            for angle in ARENA_STRIP_ANGLES:
                ca, sa = math.cos(angle), math.sin(angle)
                e_r = ca * body_y + sa * body_z
                e_tang = np.cross(e_ax, e_r)
                tang_n = np.linalg.norm(e_tang)
                if tang_n < 1e-6:
                    continue
                e_tang /= tang_n

                pts = [mid + arc_ax[i] * e_ax + arc_r[i] * e_r
                       for i in range(ARENA_ARC_SEGS + 1)]

                for s in range(ARENA_ARC_SEGS):
                    if scene.ngeom >= scene.maxgeom:
                        return

                    pa, pb = pts[s], pts[s + 1]
                    seg_mid = (pa + pb) * 0.5
                    dv = pb - pa
                    seg_len = np.linalg.norm(dv)
                    if seg_len < 1e-6:
                        continue
                    z_dir = dv / seg_len

                    radial = np.cross(e_tang, z_dir)
                    rn = np.linalg.norm(radial)
                    if rn < 1e-6:
                        continue
                    radial /= rn

                    R   = np.column_stack([e_tang, radial, z_dir])
                    mat = R.flatten()
                    size = np.array([STRIP_W / 2, STRIP_T / 2,
                                     seg_len * ARC_OVERLAP / 2])

                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, BOX, size, seg_mid, mat, STRIP_RGBA)
                    geom.emission  = 0.05
                    geom.specular  = 0.3
                    geom.shininess = 0.2
                    scene.ngeom += 1


# ═══════════════════════════════════════════════════════════════════
# Camera & Rendering
# ═══════════════════════════════════════════════════════════════════

def ease_in_out(t):
    """Smoothstep: t ∈ [0,1] → [0,1]."""
    return t * t * (3.0 - 2.0 * t)


def render_arena(output_path=None, width=WIDTH, height=HEIGHT, duration=DURATION):
    """Build arena, simulate, render, save video."""
    mesh_dir  = os.path.join(PROJECT_ROOT, "meshes")
    urdf_path = os.path.join(mesh_dir, "longworm2", "longworm2.SLDASM.urdf")

    # Copy URDF if needed
    if not os.path.exists(urdf_path):
        src = os.path.join("D:/inovxio/3d/longworm2/longworm2.SLDASM/urdf",
                           "longworm2.SLDASM.urdf")
        if os.path.exists(src):
            os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
            import shutil as sh
            sh.copy2(src, urdf_path)

    n_robots = N_ROWS * N_COLS
    total_frames = int(duration * FPS)
    print(f"Training Arena — {n_robots} worm robots ({N_ROWS} stages × {N_COLS})")
    print(f"  Resolution: {width}×{height}  FPS: {FPS}  Duration: {duration}s")
    print(f"  Frames: {total_frames}")

    # ── Build XML ──
    print("  Building arena XML...")
    xml_str = create_arena_xml(mesh_dir, urdf_path, width, height)

    bin_dir = os.path.join(PROJECT_ROOT, "bin", "v3")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, "arena.xml")
    with open(xml_path, 'w') as f:
        f.write(xml_str)

    # ── Load model ──
    print("  Loading model...")
    model = mujoco.MjModel.from_xml_string(xml_str)
    data  = mujoco.MjData(model)
    print(f"  bodies={model.nbody}  DOF={model.nv}  actuators={model.nu}")

    robots = get_robot_info(model, n_robots)

    # ── Settle ──
    print("  Settling 2s...")
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)):
        print("FATAL: NaN after settle!")
        return
    print("  Settle OK")

    # ── Renderer ──
    try:
        renderer = mujoco.Renderer(model, height=height, width=width,
                                   max_geom=25000)
    except TypeError:
        renderer = mujoco.Renderer(model, height=height, width=width)

    # ── Output path ──
    if output_path is None:
        vid_dir = os.path.join(PROJECT_ROOT, "record", "v6", "videos")
        os.makedirs(vid_dir, exist_ok=True)
        tag = "4k" if width >= 3840 else f"{width}x{height}"
        output_path = os.path.join(vid_dir, f"training_arena_{tag}.mp4")

    # ── Video writer (streaming — avoids holding all frames in RAM) ──
    ffmpeg_bin = shutil.which('ffmpeg')
    writer = None
    frames_buf = None

    if ffmpeg_bin:
        print(f"  Video encoder: ffmpeg → H.264 CRF 18")
        writer = subprocess.Popen([
            ffmpeg_bin, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(FPS),
            '-i', '-',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path,
        ], stdin=subprocess.PIPE)
    else:
        print("  Video encoder: mediapy (buffered)")
        frames_buf = []

    # ── Camera params ──
    arena_cx = 0.0
    arena_cy = 0.0
    arena_cz = BODY_Z * 0.3

    # ── Main loop ──
    total_steps    = int(duration / model.opt.timestep)
    frame_interval = max(1, int(1.0 / (FPS * model.opt.timestep)))
    frame_count    = 0
    t0 = time.time()

    print(f"  Rendering...")
    for step in range(total_steps):
        t = step * model.opt.timestep
        apply_control(data, robots, t)
        mujoco.mj_step(model, data)

        if step % frame_interval != 0:
            continue

        progress = t / duration

        # Camera orbit
        cam = mujoco.MjvCamera()
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth   = 140 + 160 * ease_in_out(progress)
        cam.elevation = -32 + 15 * ease_in_out(progress)
        cam.distance  = 5.5 - 1.0 * ease_in_out(progress)
        cam.lookat[:] = [arena_cx, arena_cy, arena_cz]

        renderer.update_scene(data, cam)
        inject_arena_strips(renderer.scene, data, robots)
        frame = renderer.render()

        if writer:
            writer.stdin.write(frame.tobytes())
        elif frames_buf is not None:
            frames_buf.append(frame.copy())

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / frame_count * (total_frames - frame_count)
            print(f"    {frame_count}/{total_frames} frames  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    renderer.close()

    # ── Finalize video ──
    if writer:
        writer.stdin.close()
        writer.wait()
        print(f"  Saved: {output_path}")
    elif frames_buf:
        try:
            import mediapy
            mediapy.write_video(output_path, frames_buf, fps=FPS)
            print(f"  Saved: {output_path}")
        except ImportError:
            print("  ERROR: No video encoder available (need ffmpeg or mediapy)")
            return

    total_time = time.time() - t0
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {frame_count} frames, {frame_count/FPS:.1f}s video, "
          f"{file_size:.1f} MB, rendered in {total_time:.1f}s")


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Training Arena — multi-robot visualization")
    ap.add_argument("--width",    type=int,   default=WIDTH)
    ap.add_argument("--height",   type=int,   default=HEIGHT)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--output",   type=str,   default=None)
    ap.add_argument("--preview",  action="store_true",
                    help="Quick 720p preview")
    args = ap.parse_args()

    if args.preview:
        args.width  = 1280
        args.height = 720

    render_arena(
        output_path=args.output,
        width=args.width,
        height=args.height,
        duration=args.duration,
    )

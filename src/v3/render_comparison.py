"""
Gait Comparison Race — Side-by-side speed comparison
=====================================================
5 robots race with different gaits. CMA-ES optimized gait highlighted.

Usage:
    python render_comparison.py
    python render_comparison.py --preview
"""

import mujoco
import numpy as np
import math
import os
import sys
import time
import copy
import json
import subprocess
import shutil
import argparse
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from worm_v6 import (
    build_xml, inject_strips,
    NUM_SLIDES, NUM_YAWS,
    SLIDE_RANGE_VAL, YAW_RANGE_VAL, BODY_Z,
    SNAKE_AMP, SNAKE_FREQ, SNAKE_WAVES, STEP_DURATION,
    STRIP_CIRCLE_R, VIS_BOW_MIN, VIS_BOW_MAX,
    ARC_OVERLAP, STRIP_W, STRIP_T, STRIP_RGBA,
    NUM_STRIPS, STRIP_ANGLES, ARC_SEGS,
)

# ─── Layout ────────────────────────────────────────────────────────
N_ROBOTS  = 3
LANE_Y    = 0.50   # lateral spacing between lanes

# ─── Video ─────────────────────────────────────────────────────────
WIDTH    = 3840
HEIGHT   = 2160
FPS      = 30
DURATION = 15.0

# ─── Gait Definitions ──────────────────────────────────────────────
# Each gait: (name, label_short, color_rgba, control_fn_factory)

def load_all_cmaes_params():
    """Load CMA-ES optimized params for all 3 gait modes."""
    results = {}
    for mode in ["serpentine", "peristaltic", "full"]:
        path = os.path.join(PROJECT_ROOT, "runs", f"cmaes_speed_{mode}", "best_gait.json")
        with open(path) as f:
            d = json.load(f)
            results[mode] = {
                'params': np.array(d["best_params"]),
                'speed': d["best_speed_mm_s"],
            }
    return results


def prefix_names(elem, prefix):
    if elem.get('name') is not None:
        elem.set('name', prefix + elem.get('name'))
    for child in elem:
        prefix_names(child, prefix)


def build_race_xml(mesh_dir, urdf_path, width, height):
    """5 robots in parallel lanes."""
    single_xml = build_xml(mesh_dir, urdf_path)
    root = ET.fromstring(single_xml)

    # Memory
    size_el = root.find('size')
    if size_el is not None:
        size_el.set('memory', '1G')

    # Visual
    visual = root.find('visual')
    if visual is not None:
        g = visual.find('global')
        if g is not None:
            g.set('offwidth', str(width))
            g.set('offheight', str(height))
        q = visual.find('quality')
        if q is not None:
            q.set('shadowsize', '8192')

    # Floor
    asset = root.find('asset')
    for tex in asset.findall('texture'):
        if tex.get('name') == 'grid':
            tex.set('rgb1', '0.22 0.23 0.26')
            tex.set('rgb2', '0.17 0.18 0.20')
    for mat in asset.findall('material'):
        if mat.get('name') == 'grid_mat':
            mat.set('reflectance', '0.12')

    worldbody = root.find('worldbody')
    floor = worldbody.find('.//geom[@name="floor"]')
    if floor is not None:
        floor.set('size', '30 10 0.1')
        floor.set('contype', '1')
        floor.set('conaffinity', '2')

    # Lighting
    for light in worldbody.findall('light'):
        worldbody.remove(light)
    def add_light(pos, d, diff, amb=None, shadow=False):
        el = ET.SubElement(worldbody, 'light')
        el.set('pos', pos); el.set('dir', d); el.set('diffuse', diff)
        if amb: el.set('ambient', amb)
        if shadow: el.set('castshadow', 'true')
    add_light('5 -5 10', '-0.2 0.3 -1', '0.85 0.82 0.78',
              amb='0.25 0.25 0.28', shadow=True)
    add_light('-5 5 7', '0.3 -0.3 -1', '0.3 0.33 0.38')

    # Clone robots into lanes
    robot_body = worldbody.find('.//body[@name="base_link"]')
    actuator_el = root.find('actuator')
    orig_acts = list(actuator_el)
    worldbody.remove(robot_body)
    for act in orig_acts:
        actuator_el.remove(act)

    y_center = (N_ROBOTS - 1) * LANE_Y / 2
    for i in range(N_ROBOTS):
        prefix = f"r{i}_"
        y = i * LANE_Y - y_center
        body = copy.deepcopy(robot_body)
        prefix_names(body, prefix)
        body.set('pos', f'0.0000 {y:.4f} {BODY_Z:.4f}')
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


def get_robot_info(model):
    robots = []
    for i in range(N_ROBOTS):
        prefix = f"r{i}_"
        slide_ids, yaw_ids = [], []
        for j in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
            if name and name.startswith(prefix + 'act_back'):
                slide_ids.append(j)
            elif name and name.startswith(prefix + 'act_front'):
                yaw_ids.append(j)
        def bid(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + n)
        slide_pairs = [
            (bid('base_link'), bid('back1_Link')),
            (bid('front2_Link'), bid('back2_Link')),
            (bid('front3_Link'), bid('back3_Link')),
            (bid('front4_Link'), bid('back4_Link')),
            (bid('front5_Link'), bid('back5_Link')),
            (bid('front6_Link'), bid('back6_Link')),
        ]
        head_id = bid('base_link')
        tail_id = bid('back6_Link')
        robots.append({
            'slide_ids': slide_ids, 'yaw_ids': yaw_ids,
            'slide_pairs': slide_pairs,
            'spacings': [0.151] + [0.1175] * 5,
            'head_id': head_id, 'tail_id': tail_id,
        })
    return robots


def apply_gaits(data, robots, t):
    """Apply CMA-ES optimized gaits: serpentine, peristaltic, full combined."""
    cma_all = apply_gaits.__cache  # cached params dict
    modes = ["serpentine", "peristaltic", "full"]

    for i, r in enumerate(robots):
        n_s = len(r['slide_ids'])
        n_y = len(r['yaw_ids'])
        p = cma_all[modes[i]]['params']

        # Slide wave (zeroed for serpentine by CMA-ES constraint)
        slide_amp = p[0] if modes[i] != 'serpentine' else 0.0
        for j in range(n_s):
            if modes[i] == 'peristaltic':
                # Retrograde wave: compression travels head→tail (front→back)
                # +j/n_s shifts later joints ahead in phase so peak hits j=0 last
                phase = 2*math.pi*(t*1.0 + j/n_s)
                data.ctrl[r['slide_ids'][j]] = -SLIDE_RANGE_VAL * math.sin(phase)
            else:
                phase = (2*math.pi*(t*p[1] - p[2]*j/n_s) + p[6+j])
                data.ctrl[r['slide_ids'][j]] = -slide_amp * (1+math.sin(phase))

        # Yaw wave (zeroed for peristaltic by CMA-ES constraint)
        yaw_amp = p[3] if modes[i] != 'peristaltic' else 0.0
        for j in range(n_y):
            phase = (2*math.pi*p[4]*t
                     + 2*math.pi*p[5]*j/n_y
                     + p[13]*2*math.pi*t*p[1])
            data.ctrl[r['yaw_ids'][j]] = yaw_amp * math.sin(phase)


def inject_race_strips(scene, data, robots):
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    for robot in robots:
        for pair_idx, (bid_p, bid_c) in enumerate(robot['slide_pairs']):
            if bid_p < 0 or bid_c < 0: continue
            p_par = data.xpos[bid_p].copy()
            p_chi = data.xpos[bid_c].copy()
            r_par = data.xmat[bid_p].reshape(3, 3)
            link = p_chi - p_par
            dist = np.linalg.norm(link)
            if dist < 0.005: continue
            e_ax = link / dist
            nat_len = robot['spacings'][pair_idx]
            comp = max(0.0, 1.0 - dist / nat_len)
            vis_bow = VIS_BOW_MIN + (VIS_BOW_MAX - VIS_BOW_MIN) * min(1.0, comp * 8.0)
            span = dist * 0.92
            half_span = span * 0.5
            body_y = r_par[:, 1]
            body_z = r_par[:, 2]
            mid = (p_par + p_chi) * 0.5
            arc_t = [s / ARC_SEGS for s in range(ARC_SEGS + 1)]
            arc_r = [STRIP_CIRCLE_R + vis_bow * 4.0 * tt * (1.0 - tt) for tt in arc_t]
            arc_ax = [-half_span + span * tt for tt in arc_t]
            for angle in STRIP_ANGLES:
                ca, sa = math.cos(angle), math.sin(angle)
                e_r = ca * body_y + sa * body_z
                e_tang = np.cross(e_ax, e_r)
                tang_n = np.linalg.norm(e_tang)
                if tang_n < 1e-6: continue
                e_tang /= tang_n
                pts = [mid + arc_ax[ii] * e_ax + arc_r[ii] * e_r for ii in range(ARC_SEGS + 1)]
                for s in range(ARC_SEGS):
                    if scene.ngeom >= scene.maxgeom: return
                    pa, pb = pts[s], pts[s + 1]
                    seg_mid = (pa + pb) * 0.5
                    dv = pb - pa
                    seg_len = np.linalg.norm(dv)
                    if seg_len < 1e-6: continue
                    z_dir = dv / seg_len
                    radial = np.cross(e_tang, z_dir)
                    rn = np.linalg.norm(radial)
                    if rn < 1e-6: continue
                    radial /= rn
                    R = np.column_stack([e_tang, radial, z_dir])
                    mat = R.flatten()
                    size = np.array([STRIP_W/2, STRIP_T/2, seg_len*ARC_OVERLAP/2])
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, BOX, size, seg_mid, mat, STRIP_RGBA)
                    geom.emission = 0.05
                    geom.specular = 0.3
                    geom.shininess = 0.2
                    scene.ngeom += 1


def add_text_overlay(frame, robots, data, t):
    """Paper-style text overlay — PIL-based for crisp fonts at any resolution."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    h, w = frame.shape[:2]
    sf = w / 1920.0  # scale relative to 1080p base

    def load_font(size_base, bold=False):
        size = max(10, int(size_base * sf))
        candidates = (
            ["C:/Windows/Fonts/arialbd.ttf",
             "C:/Windows/Fonts/calibrib.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
            if bold else
            ["C:/Windows/Fonts/arial.ttf",
             "C:/Windows/Fonts/calibri.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        )
        for p in candidates:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    cma = apply_gaits.__cache
    GAITS = [
        ("serpentine",  "Serpentine",    (150, 190, 255), (190, 215, 255)),
        ("peristaltic", "Peristaltic",   (110, 215, 150), (150, 235, 185)),
        ("full",        "Full Combined", (255, 210, 40),  (255, 240, 110)),
    ]

    # Build semi-transparent panel on RGBA copy
    base    = Image.fromarray(frame).convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ov      = ImageDraw.Draw(overlay)

    pad     = int(22 * sf)
    row_h   = int(72 * sf)
    panel_x = pad
    panel_y = int(h * 0.26)
    panel_w = int(430 * sf)
    panel_h = row_h * 3 + int(20 * sf)

    box = [panel_x - int(14*sf), panel_y - int(10*sf),
           panel_x + panel_w,    panel_y + panel_h]
    try:
        ov.rounded_rectangle(box, radius=int(8*sf), fill=(6, 6, 10, 175))
    except AttributeError:
        ov.rectangle(box, fill=(6, 6, 10, 175))

    img  = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    f_title = load_font(28, bold=True)
    f_sub   = load_font(15, bold=False)
    f_label = load_font(20, bold=True)
    f_speed = load_font(22, bold=True)
    f_small = load_font(14, bold=False)
    f_timer = load_font(18, bold=False)

    # ── Title (top, center-left) ──────────────────────────────
    tx, ty = int(w * 0.28), pad
    draw.text((tx, ty),            "CMA-ES Gait Optimization",
              font=f_title, fill=(255, 255, 255))
    draw.text((tx, ty + int(38*sf)),
              "Worm Robot  \u00b7  1 Hz Servo Limit  \u00b7  MuJoCo Simulation",
              font=f_sub, fill=(160, 165, 175))

    # ── Timer (top-right) ────────────────────────────────────
    ts = f"t = {t:.1f} s"
    try:
        bb = draw.textbbox((0, 0), ts, font=f_timer)
        tw = bb[2] - bb[0]
    except AttributeError:
        tw = int(120 * sf)
    draw.text((w - tw - pad, pad + int(6*sf)), ts,
              font=f_timer, fill=(170, 172, 178))

    # ── Lane legend (left panel) ──────────────────────────────
    stripe_w = int(4 * sf)
    for i, (key, name, col, hi) in enumerate(GAITS):
        rx = panel_x
        ry = panel_y + i * row_h + int(10 * sf)
        draw.rectangle([rx, ry, rx + stripe_w, ry + row_h - int(14*sf)], fill=col)
        tx2 = rx + stripe_w + int(12*sf)
        draw.text((tx2, ry + int(2*sf)),  name,                              font=f_label, fill=col)
        draw.text((tx2, ry + int(30*sf)), f"{cma[key]['speed']:.0f} mm/s",   font=f_speed, fill=hi)
        if i == 2:
            draw.text((tx2 + int(185*sf), ry + int(34*sf)), "\u2605 FASTEST",
                      font=f_small, fill=(255, 210, 40))

    # ── Scale bar (bottom-left, ~100 mm) ──────────────────────
    ppm    = w / (2.0 * math.tan(math.radians(22.5)) * 2.5)
    bar_px = int(ppm * 0.100)
    bx, by = pad, h - int(55 * sf)
    lw     = max(2, int(3 * sf))
    tk     = int(9 * sf)
    draw.rectangle([bx,            by,            bx + bar_px, by + lw],         fill=(205, 205, 205))
    draw.rectangle([bx,            by - tk//2,    bx + lw,     by + lw + tk//2], fill=(205, 205, 205))
    draw.rectangle([bx + bar_px - lw, by - tk//2, bx + bar_px, by + lw + tk//2], fill=(205, 205, 205))
    draw.text((bx, by + int(8*sf)), "100 mm", font=f_small, fill=(185, 185, 185))

    return np.array(img)


def render_comparison(output_path=None, width=WIDTH, height=HEIGHT, duration=DURATION):
    mesh_dir = os.path.join(PROJECT_ROOT, "meshes")
    urdf_path = os.path.join(mesh_dir, "longworm2", "longworm2.SLDASM.urdf")

    if not os.path.exists(urdf_path):
        src = os.path.join("D:/inovxio/3d/longworm2/longworm2.SLDASM/urdf",
                           "longworm2.SLDASM.urdf")
        if os.path.exists(src):
            os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
            import shutil as sh; sh.copy2(src, urdf_path)

    # Cache CMA-ES params for all modes
    apply_gaits.__cache = load_all_cmaes_params()

    print(f"Gait Comparison Race — {N_ROBOTS} robots")
    print(f"  Resolution: {width}x{height}, {FPS}fps, {duration}s")

    print("  Building XML...")
    xml_str = build_race_xml(mesh_dir, urdf_path, width, height)

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    print(f"  bodies={model.nbody}, DOF={model.nv}, actuators={model.nu}")

    robots = get_robot_info(model)

    # Settle
    print("  Settling...")
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    # Record initial positions
    x0 = [data.xpos[r['head_id'], 0] for r in robots]

    # Renderer
    try:
        renderer = mujoco.Renderer(model, height=height, width=width, max_geom=15000)
    except TypeError:
        renderer = mujoco.Renderer(model, height=height, width=width)

    # Output
    if output_path is None:
        vid_dir = os.path.join(PROJECT_ROOT, "record", "v6", "videos")
        os.makedirs(vid_dir, exist_ok=True)
        tag = "4k" if width >= 3840 else f"{width}x{height}"
        output_path = os.path.join(vid_dir, f"gait_comparison_{tag}.mp4")

    # Video writer
    ffmpeg_bin = shutil.which('ffmpeg')
    writer = None
    frames_buf = None
    if ffmpeg_bin:
        writer = subprocess.Popen([
            ffmpeg_bin, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(FPS),
            '-i', '-',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p', output_path,
        ], stdin=subprocess.PIPE)
    else:
        frames_buf = []

    # Main loop
    total_steps = int(duration / model.opt.timestep)
    frame_iv = max(1, int(1.0 / (FPS * model.opt.timestep)))
    frame_count = 0
    t0 = time.time()

    # Camera tracks the CMA-ES robot (index 4) with smooth follow
    lookat = np.array([0.0, 0.0, BODY_Z * 0.5])

    print(f"  Rendering...")
    for step in range(total_steps):
        t = step * model.opt.timestep
        apply_gaits(data, robots, t)
        mujoco.mj_step(model, data)

        if step % frame_iv != 0:
            continue

        progress = t / duration

        # Camera: side view, tracking the group center weighted toward CMA-ES
        group_x = np.mean([data.xpos[r['head_id'], 0] for r in robots])
        cma_x = data.xpos[robots[2]['head_id'], 0]
        track_x = 0.4 * group_x + 0.6 * cma_x  # bias toward CMA-ES
        target = np.array([track_x, 0.0, BODY_Z * 0.4])
        lookat += 0.05 * (target - lookat)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 160
        cam.elevation = -22
        cam.distance = 2.8 - 0.5 * progress  # slowly zoom in
        cam.lookat[:] = lookat

        renderer.update_scene(data, cam)
        inject_race_strips(renderer.scene, data, robots)
        frame = renderer.render().copy()

        # Text overlay
        frame = add_text_overlay(frame, robots, data, t)

        if writer:
            writer.stdin.write(frame.tobytes())
        elif frames_buf is not None:
            frames_buf.append(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - t0
            total_frames = int(duration * FPS)
            eta = elapsed / frame_count * (total_frames - frame_count)
            print(f"    {frame_count}/{total_frames} ({elapsed:.0f}s, ~{eta:.0f}s ETA)")

    renderer.close()

    # Print speed results
    print(f"\n  Race Results (forward distance in {duration}s):")
    names = ["Serpentine-only", "Peristaltic-only", "Full Combined"]
    for i, (r, name) in enumerate(zip(robots, names)):
        xf = data.xpos[r['head_id'], 0]
        fwd = -(xf - x0[i]) * 1000
        spd = fwd / duration
        marker = " ★" if i == 2 else ""
        print(f"    {name:>20s}: {fwd:>7.1f} mm  ({spd:>6.1f} mm/s){marker}")

    if writer:
        writer.stdin.close()
        writer.wait()
    elif frames_buf:
        try:
            import mediapy
            mediapy.write_video(output_path, frames_buf, fps=FPS)
        except ImportError:
            print("  ERROR: no video encoder")
            return

    total_time = time.time() - t0
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved: {output_path}")
    print(f"  {frame_count} frames, {file_size:.1f} MB, {total_time:.1f}s render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if args.preview:
        args.width, args.height = 1280, 720

    render_comparison(
        output_path=args.output,
        width=args.width,
        height=args.height,
        duration=args.duration,
    )

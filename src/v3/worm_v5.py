"""
Worm Robot V5 — Dual-Mode Locomotion (Worm + Snake)
====================================================
Based on V4 (cable composites, proven peristaltic locomotion).
Adds yaw position actuators on plate joints for snake-like undulation.

  - Worm mode:     V4 peristaltic contraction wave (identical to V4)
  - Snake mode:    Sinusoidal yaw wave on boundary plates
  - Combined mode: Peristalsis + yaw undulation simultaneously

Architecture: same 5 segments, 6 plates as V4.
Swing yaw at plates 2 and 4 (boundaries between body segments).

Usage:
    python worm_v5.py --mode worm --video
    python worm_v5.py --mode snake --video
    python worm_v5.py --mode combined --video
    python worm_v5.py --mode combined --snake-amp 0.4 --snake-freq 0.8

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
# Geometry constants — identical to V4
# ─────────────────────────────────────────────────────────────────────────────
NUM_SEGMENTS   = 5
SEG_LENGTH     = 0.1012
PLATE_RADIUS   = 0.055
STRIP_CIRCLE_R = 0.042
BOW_AMOUNT     = 0.023
NUM_STRIPS     = 8
NUM_VERTS      = 40
Z_CENTER       = PLATE_RADIUS + 0.001
STRIP_ANGLES   = [2 * math.pi * i / NUM_STRIPS for i in range(NUM_STRIPS)]

STRIP_W      = 0.021
STRIP_T      = 0.004
VIS_BOW      = 0.028
STRIP_RGBA   = np.array([0.10, 0.10, 0.12, 1.0], dtype=np.float32)
PLATE_RGBA   = np.array([0.08, 0.08, 0.10, 0.97], dtype=np.float32)

# Swing yaw: plates at segment boundaries
# Plate 2 = boundary between seg 1-2, Plate 4 = boundary between seg 3-4
BOUNDARY_PLATES = [2, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Visual rendering (identical to V4.1)
# ─────────────────────────────────────────────────────────────────────────────

def hide_cable_geoms(scene, m, plate_id_set, n_orig):
    OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
    for i in range(n_orig):
        g = scene.geoms[i]
        keep = False
        if int(g.objtype) == OBJ_GEOM and 0 <= g.objid < m.ngeom:
            bid = int(m.geom_bodyid[g.objid])
            if bid == 0 or bid in plate_id_set:
                keep = True
        elif int(g.objtype) != OBJ_GEOM:
            keep = True
        if not keep:
            g.rgba[3] = 0.0


def fix_plate_orientations(scene, m, d, plate_ids, plate_id_set):
    world_up = np.array([0.0, 0.0, 1.0])
    n = len(plate_ids)
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
        e_up = np.cross(e_lat, e_fwd)
        e_up /= (np.linalg.norm(e_up) + 1e-12)
        target_mats[pid] = np.array([
            e_lat[0], e_lat[1], e_lat[2],
            e_up[0],  e_up[1],  e_up[2],
            e_fwd[0], e_fwd[1], e_fwd[2],
        ], dtype=np.float32)

    OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if int(g.objtype) == OBJ_GEOM and 0 <= g.objid < m.ngeom:
            bid = int(m.geom_bodyid[g.objid])
            if bid in target_mats:
                g.mat[:] = target_mats[bid].reshape(3, 3)


def inject_flat_strips(scene, d, plate_ids):
    BOX      = int(mujoco.mjtGeom.mjGEOM_BOX)
    num_segs = len(plate_ids) - 1
    world_up = np.array([0.0, 0.0, 1.0])

    for seg in range(num_segs):
        pi = plate_ids[seg]
        pj = plate_ids[seg + 1]
        pos_i = d.xpos[pi].copy()
        pos_j = d.xpos[pj].copy()
        body_axis = pos_j - pos_i
        body_len = np.linalg.norm(body_axis)
        if body_len < 1e-6:
            continue
        e_fwd = body_axis / body_len
        mat_i = d.xmat[pi].reshape(3, 3)
        mat_j = d.xmat[pj].reshape(3, 3)

        def make_ref(mat_col):
            ref = mat_col - np.dot(mat_col, e_fwd) * e_fwd
            n = np.linalg.norm(ref)
            if n < 1e-6:
                ref = np.cross(e_fwd, world_up)
                n = np.linalg.norm(ref)
            return ref / (n + 1e-12)

        ref_i = make_ref(mat_i[:, 0])
        ref_j = make_ref(mat_j[:, 0])
        bow_scale = min(3.0, max(0.2, SEG_LENGTH / max(body_len, 0.012)))
        vis_bow = VIS_BOW * bow_scale

        for si in range(NUM_STRIPS):
            ca = math.cos(STRIP_ANGLES[si])
            sa = math.sin(STRIP_ANGLES[si])
            verts, e_lats, e_ups = [], [], []
            for k in range(NUM_VERTS):
                t = k / (NUM_VERTS - 1)
                e_lat_k = (1.0 - t) * ref_i + t * ref_j
                e_lat_k -= np.dot(e_lat_k, e_fwd) * e_fwd
                n = np.linalg.norm(e_lat_k)
                e_lat_k = e_lat_k / (n + 1e-12)
                e_up_k = np.cross(e_lat_k, e_fwd)
                e_up_k /= (np.linalg.norm(e_up_k) + 1e-12)
                center = (1.0 - t) * pos_i + t * pos_j
                bow_r = vis_bow * 4.0 * t * (1.0 - t)
                r = STRIP_CIRCLE_R + bow_r
                verts.append(center + r * ca * e_lat_k + (Z_CENTER + r * sa) * e_up_k)
                e_lats.append(e_lat_k)
                e_ups.append(e_up_k)

            for k in range(NUM_VERTS - 1):
                if scene.ngeom >= scene.maxgeom:
                    return
                p0, p1 = verts[k], verts[k + 1]
                mid = (p0 + p1) * 0.5
                dv = p1 - p0
                L = np.linalg.norm(dv)
                if L < 1e-10:
                    continue
                half_len = L * 0.5 * 1.25
                z_ax = dv / L
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
                    tang[0], tang[1], tang[2],
                    radial[0], radial[1], radial[2],
                    z_ax[0], z_ax[1], z_ax[2],
                ], dtype=np.float64)
                size = np.array([STRIP_W / 2, STRIP_T / 2, half_len], dtype=np.float64)
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, BOX, size, mid.astype(np.float64), mat, STRIP_RGBA)
                geom.emission  = 0.12
                geom.specular  = 0.5
                geom.shininess = 0.4
                scene.ngeom += 1


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run(mode='combined', record_video=False, sim_time=None,
        snake_amp=0.3, snake_freq=0.5, snake_waves=1.0):
    num_plates = NUM_SEGMENTS + 1  # 6

    # V4 proven parameters + yaw actuators
    params = dict(
        num_segments      = NUM_SEGMENTS,
        cable_constraint  = 'weld',
        cable_weld_solref = '0.002 1',
        bend_stiff        = 1e8,
        twist_stiff       = 2e6,
        plate_stiff_x     = 500.0,
        plate_stiff_y     = 0.0,
        plate_stiff_yaw   = 0.0,
        axial_muscle_force= 100,
        gait_s0           = '0,0,0,1,1',
        step_duration     = 0.5,
        settle_time       = 1.0,
        sim_time          = sim_time or 25.0,
        # V5 addition: yaw position actuators on all plate joints
        yaw_actuator      = 1,
        yaw_actuator_kp   = 50.0,
        yaw_actuator_range= 0.5,
    )

    exp_id = f"v5_{mode}"
    xml_str, P = build_model_xml(exp_id, params)

    num_axial = NUM_SEGMENTS * 4
    num_ring  = NUM_SEGMENTS

    if record_video and '<visual>' not in xml_str:
        xml_str = xml_str.replace(
            '<size memory=',
            '<visual><global offwidth="1280" offheight="720"/></visual>\n  <size memory='
        )

    src_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    bin_dir      = os.path.join(project_root, "bin", "v3")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, f"worm_{exp_id}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    plate_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}")
                 for p in range(num_plates)]
    head_id = plate_ids[-1]
    tail_id = plate_ids[0]

    # Steer muscle base index
    num_steer_base = num_axial + num_ring

    # Find yaw actuator indices at boundary plates
    yaw_act_ids = []
    for p in BOUNDARY_PLATES:
        try:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"yaw_p{p}")
            yaw_act_ids.append(aid)
        except Exception:
            print(f"WARNING: yaw_p{p} not found")

    print(f"Model: bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}")
    print(f"  Yaw actuators at plates {BOUNDARY_PLATES} → {yaw_act_ids}")
    print(f"Mode: {mode.upper()}")
    if mode != 'worm':
        print(f"  snake_amp={snake_amp:.2f} rad ({math.degrees(snake_amp):.1f}°), "
              f"freq={snake_freq:.2f} Hz, waves={snake_waves:.1f}")

    # Hide cable geoms + colour plates
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

    # Settle
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

    # Gait
    gait_s0  = [int(x) for x in str(P['gait_s0']).split(',')]
    step_dur = P['step_duration']
    snake_phase_offset = 2 * math.pi * snake_waves / max(len(yaw_act_ids), 1)

    def get_states(t):
        j = int(t / step_dur) % NUM_SEGMENTS
        return [gait_s0[(k + j) % NUM_SEGMENTS] for k in range(NUM_SEGMENTS)]

    def apply_control(d, t):
        d.ctrl[:] = 0

        # Worm: peristaltic wave (identical to V4)
        if mode in ('worm', 'combined'):
            states = get_states(t)
            for seg, s in enumerate(states):
                if s == 1:
                    for mi in range(4):
                        d.ctrl[seg * 4 + mi] = 1.0
                elif s == 2:
                    steer_idx = num_steer_base + seg * 2
                    if steer_idx < m.nu:
                        d.ctrl[steer_idx] = 1.0
                elif s == 3:
                    steer_idx = num_steer_base + seg * 2 + 1
                    if steer_idx < m.nu:
                        d.ctrl[steer_idx] = 1.0
                ring_idx = num_axial + seg
                if ring_idx < m.nu:
                    d.ctrl[ring_idx] = 1.0 if s != 1 else 0.0

        # Snake: sinusoidal yaw on boundary plates
        if mode in ('snake', 'combined'):
            for i, aid in enumerate(yaw_act_ids):
                phase = 2 * math.pi * snake_freq * t - snake_phase_offset * i
                d.ctrl[aid] = snake_amp * math.sin(phase)

    # Video
    renderer = None
    frames   = []
    fps      = 30
    if record_video:
        renderer = mujoco.Renderer(m, 720, 1280, max_geom=10000)
        vopt = mujoco.MjvOption()
        vopt.geomgroup[4] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0

    # Main loop
    sim_total   = P['sim_time']
    total_steps = int(sim_total / m.opt.timestep)
    frame_iv    = max(1, int(1.0 / (fps * m.opt.timestep)))
    dt          = m.opt.timestep
    lookat_smooth = ((d.xpos[head_id] + d.xpos[tail_id]) / 2.0).copy()
    t0_wall      = time.time()
    print(f"Simulating {sim_total:.0f}s ({total_steps} steps) ...")

    for step in range(total_steps):
        t = step * dt
        apply_control(d, t)
        mujoco.mj_step(m, d)

        if step % 5000 == 0 and step > 0:
            if np.any(np.isnan(d.qpos)):
                print(f"NaN at t={t:.2f}s!"); break

        if step % 10000 == 0 and step > 0:
            hx = d.xpos[head_id, 0] * 1000
            hy = d.xpos[head_id, 1] * 1000
            tx = d.xpos[tail_id, 0] * 1000
            ty = d.xpos[tail_id, 1] * 1000
            hdg = math.degrees(math.atan2(hx - tx, hy - ty))
            rate = step / (time.time() - t0_wall)
            print(f"  t={t:.1f}s  head=({hx:.1f},{hy:.1f})mm  hdg={hdg:.1f}°  "
                  f"ncon={d.ncon}  [{rate:.0f} steps/s]")

        if record_video and step % frame_iv == 0 and renderer is not None:
            cam          = mujoco.MjvCamera()
            cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 1.0
            cam.elevation = -25
            cam.azimuth  = 45
            mid = (d.xpos[head_id] + d.xpos[tail_id]) / 2.0
            lookat_smooth += 0.03 * (mid - lookat_smooth)
            cam.lookat[:] = lookat_smooth

            renderer.update_scene(d, cam, scene_option=vopt)
            n_orig = renderer.scene.ngeom
            hide_cable_geoms(renderer.scene, m, plate_id_set, n_orig)
            fix_plate_orientations(renderer.scene, m, d, plate_ids, plate_id_set)
            inject_flat_strips(renderer.scene, d, plate_ids)
            frames.append(renderer.render().copy())

    elapsed = time.time() - t0_wall
    hx_f = d.xpos[head_id, 0] * 1000
    hy_f = d.xpos[head_id, 1] * 1000
    heading = math.degrees(math.atan2(
        d.xpos[head_id, 0] - d.xpos[tail_id, 0],
        d.xpos[head_id, 1] - d.xpos[tail_id, 1]))
    displacement = math.hypot(hx_f - hx0, hy_f - hy0)

    print(f"\n{'─'*60}")
    print(f"  Worm V5 — {mode.upper()}")
    print(f"  Head final:    ({hx_f:.1f}, {hy_f:.1f}) mm")
    print(f"  Displacement:  {displacement:.1f} mm in {sim_total:.0f}s")
    print(f"  Speed:         {displacement/sim_total:.2f} mm/s")
    print(f"  Heading:       {heading:.1f}°")
    print(f"  Wall time:     {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")
    print(f"{'─'*60}")

    if record_video and frames:
        try:
            import mediapy
            vid_dir  = os.path.join(project_root, "record", "v5", "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"worm_v5_{mode}.mp4")
            mediapy.write_video(vid_path, frames, fps=fps)
            print(f"Video: {vid_path}  ({len(frames)} frames)")
        except ImportError:
            print("mediapy not found — video not saved")

    if renderer:
        renderer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Worm V5 — dual-mode locomotion")
    ap.add_argument("--mode", choices=["worm", "snake", "combined"], default="combined")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--snake-amp", type=float, default=0.3)
    ap.add_argument("--snake-freq", type=float, default=0.5)
    ap.add_argument("--snake-waves", type=float, default=1.0)
    args = ap.parse_args()

    run(mode=args.mode, record_video=args.video, sim_time=args.time,
        snake_amp=args.snake_amp, snake_freq=args.snake_freq,
        snake_waves=args.snake_waves)

"""
Record video of worm locomotion using MuJoCo offscreen rendering.
=================================================================
Uses exp_runner.build_model_xml() for the FULL model (cables/steel strips).

Usage:
  python record_video.py [--output path.mp4] [--duration 60] [--fps 30]
                         [--width 1280] [--height 720] [--camera top|side|track|orbit]
                         [--params key=val ...]

Default: records the optimal S23 config (body_head 0.3N, circular locomotion).
"""
import mujoco
import numpy as np
import math
import os
import sys
import time
import imageio

from exp_runner import build_model_xml, DEFAULTS


def prepare_render_xml(xml_str, width=1280, height=720, ground_size=5.0):
    """Add visual rendering elements (offscreen buffer, grid, lights) to model XML."""
    # 1. Insert <visual> block before <worldbody>
    visual_block = (
        f'  <visual>\n'
        f'    <global azimuth="135" elevation="-25" offwidth="{width}" offheight="{height}"/>\n'
        f'    <quality shadowsize="2048"/>\n'
        f'    <map force="0.1"/>\n'
        f'  </visual>\n'
    )
    xml_str = xml_str.replace('  <worldbody>\n', visual_block + '  <worldbody>\n')

    # 2. Replace single light with two-light setup
    xml_str = xml_str.replace(
        '    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>',
        '    <light diffuse=".8 .8 .8" dir="0 -0.3 -1" directional="true" pos="0 0 3"/>\n'
        '    <light diffuse=".3 .3 .3" dir="0 0.3 -1" directional="true" pos="0 0 3"/>'
    )

    # 3. Enlarge ground plane
    xml_str = xml_str.replace('size="2 2 0.01"', f'size="{ground_size} {ground_size} 0.01"')

    # 4. Add grid lines and ground color before </worldbody>
    grid_xml = ""
    grid_spacing = 0.1  # 100mm
    n = int(ground_size / grid_spacing)
    for i in range(-n, n + 1):
        pos = i * grid_spacing
        alpha = "0.5" if i % 10 == 0 else "0.15"
        w = "0.001" if i % 10 == 0 else "0.0004"
        grid_xml += f'    <geom type="box" size="{ground_size} {w} 0.00005" pos="0 {pos:.4f} 0.00005" rgba="0.5 0.5 0.5 {alpha}" contype="0" conaffinity="0"/>\n'
        grid_xml += f'    <geom type="box" size="{w} {ground_size} 0.00005" pos="{pos:.4f} 0 0.00005" rgba="0.5 0.5 0.5 {alpha}" contype="0" conaffinity="0"/>\n'
    xml_str = xml_str.replace('  </worldbody>', grid_xml + '  </worldbody>')

    return xml_str


def color_plates(m, num_plates):
    """Color head plate red, tail plate blue using model geom_rgba."""
    for gi in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        body_id = m.geom_bodyid[gi]
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        # Head plate (last plate)
        if body_name == f"plate{num_plates - 1}" and name == "":
            # Only the main cylinder geom (unnamed)
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                m.geom_rgba[gi] = [0.9, 0.25, 0.2, 0.95]
        # Tail plate (plate0)
        elif body_name == "plate0" and name == "":
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                m.geom_rgba[gi] = [0.2, 0.25, 0.9, 0.95]


def run_and_record(params, output_path, duration=60.0, fps=30, width=1280, height=720, camera_mode="top"):
    """Run simulation and record video using the full worm model."""
    # Build model XML using exp_runner's full model generation
    xml_str, P = build_model_xml("video", params)

    # Add rendering elements
    ground_size = max(5.0, float(P.get('ground_size', 5.0)))
    xml_str = prepare_render_xml(xml_str, width=width, height=height, ground_size=ground_size)

    # Save XML and load model
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
    bin_dir = os.path.join(project_root, "bin", "v3", "experiments")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, "video_model.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    num_segments = P['num_segments']
    num_plates = num_segments + 1
    no_cables = int(P.get('no_cables', 0))
    num_axial = num_segments * 4
    num_ring = 0 if no_cables else num_segments

    # Color head/tail plates
    color_plates(m, num_plates)

    # Renderer
    renderer = mujoco.Renderer(m, height=height, width=width)

    pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
    head_id = pids[-1]
    tail_id = pids[0]

    # Yaw joint info
    yaw_jnt_ids = []
    yaw_dof_ids = []
    for p in range(num_plates):
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{p}_yaw")
        yaw_jnt_ids.append(jnt_id)
        yaw_dof_ids.append(m.jnt_dofadr[jnt_id])

    # Gait
    gait_str = str(P['gait_s0'])
    s0_gait = [int(x) for x in gait_str.split(',')]
    TN = num_segments
    nP = 1
    step_dur = P['step_duration']
    state2_axial = P['state2_axial']
    state2_mode = str(P['state2_mode'])
    steer_in_s2 = int(P['steer_in_state2'])

    def get_states(t_act):
        j = int(t_act / step_dur) % TN
        return [s0_gait[(k + j * nP) % TN] for k in range(TN)]

    # Settle
    mujoco.mj_forward(m, d)
    for _ in range(4000):
        mujoco.mj_step(m, d)

    # Video writer
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p', '-crf', '23'])

    dt = m.opt.timestep
    settle_time = P['settle_time']
    total_steps = int(duration / dt)
    steps_per_frame = max(1, int(1.0 / (fps * dt)))
    steer_base = num_axial + num_ring  # account for ring muscles

    # Camera + scene
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

    # Yaw mode params
    yaw_mode = str(P.get('yaw_mode', 'none'))
    yaw_angle_rad = math.radians(float(P.get('yaw_angle', 18.0)))
    yaw_kp = float(P.get('yaw_kp', 50.0))
    yaw_kd = float(P.get('yaw_kd', 2.0))

    print(f"Recording {duration}s at {fps}fps ({total_steps} steps, {total_steps//steps_per_frame} frames)...")
    print(f"Model: {'no_cables' if no_cables else 'full cables'}, "
          f"actuators={m.nu}, DOF={m.nv}, bodies={m.nbody}")
    t0 = time.time()
    frame_count = 0

    for step in range(total_steps):
        t = step * dt

        # Control — identical to exp_runner.run_experiment
        d.ctrl[:] = 0
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        if t >= settle_time:
            t_act = t - settle_time
            states = get_states(t_act)
            for seg in range(num_segments):
                s = states[seg]
                if s == 0:
                    pass
                elif s == 1:
                    for mi in range(4):
                        d.ctrl[seg * 4 + mi] = 1.0
                elif s == 2:
                    if state2_mode == "differential":
                        d.ctrl[seg * 4 + 2] = 1.0  # left ON
                    else:
                        for mi in range(4):
                            d.ctrl[seg * 4 + mi] = state2_axial
                elif s == 3:
                    if state2_mode == "differential":
                        d.ctrl[seg * 4 + 0] = 1.0  # right ON
                    else:
                        for mi in range(4):
                            d.ctrl[seg * 4 + mi] = state2_axial
                # Ring muscle (squeeze on extend, relax on contract)
                if num_ring > 0:
                    d.ctrl[num_axial + seg] = 1.0 if s == 0 else 0.0
                # Diagonal steering
                if steer_in_s2:
                    if s == 2:
                        d.ctrl[steer_base + seg * 2] = 1.0
                    elif s == 3:
                        d.ctrl[steer_base + seg * 2 + 1] = 1.0

            # Yaw spring controller
            if yaw_mode == "spring":
                anchor_damp = float(P.get('anchor_yaw_damp', 0.0))
                max_torque = float(P.get('yaw_max_torque', 0.005))
                for seg in range(num_segments):
                    s = states[seg]
                    desired_rel = 0.0
                    if s == 2:
                        desired_rel = yaw_angle_rad
                    elif s == 3:
                        desired_rel = -yaw_angle_rad
                    qpos_head = d.qpos[m.jnt_qposadr[yaw_jnt_ids[seg+1]]]
                    qpos_tail = d.qpos[m.jnt_qposadr[yaw_jnt_ids[seg]]]
                    actual_rel = qpos_head - qpos_tail
                    qvel_head = d.qvel[yaw_dof_ids[seg+1]]
                    qvel_tail = d.qvel[yaw_dof_ids[seg]]
                    actual_rel_vel = qvel_head - qvel_tail
                    error = desired_rel - actual_rel
                    torque = yaw_kp * error - yaw_kd * actual_rel_vel
                    torque = max(min(torque, max_torque), -max_torque)
                    d.qfrc_applied[yaw_dof_ids[seg+1]] += torque
                    d.qfrc_applied[yaw_dof_ids[seg]] -= torque
                    if s == 1 and anchor_damp > 0:
                        d.qfrc_applied[yaw_dof_ids[seg]] -= anchor_damp * d.qvel[yaw_dof_ids[seg]]
                        d.qfrc_applied[yaw_dof_ids[seg+1]] -= anchor_damp * d.qvel[yaw_dof_ids[seg+1]]

            # Lateral steering force
            steer_f = float(P.get('steer_force', 0))
            if steer_f != 0:
                steer_m = P.get('steer_mode', 'extend')
                body_frame = steer_m.startswith("body")
                if body_frame:
                    _hx = d.xpos[head_id, 0]
                    _hy = d.xpos[head_id, 1]
                    _tx = d.xpos[tail_id, 0]
                    _ty = d.xpos[tail_id, 1]
                    _heading = math.atan2(_hx - _tx, _hy - _ty)
                    fx = -steer_f * math.cos(_heading)
                    fy = steer_f * math.sin(_heading)
                else:
                    fx, fy = -steer_f, 0.0
                effective_mode = steer_m.replace("body_", "") if body_frame else steer_m
                if effective_mode == "all":
                    for p in range(num_plates):
                        d.xfrc_applied[pids[p], 0] += fx
                        d.xfrc_applied[pids[p], 1] += fy
                elif effective_mode == "extend":
                    applied = set()
                    for seg in range(num_segments):
                        if states[seg] == 0:
                            for pi in (seg, seg + 1):
                                if pi not in applied:
                                    d.xfrc_applied[pids[pi], 0] += fx
                                    d.xfrc_applied[pids[pi], 1] += fy
                                    applied.add(pi)
                elif effective_mode == "anchor":
                    applied = set()
                    for seg in range(num_segments):
                        if states[seg] == 1:
                            for pi in (seg, seg + 1):
                                if pi not in applied:
                                    d.xfrc_applied[pids[pi], 0] += fx
                                    d.xfrc_applied[pids[pi], 1] += fy
                                    applied.add(pi)
                elif effective_mode == "head":
                    d.xfrc_applied[pids[num_plates - 1], 0] += fx
                    d.xfrc_applied[pids[num_plates - 1], 1] += fy

        mujoco.mj_step(m, d)

        # Render frame
        if step % steps_per_frame == 0:
            com_x = np.mean([d.xpos[pids[p], 0] for p in range(num_plates)])
            com_y = np.mean([d.xpos[pids[p], 1] for p in range(num_plates)])
            com_z = np.mean([d.xpos[pids[p], 2] for p in range(num_plates)])

            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if camera_mode == "top":
                cam.lookat[:] = [com_x, com_y, 0]
                cam.distance = 0.55
                cam.azimuth = 90
                cam.elevation = -90
            elif camera_mode == "side":
                cam.lookat[:] = [com_x, com_y, com_z]
                cam.distance = 0.45
                cam.azimuth = 135
                cam.elevation = -20
            elif camera_mode == "track":
                cam.lookat[:] = [com_x, com_y, com_z]
                cam.distance = 0.40
                cam.azimuth = 90 + t * 3
                cam.elevation = -30
            elif camera_mode == "orbit":
                cam.lookat[:] = [com_x, com_y, 0]
                cam.distance = 3.5
                cam.azimuth = 90
                cam.elevation = -90

            renderer.update_scene(d, cam, opt)
            frame = renderer.render()
            writer.append_data(frame)
            frame_count += 1

            if frame_count % (fps * 10) == 0:
                elapsed = time.time() - t0
                print(f"  t={t:.1f}s, frames={frame_count}, elapsed={elapsed:.1f}s")

    writer.close()
    renderer.close()
    elapsed = time.time() - t0
    print(f"Done! {frame_count} frames, {elapsed:.1f}s wall time")
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    # Default: optimal S23 config WITH CABLES (full visual model)
    params = {
        'gait_s0': '2,0,0,0,1',
        'state2_mode': 'symmetric',
        'steer_in_state2': 0,
        # no_cables=0 (default) → full cable model with steel strips
        'plate_stiff_y': 0,
        'plate_stiff_x': 0,
        'tendon_stiffness': 5000,
        'tendon_damping': 10,
        'yaw_mode': 'none',
        'steer_force': 0.3,
        'steer_mode': 'body_head',
    }

    output_path = None
    duration = 60.0
    fps = 30
    width = 1280
    height = 720
    camera_mode = "top"

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--output' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif arg == '--duration' and i + 1 < len(sys.argv):
            duration = float(sys.argv[i + 1])
            i += 2
        elif arg == '--fps' and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])
            i += 2
        elif arg == '--width' and i + 1 < len(sys.argv):
            width = int(sys.argv[i + 1])
            i += 2
        elif arg == '--height' and i + 1 < len(sys.argv):
            height = int(sys.argv[i + 1])
            i += 2
        elif arg == '--camera' and i + 1 < len(sys.argv):
            camera_mode = sys.argv[i + 1]
            i += 2
        elif arg == '--params':
            i += 1
            while i < len(sys.argv):
                kv = sys.argv[i]
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    try:
                        v = float(v)
                        if v == int(v):
                            v = int(v)
                    except ValueError:
                        pass
                    params[k] = v
                i += 1
        else:
            i += 1

    if output_path is None:
        project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_path = os.path.join(project_root, "record", "v3", "videos", f"circular_{camera_mode}.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_and_record(params, output_path, duration=duration, fps=fps,
                   width=width, height=height, camera_mode=camera_mode)

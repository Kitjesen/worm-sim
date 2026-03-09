"""Quick snapshot test: compare cable shape at different bend_stiff values."""
import mujoco
import numpy as np
import math, os, sys, re, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_runner import build_model_xml
from pipe_crawl import generate_pipe_xml

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, "..", ".."))
VID_DIR = os.path.join(ROOT, "record", "v3", "videos")
os.makedirs(VID_DIR, exist_ok=True)

configs = [
    ("bend1e8_weld",    1e8, 2e6, "weld",    "0.002 1"),
    ("bend1e8_connect", 1e8, 2e6, "connect", "0.005 1"),
    ("bend5e7_connect", 5e7, 1e6, "connect", "0.005 1"),
    ("bend1e7_connect", 1e7, 2e5, "connect", "0.005 1"),
]

for name, bend, twist, ctype, solref in configs:
    print(f"\n{'='*60}")
    print(f"  {name}: bend={bend:.0e}, constraint={ctype}, solref={solref}")
    print(f"{'='*60}")

    params = {
        'num_segments': 5, 'no_cables': 0,
        'cable_constraint': ctype,
        'cable_weld_solref': solref,
        'bend_stiff': bend, 'twist_stiff': twist,
        'plate_stiff_x': 0.0, 'plate_stiff_y': 0.0, 'plate_stiff_yaw': 0.0,
        'tendon_stiffness': 10000, 'tendon_damping': 15,
        'axial_muscle_force': 50, 'gait_s0': '0,0,0,1,1',
        'step_duration': 0.5, 'sim_time': 20.0, 'settle_time': 2.0,
        'steer_in_state2': 0, 'state2_mode': 'symmetric', 'state2_axial': 0.0,
    }

    xml_str, P = build_model_xml("snap_test", params)

    # Collision groups
    xml_str = re.sub(
        r'(<composite type="cable"[^>]*>.*?)(contype="1" conaffinity="1")(.*?</composite>)',
        lambda m: m.group(1) + 'contype="4" conaffinity="4"' + m.group(3),
        xml_str, flags=re.DOTALL
    )
    xml_str = xml_str.replace('contype="3" conaffinity="3"', 'contype="7" conaffinity="7"', 1)

    # Channel
    pipe_xml, _ = generate_pipe_xml(channel_width=0.056, bend_radius=0.20,
                                     straight_length=0.40, ceiling_z=0.055)
    visual_block = '  <visual>\n    <global offwidth="1280" offheight="720"/>\n    <quality shadowsize="2048"/>\n    <map znear="0.001" zfar="5.0"/>\n  </visual>\n'
    xml_str = xml_str.replace('<worldbody>', visual_block + '  <worldbody>')
    xml_str = xml_str.replace('  </worldbody>', pipe_xml + '\n  </worldbody>')

    bin_dir = os.path.join(ROOT, "bin", "v3", "experiments")
    os.makedirs(bin_dir, exist_ok=True)
    xml_path = os.path.join(bin_dir, f"exp_snap_{name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    num_segments = P['num_segments']
    num_plates = num_segments + 1
    num_axial = num_segments * 4
    pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
    head_id, tail_id = pids[-1], pids[0]

    # Color plates
    for gi in range(m.ngeom):
        bid = m.geom_bodyid[gi]
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if gn.startswith(("wall_", "ceil")):
            continue
        if bid == head_id:
            m.geom_rgba[gi] = [0.9, 0.25, 0.2, 0.95]
        elif bid == tail_id:
            m.geom_rgba[gi] = [0.2, 0.25, 0.9, 0.95]

    # Settle
    dt = m.opt.timestep
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(m, d)

    # Gait
    gait_s0 = [int(x) for x in str(P['gait_s0']).split(',')]
    TN = num_segments

    def get_states(t):
        j = int(t / P['step_duration']) % TN
        return [gait_s0[(k + j) % TN] for k in range(TN)]

    # Simulate 15s
    t0 = time.time()
    for step in range(int(15.0 / dt)):
        t = step * dt
        d.ctrl[:] = 0
        states = get_states(t)
        for seg in range(num_segments):
            s = states[seg]
            if s == 1:
                for mi in range(4):
                    d.ctrl[seg * 4 + mi] = 1.0
            d.ctrl[num_axial + seg] = 1.0 if s == 0 else 0.0
        mujoco.mj_step(m, d)

    elapsed = time.time() - t0
    hx = d.xpos[head_id, 0] * 1000
    hy = d.xpos[head_id, 1] * 1000
    tx = d.xpos[tail_id, 0] * 1000
    ty = d.xpos[tail_id, 1] * 1000
    hdg = math.degrees(math.atan2(hx - tx, hy - ty))
    ncon = d.ncon
    sps = int(15.0 / dt) / elapsed
    print(f"  head=({hx:.1f}, {hy:.1f})mm  hdg={hdg:.1f}°  ncon={ncon}  {sps:.0f} sps")

    # Render 2 views
    renderer = mujoco.Renderer(m, 720, 1280)

    # Top-down view
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.10, 0.50, 0.02]
    cam.distance = 0.9
    cam.azimuth = 90
    cam.elevation = -75
    renderer.update_scene(d, cam)
    frame = renderer.render()
    import mediapy
    mediapy.write_image(os.path.join(VID_DIR, f"snap_{name}_top.png"), frame)

    # 3/4 view
    cam.lookat[:] = [0.10, 0.45, 0.02]
    cam.distance = 0.6
    cam.azimuth = 135
    cam.elevation = -30
    renderer.update_scene(d, cam)
    frame = renderer.render()
    mediapy.write_image(os.path.join(VID_DIR, f"snap_{name}_3q.png"), frame)

    renderer.close()
    print(f"  Snapshots saved: snap_{name}_top.png, snap_{name}_3q.png")

print("\nDone!")

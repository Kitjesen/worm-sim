"""Quick render check — capture 4 frames showing rest, compress, yaw, combined."""
import os, sys, math
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worm_v5_1 import (
    build_xml, inject_strips, NUM_SEGMENTS,
    SEG_SPACING_01, SEG_SPACING_REST, SLIDE_RANGE, YAW_RANGE,
)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SRC_DIR, "..", ".."))
MESH_DIR     = os.path.join(PROJECT_ROOT, "meshes")
BIN_DIR      = os.path.join(PROJECT_ROOT, "bin", "v3")
FRAME_DIR    = os.path.join(PROJECT_ROOT, "record", "v5_1", "frames")
os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# Build model
xml_str  = build_xml(MESH_DIR)
xml_path = os.path.join(BIN_DIR, "worm_v5_1_render.xml")
with open(xml_path, "w") as f:
    f.write(xml_str)

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

seg_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"seg{i}")
           for i in range(NUM_SEGMENTS)]
natural_spacings = [SEG_SPACING_01] + [SEG_SPACING_REST] * (NUM_SEGMENTS - 2)

n_slides = NUM_SEGMENTS - 1
n_yaws   = NUM_SEGMENTS - 1
slide_ids = list(range(n_slides))
yaw_ids   = list(range(n_slides, n_slides + n_yaws))

# Settle
for _ in range(int(1.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
print(f"Model: bodies={m.nbody}, DOF={m.nv}, actuators={m.nu}")

def render_frame(m, d, seg_ids, spacings, name, cam_dist=0.9, cam_elev=-20, cam_az=135):
    renderer = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_dist
    cam.elevation = cam_elev
    cam.azimuth = cam_az
    mid = (d.xpos[seg_ids[0]] + d.xpos[seg_ids[-1]]) / 2.0
    cam.lookat[:] = mid
    renderer.update_scene(d, cam)
    inject_strips(renderer.scene, d, seg_ids, spacings)
    img = renderer.render().copy()
    renderer.close()
    path = os.path.join(FRAME_DIR, f"{name}.png")
    Image.fromarray(img).save(path)
    print(f"  Saved: {path}")
    return path

# 1. Rest pose
d.ctrl[:] = 0
for _ in range(500):
    mujoco.mj_step(m, d)
p1 = render_frame(m, d, seg_ids, natural_spacings, "v2_rest")

# 2. Compress (peristaltic)
d.ctrl[:] = 0
d.ctrl[slide_ids[0]] = -SLIDE_RANGE
d.ctrl[slide_ids[1]] = -SLIDE_RANGE
d.ctrl[slide_ids[2]] =  SLIDE_RANGE
d.ctrl[slide_ids[3]] =  SLIDE_RANGE
for _ in range(int(0.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)
p2 = render_frame(m, d, seg_ids, natural_spacings, "v2_compress")

# 3. Yaw (snake bend)
d.ctrl[:] = 0
d.ctrl[yaw_ids[0]] =  YAW_RANGE * 0.6
d.ctrl[yaw_ids[1]] = -YAW_RANGE * 0.6
d.ctrl[yaw_ids[2]] =  YAW_RANGE * 0.6
d.ctrl[yaw_ids[3]] = -YAW_RANGE * 0.6
for _ in range(int(0.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)
p3 = render_frame(m, d, seg_ids, natural_spacings, "v2_yaw")

# 4. Combined
d.ctrl[:] = 0
d.ctrl[slide_ids[0]] = -SLIDE_RANGE
d.ctrl[slide_ids[1]] =  SLIDE_RANGE
d.ctrl[slide_ids[2]] = -SLIDE_RANGE
d.ctrl[slide_ids[3]] =  SLIDE_RANGE
d.ctrl[yaw_ids[0]] =  YAW_RANGE * 0.4
d.ctrl[yaw_ids[1]] = -YAW_RANGE * 0.4
d.ctrl[yaw_ids[2]] =  YAW_RANGE * 0.4
d.ctrl[yaw_ids[3]] = -YAW_RANGE * 0.4
for _ in range(int(0.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)
p4 = render_frame(m, d, seg_ids, natural_spacings, "v2_combined")

# 5. Close-up top view
d.ctrl[:] = 0
d.ctrl[slide_ids[1]] = -SLIDE_RANGE  # compress one joint
for _ in range(int(0.3 / m.opt.timestep)):
    mujoco.mj_step(m, d)
p5 = render_frame(m, d, seg_ids, natural_spacings, "v2_closeup",
                   cam_dist=0.55, cam_elev=-25, cam_az=120)

print("\nAll frames rendered.")

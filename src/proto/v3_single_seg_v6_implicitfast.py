"""
V3 Single Segment Prototype — Cable Strips + Axial Muscles + Ring Muscle
=========================================================================
验证拮抗双肌肉系统：
  - 纵肌 (axial muscle): 4条，拉近两板 → 体节变短，cable 弯弓膨胀
  - 环肌 (ring muscle): 1条闭环 tendon，挤压 cable 中点 → 体节变细

测试序列 (16s):
  Phase 1: 沉降 (2s)
  Phase 2: 全部纵肌收缩 (3s) — 变短变胖
  Phase 3: 释放 (2s)
  Phase 4: 环肌收缩 (3s) — 变细
  Phase 5: 释放 (2s)
  Phase 6: 左弯 (2s) — 左侧纵肌
  Phase 7: 右弯 (2s) — 右侧纵肌

坐标系: Y=轴向(前进), Z=上, X=侧向 (与 V2 蠕虫一致)
4面视角: 等轴测、侧面、正面、俯视
"""
import mujoco
import numpy as np
import mediapy as media
import time
import math
import os

# ======================== Parameters ========================
seg_length = 0.065       # 65mm axial length
plate_radius = 0.022     # 22mm structural radius (for sites)
plate_geom_r = 0.014     # 14mm visual plate (< strip_circle_r to avoid penetration)
strip_circle_r = 0.017   # cable anchor radius (outside plate geom)
num_strips = 8
num_verts = 7            # vertices per cable strip
strip_r = 0.002          # capsule radius
bow_amount = 0.007       # midpoint outward bow

bend_stiff = 1e8         # restored (ref uses 8e8, ours was too soft → vibration)
twist_stiff = 4e7        # restored (ref uses 2e8)
axial_muscle_force = 30  # N per axial muscle (ref ring-only uses 5-10N)
ring_muscle_force = 10   # N for ring muscle (ref uses 5-10N)
ground_friction = 1.5

z_center = plate_radius + 0.001  # center height

# ======================== Helpers ========================
def strip_verts(angle, seg_length, strip_r_base, bow):
    """Generate cable vertices along Y axis with outward bow."""
    verts = []
    for k in range(num_verts):
        t = k / (num_verts - 1)
        y = t * seg_length
        bow_r = bow * 4.0 * t * (1.0 - t)
        r = strip_r_base + bow_r
        x = r * math.cos(angle)
        z = z_center + r * math.sin(angle)
        verts.append(f"{x:.6f} {y:.6f} {z:.6f}")
    return "  ".join(verts)

strip_angles = [2 * math.pi * i / num_strips for i in range(num_strips)]
mid_vert_idx = num_verts // 2  # index 3 for 7 verts

# ======================== XML Generation ========================

# --- Cable strips ---
cables_xml = ""
for si, angle in enumerate(strip_angles):
    v = strip_verts(angle, seg_length, strip_circle_r, bow_amount)
    cables_xml += f"""
    <body>
      <freejoint/>
      <composite type="cable" prefix="s{si}" initial="none" vertex="{v}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="{bend_stiff}"/>
          <config key="twist" value="{twist_stiff}"/>
          <config key="vmax" value="2"/>
        </plugin>
        <joint armature="0.01" damping="0.12" kind="main"/>
        <geom type="capsule" size="{strip_r}" density="3500"
              friction="{ground_friction}" contype="1" conaffinity="1"
              rgba="0.3 0.5 0.9 1"/>
      </composite>
    </body>"""

# --- Plate sites for axial tendons (4 at 90° intervals) ---
bot_sites = ""
top_sites = ""
for mi in range(4):
    sa = strip_angles[mi * 2]  # use every other strip angle (0°, 90°, 180°, 270°)
    sx = strip_circle_r * 0.7 * math.cos(sa)
    sz = z_center + strip_circle_r * 0.7 * math.sin(sa)
    bot_sites += f'      <site name="bs{mi}" pos="{sx:.5f} 0 {sz:.5f}" size="0.0015" rgba="1 0.2 0.2 1"/>\n'
    top_sites += f'      <site name="ts{mi}" pos="{sx:.5f} 0 {sz:.5f}" size="0.0015" rgba="1 0.2 0.2 1"/>\n'

# --- Ring connector balls at cable midpoints ---
ring_balls_xml = ""
for si, angle in enumerate(strip_angles):
    mid_y = seg_length / 2
    mid_bow = bow_amount
    mid_r = strip_circle_r + mid_bow
    mx = mid_r * math.cos(angle)
    mz = z_center + mid_r * math.sin(angle)
    ring_balls_xml += f'    <body name="rb{si}" pos="{mx:.5f} {mid_y:.5f} {mz:.5f}">\n'
    ring_balls_xml += f'      <freejoint/>\n'
    ring_balls_xml += f'      <geom type="sphere" size="0.002" mass="0.0005" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0"/>\n'
    ring_balls_xml += f'      <site name="rs{si}" pos="0 0 0" size="0.0018"/>\n'
    ring_balls_xml += f'    </body>\n'

# --- Weld constraints: cable ends to plates ---
welds_xml = ""
for si in range(num_strips):
    welds_xml += f'    <weld body1="bot" body2="s{si}B_first" solref="0.002 1"/>\n'
    welds_xml += f'    <weld body1="top" body2="s{si}B_last" solref="0.002 1"/>\n'

# --- Connect ring balls to cable midpoints ---
ring_connects_xml = ""
for si in range(num_strips):
    ring_connects_xml += f'    <connect body1="rb{si}" body2="s{si}B_{mid_vert_idx}" anchor="0 0 0" solref="0.003 1"/>\n'

# --- Axial tendons (4 muscles) ---
axial_tendons_xml = ""
axial_muscles_xml = ""
for mi in range(4):
    axial_tendons_xml += f'    <spatial name="at{mi}" width="0.0012" rgba="0.15 0.75 0.15 1">\n'
    axial_tendons_xml += f'      <site site="ts{mi}"/>\n'
    axial_tendons_xml += f'      <site site="bs{mi}"/>\n'
    axial_tendons_xml += f'    </spatial>\n'
    axial_muscles_xml += f'    <muscle class="muscle" name="am{mi}" tendon="at{mi}" force="{axial_muscle_force}" lengthrange="0.03 0.08"/>\n'

# --- Ring tendon (closed loop through all 8 ring balls) ---
ring_tendon_sites = ""
for si in range(num_strips):
    ring_tendon_sites += f'      <site site="rs{si}"/>\n'
ring_tendon_sites += f'      <site site="rs0"/>\n'  # close the loop

# ======================== Full XML ========================
full_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="v3_single_seg">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <size memory="2G"/>
  <compiler autolimits="true">
    <lengthrange useexisting="true" tolrange="0.5"/>
  </compiler>
  <option timestep="0.0005" gravity="0 0 -9.81" solver="Newton" iterations="200" tolerance="1e-8" integrator="implicitfast"/>

  <statistic center="0 {seg_length/2:.3f} {z_center}" extent="{seg_length*2:.3f}"/>
  <visual>
    <global offwidth="1280" offheight="720" elevation="-25"/>
  </visual>

  <default>
    <geom solimp=".95 .99 .0001" solref="0.005 1"/>
    <site size="0.002"/>
    <default class="muscle">
      <muscle ctrllimited="true" ctrlrange="0 1"/>
    </default>
  </default>

  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <light castshadow="false" diffuse=".3 .3 .3" dir="0 0 -1" pos="0 0 1.5"/>

    <geom type="plane" size="0.5 0.5 0.01" rgba="0.93 0.93 0.93 1" condim="3"
          friction="{ground_friction}" contype="3" conaffinity="3"/>

    <!-- Bottom plate (fixed) -->
    <body name="bot" pos="0 0 0">
      <geom type="cylinder" size="{plate_geom_r} 0.003" pos="0 0 {z_center}"
            euler="90 0 0" rgba="0.45 0.45 0.50 0.9" mass="0.02" contype="2" conaffinity="2"/>
{bot_sites}
    </body>

    <!-- Top plate (6-DOF: 3 slides + 3 hinges, cables provide resistance) -->
    <body name="top" pos="0 {seg_length} 0">
      <joint name="top_y" type="slide" axis="0 1 0" damping="1.0" limited="true" range="-0.030 0.01"/>
      <joint name="top_z" type="slide" axis="0 0 1" stiffness="20" damping="0.5"/>
      <joint name="top_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="top_pitch" type="hinge" axis="1 0 0" damping="0.3"/>
      <joint name="top_roll" type="hinge" axis="0 1 0" damping="0.3"/>
      <joint name="top_yaw" type="hinge" axis="0 0 1" damping="0.3"/>
      <geom type="cylinder" size="{plate_geom_r} 0.003" pos="0 0 {z_center}"
            euler="90 0 0" rgba="0.45 0.45 0.50 0.9" mass="0.02" contype="2" conaffinity="2"/>
{top_sites}
    </body>

    <!-- Cable strips (spring steel) -->
{cables_xml}

    <!-- Ring connector balls -->
{ring_balls_xml}
  </worldbody>

  <contact>
    <exclude body1="bot" body2="top"/>
  </contact>

  <equality>
    <!-- Cable ends welded to plates -->
{welds_xml}
    <!-- Ring balls connected to cable midpoints -->
{ring_connects_xml}
  </equality>

  <tendon>
    <!-- 4 axial tendons (longitudinal muscles, green) -->
{axial_tendons_xml}
    <!-- Ring tendon (circumferential muscle, red, closed loop) -->
    <spatial name="ring_t" width="0.0015" rgba="0.85 0.2 0.2 1">
{ring_tendon_sites}
    </spatial>
  </tendon>

  <actuator>
    <!-- 4 axial muscles (longitudinal contraction) -->
{axial_muscles_xml}
    <!-- 1 ring muscle (circumferential contraction) -->
    <muscle class="muscle" name="ring_m" tendon="ring_t" force="{ring_muscle_force}" lengthrange="0.05 0.20"/>
  </actuator>

  <sensor>
    <jointpos name="top_y_pos" joint="top_y"/>
    <tendonpos name="ring_len" tendon="ring_t"/>
  </sensor>
</mujoco>
"""

# ======================== Save & Validate ========================
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
bin_dir = os.path.join(project_root, "bin", "proto")
vid_dir = os.path.join(project_root, "record", "proto", "videos")
img_dir = os.path.join(project_root, "record", "proto", "images")
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

xml_path = os.path.join(bin_dir, "v3_single_seg.xml")
with open(xml_path, "w") as f:
    f.write(full_xml)
print(f"XML: {xml_path}")

print("Loading model...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"Bodies: {model.nbody}, DOF: {model.nv}, Actuators: {model.nu}")
print(f"Total mass: {np.sum(model.body_mass):.4f} kg")

bot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bot")
top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top")

# ======================== Helpers ========================
def measure_ring(m, d):
    """Measure ring radius from ring ball positions."""
    rb_pos = []
    for si in range(num_strips):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"rb{si}")
        rb_pos.append(d.xpos[bid].copy())
    c = np.mean(rb_pos, axis=0)
    radii = [math.sqrt((p[0]-c[0])**2 + (p[2]-c[2])**2) for p in rb_pos]
    return np.mean(radii)

def measure_angle(m, d):
    """Measure angle between top and bottom plates (degrees)."""
    bid_bot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bot")
    bid_top = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "top")
    # Top plate normal (Z in body frame, rotated)
    R_bot = d.xmat[bid_bot].reshape(3, 3)
    R_top = d.xmat[bid_top].reshape(3, 3)
    # Y axis of each plate (axial direction in body frame)
    n_bot = R_bot[:, 1]  # Y column
    n_top = R_top[:, 1]
    cos_a = np.clip(np.dot(n_bot, n_top), -1, 1)
    return math.degrees(math.acos(cos_a))

# Quick settle test
print("\nSettling 2s...")
for i in range(4000):  # 2s at dt=0.0005
    mujoco.mj_step(model, data)
if np.any(np.isnan(data.qpos)):
    print("WARNING: NaN after settling!")
    exit(1)
print(f"  top Y: {data.xpos[top_id][1]*1000:.1f} mm (rest: {seg_length*1000:.0f})")
print(f"  Ring radius: {measure_ring(model, data)*1000:.1f} mm")
print(f"  Plate angle: {measure_angle(model, data):.1f}°")

# ======================== Record Multi-Angle Video ========================
print("\n--- Recording 4-angle V3 single-segment demo ---")
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

bid_bot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bot")
bid_top = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "top")

am_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"am{i}") for i in range(4)]
ring_aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ring_m")

# 4 renderers for 4 views, each 360x360
PW, PH = 360, 360
renderers = [mujoco.Renderer(m, PH, PW) for _ in range(4)]

# Camera configs: [name, distance, elevation, azimuth]
cam_cfgs = [
    ("ISO",   0.12, -20,  45),   # isometric
    ("SIDE",  0.10, -10,  90),   # side (YZ plane)
    ("FRONT", 0.08,  -5,   0),   # front (XZ plane)
    ("TOP",   0.10, -89,  90),   # top-down (XY plane)
]
cameras = []
for name, dist, elev, azi in cam_cfgs:
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.distance = dist
    c.elevation = elev
    c.azimuth = azi
    cameras.append(c)

framerate = 60
frames = []
dt = m.opt.timestep

# Timeline (16s):
#  0-2s:   settle
#  2-5s:   ALL axial contract (变短变胖)
#  5-7s:   release
#  7-10s:  ring contract (变细)
# 10-12s:  release
# 12-14s:  LEFT bend (am2)
# 14-16s:  RIGHT bend (am0)

phases = [
    (0,  2,  "SETTLE",  None),
    (2,  5,  "AXIAL",   "all"),
    (5,  7,  "RELEASE", None),
    (7,  10, "RING",    "ring"),
    (10, 12, "RELEASE", None),
    (12, 14, "LEFT",    "left"),
    (14, 16, "RIGHT",   "right"),
]

def get_phase(t):
    for t0, t1, name, action in phases:
        if t0 <= t < t1:
            return name, action, t0
    return "END", None, 16

total_time = 16.0
total_steps = int(total_time / dt)
t0_wall = time.time()

for step in range(total_steps):
    t = step * dt
    d.ctrl[:] = 0

    phase_name, action, phase_start = get_phase(t)
    ramp = min((t - phase_start) / 0.5, 1.0)

    if action == "all":
        for ai in am_ids:
            d.ctrl[ai] = ramp
    elif action == "ring":
        d.ctrl[ring_aid] = ramp
    elif action == "left":
        d.ctrl[am_ids[2]] = ramp  # am2 = 180° = -X side
    elif action == "right":
        d.ctrl[am_ids[0]] = ramp  # am0 = 0° = +X side

    mujoco.mj_step(m, d)

    if np.any(np.isnan(d.qpos)):
        print(f"  NaN at t={t:.2f}s — stopping.")
        break

    # Capture frame
    if len(frames) < t * framerate:
        center = (d.xpos[bid_bot] + d.xpos[bid_top]) / 2
        panels = []
        for ci, cam in enumerate(cameras):
            cam.lookat[:] = center
            renderers[ci].update_scene(d, cam)
            panels.append(renderers[ci].render().copy())
        # 2x2 grid
        top_row = np.concatenate([panels[0], panels[1]], axis=1)
        bot_row = np.concatenate([panels[2], panels[3]], axis=1)
        frame = np.concatenate([top_row, bot_row], axis=0)
        frames.append(frame)

    if (step + 1) % 2000 == 0:
        top_y = d.xpos[bid_top][1]
        top_x = d.xpos[bid_top][0]
        rr = measure_ring(m, d)
        ang = measure_angle(m, d)
        print(f"  t={t:.1f}s [{phase_name:>7s}] Y={top_y*1000:.1f}mm  X={top_x*1000:.1f}mm  R={rr*1000:.1f}mm  angle={ang:.1f}°")

elapsed = time.time() - t0_wall
print(f"\nSim done in {elapsed:.1f}s, {len(frames)} frames")

for r in renderers:
    r.close()

vid_path = os.path.join(vid_dir, "v3_single_seg.mp4")
if len(frames) > 0:
    media.write_video(vid_path, frames, fps=framerate)
    print(f"Video: {vid_path}")
    print(f"Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")

print("\nDone.")

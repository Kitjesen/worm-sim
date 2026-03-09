"""
Worm segment V2: Two plates + 8 spring steel strips + axial muscle
=====================================================================
Structure:
  - Bottom plate: fixed rigid disk
  - Top plate: free to slide vertically
  - 8 spring steel strips (cables) connecting them, welded at both ends
  - Pre-bowed outward slightly (like leaf springs)

Actuation:
  - Axial muscle tendon from top plate center → bottom plate center
  - Muscle ON → pulls plates together → strips bow outward (compress)
  - Muscle OFF → strips spring back → plates push apart (extend)

This mimics a worm segment:
  - Longitudinal muscle contracts → segment shortens
  - Elastic structure restores → segment extends
"""
import mujoco
import numpy as np
import mediapy as media
import time
import math
import os

# ===================== Parameters =====================
seg_height = 0.10         # 10cm segment height
plate_radius = 0.035      # 3.5cm plate radius
strip_circle_r = 0.028    # strips placed at this radius
num_strips = 8
num_verts_per_strip = 13  # more vertices = smoother bending
strip_r = 0.0012          # capsule radius (thin strip)
bow_amount = 0.006        # initial outward bow at midpoint

# Stiffness: tuned for visible deformation + clear springback
bend = 2e8                # bending stiffness
twist = 8e7               # twist stiffness

# Muscle
muscle_force = 50         # max isometric force (N)

# ===================== Generate geometry =====================
def strip_vertices(angle, n_pts, height, r_base, bow):
    """Generate pre-bowed strip from bottom (z=0) to top (z=height)."""
    verts = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        z = t * height
        # Parabolic outward bow: max at midpoint
        bow_r = bow * 4.0 * t * (1.0 - t)
        r = r_base + bow_r
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        verts.append(f"{x:.6f} {y:.6f} {z:.6f}")
    return "  ".join(verts)

angles = [2 * math.pi * i / num_strips for i in range(num_strips)]
z_base = plate_radius + 0.005  # bottom plate height above ground

# ===================== Build XML =====================
cable_xml = ""
for i, a in enumerate(angles):
    v = strip_vertices(a, num_verts_per_strip, seg_height, strip_circle_r, bow_amount)
    cable_xml += f"""
    <body pos="0 0 {z_base}">
      <freejoint/>
      <composite type="cable" prefix="s{i}" initial="none" vertex="{v}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="{bend}"/>
          <config key="twist" value="{twist}"/>
          <config key="vmax" value="2"/>
        </plugin>
        <joint armature="0.008" damping="0.1" kind="main"/>
        <geom type="capsule" size="{strip_r}" density="7850" rgba="0.25 0.50 0.85 1"/>
      </composite>
    </body>"""

# Weld constraints
weld_xml = ""
for i in range(num_strips):
    weld_xml += f'    <weld body1="bot_plate" body2="s{i}B_first" solref="0.002 1"/>\n'
    weld_xml += f'    <weld body1="top_plate" body2="s{i}B_last" solref="0.002 1"/>\n'

# Plate sites for axial tendon
top_sites = ""
bot_sites = ""
for i, a in enumerate(angles):
    x = strip_circle_r * 0.5 * math.cos(a)
    y = strip_circle_r * 0.5 * math.sin(a)
    top_sites += f'      <site name="ts{i}" pos="{x:.5f} {y:.5f} 0" size="0.002" rgba="1 0.2 0.2 1"/>\n'
    bot_sites += f'      <site name="bs{i}" pos="{x:.5f} {y:.5f} 0" size="0.002" rgba="1 0.2 0.2 1"/>\n'

# Axial tendons: 4 straight tendons connecting top-bottom plate at 90° intervals
axial_tendons = ""
axial_muscles = ""
for i in range(4):
    idx = i * 2  # use strips 0, 2, 4, 6 positions
    axial_tendons += f"""    <spatial name="axial{i}" width="0.0015" rgba="0.9 0.15 0.15 1">
      <site site="ts{idx}"/>
      <site site="bs{idx}"/>
    </spatial>
"""
    axial_muscles += f'    <muscle class="muscle" name="m{i}" tendon="axial{i}" force="{muscle_force}" lengthrange="0.05 0.12"/>\n'

full_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="worm_segment_v2">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <size memory="1G"/>
  <compiler autolimits="true">
    <lengthrange tolrange="0.5"/>
  </compiler>
  <option timestep="0.001" gravity="0 0 -9.81" solver="PGS" iterations="500" tolerance="1e-6" integrator="RK4"/>

  <visual>
    <global offwidth="960" offheight="540" elevation="-20"/>
  </visual>

  <default>
    <geom solimp=".95 .99 .0001" solref="0.01 1" friction="0.35"/>
    <site size="0.002"/>
    <default class="muscle">
      <muscle ctrllimited="true" ctrlrange="0 1"/>
    </default>
  </default>

  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <light castshadow="false" diffuse=".4 .4 .4" dir="0 0 -1" pos="0 0 1.5"/>

    <camera name="front" pos="0.18 -0.08 {z_base + seg_height/2}" xyaxes="0.4 1 0 0 0 1"/>
    <camera name="side" pos="0.0 -0.20 {z_base + seg_height/2}" xyaxes="1 0 0 0 0 1"/>
    <camera name="top" pos="0 0 {z_base + seg_height + 0.15}" xyaxes="1 0 0 0 1 0"/>

    <geom type="plane" size="0.5 0.5 0.01" rgba="0.95 0.95 0.95 1" condim="3" name="floor"/>

    <!-- ===== Bottom plate (fixed to world) ===== -->
    <body name="bot_plate" pos="0 0 {z_base}">
      <geom type="cylinder" size="{plate_radius} 0.004" rgba="0.45 0.45 0.45 0.9" mass="0.1"/>
      <site name="bot_center" pos="0 0 0" size="0.003" rgba="1 0 0 1"/>
{bot_sites}
    </body>

    <!-- ===== Top plate (slides vertically) ===== -->
    <body name="top_plate" pos="0 0 {z_base + seg_height}">
      <joint name="top_z" type="slide" axis="0 0 1" damping="0.3" limited="true" range="-0.06 0.02"/>
      <joint name="top_x" type="slide" axis="1 0 0" damping="0.3" limited="true" range="-0.01 0.01"/>
      <joint name="top_y" type="slide" axis="0 1 0" damping="0.3" limited="true" range="-0.01 0.01"/>
      <geom type="cylinder" size="{plate_radius} 0.004" rgba="0.45 0.45 0.45 0.9" mass="0.1"/>
      <site name="top_center" pos="0 0 0" size="0.003" rgba="1 0 0 1"/>
{top_sites}
    </body>

    <!-- ===== 8 Spring Steel Strips ===== -->
{cable_xml}
  </worldbody>

  <equality>
{weld_xml}
  </equality>

  <tendon>
    <!-- 4 axial tendons at 90 degree intervals -->
{axial_tendons}
  </tendon>

  <actuator>
{axial_muscles}
  </actuator>

  <sensor>
    <jointpos name="seg_height" joint="top_z"/>
    <tendonpos name="tendon0_len" tendon="axial0"/>
  </sensor>
</mujoco>
"""

# ===================== Save & Test =====================
xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worm_segment_v2.xml")
with open(xml_path, "w") as f:
    f.write(full_xml)
print(f"XML saved to {xml_path}")

print("Loading model...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"Bodies: {model.nbody}, DOF: {model.nv}, Actuators: {model.nu}")
print(f"Total mass: {np.sum(model.body_mass):.4f} kg")

top_z0 = data.xpos[2, 2]  # top plate initial z
print(f"\nInitial: top_plate z = {top_z0:.4f}m")

# Settle
for i in range(500):
    mujoco.mj_step(model, data)
top_z_settled = data.xpos[2, 2]
print(f"Settled: top_plate z = {top_z_settled:.4f}m  (gravity sag = {(top_z0-top_z_settled)*1000:.1f}mm)")

# Contract: all 4 muscles activated simultaneously
for i in range(3000):
    act = min(1.0, i / 1500.0)
    for j in range(4):
        data.ctrl[j] = act
    mujoco.mj_step(model, data)
top_z_contracted = data.xpos[2, 2]
compression = (top_z_settled - top_z_contracted) * 1000
print(f"Contracted: top_plate z = {top_z_contracted:.4f}m  (compression = {compression:.1f}mm)")

# Release
for i in range(4000):
    act = max(0.0, 1.0 - i / 1500.0)
    for j in range(4):
        data.ctrl[j] = act
    mujoco.mj_step(model, data)
top_z_released = data.xpos[2, 2]
recovery = (top_z_released - top_z_contracted) * 1000
print(f"Released: top_plate z = {top_z_released:.4f}m  (spring-back = {recovery:.1f}mm)")
print(f"\nElastic recovery: {recovery/max(compression,0.001)*100:.0f}%")

# ===================== Record Video =====================
print("\n--- Recording video ---")
model2 = mujoco.MjModel.from_xml_path(xml_path)
data2 = mujoco.MjData(model2)
mujoco.mj_forward(model2, data2)

r_front = mujoco.Renderer(model2, 500, 500)
r_top = mujoco.Renderer(model2, 250, 500)

framerate = 60
frames = []
total_steps = 12000  # 12 seconds

# Schedule: settle 1s → contract 2s → hold 3s → release 2s → recover 4s
t0 = time.time()
for i in range(total_steps):
    t = i * 0.001
    if t < 1.0:
        act = 0.0
    elif t < 3.0:
        act = min(1.0, (t - 1.0) / 2.0)
    elif t < 6.0:
        act = 1.0
    elif t < 8.0:
        act = max(0.0, 1.0 - (t - 6.0) / 2.0)
    else:
        act = 0.0

    for j in range(4):
        data2.ctrl[j] = act
    mujoco.mj_step(model2, data2)

    if len(frames) < data2.time * framerate:
        r_front.update_scene(data2, camera='front')
        px_front = r_front.render().copy()
        r_top.update_scene(data2, camera='top')
        px_top = r_top.render().copy()
        # Stack: front view on top, top view on bottom
        combined = np.concatenate([px_front, px_top], axis=0)
        frames.append(combined)

    if (i + 1) % 3000 == 0:
        tz = data2.xpos[2, 2]
        print(f"  t={t:.1f}s | ctrl={act:.2f} | top_z={tz:.4f}m")

elapsed = time.time() - t0
print(f"Render done in {elapsed:.1f}s, {len(frames)} frames")

vid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worm_segment_v2.mp4")
media.write_video(vid_path, frames, fps=framerate)
print(f"Video: {vid_path}")
r_front.close()
r_top.close()

"""
5-Segment Worm Robot V2 — Independent Plates with Per-DOF Control
=================================================================
Architecture:
  - 6 independent plates (worldbody children with named slide+hinge joints)
  - Per-DOF stiffness: yaw/lateral stiff, axial/height free
  - 8 spring steel strips per segment (cable plugin)
  - 4 axial muscle tendons per segment (Hill actuators)
  - Bulge ring per segment with passive stiffness (Phase 1)

V2 key difference from V1:
  - Each plate is independent worldbody child (NOT nested kinematic chain)
  - Named joints allow per-DOF stiffness:
    * Y (axial): free (cables + muscles handle compression)
    * Z (height): free (contracted segments can lift independently!)
    * X (lateral): stiff (prevents drift)
    * Yaw: stiff (prevents rotation drift)
    * Pitch/Roll: moderate (allows natural bending)
"""
import mujoco
import numpy as np
import mediapy as media
import time
import math
import os

# ======================== Parameters ========================
num_segments = 5
seg_length = 0.065         # axial length per segment
plate_radius = 0.022       # plate radius (disk)
plate_thickness = 0.003
strip_circle_r = 0.017     # strips anchored INSIDE plate
num_strips = 8
num_verts = 7              # vertices per strip
strip_r = 0.002            # capsule radius
bow_amount = 0.007         # outward bow at midpoint

bend_stiff = 1e8
twist_stiff = 4e7          # same as V1
muscle_force = 38          # N per muscle tendon (V2: tuned for independent plate arch.)
ground_friction = 1.5

# V2 per-DOF stiffness for plates (replaces kinematic chain)
plate_stiff_x = 500.0     # N/m — lateral anti-drift (increased for weld combo)
plate_stiff_y = 0.0        # N/m — axial (free, relative coupling via equality)
plate_stiff_z = 20.0       # N/m — moderate Z restore (prevents cumulative drift)
plate_stiff_pitch = 10.0   # Nm/rad — vertical bending (same as V1)
plate_stiff_roll = 5.0     # Nm/rad — moderate
plate_stiff_yaw = 100.0    # Nm/rad — anti-yaw drift (increased for weld combo)

# V2 inter-plate soft weld for general structural coupling
plate_weld_solref = "0.010 1.0"   # soft weld — balanced axial spring-back + height independence
plate_weld_solimp = "0.90 0.95 0.002 0.5 2"

plate_damp_x = 0.5
plate_damp_y = 2.0         # higher for axial stability
plate_damp_z = 0.5         # moderate damping — allows faster settling back down
plate_damp_pitch = 0.5
plate_damp_roll = 0.5
plate_damp_yaw = 0.5

# Radial expansion — Phase 1: passive only
num_bulge = 8
bulge_sphere_r = 0.005
bulge_circle_r = 0.020
bulge_range_min = -0.003
bulge_range_max = 0.015
bulge_passive_stiffness = 0.5

# Derived
num_plates = num_segments + 1
z_center = plate_radius + 0.001
total_length = num_segments * seg_length

# Copper pillar params
copper_r = 0.002
copper_len = 0.004
num_copper_per_plate = 4

# ======================== Helpers ========================
def strip_verts(angle, seg_idx):
    """Generate vertices for one strip in one segment (world Y axis)."""
    y_start = seg_idx * seg_length
    verts = []
    for k in range(num_verts):
        t = k / (num_verts - 1)
        y = y_start + t * seg_length
        bow_r = bow_amount * 4.0 * t * (1.0 - t)
        r = strip_circle_r + bow_r
        x = r * math.cos(angle)
        z = z_center + r * math.sin(angle)
        verts.append(f"{x:.6f} {y:.6f} {z:.6f}")
    return "  ".join(verts)

strip_angles = [2 * math.pi * i / num_strips for i in range(num_strips)]
copper_angles = [2 * math.pi * i / num_copper_per_plate for i in range(num_copper_per_plate)]
bulge_angles = [2 * math.pi * i / num_bulge for i in range(num_bulge)]

# ======================== XML Generation ========================

def make_plate_content(p):
    """Generate geoms, sites, copper pillars for one plate (local frame)."""
    content = ""
    content += f'      <geom type="cylinder" size="{plate_radius} {plate_thickness}" pos="0 0 {z_center}"\n'
    content += f'            euler="90 0 0" rgba="0.5 0.5 0.5 0.85" mass="0.02" friction="0.3" contype="2" conaffinity="2"/>\n'

    if p > 0 and p < num_plates - 1:
        for ca in copper_angles:
            cx = (plate_radius - 0.003) * math.cos(ca)
            cz = z_center + (plate_radius - 0.003) * math.sin(ca)
            content += f'      <geom type="cylinder" fromto="{cx:.5f} {-copper_len:.5f} {cz:.5f} {cx:.5f} {copper_len:.5f} {cz:.5f}" size="{copper_r}" rgba="0.85 0.55 0.2 1" mass="0.002" contype="2" conaffinity="2"/>\n'
    elif p == 0:
        for ca in copper_angles:
            cx = (plate_radius - 0.003) * math.cos(ca)
            cz = z_center + (plate_radius - 0.003) * math.sin(ca)
            content += f'      <geom type="cylinder" fromto="{cx:.5f} 0 {cz:.5f} {cx:.5f} {copper_len:.5f} {cz:.5f}" size="{copper_r}" rgba="0.85 0.55 0.2 1" mass="0.002" contype="2" conaffinity="2"/>\n'
    else:
        for ca in copper_angles:
            cx = (plate_radius - 0.003) * math.cos(ca)
            cz = z_center + (plate_radius - 0.003) * math.sin(ca)
            content += f'      <geom type="cylinder" fromto="{cx:.5f} {-copper_len:.5f} {cz:.5f} {cx:.5f} 0 {cz:.5f}" size="{copper_r}" rgba="0.85 0.55 0.2 1" mass="0.002" contype="2" conaffinity="2"/>\n'

    for si in range(4):
        sa = strip_angles[si * 2]
        sx = plate_radius * 0.6 * math.cos(sa)
        sz = z_center + plate_radius * 0.6 * math.sin(sa)
        content += f'      <site name="p{p}_s{si}" pos="{sx:.5f} 0 {sz:.5f}" size="0.0015" rgba="1 0.2 0.2 1"/>\n'

    return content

def make_bulge_bodies(seg):
    """Generate radial expansion ring nested inside plate{seg} body."""
    content = ""
    for bi in range(num_bulge):
        a = bulge_angles[bi]
        bx = bulge_circle_r * math.cos(a)
        bz = z_center + bulge_circle_r * math.sin(a)
        by = seg_length / 2
        rx = math.cos(a)
        rz = math.sin(a)
        name = f"blg_s{seg}_b{bi}"
        content += f'      <body name="{name}" pos="{bx:.5f} {by:.5f} {bz:.5f}">\n'
        content += f'        <joint name="{name}_r" type="slide" axis="{rx:.5f} 0 {rz:.5f}" '
        content += f'stiffness="{bulge_passive_stiffness}" damping="0.05" range="{bulge_range_min} {bulge_range_max}"/>\n'
        content += f'        <geom type="sphere" size="{bulge_sphere_r}" '
        content += f'rgba="0.6 0.58 0.55 0.5" friction="{ground_friction}" mass="0.001" contype="1" conaffinity="1"/>\n'
        content += f'      </body>\n'
    return content

# --- V2: Flat plate list with named per-DOF joints ---
plates_xml = ""
for p in range(num_plates):
    y_world = p * seg_length
    plates_xml += f'    <body name="plate{p}" pos="0 {y_world:.5f} 0">\n'
    # Named joints: slides first (translation), then hinges (rotation)
    plates_xml += f'      <joint name="p{p}_x" type="slide" axis="1 0 0" stiffness="{plate_stiff_x}" damping="{plate_damp_x}"/>\n'
    plates_xml += f'      <joint name="p{p}_y" type="slide" axis="0 1 0" stiffness="{plate_stiff_y}" damping="{plate_damp_y}"/>\n'
    plates_xml += f'      <joint name="p{p}_z" type="slide" axis="0 0 1" stiffness="{plate_stiff_z}" damping="{plate_damp_z}"/>\n'
    plates_xml += f'      <joint name="p{p}_pitch" type="hinge" axis="1 0 0" stiffness="{plate_stiff_pitch}" damping="{plate_damp_pitch}"/>\n'
    plates_xml += f'      <joint name="p{p}_roll" type="hinge" axis="0 1 0" stiffness="{plate_stiff_roll}" damping="{plate_damp_roll}"/>\n'
    plates_xml += f'      <joint name="p{p}_yaw" type="hinge" axis="0 0 1" stiffness="{plate_stiff_yaw}" damping="{plate_damp_yaw}"/>\n'
    plates_xml += make_plate_content(p)
    if p < num_plates - 1:
        plates_xml += make_bulge_bodies(p)
    plates_xml += f'    </body>\n'

# --- Cable strips (unchanged) ---
cables_xml = ""
for seg in range(num_segments):
    for si, angle in enumerate(strip_angles):
        v = strip_verts(angle, seg)
        prefix = f"c{seg}s{si}"
        cables_xml += f"""
    <body>
      <freejoint/>
      <composite type="cable" prefix="{prefix}" initial="none" vertex="{v}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="{bend_stiff}"/>
          <config key="twist" value="{twist_stiff}"/>
          <config key="vmax" value="2"/>
        </plugin>
        <joint armature="0.01" damping="0.5" kind="main"/>
        <geom type="capsule" size="{strip_r}" density="3500" material="MatSteel" friction="{ground_friction}" contype="1" conaffinity="1"/>
      </composite>
    </body>"""

# --- Weld constraints: cable ends to plates ---
welds_xml = ""
for seg in range(num_segments):
    p_start = seg
    p_end = seg + 1
    for si in range(num_strips):
        prefix = f"c{seg}s{si}"
        welds_xml += f'    <weld body1="plate{p_start}" body2="{prefix}B_first" solref="0.005 1"/>\n'
        welds_xml += f'    <weld body1="plate{p_end}" body2="{prefix}B_last" solref="0.005 1"/>\n'

# --- Contact exclusions ---
excludes_xml = ""
for seg in range(num_segments):
    excludes_xml += f'    <exclude body1="plate{seg}" body2="plate{seg+1}"/>\n'

# --- V2: Inter-plate soft welds with relpose ---
# Provides general structural coupling (axial spring-back, height coupling, etc.)
# Named joint stiffnesses add per-DOF control on top (yaw/lateral anti-drift)
plate_welds_xml = ""
for seg in range(num_segments):
    p = seg
    plate_welds_xml += (
        f'    <weld body1="plate{p}" body2="plate{p+1}"\n'
        f'          relpose="0 {seg_length:.5f} 0  1 0 0 0"\n'
        f'          solref="{plate_weld_solref}" solimp="{plate_weld_solimp}"/>\n'
    )

# --- Axial tendons and muscles (unchanged) ---
tendons_xml = ""
muscles_xml = ""
for seg in range(num_segments):
    p_start = seg
    p_end = seg + 1
    for mi in range(4):
        tname = f"tendon_seg{seg}_m{mi}"
        mname = f"muscle_seg{seg}_m{mi}"
        tendons_xml += f"""    <spatial name="{tname}" width="0.001" rgba="0.9 0.15 0.15 1">
      <site site="p{p_end}_s{mi}"/>
      <site site="p{p_start}_s{mi}"/>
    </spatial>
"""
        muscles_xml += f'    <muscle class="muscle" name="{mname}" tendon="{tname}" force="{muscle_force}" lengthrange="0.03 0.08"/>\n'

# --- Sensors ---
sensors_xml = ""
for p in range(num_plates):
    sensors_xml += f'    <framepos name="plate{p}_pos" objtype="body" objname="plate{p}"/>\n'
for seg in range(num_segments):
    sensors_xml += f'    <tendonpos name="len_seg{seg}" tendon="tendon_seg{seg}_m0"/>\n'

# ======================== Full XML ========================
full_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="worm_5seg_v2">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <size memory="2G"/>
  <compiler autolimits="true">
    <lengthrange useexisting="true"/>
  </compiler>
  <option timestep="0.001" gravity="0 0 -9.81" solver="Newton" iterations="100" tolerance="1e-8" integrator="Euler"/>

  <statistic center="0 {total_length/2:.3f} {z_center}" extent="{total_length*1.2:.3f}"/>
  <visual>
    <global offwidth="960" offheight="540" elevation="-25"/>
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

    <geom type="plane" size="2 2 0.01" rgba="0.93 0.93 0.93 1" condim="3" name="floor"
          friction="{ground_friction}" material="MatPlane" contype="3" conaffinity="3"/>

    <!-- V2: Independent plate bodies with named per-DOF joints -->
{plates_xml}
    <!-- Cable strips -->
{cables_xml}
  </worldbody>

  <contact>
{excludes_xml}
  </contact>

  <equality>
    <!-- Cable end welds to plates -->
{welds_xml}
    <!-- V2: Inter-plate soft welds (structural coupling) -->
{plate_welds_xml}
  </equality>

  <tendon>
{tendons_xml}
  </tendon>

  <actuator>
{muscles_xml}
  </actuator>

  <sensor>
{sensors_xml}
  </sensor>

  <asset>
    <texture builtin="checker" height="512" mark="cross" markrgb=".8 .8 .8" name="texplane"
             rgb1=".3 .3 .3" rgb2="0.2 0.2 0.2" type="2d" width="512"/>
    <material name="MatPlane" reflectance="0" texrepeat="10 10" texture="texplane" texuniform="true"/>
    <material name="MatSteel" rgba="0.82 0.78 0.72 1" specular="0.6" shininess="0.8" reflectance="0.1"/>
  </asset>
</mujoco>
"""

# ======================== Save & Validate ========================
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
bin_dir = os.path.join(project_root, "bin", "v2")
record_vid_dir = os.path.join(project_root, "record", "v2", "videos")
record_plot_dir = os.path.join(project_root, "record", "v2", "plots")
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(record_vid_dir, exist_ok=True)
os.makedirs(record_plot_dir, exist_ok=True)

xml_path = os.path.join(bin_dir, "worm_5seg_v2.xml")
with open(xml_path, "w") as f:
    f.write(full_xml)
print(f"XML: {xml_path}")

print("Loading model...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

strip_color = np.array([0.82, 0.78, 0.72, 1.0], dtype=np.float32)
steel_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatSteel")
recolored = 0
for gi in range(model.ngeom):
    if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        model.geom_matid[gi] = steel_mat_id
        model.geom_rgba[gi] = strip_color
        recolored += 1
print(f"Recolored {recolored} cable geoms to MatSteel")

mujoco.mj_forward(model, data)

print(f"Bodies: {model.nbody}, DOF: {model.nv}, Actuators: {model.nu}")
print(f"Total mass: {np.sum(model.body_mass):.4f} kg")
print(f"V2: {num_plates} plates × 6 DOF (named joints), cable-coupled")

plate_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
print(f"Plate body IDs: {plate_ids}")

print("\nInitial plate positions:")
for p in range(num_plates):
    pos = data.xpos[plate_ids[p]]
    print(f"  Plate {p}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")

for i in range(1000):
    mujoco.mj_step(model, data)
print(f"\nAfter settling (1s):")
for p in range(num_plates):
    pos = data.xpos[plate_ids[p]]
    print(f"  Plate {p}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")

if np.any(np.isnan(data.qpos)):
    print("WARNING: NaN detected after settling!")
else:
    print("OK: No NaN after settling.")

# Scene-level visual: flat strip boxes
strip_rgba_scene = np.array([0.92, 0.90, 0.86, 1.0], dtype=np.float32)

def reshape_to_flat_strips(scene, plate_y_positions=None):
    """Replace round plugin cable tubes with flat strip boxes."""
    proximity_margin = 0.006
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if g.rgba[2] > 0.5 and g.rgba[0] < 0.3 and g.rgba[1] < 0.3:
            pos = np.array(g.pos, dtype=np.float64)

            if plate_y_positions is not None:
                radial_dist = math.sqrt(pos[0]**2 + (pos[2] - z_center)**2)
                if radial_dist < plate_radius:
                    hide_seg = False
                    for py in plate_y_positions:
                        if abs(pos[1] - py) < proximity_margin:
                            hide_seg = True
                            break
                    if hide_seg:
                        g.rgba[3] = 0.0
                        continue

            mat = np.array(g.mat, dtype=np.float64).reshape(3, 3)
            cable_dir = mat[:, 2].copy()
            radial = np.array([pos[0], 0.0, pos[2] - z_center])
            r_norm = np.linalg.norm(radial)
            if r_norm < 1e-6:
                radial = np.array([1.0, 0.0, 0.0])
            else:
                radial /= r_norm

            tangent = np.cross(cable_dir, radial)
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                tangent = np.cross(cable_dir, np.array([0, 1, 0]))
                t_norm = np.linalg.norm(tangent)
                if t_norm < 1e-6:
                    tangent = np.array([1, 0, 0])
                else:
                    tangent /= t_norm
            else:
                tangent /= t_norm

            radial = np.cross(tangent, cable_dir)
            r2_norm = np.linalg.norm(radial)
            if r2_norm > 1e-6:
                radial /= r2_norm

            new_mat = np.column_stack([tangent, radial, cable_dir])
            half_len = float(g.size[2])
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size[0] = 0.003
            g.size[1] = 0.0004
            g.size[2] = half_len
            g.mat[:] = new_mat.astype(np.float32)
            g.rgba[:] = strip_rgba_scene

        elif g.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            g.rgba[3] = 0.0

# ======================== Peristaltic Wave ========================
wave_period = 1500.0

def peristaltic_signal(t_ms, seg_idx, num_segs):
    """Sinusoidal peristaltic wave: prograde (tail->head), 50% duty."""
    phase = 2.0 * math.pi * (t_ms / wave_period - (num_segs - 1 - seg_idx) / num_segs)
    raw = math.sin(phase)
    if raw <= 0.0:
        return 0.0
    return raw

# ======================== Record Video ========================
print("\n--- Recording V2 crawling video ---")
model2 = mujoco.MjModel.from_xml_path(xml_path)
steel_mat_id2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_MATERIAL, "MatSteel")
for gi in range(model2.ngeom):
    if model2.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        model2.geom_matid[gi] = steel_mat_id2
        model2.geom_rgba[gi] = strip_color
data2 = mujoco.MjData(model2)
mujoco.mj_forward(model2, data2)

plate_ids2 = [mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
head_id = plate_ids2[0]
tail_id = plate_ids2[-1]

r_side = mujoco.Renderer(model2, 400, 900)
r_top = mujoco.Renderer(model2, 200, 900)

framerate = 60
frames = []
total_steps = 15000

cam_side = mujoco.MjvCamera()
cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
cam_side.distance = 0.35
cam_side.elevation = -15
cam_side.azimuth = 90

cam_top = mujoco.MjvCamera()
cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
cam_top.distance = 0.40
cam_top.elevation = -90
cam_top.azimuth = 0

record_interval = 10
time_history = []
plate_x_history = {p: [] for p in range(num_plates)}
plate_y_history = {p: [] for p in range(num_plates)}
plate_z_history = {p: [] for p in range(num_plates)}

settle_ms = 1000
t0 = time.time()

for i in range(total_steps):
    t_ms = i

    if t_ms % record_interval == 0:
        time_history.append(t_ms / 1000.0)
        for p in range(num_plates):
            plate_x_history[p].append(data2.xpos[plate_ids2[p], 0])
            plate_y_history[p].append(data2.xpos[plate_ids2[p], 1])
            plate_z_history[p].append(data2.xpos[plate_ids2[p], 2])

    if t_ms < settle_ms:
        for j in range(model2.nu):
            data2.ctrl[j] = 0.0
    else:
        t_active = t_ms - settle_ms
        for seg in range(num_segments):
            act = peristaltic_signal(t_active, seg, num_segments)
            for mi in range(4):
                data2.ctrl[seg * 4 + mi] = act

    mujoco.mj_step(model2, data2)

    if i == 2000 and np.any(np.isnan(data2.qpos)):
        print("FATAL: NaN at t=2s — aborting.")
        break

    if len(frames) < data2.time * framerate:
        head_pos = data2.xpos[head_id]
        tail_pos = data2.xpos[tail_id]
        center = (head_pos + tail_pos) / 2.0
        cam_side.lookat[:] = center
        cam_top.lookat[:] = center

        plate_y_pos = [data2.xpos[plate_ids2[p], 1] for p in range(num_plates)]
        r_side.update_scene(data2, cam_side)
        reshape_to_flat_strips(r_side.scene, plate_y_pos)
        px_side = r_side.render().copy()
        r_top.update_scene(data2, cam_top)
        reshape_to_flat_strips(r_top.scene, plate_y_pos)
        px_top = r_top.render().copy()
        frames.append(np.concatenate([px_side, px_top], axis=0))

    if (i + 1) % 3000 == 0:
        head_x = data2.xpos[head_id, 0]
        head_y = data2.xpos[head_id, 1]
        elapsed = time.time() - t0
        z_vals = [data2.xpos[plate_ids2[p], 2] for p in range(num_plates)]
        z_range = max(z_vals) - min(z_vals)
        print(f"  t={t_ms/1000:.1f}s | head x={head_x:+.4f} y={head_y:+.4f} | Z range={z_range*1000:.2f}mm | elapsed={elapsed:.1f}s")

elapsed = time.time() - t0
print(f"Simulation done in {elapsed:.1f}s, {len(frames)} frames")

head_y_final = data2.xpos[head_id, 1]
head_x_final = data2.xpos[head_id, 0]
fwd_dist = head_y_final * 1000
lat_drift = head_x_final * 1000
print(f"Forward: {fwd_dist:.1f}mm, Lateral drift: {lat_drift:.1f}mm")

z_final = [data2.xpos[plate_ids2[p], 2] for p in range(num_plates)]
print(f"Final Z range: {(max(z_final)-min(z_final))*1000:.2f}mm")

if len(frames) > 0:
    vid_path = os.path.join(record_vid_dir, "worm_5seg_v2.mp4")
    media.write_video(vid_path, frames, fps=framerate)
    print(f"Video: {vid_path}")
r_side.close()
r_top.close()

# ======================== Plot ========================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t_arr = np.array(time_history)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
plate_labels = [f'Plate {p}' for p in range(num_plates)]

fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

ax = axes[0]
for p in range(num_plates):
    y_mm = np.array(plate_y_history[p]) * 1000
    ax.plot(t_arr, y_mm, color=colors[p], linewidth=1.2, label=plate_labels[p])
ax.set_ylabel('Y position (mm)')
ax.set_title('V2: Plate Y (forward) vs Time')
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
ax.axvline(x=settle_ms/1000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

ax = axes[1]
for p in range(num_plates):
    x_mm = np.array(plate_x_history[p]) * 1000
    ax.plot(t_arr, x_mm, color=colors[p], linewidth=1.2, label=plate_labels[p])
ax.set_ylabel('X position (mm)')
ax.set_title('V2: Plate X (lateral drift) vs Time')
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
ax.axvline(x=settle_ms/1000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

ax = axes[2]
for p in range(num_plates):
    z_mm = np.array(plate_z_history[p]) * 1000
    ax.plot(t_arr, z_mm, color=colors[p], linewidth=1.2, label=plate_labels[p])
ax.set_ylabel('Z position (mm)')
ax.set_title('V2: Plate Z (height) — contracted segments should show independent lift')
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
ax.axvline(x=settle_ms/1000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

ax = axes[3]
seg_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for seg in range(num_segments):
    y_start = np.array(plate_y_history[seg])
    y_end = np.array(plate_y_history[seg + 1])
    gap_mm = (y_end - y_start) * 1000
    ax.plot(t_arr, gap_mm, color=seg_colors[seg], linewidth=1.0, label=f'Seg {seg}')
ax.set_ylabel('Segment length (mm)')
ax.set_xlabel('Time (s)')
ax.set_title('V2: Segment axial length vs Time')
ax.legend(loc='upper left', fontsize=8, ncol=5)
ax.grid(True, alpha=0.3)
ax.axvline(x=settle_ms/1000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.axhline(y=seg_length*1000, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
ax.text(0.1, seg_length*1000 + 0.5, f'rest={seg_length*1000:.0f}mm', fontsize=7, color='gray')

plt.tight_layout()
plot_path = os.path.join(record_plot_dir, "worm_5seg_v2_motion.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Motion plot: {plot_path}")

"""
5-Segment Worm Robot V3 — Antagonistic Muscles (Axial + Ring)
=============================================================
Based on V2 independent-plate architecture, adding:
  - Ring muscles (1 per segment): 8 connector balls + closed-loop spatial tendon
  - implicitfast integrator (eliminates Euler vibration)
  - Cable damping=0.12 (matched to reference paper)
  - Peristaltic wave: axial + ring in anti-phase

Muscle coordination:
  - Axial contract (段变短变胖) → ground anchor
  - Ring contract  (段变细) → slide forward
  - Anti-phase wave: when axial ON → ring OFF, and vice versa

坐标系: Y=前进, Z=上, X=侧向
"""
import mujoco
import numpy as np
import mediapy as media
import time
import math
import os

# ======================== Parameters ========================
num_segments = 5
seg_length = 0.065
plate_radius = 0.022
plate_geom_r = 0.022       # visual plate (>= strip_circle_r, strips inside)
plate_thickness = 0.003
strip_circle_r = 0.017
num_strips = 8
num_verts = 7
strip_r = 0.002
bow_amount = 0.007

bend_stiff = 1e8
twist_stiff = 4e7
axial_muscle_force = 30
ring_muscle_force = 10
ground_friction = 1.5

# Plate joint parameters (from V2, with minor V3 adjustments)
plate_stiff_x = 500.0
plate_stiff_y = 0.0
plate_stiff_z = 20.0
plate_stiff_pitch = 5.0     # reduced from V2's 10 for ring muscle bending
plate_stiff_roll = 5.0
plate_stiff_yaw = 100.0

plate_damp_x = 0.5
plate_damp_y = 1.5
plate_damp_z = 0.5
plate_damp_pitch = 0.5
plate_damp_roll = 0.5
plate_damp_yaw = 0.5

plate_weld_solref = "0.010 1.0"
plate_weld_solimp = "0.90 0.95 0.002 0.5 2"

# Derived
num_plates = num_segments + 1
z_center = plate_radius + 0.001
total_length = num_segments * seg_length
strip_angles = [2 * math.pi * i / num_strips for i in range(num_strips)]
mid_vert_idx = num_verts // 2  # index 3

# ======================== Helpers ========================
def strip_verts(angle, seg_idx):
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

# ======================== XML Generation ========================

# --- Plates (V2 style: independent, 6 DOF each) ---
plates_xml = ""
for p in range(num_plates):
    y_world = p * seg_length
    plates_xml += f'    <body name="plate{p}" pos="0 {y_world:.5f} 0">\n'
    plates_xml += f'      <joint name="p{p}_x" type="slide" axis="1 0 0" stiffness="{plate_stiff_x}" damping="{plate_damp_x}"/>\n'
    plates_xml += f'      <joint name="p{p}_y" type="slide" axis="0 1 0" stiffness="{plate_stiff_y}" damping="{plate_damp_y}"/>\n'
    plates_xml += f'      <joint name="p{p}_z" type="slide" axis="0 0 1" stiffness="{plate_stiff_z}" damping="{plate_damp_z}"/>\n'
    plates_xml += f'      <joint name="p{p}_pitch" type="hinge" axis="1 0 0" stiffness="{plate_stiff_pitch}" damping="{plate_damp_pitch}"/>\n'
    plates_xml += f'      <joint name="p{p}_roll" type="hinge" axis="0 1 0" stiffness="{plate_stiff_roll}" damping="{plate_damp_roll}"/>\n'
    plates_xml += f'      <joint name="p{p}_yaw" type="hinge" axis="0 0 1" stiffness="{plate_stiff_yaw}" damping="{plate_damp_yaw}"/>\n'
    plates_xml += f'      <geom type="cylinder" size="{plate_geom_r} {plate_thickness}" pos="0 0 {z_center}"\n'
    plates_xml += f'            euler="90 0 0" rgba="0.45 0.45 0.50 0.9" mass="0.02" contype="2" conaffinity="2"/>\n'
    # Axial tendon sites (4 per plate)
    for mi in range(4):
        sa = strip_angles[mi * 2]
        sx = strip_circle_r * 0.7 * math.cos(sa)
        sz = z_center + strip_circle_r * 0.7 * math.sin(sa)
        plates_xml += f'      <site name="p{p}_s{mi}" pos="{sx:.5f} 0 {sz:.5f}" size="0.0015" rgba="1 0.2 0.2 1"/>\n'
    plates_xml += f'    </body>\n'

# --- Cable strips ---
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
        <joint armature="0.01" damping="0.12" kind="main"/>
        <geom type="capsule" size="{strip_r}" density="3500"
              friction="{ground_friction}" contype="1" conaffinity="1"
              material="MatSteel"/>
      </composite>
    </body>"""

# --- Ring connector balls (8 per segment) ---
ring_balls_xml = ""
for seg in range(num_segments):
    mid_y = seg * seg_length + seg_length / 2
    for si, angle in enumerate(strip_angles):
        mid_r = strip_circle_r + bow_amount
        mx = mid_r * math.cos(angle)
        mz = z_center + mid_r * math.sin(angle)
        ring_balls_xml += f'    <body name="rb{seg}_{si}" pos="{mx:.5f} {mid_y:.5f} {mz:.5f}">\n'
        ring_balls_xml += f'      <freejoint/>\n'
        ring_balls_xml += f'      <geom type="sphere" size="0.002" mass="0.0005" rgba="0.9 0.2 0.2 0" contype="0" conaffinity="0"/>\n'
        ring_balls_xml += f'      <site name="rs{seg}_{si}" pos="0 0 0" size="0.0018" rgba="0 0 0 0"/>\n'
        ring_balls_xml += f'    </body>\n'

# --- Contact exclusions ---
excludes_xml = ""
for seg in range(num_segments):
    excludes_xml += f'    <exclude body1="plate{seg}" body2="plate{seg+1}"/>\n'

# --- Cable weld constraints ---
cable_welds_xml = ""
for seg in range(num_segments):
    for si in range(num_strips):
        prefix = f"c{seg}s{si}"
        cable_welds_xml += f'    <weld body1="plate{seg}" body2="{prefix}B_first" solref="0.002 1"/>\n'
        cable_welds_xml += f'    <weld body1="plate{seg+1}" body2="{prefix}B_last" solref="0.002 1"/>\n'

# --- Inter-plate soft welds (V2 structural coupling) ---
plate_welds_xml = ""
for seg in range(num_segments):
    plate_welds_xml += (
        f'    <weld body1="plate{seg}" body2="plate{seg+1}"\n'
        f'          relpose="0 {seg_length:.5f} 0  1 0 0 0"\n'
        f'          solref="{plate_weld_solref}" solimp="{plate_weld_solimp}"/>\n'
    )

# --- Ring ball → cable midpoint connects ---
ring_connects_xml = ""
for seg in range(num_segments):
    for si in range(num_strips):
        prefix = f"c{seg}s{si}"
        ring_connects_xml += f'    <connect body1="rb{seg}_{si}" body2="{prefix}B_{mid_vert_idx}" anchor="0 0 0" solref="0.003 1"/>\n'

# --- Axial tendons & muscles (4 per segment) ---
axial_tendons_xml = ""
axial_muscles_xml = ""
for seg in range(num_segments):
    p_start = seg
    p_end = seg + 1
    for mi in range(4):
        tname = f"at{seg}_{mi}"
        mname = f"am{seg}_{mi}"
        axial_tendons_xml += f'    <spatial name="{tname}" width="0.0012" rgba="0.15 0.75 0.15 1">\n'
        axial_tendons_xml += f'      <site site="p{p_end}_s{mi}"/>\n'
        axial_tendons_xml += f'      <site site="p{p_start}_s{mi}"/>\n'
        axial_tendons_xml += f'    </spatial>\n'
        axial_muscles_xml += f'    <muscle class="muscle" name="{mname}" tendon="{tname}" force="{axial_muscle_force}" lengthrange="0.03 0.08"/>\n'

# --- Ring tendons & muscles (1 per segment) ---
ring_tendons_xml = ""
ring_muscles_xml = ""
for seg in range(num_segments):
    tname = f"rt{seg}"
    ring_tendons_xml += f'    <spatial name="{tname}" width="0.0015" rgba="0.85 0.2 0.2 0">\n'
    for si in range(num_strips):
        ring_tendons_xml += f'      <site site="rs{seg}_{si}"/>\n'
    ring_tendons_xml += f'      <site site="rs{seg}_0"/>\n'  # close loop
    ring_tendons_xml += f'    </spatial>\n'
    ring_muscles_xml += f'    <muscle class="muscle" name="rm{seg}" tendon="{tname}" force="{ring_muscle_force}" lengthrange="0.05 0.20"/>\n'

# --- Sensors ---
sensors_xml = ""
for p in range(num_plates):
    sensors_xml += f'    <framepos name="plate{p}_pos" objtype="body" objname="plate{p}"/>\n'

# ======================== Full XML ========================
full_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="worm_5seg_v3">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <size memory="4G"/>
  <compiler autolimits="true">
    <lengthrange useexisting="true" tolrange="0.5"/>
  </compiler>
  <option timestep="0.0005" gravity="0 0 -9.81" solver="Newton" iterations="200" tolerance="1e-8" integrator="implicitfast"/>

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

  <asset>
    <texture builtin="checker" height="512" mark="cross" markrgb=".8 .8 .8" name="texplane"
             rgb1=".3 .3 .3" rgb2="0.2 0.2 0.2" type="2d" width="512"/>
    <material name="MatPlane" reflectance="0" texrepeat="10 10" texture="texplane" texuniform="true"/>
    <material name="MatSteel" rgba="0.82 0.78 0.72 1" specular="0.6" shininess="0.8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <light castshadow="false" diffuse=".3 .3 .3" dir="0 0 -1" pos="0 0 1.5"/>

    <geom type="plane" size="2 2 0.01" material="MatPlane" condim="3"
          friction="{ground_friction}" contype="3" conaffinity="3"/>

{plates_xml}
{cables_xml}

{ring_balls_xml}
  </worldbody>

  <contact>
{excludes_xml}
  </contact>

  <equality>
{cable_welds_xml}
{plate_welds_xml}
{ring_connects_xml}
  </equality>

  <tendon>
{axial_tendons_xml}
{ring_tendons_xml}
  </tendon>

  <actuator>
{axial_muscles_xml}
{ring_muscles_xml}
  </actuator>

  <sensor>
{sensors_xml}
  </sensor>
</mujoco>
"""

# ======================== Save & Validate ========================
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
bin_dir = os.path.join(project_root, "bin", "v3")
vid_dir = os.path.join(project_root, "record", "v3", "videos")
plot_dir = os.path.join(project_root, "record", "v3", "plots")
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

xml_path = os.path.join(bin_dir, "worm_5seg_v3.xml")
with open(xml_path, "w") as f:
    f.write(full_xml)
print(f"XML: {xml_path}")

print("Loading model...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

num_axial = num_segments * 4
num_ring = num_segments
print(f"Bodies: {model.nbody}, DOF: {model.nv}, Actuators: {model.nu} ({num_axial} axial + {num_ring} ring)")
print(f"Total mass: {np.sum(model.body_mass):.4f} kg")

plate_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]

# Settle
print("\nSettling 2s...")
for i in range(4000):
    mujoco.mj_step(model, data)
if np.any(np.isnan(data.qpos)):
    print("FATAL: NaN after settling!")
    exit(1)
print("Plate positions after settling:")
for p in range(num_plates):
    pos = data.xpos[plate_ids[p]]
    print(f"  Plate {p}: y={pos[1]*1000:.1f}mm z={pos[2]*1000:.1f}mm")

# ======================== Peristaltic Wave ========================
wave_period = 1500.0  # ms

def peristaltic_axial(t_ms, seg_idx):
    """Axial muscle signal: positive half of sine wave (prograde tail→head)."""
    phase = 2.0 * math.pi * (t_ms / wave_period - (num_segments - 1 - seg_idx) / num_segments)
    return max(0.0, math.sin(phase))

def peristaltic_ring(t_ms, seg_idx):
    """Ring muscle signal: anti-phase to axial (negative half of sine)."""
    phase = 2.0 * math.pi * (t_ms / wave_period - (num_segments - 1 - seg_idx) / num_segments)
    return max(0.0, -math.sin(phase))

# ======================== Flat Steel Strip Rendering ========================
strip_rgba_scene = np.array([0.92, 0.90, 0.86, 1.0], dtype=np.float32)

def reshape_to_flat_strips(scene):
    """Replace round cable capsules with flat strip boxes in scene."""
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        # Detect cable geoms: capsule type OR blue-dominant plugin tubes
        is_capsule = (g.type == mujoco.mjtGeom.mjGEOM_CAPSULE)
        is_blue_tube = (g.rgba[2] > 0.5 and g.rgba[0] < 0.4 and g.rgba[1] < 0.6)
        if not is_capsule and not is_blue_tube:
            continue
        if g.rgba[3] < 0.01:
            continue  # skip hidden geoms (diagonal strips)

        pos = np.array(g.pos, dtype=np.float64)

        # Compute flat strip orientation
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
        g.size[0] = 0.003      # width (tangential)
        g.size[1] = 0.0004     # thickness (radial)
        g.size[2] = half_len   # length (axial)
        g.mat[:] = new_mat.astype(np.float32)
        g.rgba[:] = strip_rgba_scene

# ======================== Record Video ========================
print("\n--- Recording V3 crawling video ---")
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# Recolor cable geoms to MatSteel; hide diagonal strips (si=1,3,5,7)
strip_color = np.array([0.82, 0.78, 0.72, 1.0], dtype=np.float32)
strip_hidden = np.array([0.82, 0.78, 0.72, 0.0], dtype=np.float32)
steel_mat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "MatSteel")
diagonal_tags = ["s1B", "s3B", "s5B", "s7B"]
recolored = 0
hidden = 0
for gi in range(m.ngeom):
    if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        m.geom_matid[gi] = steel_mat_id
        body_id = m.geom_bodyid[gi]
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and any(tag in body_name for tag in diagonal_tags):
            m.geom_rgba[gi] = strip_hidden
            hidden += 1
        else:
            m.geom_rgba[gi] = strip_color
            recolored += 1
print(f"Recolored {recolored} cable geoms to MatSteel, hidden {hidden} diagonal strips")

mujoco.mj_forward(m, d)

pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
head_id = pids[0]
tail_id = pids[-1]

r_side = mujoco.Renderer(m, 400, 900)
r_top = mujoco.Renderer(m, 200, 900)

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

framerate = 60
frames = []
dt = m.opt.timestep
sim_time = 15.0  # seconds
total_steps = int(sim_time / dt)
settle_time = 1.0  # 1s settle

# Tracking
record_interval = 20  # every 20 steps = 10ms
time_history = []
plate_y_history = {p: [] for p in range(num_plates)}

t0_wall = time.time()

for step in range(total_steps):
    t = step * dt
    t_ms = t * 1000.0

    # Record tracking data
    if step % record_interval == 0:
        time_history.append(t)
        for p in range(num_plates):
            plate_y_history[p].append(d.xpos[pids[p], 1])

    # Control
    d.ctrl[:] = 0
    if t >= settle_time:
        t_active_ms = (t - settle_time) * 1000.0
        for seg in range(num_segments):
            # Axial muscles (first 20 actuators: seg*4 + mi)
            ax_act = peristaltic_axial(t_active_ms, seg)
            for mi in range(4):
                d.ctrl[seg * 4 + mi] = ax_act
            # Ring muscles (last 5 actuators: num_axial + seg)
            d.ctrl[num_axial + seg] = peristaltic_ring(t_active_ms, seg)

    mujoco.mj_step(m, d)

    if step == 4000 and np.any(np.isnan(d.qpos)):
        print("FATAL: NaN at t=2s — aborting.")
        break

    # Capture frame
    if len(frames) < t * framerate:
        center = (d.xpos[head_id] + d.xpos[tail_id]) / 2.0
        cam_side.lookat[:] = center
        cam_top.lookat[:] = center
        r_side.update_scene(d, cam_side)
        reshape_to_flat_strips(r_side.scene)
        px_side = r_side.render().copy()
        r_top.update_scene(d, cam_top)
        reshape_to_flat_strips(r_top.scene)
        px_top = r_top.render().copy()
        frames.append(np.concatenate([px_side, px_top], axis=0))

    if (step + 1) % 6000 == 0:
        hy = d.xpos[head_id, 1]
        hx = d.xpos[head_id, 0]
        z_vals = [d.xpos[pids[p], 2] for p in range(num_plates)]
        z_range = max(z_vals) - min(z_vals)
        print(f"  t={t:.1f}s | head y={hy*1000:+.1f}mm x={hx*1000:+.1f}mm | Z range={z_range*1000:.1f}mm")

elapsed = time.time() - t0_wall
print(f"\nSim done in {elapsed:.1f}s, {len(frames)} frames")

# Results
head_y = d.xpos[head_id, 1] * 1000
head_x = d.xpos[head_id, 0] * 1000
z_vals = [d.xpos[pids[p], 2] for p in range(num_plates)]
z_range = (max(z_vals) - min(z_vals)) * 1000
print(f"Forward: {head_y:.1f}mm / {sim_time:.0f}s ({head_y/sim_time:.1f} mm/s)")
print(f"Lateral drift: {head_x:.1f}mm")
print(f"Z range: {z_range:.1f}mm")

r_side.close()
r_top.close()

vid_path = os.path.join(vid_dir, "worm_5seg_v3.mp4")
if len(frames) > 0:
    media.write_video(vid_path, frames, fps=framerate)
    print(f"Video: {vid_path}")

# Plot Y trajectories
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for p in range(num_plates):
        ax.plot(time_history, [y*1000 for y in plate_y_history[p]], label=f"Plate {p}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("V3 Worm — Plate Y Trajectories (Antagonistic Muscles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(plot_dir, "plate_y_trajectories.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {plot_path}")
except Exception as e:
    print(f"Plot skipped: {e}")

print("\nDone.")

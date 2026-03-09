"""
5-Segment Worm Robot V3 — Circular Locomotion (Discrete Gait State Machine)
===========================================================================
Implements the paper's discrete gait state machine for circular locomotion.
Zhan et al., IJRR 2019: "Planar locomotion of earthworm-like metameric robots"

Gait: {n₂=1, n₃=0, n₁=1 | nP=1}, TN=5
- θ_T = μ(n₂-n₃)θ₀ = 18° per period (5 transitions)
- State 0: relaxed (all muscles off)
- State 1: contracted (left + right muscles on, anchor)
- State 2: left-contracted (left muscle only → bends left)
- State 3: right-contracted (right muscle only → bends right)
- Retrograde peristalsis: s_i^{t_j} = s_{Mod[i-1+j·nP, TN]+1}^{t₀}

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

bend_stiff = 1e8            # original: structural stiffness for spring steel strips
twist_stiff = 2e6
axial_muscle_force = 50
ring_muscle_force = 10
ground_friction = 1.5

# Plate joint parameters (V3.4 values, reduced yaw stiffness for turning)
plate_stiff_x = 500.0
plate_stiff_y = 0.0
plate_stiff_z = 20.0
plate_stiff_pitch = 5.0
plate_stiff_roll = 5.0
plate_stiff_yaw = 0.0      # no world yaw spring — cable bend_stiff provides inter-plate angular stiffness

plate_damp_x = 0.5
plate_damp_y = 1.5
plate_damp_z = 0.5
plate_damp_pitch = 0.5
plate_damp_roll = 0.5
plate_damp_yaw = 0.5

plate_weld_solref = "0.010 1.0"   # position coupling (restored original stiffness)
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
    # Steering tendon sites (2 per plate: left and right at plate edge)
    steer_r = plate_radius * 0.85   # near plate edge for maximum lever arm
    steer_z = z_center              # at center height
    plates_xml += f'      <site name="p{p}_stL" pos="{-steer_r:.5f} 0 {steer_z:.5f}" size="0.0015" rgba="0.2 0.2 1 1"/>\n'
    plates_xml += f'      <site name="p{p}_stR" pos="{steer_r:.5f} 0 {steer_z:.5f}" size="0.0015" rgba="0.2 0.2 1 1"/>\n'
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

# --- Inter-plate connect constraints (position coupling, free orientation) ---
plate_welds_xml = ""
for seg in range(num_segments):
    plate_welds_xml += (
        f'    <connect body1="plate{seg}" body2="plate{seg+1}"\n'
        f'          anchor="0 {seg_length:.5f} 0"\n'
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

# --- Steering diagonal tendons & muscles (2 per segment: left-turn, right-turn) ---
steer_muscle_force = 500  # optimal: H1 experiment (500N turns ~10.9°/period)
steer_tendons_xml = ""
steer_muscles_xml = ""
for seg in range(num_segments):
    p_start = seg
    p_end = seg + 1
    # Left-turn tendon: right(start) → left(end) → diagonal creates leftward force
    tname_L = f"stL{seg}"
    mname_L = f"smL{seg}"
    steer_tendons_xml += f'    <spatial name="{tname_L}" width="0.0012" rgba="0.2 0.2 0.9 0.6">\n'
    steer_tendons_xml += f'      <site site="p{p_start}_stR"/>\n'
    steer_tendons_xml += f'      <site site="p{p_end}_stL"/>\n'
    steer_tendons_xml += f'    </spatial>\n'
    steer_muscles_xml += f'    <muscle class="muscle" name="{mname_L}" tendon="{tname_L}" force="{steer_muscle_force}" lengthrange="0.03 0.10"/>\n'
    # Right-turn tendon: left(start) → right(end) → diagonal creates rightward force
    tname_R = f"stR{seg}"
    mname_R = f"smR{seg}"
    steer_tendons_xml += f'    <spatial name="{tname_R}" width="0.0012" rgba="0.9 0.2 0.2 0.6">\n'
    steer_tendons_xml += f'      <site site="p{p_start}_stL"/>\n'
    steer_tendons_xml += f'      <site site="p{p_end}_stR"/>\n'
    steer_tendons_xml += f'    </spatial>\n'
    steer_muscles_xml += f'    <muscle class="muscle" name="{mname_R}" tendon="{tname_R}" force="{steer_muscle_force}" lengthrange="0.03 0.10"/>\n'
num_steer = num_segments * 2  # 5 left-turn + 5 right-turn = 10

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
{steer_tendons_xml}
  </tendon>

  <actuator>
{axial_muscles_xml}
{ring_muscles_xml}
{steer_muscles_xml}
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
print(f"Bodies: {model.nbody}, DOF: {model.nv}, Actuators: {model.nu} ({num_axial} axial + {num_ring} ring + {num_steer} steer)")
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

# ======================== Discrete Gait State Machine ========================
# Paper gait: {n₂=1, n₃=0, n₁=1 | nP=1}, TN=5
# Modified: 1-anchor variant (na=1) for stronger turning in cable-plate model
#   s^t₀ = (2, 0, 0, 0, 1) — 1 bending + 3 relaxed + 1 anchor
# At each step: 1 anchor + 1 bending + 3 relaxed → more segments extend, stronger propulsion
# θ_T ≈ 10.9°/period (measured), paper kinematic model predicts 18°
#
# State sequence (tail→head):
#   t₀: [2, 0, 0, 0, 1]  — seg0 bends, seg4 anchors
#   t₁: [0, 0, 0, 1, 2]  — seg3 anchors, seg4 bends
#   t₂: [0, 0, 1, 2, 0]  — seg2 anchors, seg3 bends
#   t₃: [0, 1, 2, 0, 0]  — seg1 anchors, seg2 bends
#   t₄: [1, 2, 0, 0, 0]  — seg0 anchors, seg1 bends

s0_gait = [2, 0, 0, 0, 1]  # 1-anchor gait: 1 bend + 3 relaxed + 1 anchor
TN = num_segments            # total segments = 5
nP_gait = 1                  # propagation step
gait_period = TN             # 5 transitions per period
step_duration = 0.5          # seconds per transition (Δt)

def get_segment_states(t_active):
    """Get segment states at time t_active.
    Returns list of states for [seg0(tail), seg1, seg2, seg3, seg4(head)].
    Uses paper formula: s_i^{t_j} = s_{Mod[i-1+j·nP, TN]+1}^{t₀}
    Code seg k = paper segment #(k+1), so state = s0[(k + j) % TN]
    """
    j = int(t_active / step_duration) % gait_period
    return [s0_gait[(k + j * nP_gait) % TN] for k in range(TN)]

def state_to_axial(state):
    """Map segment state to 4 axial muscle activations [m0_right, m1_top, m2_left, m3_bottom].
    State 0: relaxed (all off)
    State 1: contracted (all on, symmetric anchor)
    State 2: left-bent (partial contraction via axial, bending via diagonal tendon)
    State 3: right-bent (partial contraction via axial, bending via diagonal tendon)
    """
    if state == 0:   return [0.0, 0.0, 0.0, 0.0]
    elif state == 1: return [1.0, 1.0, 1.0, 1.0]
    elif state == 2: return [0.5, 0.5, 0.5, 0.5]  # partial axial contraction
    elif state == 3: return [0.5, 0.5, 0.5, 0.5]  # partial axial contraction
    return [0.0, 0.0, 0.0, 0.0]

def state_to_steer(state):
    """Map segment state to [left_turn, right_turn] diagonal tendon activations.
    Diagonal tendons provide the lateral force for bending (axial tendons can't).
    Left-turn tendon: right(start) → left(end) → bends segment leftward.
    """
    if state == 2:   return [1.0, 0.0]  # left-bent: activate left-turn diagonal
    elif state == 3: return [0.0, 1.0]  # right-bent: activate right-turn diagonal
    return [0.0, 0.0]

def state_to_ring(state):
    """Ring muscle: ON when relaxed (maintain slim shape), OFF when contracting."""
    return 1.0 if state == 0 else 0.0

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

# Recolor cable geoms to MatSteel at model level
strip_color = np.array([0.82, 0.78, 0.72, 1.0], dtype=np.float32)
steel_mat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "MatSteel")
recolored = 0
for gi in range(m.ngeom):
    if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        m.geom_matid[gi] = steel_mat_id
        m.geom_rgba[gi] = strip_color
        recolored += 1
print(f"Recolored {recolored} cable geoms to MatSteel")

mujoco.mj_forward(m, d)

# Pre-settle 2s (critical: match exp_runner, prevents QACC instability)
print("Pre-settling 2s...")
for _ in range(4000):
    mujoco.mj_step(m, d)
if np.any(np.isnan(d.qpos)):
    print("FATAL: NaN after pre-settling video model!")
    exit(1)

pids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
head_id = pids[-1]   # plate 5 = head (highest Y, frontmost)
tail_id = pids[0]    # plate 0 = tail (lowest Y, rearmost)

r_side = mujoco.Renderer(m, 400, 900)
r_top = mujoco.Renderer(m, 200, 900)

cam_side = mujoco.MjvCamera()
cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
cam_side.distance = 0.50
cam_side.elevation = -15
cam_side.azimuth = 90

cam_top = mujoco.MjvCamera()
cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
cam_top.distance = 0.80
cam_top.elevation = -90
cam_top.azimuth = 0

framerate = 60
frames = []
dt = m.opt.timestep
sim_time = 30.0  # seconds: ~11.6 gait periods @ 2.5s each → expect ~92° heading (7.9°/period)
total_steps = int(sim_time / dt)
settle_time = 1.0  # 1s settle

# Tracking
record_interval = 20  # every 20 steps = 10ms
time_history = []
plate_y_history = {p: [] for p in range(num_plates)}
head_xy_history = []  # for XY trajectory plot

t0_wall = time.time()

for step in range(total_steps):
    t = step * dt
    t_ms = t * 1000.0

    # Record tracking data
    if step % record_interval == 0:
        time_history.append(t)
        for p in range(num_plates):
            plate_y_history[p].append(d.xpos[pids[p], 1])
        head_xy_history.append((d.xpos[head_id, 0], d.xpos[head_id, 1]))

    # Control — Discrete Gait State Machine + Diagonal Tendons for Bending
    d.ctrl[:] = 0
    if t >= settle_time:
        t_active = t - settle_time
        seg_states = get_segment_states(t_active)
        steer_base = num_axial + num_ring  # actuator index offset for steering
        for seg in range(num_segments):
            s = seg_states[seg]
            # Axial muscles (4 per segment)
            axial_acts = state_to_axial(s)
            for mi in range(4):
                d.ctrl[seg * 4 + mi] = axial_acts[mi]
            # Ring muscles
            d.ctrl[num_axial + seg] = state_to_ring(s)
            # Diagonal steering tendons: provide lateral force for States 2/3
            steer_acts = state_to_steer(s)
            d.ctrl[steer_base + seg * 2] = steer_acts[0]      # left-turn
            d.ctrl[steer_base + seg * 2 + 1] = steer_acts[1]  # right-turn

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

    if (step + 1) % 10000 == 0:
        hy = d.xpos[head_id, 1]
        hx = d.xpos[head_id, 0]
        ty = d.xpos[tail_id, 1]
        tx = d.xpos[tail_id, 0]
        z_vals = [d.xpos[pids[p], 2] for p in range(num_plates)]
        z_range = max(z_vals) - min(z_vals)
        hdg = math.degrees(math.atan2(hx - tx, hy - ty))
        # Show current gait state
        if t >= settle_time:
            states = get_segment_states(t - settle_time)
            state_str = str(states)
        else:
            state_str = "settling"
        print(f"  t={t:.1f}s | head x={hx*1000:+.1f} y={hy*1000:+.1f}mm | heading={hdg:+.1f}° | Z={z_range*1000:.1f}mm | states={state_str}")

elapsed = time.time() - t0_wall
print(f"\nSim done in {elapsed:.1f}s, {len(frames)} frames")

# Results
head_y = d.xpos[head_id, 1] * 1000
head_x = d.xpos[head_id, 0] * 1000
tail_y = d.xpos[tail_id, 1] * 1000
tail_x = d.xpos[tail_id, 0] * 1000
z_vals = [d.xpos[pids[p], 2] for p in range(num_plates)]
z_range = (max(z_vals) - min(z_vals)) * 1000

# Heading angle: atan2(head_x - tail_x, head_y - tail_y) relative to +Y
heading = math.degrees(math.atan2(d.xpos[head_id, 0] - d.xpos[tail_id, 0],
                                   d.xpos[head_id, 1] - d.xpos[tail_id, 1]))
# Head starts at y = 5*seg_length = 0.325m, x = 0
head_y0 = num_segments * seg_length * 1000  # initial head Y in mm
displacement = math.sqrt(head_x**2 + (head_y - head_y0)**2)  # from initial head position

print(f"Head: x={head_x:.1f}mm  y={head_y:.1f}mm")
print(f"Tail: x={tail_x:.1f}mm  y={tail_y:.1f}mm")
print(f"Displacement: {displacement:.1f}mm / {sim_time:.0f}s")
print(f"Heading: {heading:.1f}° (0°=+Y, positive=CCW)")
print(f"Z range: {z_range:.1f}mm")

r_side.close()
r_top.close()

vid_path = os.path.join(vid_dir, "worm_5seg_v3_circular.mp4")
if len(frames) > 0:
    media.write_video(vid_path, frames, fps=framerate)
    print(f"Video: {vid_path}")

# Plot XY trajectory (top-down view of circular motion)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: XY trajectory
    ax1 = axes[0]
    xs = [p[0]*1000 for p in head_xy_history]
    ys = [p[1]*1000 for p in head_xy_history]
    ax1.plot(xs, ys, 'b-', linewidth=1.5, label='Head trajectory')
    ax1.plot(xs[0], ys[0], 'go', markersize=8, label='Start')
    ax1.plot(xs[-1], ys[-1], 'r*', markersize=12, label='End')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title(f"V3 Circular Gait — XY Trajectory\n"
                  f"Gait {{n₂=1, n₃=0, n₁=1 | nP=1}}, heading={heading:.1f}°")
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right: Plate Y over time
    ax2 = axes[1]
    for p in range(num_plates):
        ax2.plot(time_history, [y*1000 for y in plate_y_history[p]], label=f"Plate {p}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y position (mm)")
    ax2.set_title("Plate Y Trajectories")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(plot_dir, "circular_gait_trajectory.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {plot_path}")
except Exception as e:
    print(f"Plot skipped: {e}")

print("\nDone.")

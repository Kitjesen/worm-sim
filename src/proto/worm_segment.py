"""
Worm segment prototype: 2 rigid plates + 8 spring steel strips + muscle actuator
- 8 cables arranged in a circle, welded to both plates
- Ring tendon at midpoint for circumferential contraction
- Hill muscle actuator drives the ring tendon
- When muscle activates: ring contracts → segment bulges axially (extends)
- When muscle relaxes: spring steel restores → segment returns to rest
"""
import mujoco
import numpy as np
import mediapy as media
import time
import math

# ===== Geometric parameters =====
seg_height = 0.06        # segment height (distance between plates)
plate_radius = 0.03      # plate radius
strip_radius = 0.025     # radius of the circle where strips are placed
num_strips = 8           # number of spring steel strips
num_verts = 9            # vertices per strip (odd number for midpoint)
strip_capsule_r = 0.0015 # capsule radius (smaller = less stiff = more visible deformation)
bend_stiffness = 1e8     # lower bend stiffness for more visible deformation
twist_stiffness = 4e7    # twist stiffness

# Cable bow: slight outward curvature at midpoint
bow_amount = 0.005       # how much the strip bows outward at middle

# ===== Generate XML =====
def make_strip_vertices(angle, num_v, height, r_circle, bow):
    """Generate vertices for one strip from bottom plate to top plate with slight bow."""
    cx = r_circle * math.cos(angle)
    cy = r_circle * math.sin(angle)
    verts = []
    for i in range(num_v):
        t = i / (num_v - 1)  # 0 to 1
        z = t * height
        # Parabolic bow outward at midpoint
        bow_offset = bow * 4 * t * (1 - t)  # max at t=0.5
        x = cx + bow_offset * math.cos(angle)
        y = cy + bow_offset * math.sin(angle)
        verts.append(f"{x:.6f} {y:.6f} {z:.6f}")
    return "  ".join(verts)

# Generate strip angles
angles = [2 * math.pi * i / num_strips for i in range(num_strips)]

# Build cable bodies
cable_bodies = ""
for i, ang in enumerate(angles):
    verts = make_strip_vertices(ang, num_verts, seg_height, strip_radius, bow_amount)
    cable_bodies += f"""
    <body>
      <freejoint/>
      <composite type="cable" prefix="strip{i}" initial="none" vertex="{verts}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="{bend_stiffness}"/>
          <config key="twist" value="{twist_stiffness}"/>
          <config key="vmax" value="2"/>
        </plugin>
        <joint armature="0.01" damping="0.15" kind="main"/>
        <geom type="capsule" size="{strip_capsule_r}" density="7850" rgba="0.3 0.5 0.8 1"/>
      </composite>
    </body>
"""

# Build weld constraints (both ends of each strip to plates)
welds = ""
for i in range(num_strips):
    welds += f'    <weld body1="bottom_plate" body2="strip{i}B_first" solref="0.001 1"/>\n'
    welds += f'    <weld body1="top_plate" body2="strip{i}B_last" solref="0.001 1"/>\n'

# Build ring tendon: goes through midpoint of each strip
mid_idx = num_verts // 2  # index of midpoint body in each cable
# For composite cable with N vertices, bodies are: prefix_B_first, prefix_B_1, ..., prefix_B_{N-3}, prefix_B_last
# So midpoint body index = mid_idx - 1 (0-indexed from B_first)
# With 9 vertices: B_first(0), B_1(1), B_2(2), B_3(3=mid), B_4(4), B_5(5), B_6(6), B_last(7)
# midpoint body = B_3

# We need sites on the midpoint bodies for the ring tendon
# Sites are placed at body origin (0,0,0) in body frame
mid_sites = ""
for i in range(num_strips):
    if mid_idx == 0:
        body_name = f"strip{i}B_first"
    elif mid_idx == num_verts - 2:
        body_name = f"strip{i}B_last"
    else:
        body_name = f"strip{i}B_{mid_idx}"
    mid_sites += f'      <site name="ring_site{i}" size="0.003" rgba="1 0 0 1"/>\n'

# Build ring tendon path (loop through all midpoint sites + back to first)
ring_tendon_sites = ""
for i in range(num_strips):
    ring_tendon_sites += f'      <site site="ring_site{i}"/>\n'
ring_tendon_sites += f'      <site site="ring_site0"/>\n'  # close the loop

# For the sites, we need to attach them to the midpoint bodies
# Since composite doesn't allow adding sites to generated bodies,
# we'll use ball bodies at the midpoints (like the worm model)
ball_bodies = ""
ball_constraints = ""
for i, ang in enumerate(angles):
    t = 0.5  # midpoint
    z = t * seg_height
    bow_offset = bow_amount * 4 * t * (1 - t)
    x = (strip_radius + bow_offset) * math.cos(ang)
    y = (strip_radius + bow_offset) * math.sin(ang)

    ball_bodies += f"""
    <body name="ring_ball{i}" pos="{x:.6f} {y:.6f} {z:.6f}">
      <freejoint/>
      <geom type="sphere" size="0.003" mass="0.001" margin="-1.0" rgba="1 0.3 0.3 1"/>
      <site name="ring_site{i}" pos="0 0 0" size="0.003"/>
    </body>
"""
    # Connect ball to cable midpoint body
    if mid_idx == 0:
        cable_body = f"strip{i}B_first"
    elif mid_idx >= num_verts - 2:
        cable_body = f"strip{i}B_last"
    else:
        cable_body = f"strip{i}B_{mid_idx}"
    ball_constraints += f'    <connect body1="ring_ball{i}" body2="{cable_body}" anchor="0 0 0" solref="0.001 1"/>\n'

# Plate sites for position tracking
plate_site_top = ""
plate_site_bot = ""
for i, ang in enumerate(angles):
    x = strip_radius * math.cos(ang)
    y = strip_radius * math.sin(ang)
    plate_site_top += f'      <site name="top_site{i}" pos="{x:.6f} {y:.6f} 0" size="0.002"/>\n'
    plate_site_bot += f'      <site name="bot_site{i}" pos="{x:.6f} {y:.6f} 0" size="0.002"/>\n'

# ===== Full XML =====
xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="worm_segment">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <size memory="1G"/>
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
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 2"/>
    <light castshadow="false" diffuse=".3 .3 .3" dir="0 0 -1" pos="0 0 1"/>
    <camera name="side" pos="0.15 -0.15 0.05" xyaxes="0.7 0.7 0 -0.1 0.1 1"/>
    <camera name="top" pos="0 0 0.25" xyaxes="1 0 0 0 1 0"/>

    <geom type="plane" size="0.5 0.5 0.01" rgba="0.95 0.95 0.95 1" condim="3"/>

    <!-- Bottom plate (fixed to ground) -->
    <body name="bottom_plate" pos="0 0 {plate_radius + 0.01}">
      <geom type="cylinder" size="{plate_radius} 0.003" rgba="0.5 0.5 0.5 0.8" mass="0.05"/>
{plate_site_bot}
    </body>

    <!-- Top plate (free to move vertically) -->
    <body name="top_plate" pos="0 0 {plate_radius + 0.01 + seg_height}">
      <joint name="top_slide" type="slide" axis="0 0 1" damping="0.5"/>
      <geom type="cylinder" size="{plate_radius} 0.003" rgba="0.5 0.5 0.5 0.8" mass="0.05"/>
{plate_site_top}
    </body>

    <!-- 8 spring steel strips (cables) -->
{cable_bodies}

    <!-- Ball bodies at strip midpoints (for ring tendon) -->
{ball_bodies}
  </worldbody>

  <equality>
    <!-- Weld strip ends to plates -->
{welds}
    <!-- Connect midpoint balls to cable bodies -->
{ball_constraints}
  </equality>

  <contact>
    <!-- Exclude self-collisions between strips and balls -->
  </contact>

  <tendon>
    <!-- Ring tendon through midpoints of all 8 strips -->
    <spatial name="ring_muscle" width="0.002" springlength="0.10 0.20" stiffness="100000"
            rgba="0.9 0.2 0.2 1">
{ring_tendon_sites}
    </spatial>
  </tendon>

  <actuator>
    <muscle class="muscle" name="ring_actuator" tendon="ring_muscle" force="40"
            lengthrange="0.10 0.20"/>
  </actuator>

  <sensor>
    <tendonpos name="ring_length" tendon="ring_muscle"/>
    <actuatorfrc name="muscle_force" actuator="ring_actuator"/>
  </sensor>
</mujoco>
"""

print("Generating XML...")
xml_path = "worm_segment.xml"
with open(xml_path, "w") as f:
    f.write(xml)

print("Loading model...")
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print(f"Model: {model.nbody} bodies, {model.nv} DOF, {model.nu} actuators")
    print(f"Mass: {np.sum(model.body_mass):.4f} kg")

    mujoco.mj_forward(model, data)
    print(f"Ring tendon length: {data.ten_length[0]:.4f}m")
    print(f"Top plate z: {data.xpos[2, 2]:.4f}m")

    # Quick test: let it settle under gravity
    for i in range(500):
        mujoco.mj_step(model, data)
    print(f"After settling - Top plate z: {data.xpos[2, 2]:.4f}m, Ring len: {data.ten_length[0]:.4f}m")

    # Activate muscle
    for i in range(2000):
        data.ctrl[0] = min(1.0, i / 1000.0)  # ramp up
        mujoco.mj_step(model, data)
    print(f"After contraction - Top plate z: {data.xpos[2, 2]:.4f}m, Ring len: {data.ten_length[0]:.4f}m")

    # Deactivate muscle
    for i in range(2000):
        data.ctrl[0] = max(0.0, 1.0 - i / 1000.0)  # ramp down
        mujoco.mj_step(model, data)
    print(f"After release - Top plate z: {data.xpos[2, 2]:.4f}m, Ring len: {data.ten_length[0]:.4f}m")

    print("\nModel loaded and validated successfully!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

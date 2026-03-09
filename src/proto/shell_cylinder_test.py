"""
P0: flexcomp 圆筒壳体原型验证
================================
验证 MuJoCo flexcomp type="direct" dim="2" + elasticity elastic2d="bend"
能否创建闭合圆筒壳体，用于替代 V2 的 8 条离散 cable 条带。

Step 1: 静态加载
Step 2: 重力沉降
Step 3: 轴向压缩（position actuator）
Step 4: 环肌原型（spatial tendon 闭环）
Step 5: 性能评估
"""
import mujoco
import numpy as np
import math
import os
import time

# ======================== Geometry Parameters ========================
N_circ = 12          # vertices per ring (30° spacing)
N_axial = 7          # axial rows
radius = 0.022       # 22mm (= V2 plate_radius)
length = 0.065       # 65mm (= V2 seg_length)
z_center = 0.023     # center height above ground

# Shell elasticity
shell_young = 2e5    # Pa — conservative start
shell_poisson = 0.3  # steel standard
shell_thickness = 3e-4  # 0.3mm
shell_elastic2d = "bend"  # enable bending stiffness

# ======================== Generate Cylinder Mesh ========================
def generate_cylinder_mesh():
    """Generate vertices and triangle elements for a closed cylinder shell."""
    points = []
    for j in range(N_axial):
        y = j * length / (N_axial - 1)
        for i in range(N_circ):
            angle = 2 * math.pi * i / N_circ
            x = radius * math.cos(angle)
            z = z_center + radius * math.sin(angle)
            points.append((x, y, z))

    elements = []
    for j in range(N_axial - 1):
        for i in range(N_circ):
            i_next = (i + 1) % N_circ  # wrap around to close cylinder
            v0 = j * N_circ + i
            v1 = j * N_circ + i_next
            v2 = (j + 1) * N_circ + i
            v3 = (j + 1) * N_circ + i_next
            elements.append((v0, v1, v2))
            elements.append((v1, v3, v2))

    return points, elements

points, elements = generate_cylinder_mesh()
n_verts = len(points)
n_tris = len(elements)
print(f"Cylinder mesh: {n_verts} vertices, {n_tris} triangles")
print(f"  N_circ={N_circ}, N_axial={N_axial}")
print(f"  Bottom ring: indices 0..{N_circ-1}")
print(f"  Top ring: indices {(N_axial-1)*N_circ}..{n_verts-1}")
mid_row = (N_axial - 1) // 2
print(f"  Mid ring (row {mid_row}): indices {mid_row*N_circ}..{(mid_row+1)*N_circ-1}")

# Format for XML
point_str = "\n                ".join(
    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points
)
element_str = "\n                ".join(
    f"{e[0]} {e[1]} {e[2]}" for e in elements
)

# ======================== Connect constraints ========================
bot_connects = ""
top_connects = ""

for i in range(N_circ):
    # Bottom ring (row 0) → bot plate
    vi = i
    px, py, pz = points[vi]
    bot_connects += f'    <connect body1="bot" body2="shell_{vi}" anchor="{px:.6f} {py:.6f} {pz:.6f}" solref="0.005 1"/>\n'

    # Top ring (last row) → top plate
    vi_top = (N_axial - 1) * N_circ + i
    px, py, pz = points[vi_top]
    top_connects += f'    <connect body1="top" body2="shell_{vi_top}" anchor="{px:.6f} {py:.6f} {pz:.6f}" solref="0.005 1"/>\n'

# ======================== Ring tendon (mid ring) ========================
mid_start = mid_row * N_circ
ring_ball_xml = ""
ring_connect_xml = ""
ring_sites = ""

for i in range(N_circ):
    vi = mid_start + i
    px, py, pz = points[vi]
    bname = f"ring_ball{i}"
    ring_ball_xml += f'    <body name="{bname}" pos="{px:.6f} {py:.6f} {pz:.6f}">\n'
    ring_ball_xml += f'      <freejoint/>\n'
    ring_ball_xml += f'      <geom type="sphere" size="0.002" mass="0.0005" rgba="1 0.2 0.2 0.8"/>\n'
    ring_ball_xml += f'      <site name="ring_site{i}" pos="0 0 0" size="0.002"/>\n'
    ring_ball_xml += f'    </body>\n'
    ring_connect_xml += f'    <connect body1="ring_ball{i}" body2="shell_{vi}" anchor="0 0 0" solref="0.005 1"/>\n'

# Ring tendon: closed loop through all ring sites
ring_tendon_sites = ""
for i in range(N_circ):
    ring_tendon_sites += f'      <site site="ring_site{i}"/>\n'
ring_tendon_sites += f'      <site site="ring_site0"/>\n'  # close the loop

# ======================== Full XML ========================
xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="shell_cylinder_test">
  <size memory="1G"/>

  <option timestep="0.002" solver="CG" tolerance="1e-6" integrator="implicitfast">
    <flag energy="enable"/>
  </option>

  <visual>
    <global offwidth="960" offheight="540" elevation="-20"/>
  </visual>

  <default>
    <geom solimp=".95 .99 .0001" solref="0.005 1"/>
    <default class="muscle">
      <muscle ctrllimited="true" ctrlrange="0 1"/>
    </default>
  </default>

  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 2"/>

    <geom type="plane" size="0.5 0.5 0.01" rgba="0.93 0.93 0.93 1" friction="1.5"/>

    <!-- Bottom plate (fixed) -->
    <body name="bot" pos="0 0 0">
      <geom type="cylinder" size="0.022 0.003" pos="0 0 {z_center}" euler="90 0 0"
            rgba="0.5 0.5 0.5 0.85" mass="0.02"/>
    </body>

    <!-- Top plate (Y + Z free) -->
    <body name="top" pos="0 {length} 0">
      <joint name="top_y" type="slide" axis="0 1 0" damping="0.5"/>
      <joint name="top_z" type="slide" axis="0 0 1" damping="0.5"/>
      <geom type="cylinder" size="0.022 0.003" pos="0 0 {z_center}" euler="90 0 0"
            rgba="0.5 0.5 0.5 0.85" mass="0.02"/>
    </body>

    <!-- Flexcomp cylinder shell -->
    <flexcomp name="shell" type="direct" dim="2" radius="0.002" mass="0.05"
              rgba="0.75 0.72 0.68 0.7"
              point="{point_str}"
              element="{element_str}">
      <contact condim="3" friction="1.5" solref="0.01 1" selfcollide="none"/>
      <edge equality="true" damping="0.01"/>
      <elasticity young="{shell_young}" poisson="{shell_poisson}"
                  thickness="{shell_thickness}" elastic2d="{shell_elastic2d}"/>
    </flexcomp>

    <!-- Ring connector balls (mid-ring) -->
{ring_ball_xml}
  </worldbody>

  <equality>
    <!-- Bottom ring pin to bot plate -->
{bot_connects}
    <!-- Top ring pin to top plate -->
{top_connects}
    <!-- Ring balls connected to shell mid-ring -->
{ring_connect_xml}
  </equality>

  <tendon>
    <spatial name="ring_tendon" width="0.001" rgba="0.9 0.15 0.15 1">
{ring_tendon_sites}
    </spatial>
  </tendon>

  <actuator>
    <!-- Axial compression: position actuator pulling top toward bot -->
    <position name="compress" joint="top_y" kp="50" ctrlrange="-0.04 0"/>
    <!-- Ring muscle -->
    <muscle class="muscle" name="ring_muscle" tendon="ring_tendon"
            force="20" lengthrange="0.05 0.20"/>
  </actuator>

  <sensor>
    <jointpos name="top_y_pos" joint="top_y"/>
    <tendonpos name="ring_len" tendon="ring_tendon"/>
  </sensor>
</mujoco>
"""

# ======================== Save & Load ========================
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(src_dir, "..", ".."))
bin_dir = os.path.join(project_root, "bin", "proto")
record_dir = os.path.join(project_root, "record", "proto", "images")
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(record_dir, exist_ok=True)

xml_path = os.path.join(bin_dir, "shell_test.xml")
with open(xml_path, "w") as f:
    f.write(xml)
print(f"\nXML saved: {xml_path}")

# ======================== Step 1: Static Load ========================
print("\n=== Step 1: Static Load ===")
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"  Bodies: {model.nbody}")
    print(f"  DOF (nv): {model.nv}")
    print(f"  Actuators: {model.nu}")
    print(f"  Constraints (nefc): {data.nefc}")
    print(f"  Total mass: {np.sum(model.body_mass):.4f} kg")
    print(f"  NaN check: {'FAIL' if np.any(np.isnan(data.qpos)) else 'OK'}")
except Exception as e:
    print(f"  FAILED: {e}")
    print("\n  Trying to diagnose...")
    # Try loading with more verbose error
    import traceback
    traceback.print_exc()
    exit(1)

# Check flex body naming
print("\n  Flex body naming check:")
for vi in [0, 1, N_circ-1, mid_start, n_verts-1]:
    name = f"shell_{vi}"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        pos = data.xpos[bid]
        print(f"    {name} → body {bid}, pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    else:
        print(f"    {name} → NOT FOUND")
        # Try alternative naming
        for prefix in ["shell_", "shellv", "shell"]:
            alt = f"{prefix}{vi}"
            bid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, alt)
            if bid2 >= 0:
                print(f"    → Found as '{alt}' (body {bid2})")
                break

# ======================== Step 2: Gravity Settling ========================
print("\n=== Step 2: Gravity Settling (1s) ===")
bot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bot")
top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top")

print(f"  Before settling:")
print(f"    bot: {data.xpos[bot_id]}")
print(f"    top: {data.xpos[top_id]}")

for i in range(500):  # 1s at dt=0.002
    data.ctrl[:] = 0  # no actuation
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)):
        print(f"  NaN at step {i} (t={i*0.002:.3f}s)!")
        break

print(f"  After settling:")
print(f"    bot: {data.xpos[bot_id]}")
print(f"    top: {data.xpos[top_id]}")
print(f"    top_y displacement: {data.xpos[top_id][1] - length:.4f} m")
print(f"    NaN: {'FAIL' if np.any(np.isnan(data.qpos)) else 'OK'}")

# Take screenshot - rest state
renderer = mujoco.Renderer(model, 540, 960)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 0.15
cam.elevation = -20
cam.azimuth = 135
cam.lookat[:] = [0, length/2, z_center]

renderer.update_scene(data, cam)
img_rest = renderer.render().copy()

# ======================== Step 3: Axial Compression ========================
print("\n=== Step 3: Axial Compression ===")
compress_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "compress")

# Gradually compress
for i in range(1000):  # 2s
    data.ctrl[compress_id] = -0.02  # pull top 20mm toward bot
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)):
        print(f"  NaN at compression step {i}!")
        break

top_y_now = data.xpos[top_id][1]
compression = length - (top_y_now - data.xpos[bot_id][1])
print(f"  top Y: {top_y_now:.4f} m")
print(f"  Axial compression: {compression*1000:.2f} mm")

# Measure mid-ring diameter change
mid_positions = []
for i in range(N_circ):
    bname = f"ring_ball{i}"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid >= 0:
        mid_positions.append(data.xpos[bid].copy())

if len(mid_positions) == N_circ:
    center = np.mean(mid_positions, axis=0)
    radii = [np.sqrt((p[0]-center[0])**2 + (p[2]-center[2])**2) for p in mid_positions]
    avg_radius = np.mean(radii)
    print(f"  Mid-ring avg radius: {avg_radius*1000:.2f} mm (rest: {radius*1000:.1f} mm)")
    print(f"  Radial change: {(avg_radius - radius)*1000:.2f} mm")

# Take screenshot - compressed
renderer.update_scene(data, cam)
img_compressed = renderer.render().copy()

# ======================== Step 4: Ring Muscle Test ========================
print("\n=== Step 4: Ring Muscle Test ===")
ring_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ring_muscle")

# Reset compression, activate ring muscle
data.ctrl[compress_id] = 0
for i in range(500):
    mujoco.mj_step(model, data)

# Now activate ring muscle
for i in range(1000):
    data.ctrl[ring_act_id] = 1.0  # full ring contraction
    data.ctrl[compress_id] = 0
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)):
        print(f"  NaN at ring step {i}!")
        break

# Measure ring diameter after contraction
mid_positions2 = []
for i in range(N_circ):
    bname = f"ring_ball{i}"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid >= 0:
        mid_positions2.append(data.xpos[bid].copy())

if len(mid_positions2) == N_circ:
    center2 = np.mean(mid_positions2, axis=0)
    radii2 = [np.sqrt((p[0]-center2[0])**2 + (p[2]-center2[2])**2) for p in mid_positions2]
    avg_radius2 = np.mean(radii2)
    print(f"  Ring muscle activated:")
    print(f"    Mid-ring avg radius: {avg_radius2*1000:.2f} mm")
    print(f"    Radial contraction: {(radius - avg_radius2)*1000:.2f} mm")
    top_y_ring = data.xpos[top_id][1]
    elongation = (top_y_ring - data.xpos[bot_id][1]) - length
    print(f"    Axial elongation: {elongation*1000:.2f} mm")

# Take screenshot - ring contracted
renderer.update_scene(data, cam)
img_ring = renderer.render().copy()

# ======================== Step 5: Performance ========================
print("\n=== Step 5: Performance ===")
data.ctrl[:] = 0
t0 = time.time()
n_bench = 1000
for i in range(n_bench):
    mujoco.mj_step(model, data)
elapsed = time.time() - t0
print(f"  {n_bench} steps in {elapsed:.3f}s")
print(f"  Step time: {elapsed/n_bench*1000:.2f} ms")
print(f"  Real-time ratio: {n_bench * model.opt.timestep / elapsed:.1f}x")

# ======================== Save Images ========================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rest)
axes[0].set_title("Rest (after settling)")
axes[0].axis('off')

axes[1].imshow(img_compressed)
axes[1].set_title("Axial compression (top pushed -20mm)")
axes[1].axis('off')

axes[2].imshow(img_ring)
axes[2].set_title("Ring muscle contraction")
axes[2].axis('off')

plt.tight_layout()
img_path = os.path.join(record_dir, "shell_cylinder_test.png")
plt.savefig(img_path, dpi=150)
plt.close()
print(f"\nScreenshots: {img_path}")

renderer.close()

# ======================== Record Video ========================
print("\n=== Recording demo video ===")
import mediapy as media

model2 = mujoco.MjModel.from_xml_path(xml_path)
data2 = mujoco.MjData(model2)
mujoco.mj_forward(model2, data2)

compress_id2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_ACTUATOR, "compress")
ring_id2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_ACTUATOR, "ring_muscle")
bot_id2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_BODY, "bot")
top_id2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_BODY, "top")

r_main = mujoco.Renderer(model2, 480, 640)
r_front = mujoco.Renderer(model2, 480, 320)

cam_main = mujoco.MjvCamera()
cam_main.type = mujoco.mjtCamera.mjCAMERA_FREE
cam_main.distance = 0.15
cam_main.elevation = -20
cam_main.azimuth = 135

cam_front = mujoco.MjvCamera()
cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
cam_front.distance = 0.12
cam_front.elevation = -10
cam_front.azimuth = 180  # front view (looking along -Y)

framerate = 30
frames = []
dt = model2.opt.timestep

# Timeline: 0-1s settle, 1-3s compress, 3-4s hold, 4-6s ring, 6-7s release
total_time = 7.0
total_steps = int(total_time / dt)

for step in range(total_steps):
    t = step * dt

    # Phase control
    if t < 1.0:
        data2.ctrl[:] = 0
    elif t < 3.0:
        # Gradual compression
        frac = (t - 1.0) / 2.0
        data2.ctrl[compress_id2] = -0.02 * frac
        data2.ctrl[ring_id2] = 0
    elif t < 4.0:
        data2.ctrl[compress_id2] = -0.02
        data2.ctrl[ring_id2] = 0
    elif t < 6.0:
        # Release compression, activate ring
        data2.ctrl[compress_id2] = 0
        frac = min((t - 4.0) / 1.0, 1.0)
        data2.ctrl[ring_id2] = 0.8 * frac  # 80% to avoid NaN
    else:
        # Release all
        data2.ctrl[:] = 0

    mujoco.mj_step(model2, data2)

    if np.any(np.isnan(data2.qpos)):
        print(f"  NaN at t={t:.2f}s — stopping video.")
        break

    # Capture frame
    if len(frames) < t * framerate:
        center = (data2.xpos[bot_id2] + data2.xpos[top_id2]) / 2
        cam_main.lookat[:] = center
        cam_front.lookat[:] = center

        r_main.update_scene(data2, cam_main)
        px_main = r_main.render().copy()
        r_front.update_scene(data2, cam_front)
        px_front = r_front.render().copy()
        frames.append(np.concatenate([px_main, px_front], axis=1))

r_main.close()
r_front.close()

vid_dir = os.path.join(project_root, "record", "proto", "videos")
os.makedirs(vid_dir, exist_ok=True)
vid_path = os.path.join(vid_dir, "shell_cylinder_test.mp4")
if len(frames) > 0:
    media.write_video(vid_path, frames, fps=framerate)
    print(f"Video ({len(frames)} frames): {vid_path}")
else:
    print("No frames captured!")

print("\nDone.")

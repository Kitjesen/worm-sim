"""Demo: spring steel strip (cable) vs spring steel plate (flexcomp) in MuJoCo 3.5"""
import mujoco
import numpy as np
import mediapy as media
import time

# ====== Combined scene: cable strip + flexcomp plate ======
xml = '''
<mujoco model="spring_steel_demo">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <option timestep="0.0005" gravity="0 0 -9.81" solver="CG" iterations="200"/>
  <visual>
    <global offwidth="960" offheight="540" elevation="-25"/>
  </visual>

  <worldbody>
    <light diffuse=".8 .8 .8" dir="0 0 -1" directional="true" pos="0 0 3"/>
    <light castshadow="false" diffuse=".3 .3 .3" dir="0 0 -1" pos="0 0 2"/>
    <geom type="plane" size="1 1 0.01" rgba="0.95 0.95 0.95 1"/>

    <camera name="main" pos="0.15 -0.35 0.2" xyaxes="1 0 0 0 0.5 1"/>

    <!-- ========= 方案A: 弹簧钢窄条 (1D cable, bend=200GPa) ========= -->
    <!-- 左侧：一端固定的悬臂钢条 -->
    <body name="anchor_strip" pos="0 0 0.15">
      <geom type="box" size="0.01 0.01 0.005" rgba="0.3 0.3 0.3 1"/>
      <site name="anchor_site" pos="0 0 0"/>
    </body>

    <body>
      <freejoint/>
      <composite type="cable" prefix="strip" initial="none"
                 vertex="0 0 0.15  0.015 0 0.15  0.03 0 0.15  0.045 0 0.15
                          0.06 0 0.15  0.075 0 0.15  0.09 0 0.15  0.105 0 0.15
                          0.12 0 0.15  0.135 0 0.15  0.15 0 0.15  0.165 0 0.15
                          0.18 0 0.15  0.195 0 0.15  0.21 0 0.15  0.225 0 0.15">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="200e9"/>
          <config key="twist" value="80e9"/>
          <config key="vmax" value="0"/>
        </plugin>
        <joint armature="0.0001" damping="0.02" kind="main"/>
        <geom type="capsule" size="0.0015" density="7850" rgba="0.4 0.4 0.8 1"/>
      </composite>
    </body>

    <!-- ========= 方案B: 弹簧钢宽板 (2D flexcomp) ========= -->
    <body name="anchor_plate" pos="0 0.08 0.15">
      <geom type="box" size="0.01 0.02 0.005" rgba="0.3 0.3 0.3 1"/>
      <site name="anchor_plate_site" pos="0 0 0"/>
    </body>

    <flexcomp name="plate" type="grid" count="12 4 1" spacing="0.015 0.015 0.015"
              pos="0 0.06 0.15" radius="0.0015" dim="2" mass="0.05"
              rgba="0.8 0.4 0.3 1">
      <contact selfcollide="none" solref="0.01 1"/>
      <edge damping="0.3"/>
      <elasticity young="200e9" poisson="0.3" thickness="0.0003"/>
    </flexcomp>
  </worldbody>

  <!-- Pin the first end of both to simulate cantilever -->
  <equality>
    <connect body1="anchor_strip" body2="stripB_first" anchor="0 0 0" solref="0.001 1"/>
    <connect body1="anchor_plate" body2="plate_0" anchor="0 0 0" solref="0.001 1"/>
    <connect body1="anchor_plate" body2="plate_1" anchor="0 0.015 0" solref="0.001 1"/>
    <connect body1="anchor_plate" body2="plate_2" anchor="0 0.030 0" solref="0.001 1"/>
    <connect body1="anchor_plate" body2="plate_3" anchor="0 0.045 0" solref="0.001 1"/>
  </equality>
</mujoco>
'''

print("Loading model...")
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print(f"Model: {model.nbody} bodies, {model.nv} DOF")

renderer = mujoco.Renderer(model, 540, 960)
mujoco.mj_forward(model, data)

framerate = 60
frames = []
num_steps = 4000  # 2 seconds at dt=0.0005

print(f"Running {num_steps} steps (2s sim time)...")
t0 = time.time()

for i in range(num_steps):
    mujoco.mj_step(model, data)

    if len(frames) < data.time * framerate:
        renderer.update_scene(data, camera='main')
        pixels = renderer.render().copy()
        frames.append(pixels)

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        print(f"  Step {i+1}/{num_steps} | {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s, {len(frames)} frames")

vid_path = "spring_steel_demo.mp4"
media.write_video(vid_path, frames, fps=framerate)
print(f"Video saved: {vid_path}")
renderer.close()

"""Record worm segment dual-view video: side + top down"""
import mujoco
import numpy as np
import mediapy as media
import time

model = mujoco.MjModel.from_xml_path("worm_segment.xml")
data = mujoco.MjData(model)
print(f"Model: {model.nbody} bodies, {model.nv} DOF")

# Two renderers for dual view
renderer_side = mujoco.Renderer(model, 400, 480)
renderer_top = mujoco.Renderer(model, 400, 480)
mujoco.mj_forward(model, data)

framerate = 60
frames = []
num_steps = 10000  # 10 seconds

# Schedule:
# 0-1s: settle
# 1-2.5s: ramp up
# 2.5-5s: hold
# 5-6.5s: ramp down
# 6.5-10s: recover

print("Running simulation...")
t0 = time.time()

for i in range(num_steps):
    t = i * 0.001

    if t < 1.0:
        data.ctrl[0] = 0.0
    elif t < 2.5:
        data.ctrl[0] = (t - 1.0) / 1.5
    elif t < 5.0:
        data.ctrl[0] = 1.0
    elif t < 6.5:
        data.ctrl[0] = 1.0 - (t - 5.0) / 1.5
    else:
        data.ctrl[0] = 0.0

    mujoco.mj_step(model, data)

    if len(frames) < data.time * framerate:
        renderer_side.update_scene(data, camera='side')
        px_side = renderer_side.render().copy()
        renderer_top.update_scene(data, camera='top')
        px_top = renderer_top.render().copy()
        # Combine side-by-side
        combined = np.concatenate([px_side, px_top], axis=1)
        frames.append(combined)

    if (i + 1) % 2000 == 0:
        rl = data.ten_length[0]
        tz = data.xpos[2, 2]
        print(f"  t={t:.1f}s | ctrl={data.ctrl[0]:.2f} | ring={rl:.4f}m | top_z={tz:.4f}m")

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s, {len(frames)} frames")

media.write_video("worm_segment_demo.mp4", frames, fps=framerate)
print("Video saved: worm_segment_demo.mp4")
renderer_side.close()
renderer_top.close()

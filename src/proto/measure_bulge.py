"""Measure cable + bulge radial expansion during contraction — diagnostic script."""
import mujoco
import numpy as np
import os

xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worm_5seg.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

num_segments = 5
num_strips = 8
num_bulge = 8
num_verts = 5
num_plates = 6
z_center = 0.023

mid_idx = 2

def get_cable_mid_bodies(seg):
    mids = []
    for si in range(num_strips):
        name = f"c{seg}s{si}B_{mid_idx}"
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            mids.append(bid)
        except:
            pass
    return mids

def get_bulge_bodies(seg):
    bids = []
    for bi in range(num_bulge):
        name = f"blg_s{seg}_b{bi}"
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            bids.append(bid)
        except:
            pass
    return bids

def measure_radii(seg, data):
    """Measure cable and bulge radial distances."""
    cable_r = []
    for bid in get_cable_mid_bodies(seg):
        pos = data.xpos[bid]
        r = np.sqrt(pos[0]**2 + (pos[2] - z_center)**2)
        cable_r.append(r)
    bulge_r = []
    for bid in get_bulge_bodies(seg):
        pos = data.xpos[bid]
        r = np.sqrt(pos[0]**2 + (pos[2] - z_center)**2)
        bulge_r.append(r)
    return np.mean(cable_r) if cable_r else 0, np.mean(bulge_r) if bulge_r else 0

# Get plate body IDs by name
plate_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"plate{p}") for p in range(num_plates)]
print(f"Plate body IDs: {plate_ids}")

def measure_plate_gap(seg, data):
    p_start = seg
    p_end = seg + 1
    y_start = data.xpos[plate_ids[p_start], 1]
    y_end = data.xpos[plate_ids[p_end], 1]
    return y_end - y_start

# Settle
print("Settling 1s...")
for i in range(1000):
    mujoco.mj_step(model, data)

print("\n=== REST STATE ===")
print(f"  {'Seg':>3s} | {'Gap':>7s} | {'Cable_r':>8s} | {'Bulge_r':>8s}")
rest_gaps, rest_cr, rest_br = [], [], []
for seg in range(num_segments):
    gap = measure_plate_gap(seg, data)
    cr, br = measure_radii(seg, data)
    rest_gaps.append(gap); rest_cr.append(cr); rest_br.append(br)
    print(f"  {seg:>3d} | {gap*1000:>6.1f}mm | {cr*1000:>7.2f}mm | {br*1000:>7.2f}mm")

# Contract all
print("\nContracting ALL muscles (2s)...")
for i in range(2000):
    act = min(1.0, i / 1000.0)
    for j in range(model.nu):
        data.ctrl[j] = act
    mujoco.mj_step(model, data)

print("\n=== ALL CONTRACTED ===")
print(f"  {'Seg':>3s} | {'Gap':>7s} {'Δ':>5s} | {'Cable_r':>8s} {'Δ':>6s} | {'Bulge_r':>8s} {'Δ':>6s}")
for seg in range(num_segments):
    gap = measure_plate_gap(seg, data)
    cr, br = measure_radii(seg, data)
    dg = (gap - rest_gaps[seg]) * 1000
    dc = (cr - rest_cr[seg]) * 1000
    db = (br - rest_br[seg]) * 1000
    print(f"  {seg:>3d} | {gap*1000:>6.1f}mm {dg:>+5.1f} | {cr*1000:>7.2f}mm {dc:>+5.2f} | {br*1000:>7.2f}mm {db:>+5.2f}")

# Single segment test
print("\nResetting for single-segment test...")
data2 = mujoco.MjData(model)
mujoco.mj_forward(model, data2)
for i in range(1000):
    mujoco.mj_step(model, data2)

rest2_gaps, rest2_cr, rest2_br = [], [], []
for seg in range(num_segments):
    gap = measure_plate_gap(seg, data2)
    cr, br = measure_radii(seg, data2)
    rest2_gaps.append(gap); rest2_cr.append(cr); rest2_br.append(br)

print("\nContracting ONLY seg 2 (2s)...")
for i in range(2000):
    act = min(1.0, i / 1000.0)
    for mi in range(4):
        data2.ctrl[2 * 4 + mi] = act
    mujoco.mj_step(model, data2)

print("\n=== SEG 2 CONTRACTED ===")
print(f"  {'Seg':>3s} | {'Gap':>7s} {'Δ':>5s} | {'Cable_r':>8s} {'Δ':>6s} | {'Bulge_r':>8s} {'Δ':>6s}")
for seg in range(num_segments):
    gap = measure_plate_gap(seg, data2)
    cr, br = measure_radii(seg, data2)
    dg = (gap - rest2_gaps[seg]) * 1000
    dc = (cr - rest2_cr[seg]) * 1000
    db = (br - rest2_br[seg]) * 1000
    marker = " <<<" if seg == 2 else ""
    print(f"  {seg:>3d} | {gap*1000:>6.1f}mm {dg:>+5.1f} | {cr*1000:>7.2f}mm {dc:>+5.2f} | {br*1000:>7.2f}mm {db:>+5.2f}{marker}")

# Ground contact check: bulge bodies at bottom of seg 2
print("\n=== BULGE GROUND CONTACT (seg 2 contracted) ===")
for bi, bid in enumerate(get_bulge_bodies(2)):
    pos = data2.xpos[bid]
    touching = "GROUND!" if pos[2] < 0.005 else ""
    print(f"  Bulge {bi}: x={pos[0]*1000:>+6.1f}mm z={pos[2]*1000:>5.1f}mm {touching}")

# Worm V2 — Independent-Plate Architecture Design Document

## 1. Problem Summary: Why V1 Fails at Per-Segment Height Change

V1 uses a **nested kinematic chain**:

```
worldbody
  └─ plate0 (freejoint)
       └─ plate1 (slide + hingeX + hingeZ relative to plate0)
            └─ plate2 (slide + hingeX + hingeZ relative to plate1)
                 └─ ...
```

When segment `seg_k` contracts, the cable strips bow outward at the bottom and generate upward ground reaction force. But because `plate_k+1` through `plate_5` are all **children** of the kinematic chain, they rigidly follow `plate_k`. The entire downstream chain lifts, not just the contracted segment's plate. This prevents differential height change and undermines the peristaltic anchoring mechanism.

## 2. V2 Core Principle: Independent Plates + Connect Constraints

Each plate has its own **freejoint** into the worldbody. Adjacent plates are coupled via MuJoCo `<connect>` equality constraints anchored at two axial sites on each plate face. This replaces the V1 slide+hinge joints.

**Key insight from reference model** (`worm_test_connector_mass.xml`): The reference uses independent `ball` bodies (freejoint) connected to cable intermediate nodes via `<connect anchor="0 0 0" body1="..." body2="..." solref="0.001 1"/>`. This is exactly the pattern we adapt for plates.

## 3. XML Structure Overview

```
worldbody
  ├─ plate0 (freejoint)       — independent, worldbody child
  ├─ plate1 (freejoint)       — independent, worldbody child
  ├─ ...
  ├─ plate5 (freejoint)       — independent, worldbody child
  ├─ [cable strip bodies]     — 8 × 5 = 40 cable bodies (same as V1)
  └─ [bulge bodies]           — 8 × 5 = 40 bulge bodies (parent = worldbody)

equality
  ├─ weld: cable ends → plates  (same as V1)
  ├─ connect: plate{p} → plate{p+1}  (NEW — replaces slide+hinge chain)
  └─ joint: bulge_r coupled to segment_length_estimate  (ADAPTED — see §7)

tendon + actuator + sensor
  — same as V1, no structural change
```

## 4. Plate Bodies

### 4.1 Geometry (unchanged from V1)

Each plate is an independent worldbody child with freejoint:

```xml
<body name="plate{p}" pos="0 {p * seg_length:.5f} 0">
  <freejoint/>
  <!-- cylinder disk -->
  <geom type="cylinder" size="{plate_radius} {plate_thickness}"
        pos="0 0 {z_center}" euler="90 0 0"
        rgba="0.5 0.5 0.5 0.85" mass="0.02" friction="0.3"
        contype="2" conaffinity="2"/>
  <!-- copper pillars (same logic as V1, based on p position) -->
  ...
  <!-- 4 tendon sites at 90° intervals -->
  <site name="p{p}_s{si}" pos="{sx:.5f} 0 {sz:.5f}" size="0.0015"/>
  <!-- 2 axial connect sites: front face and back face -->
  <site name="p{p}_front" pos="0 {+plate_thickness:.5f} {z_center:.5f}" size="0.001"/>
  <site name="p{p}_back"  pos="0 {-plate_thickness:.5f} {z_center:.5f}" size="0.001"/>
</body>
```

**New in V2**: Two additional sites per plate — `p{p}_front` and `p{p}_back` — located at the axial faces of the cylinder at its geometric center. These are used as anchor points for `<connect>` constraints.

### 4.2 Plate Initial Positions

All plates positioned at world coordinates directly:
- `plate{p}` at `pos="0 {p * seg_length} 0"` (Y axis is axial direction)
- No parent-relative offsets needed (no nested transform chain)

### 4.3 Inertia / Mass

Keep V1 values: `mass="0.02"` for disk, `mass="0.002"` per copper pillar.
With freejoint, MuJoCo requires explicit inertia or non-zero mass for stability. Current geom-derived inertia is sufficient.

## 5. Cable Strips

### 5.1 strip_verts() Adaptation

V1's `strip_verts(angle, seg_idx)` generates vertices in **world coordinates** assuming plates are at fixed positions along Y. This remains valid in V2 because plates are also initialized at the same world positions. The function does not need to change.

```python
def strip_verts(angle, seg_idx):
    """Generate vertices for one strip in one segment (world Y axis).
    Valid for V2 because plates{seg_idx} starts at world y = seg_idx * seg_length
    and plates{seg_idx+1} starts at world y = (seg_idx+1) * seg_length.
    Cable endpoints will be welded to those plates via equality constraints.
    """
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
```

### 5.2 Cable Body XML (unchanged)

```xml
<body>
  <freejoint/>
  <composite type="cable" prefix="c{seg}s{si}" initial="none" vertex="{v}">
    <plugin plugin="mujoco.elasticity.cable">
      <config key="bend" value="1e8"/>
      <config key="twist" value="4e7"/>
      <config key="vmax" value="2"/>
    </plugin>
    <joint armature="0.01" damping="0.5" kind="main"/>
    <geom type="capsule" size="0.002" density="3500"
          material="MatSteel" friction="1.5" contype="1" conaffinity="1"/>
  </composite>
</body>
```

## 6. Constraint Topology

### 6.1 Cable End Welds to Plates (retained from V1)

Weld the first and last cable element to the bounding plates. The weld in V1 uses `solref="0.005 1"` which should be kept:

```xml
<!-- segment seg: cable c{seg}s{si} welded to plate{seg} (start) and plate{seg+1} (end) -->
<weld body1="plate{seg}"   body2="c{seg}s{si}B_first" solref="0.005 1"/>
<weld body1="plate{seg+1}" body2="c{seg}s{si}B_last"  solref="0.005 1"/>
```

**Why weld not connect**: Cable composite bodies have full 6-DOF freejoints. Weld locks all 6 DOF (position + orientation) of the cable endpoint relative to the plate. Connect only locks 3 DOF (position). Orientation locking is needed to prevent the cable end from spinning.

### 6.2 NEW: Inter-Plate Connect Constraints

This is the key change replacing the kinematic slide+hinge chain.

**MuJoCo `<connect>` semantics**: Constrains two anchor points in world space to coincide. The anchor is specified in body1's local frame. MuJoCo drives the constraint so that `body1.anchor_world == body2.anchor_world`. This is a 3-DOF positional constraint with no rotational locking.

**V2 approach — 2 connects per plate-pair, offset axially**:

For adjacent plates `p` and `p+1`, we create two connect constraints using offset anchor positions:

```xml
<!-- Inter-plate connect for segment seg = p connecting plate{p} to plate{p+1} -->

<!-- Anchor 1: plate{p}'s front face center → plate{p+1}'s back face center -->
<connect body1="plate{p}"   body2="plate{p+1}"
         anchor1="0 {+plate_thickness:.5f} {z_center:.5f}"
         anchor2="0 {-plate_thickness:.5f} {z_center:.5f}"
         solref="{connect_solref}" solimp="{connect_solimp}"/>
```

**Wait** — MuJoCo `<connect>` uses a **single** anchor point in body1 frame, interpreted as the world constraint point. The correct form is:

```xml
<connect body1="plate{p}" body2="plate{p+1}"
         anchor="0 {p_front_y_world:.5f} {z_center:.5f}"
         solref="{connect_solref}" solimp="{connect_solimp}"/>
```

Where `anchor` is in **world coordinates at the initial configuration** (MuJoCo interprets connect anchor in worldbody frame for equality constraints). The constraint drives `xpos_in_body1_frame == xpos_in_body2_frame`.

**Actually** — MuJoCo `<connect>` XML attribute `anchor` gives the point in body1's **local** frame. The constraint enforces: `body1_pos + body1_rot @ anchor == body2_pos + body2_rot @ anchor`.

For our cylindrical plate centered at `pos="0 {p*seg_length} 0"` with `z_center` height:
- `anchor` in body1 (plate{p}) local frame: `(0, plate_thickness, z_center)` — the front face of plate p
- MuJoCo will constrain this world point to match the same local-frame `anchor` evaluated on body2 (plate{p+1})
- plate{p+1} is initially at `y = (p+1) * seg_length`, so anchor in its local frame `(0, -plate_thickness, z_center)` would better represent its back face

**Recommended**: Use two separate connect constraints per segment, with different anchors to prevent plate rotation (a single connect allows spin around the constraint axis):

```xml
<!-- Connect pair for segment seg, between plate{seg} and plate{seg+1} -->

<!-- Primary: plate center axis (0, 0, z_center) local to each -->
<connect body1="plate{seg}" body2="plate{seg+1}"
         anchor="0 0 {z_center:.5f}"
         solref="{connect_solref}" solimp="{connect_solimp}"/>

<!-- Secondary: offset radially (prevents axial spin) -->
<connect body1="plate{seg}" body2="plate{seg+1}"
         anchor="{plate_radius*0.5:.5f} 0 {z_center:.5f}"
         solref="{connect_solref}" solimp="{connect_solimp}"/>
```

**Wait — this is wrong**. A `<connect>` constraint constrains the same anchor point as seen from both bodies to coincide in world space. With `body1=plate{seg}` and `body2=plate{seg+1}`, this would pull the two plates to the same world position, collapsing the segment. We need a **distance** or **target displacement** constraint instead.

### 6.3 Correct V2 Plate Coupling Strategy

After carefully re-examining MuJoCo's constraint types:

**Option A: Weld with relaxed solref (compliant weld)**

Use a weld between adjacent plates with very soft parameters:

```xml
<weld body1="plate{p}" body2="plate{p+1}"
      relpose="0 {seg_length:.5f} 0  1 0 0 0"
      solref="0.020 0.8" solimp="0.9 0.95 0.01"/>
```

- `relpose` specifies the desired relative pose of body2 in body1's frame
- `solref="timeconst dampratio"`: timeconst=0.020s gives compliance, dampratio=0.8 is underdamped for spring-like behavior
- This allows the weld to stretch under external force (cable contraction) but spring back to rest length

**This is the cleanest option** — it directly controls the rest inter-plate distance AND allows bending (via soft rotational compliance). However, it doesn't allow compression past the rest length by default.

**Option B: Spatial tendon with length limit (passive spring)**

Add a passive spatial tendon between p{p}_s0 and p{p+1}_s0 acting as a maximum-length constraint. This prevents over-extension but allows free compression.

**Option C: Connect at a virtual intermediate body (pivot body)**

The reference model (`worm_test_connector_mass.xml`) uses ball bodies as rigid intermediaries. We could add a lightweight pivot body between each plate pair:

```xml
<body name="pivot_seg{seg}" pos="0 {(seg+0.5)*seg_length:.5f} {z_center:.5f}">
  <freejoint/>
  <geom type="sphere" size="0.001" mass="0.0001" contype="0" conaffinity="0"/>
</body>
```

Then connect pivot to plate{seg} and plate{seg+1}:

```xml
<connect body1="pivot_seg{seg}" body2="plate{seg}"   anchor="0 0 0" solref="0.005 1"/>
<connect body1="pivot_seg{seg}" body2="plate{seg+1}" anchor="0 0 0" solref="0.005 1"/>
```

This was the approach in the reference model but those balls were connectors for the cable mesh, not for plate-to-plate coupling.

### 6.4 RECOMMENDED: Compliant Weld with relpose

The cleanest V2 design uses a **soft weld** with explicit `relpose` for each plate pair:

```python
# Constraint parameters for inter-plate coupling
weld_timeconst = 0.015      # s — softer = more compliance under cable forces
weld_dampratio = 0.8        # slightly underdamped (spring-like return)
weld_solimp = "0.90 0.95 0.005 0.5 2"  # standard solimp
```

```xml
<equality>
  <!-- Cable welds (same as V1) -->
  <weld body1="plate{seg}" body2="c{seg}s{si}B_first" solref="0.005 1"/>
  <weld body1="plate{seg+1}" body2="c{seg}s{si}B_last" solref="0.005 1"/>
  ...

  <!-- Inter-plate soft welds (NEW in V2) -->
  <!-- relpose: plate{p+1} in plate{p}'s local frame = translate Y by seg_length, no rotation -->
  <weld body1="plate{p}" body2="plate{p+1}"
        relpose="0 {seg_length:.5f} 0  1 0 0 0"
        solref="{weld_timeconst} {weld_dampratio}"
        solimp="{weld_solimp}"/>
  ...
</equality>
```

**Why soft weld works for V2 goal**:
- At rest: plates held at exactly `seg_length` apart, no rotation
- Under cable contraction: the constraint force is `F = k * (disp - rest)` where k is determined by `solref`. With soft solref (timeconst=0.015s), the constraint allows ~1-3mm of displacement before resisting strongly
- **Height independence**: When segment `seg_k` is contracted, its cable bow force lifts `plate_k` and `plate_k+1` via weld constraints between those plates and the cable ends. The soft weld between `plate_k+1` and `plate_k+2` has finite compliance, so plate_k+1 can lift slightly relative to plate_k+2. This is the desired behavior.

**Critical tuning**: `weld_timeconst` must be tuned relative to cable stiffness. Start with `0.010` and increase toward `0.025` to observe height change. If too stiff, downstream plates lift (V1 behavior). If too soft, worm accordion-collapses.

**Alternative solimp strategy**: Use `solimp="0.95 0.99 0.0001 0.5 2"` (V1 default) for tight coupling during settling, then consider relaxing to `"0.85 0.92 0.005 0.5 2"` for locomotion. This can be tuned from Python via `model.eq_solimp`.

## 7. Axial Actuation (Muscle Tendons)

Unchanged from V1. Spatial tendons between plate sites:

```xml
<spatial name="tendon_seg{seg}_m{mi}" width="0.001" rgba="0.9 0.15 0.15 1">
  <site site="p{seg+1}_s{mi}"/>
  <site site="p{seg}_s{mi}"/>
</spatial>
```

```xml
<muscle class="muscle" name="muscle_seg{seg}_m{mi}"
        tendon="tendon_seg{seg}_m{mi}"
        force="28" lengthrange="0.03 0.08"/>
```

Because plates are now independent, tendon length is simply `||p{seg+1}_s{mi}_world - p{seg}_s{mi}_world||` — computed correctly by MuJoCo regardless of chain topology. No change needed.

## 8. Volume Conservation (Bulge Bodies)

### 8.1 V1 Bulge Architecture

In V1, each bulge body is **nested inside the plate body** (plate{p} → bulge children). The radial slide joint is in the plate's local frame. The `<joint>` equality constraint couples `bulge_r` to `seg_slide` (the plate's slide joint in the kinematic chain).

### 8.2 V2 Adaptation — Worldbody Bulge with Derived Constraint

With independent plates, bulge bodies must be **worldbody children** (cannot be nested inside plates without V1's chain problem). The bulge ring sits at the midpoint of each segment.

**V2 bulge body structure**:

```xml
<!-- Bulge ring for segment seg, bulge bi -->
<!-- Initial world position: midpoint between plate{seg} and plate{seg+1} -->
<body name="blg_s{seg}_b{bi}" pos="{bx:.5f} {(seg + 0.5)*seg_length:.5f} {bz:.5f}">
  <freejoint/>
  <geom type="sphere" size="0.005"
        rgba="0.6 0.58 0.55 0.5" friction="1.5" mass="0.001"
        contype="1" conaffinity="1"/>
</body>
```

**Problem**: With freejoint, the bulge body has 6 DOF and will fall or drift freely. We need to constrain it to follow the plate midpoint.

### 8.3 Recommended: Bulge as Child of a Helper Weld

Use a weld to pin each bulge to the midpoint of the plate-to-plate segment:

```xml
<!-- Constrain bulge to follow plate{seg} (approximately, with radial freedom) -->
<!-- Weld bulge body to plate{seg} with offset to midpoint and allow radial DOF -->
```

This reintroduces the nesting problem. A cleaner V2 approach:

**Use a fixed helper site on plate{seg} and weld the bulge to it with offset, replacing radial joint with soft compliance.**

Actually the simplest working approach: **keep bulge bodies as children of plate{seg}** (the starting plate of each segment) with the same radial slide joint. In V2, plate{seg} is independent (freejoint), so bulge children will follow their parent plate's rigid body motion BUT will not be dragged by downstream plates.

```xml
<!-- Inside plate{seg} body definition (plate{seg} has freejoint) -->
<body name="plate{seg}">
  <freejoint/>
  ...geom, sites...
  <!-- Bulge ring attached to this plate -->
  <body name="blg_s{seg}_b{bi}" pos="{bx:.5f} {by:.5f} {bz:.5f}">
    <joint name="blg_s{seg}_b{bi}_r" type="slide"
           axis="{rx:.5f} 0 {rz:.5f}"
           damping="0.1" range="-0.003 0.015"/>
    <geom type="sphere" size="0.005" .../>
  </body>
</body>
```

**This correctly achieves V2 independence**: when `plate{seg}` lifts, only the bulge ring attached to it lifts — not the downstream plates.

### 8.4 Bulge-Axial Coupling (Volume Conservation)

In V1, the bulge radial joint is coupled to the `seg_slide` joint (linear DOF between plates). In V2, there is no single scalar "slide joint" — the inter-plate distance emerges from the soft weld compliance + cable tension.

**V2 approach**: Define a **virtual measurement tendon** to track segment length, then use a `<joint>` equality to couple bulge_r to a derived joint.

**Problem**: MuJoCo `<joint>` equality only couples joints to joints, not to computed distances.

**Recommended alternative**: Use the axial muscle tendon length as the proxy for segment contraction:

The muscle tendon `tendon_seg{seg}_m0` has length ≈ `seg_length` at rest and shortens as the segment contracts. Add a `<tendon>` equality to couple bulge_r to this tendon length:

MuJoCo has no direct tendon-to-joint equality. Instead, use a **passive measurement joint**:

Add a separate measurement body with a prismatic joint:

```xml
<!-- Measurement body for segment seg length -->
<body name="meas_seg{seg}" pos="0 {seg * seg_length:.5f} {z_center:.5f}">
  <joint name="seg{seg}_len" type="slide" axis="0 1 0"
         stiffness="0" damping="0" range="-0.03 0.005"/>
  <geom type="sphere" size="0.001" mass="0.00001" contype="0" conaffinity="0"/>
</body>
```

Then weld `meas_seg{seg}` to `plate{seg}` and constrain `meas_seg{seg}.len` to track the Y-distance from `plate{seg}` to `plate{seg+1}`.

**This adds complexity without clear benefit.** For V2, the pragmatic solution is:

**Simplest working V2 bulge coupling**: Since bulge bodies are children of `plate{seg}`, use the **inter-plate weld constraint violation** (the residual) as an implicit coupling. Alternatively, simply use the axial tendon length sensor as a driving signal in Python (closed-loop coupling) rather than purely XML-driven coupling.

**Recommended V2 bulge approach for first implementation**:
1. Bulge bodies nested in `plate{seg}` (same as V1 structure within each plate body)
2. Bulge radial joints exist but with NO equality constraint (open loop)
3. Bulge springs back by passive stiffness: give the radial joint small stiffness=0.5 N/m
4. Leave volume coupling for Phase 2 once basic locomotion is confirmed

This is a deliberate simplification for V2 Phase 1. The bulge effect (ground contact during contraction) comes primarily from **cable bow outward** — bulge spheres add to this but aren't strictly necessary for the height-change mechanism to work.

## 9. Inter-Plate Distance Measurement

### 9.1 V1 Method

V1 uses `<jointpos>` sensor on the slide joint: `<jointpos name="slide_seg{seg}" joint="seg{seg}_slide"/>`.

### 9.2 V2 Method

With independent plates and soft welds (no slide joint), use:

**Option A: framepos sensor pair**

```xml
<!-- Position of plate{seg} center of mass -->
<framepos name="plate{seg}_pos" objtype="body" objname="plate{seg}"/>
<!-- Position of plate{seg+1} center of mass -->
<framepos name="plate{seg+1}_pos" objtype="body" objname="plate{seg+1}"/>
```

Compute segment length in Python as `||plate{seg+1}.pos - plate{seg}.pos||`.

**Option B: tendonpos sensor on muscle tendon**

```xml
<tendonpos name="len_seg{seg}" tendon="tendon_seg{seg}_m0"/>
```

The muscle tendon length is approximately equal to the axial distance between the plates (site-to-site distance ≈ plate center distance for small tilt). This was already in V1 and requires no change.

**Recommended**: Keep V1's `<tendonpos>` sensors. Add `<framepos>` for all 6 plates as supplementary ground truth. In Python:

```python
seg_len = np.linalg.norm(
    data.xpos[plate_ids[seg+1]] - data.xpos[plate_ids[seg]]
)
```

## 10. Constraint Parameter Recommendations

### 10.1 Cable End Welds

```
solref="0.005 1"   # 5ms timeconst — stiff, prevents cable end sliding
solimp="0.95 0.99 0.0001 0.5 2"  (default)
```

Unchanged from V1.

### 10.2 Inter-Plate Soft Welds

```
solref="0.015 0.9"    # 15ms timeconst, slightly underdamped
solimp="0.90 0.95 0.002 0.5 2"
```

**Rationale**:
- `solref[0] = 0.015`: with timestep=0.001s and Newton solver at 100 iterations, 15ms timeconst gives ~15 steps of compliance window. Cable forces (peak ~28N per muscle) need to deform the segment ~10-25mm. With plate mass 0.02kg and cable force 28N: `a = F/m = 28/0.02 = 1400 m/s²`. In 15ms: `d = ½at² = ½ * 1400 * 0.015² = 0.157m` — too much. Scale down: `solref="0.008 1.0"` gives tighter coupling.
- Start with `solref="0.008 1.0"` (8ms, critically damped). If downstream plates still lift (V1 behavior), soften to `0.012`. If robot collapses, stiffen to `0.005`.

**Quick-tune from Python** (no XML re-generation needed):
```python
# After model load, find inter-plate weld equality IDs and adjust:
for eq_id in plate_eq_ids:
    model.eq_solref[eq_id] = [0.010, 1.0]
```

### 10.3 Bulge Radial Joints (Phase 1 — passive only)

```
stiffness=0.5    # light spring return
damping=0.05
range="-0.003 0.015"
```

### 10.4 Global Solver

Keep V1's `solver="Newton" iterations="100" tolerance="1e-8"`. The increase in equality constraints (from V1: 2×8×5=80 cable welds + 5×8 bulge couplings = 120 total, to V2: 80 cable welds + 5 inter-plate welds = 85 total) actually **reduces** the constraint count. However, 6 independent plate freejoints (vs 1 freejoint + 15 joints) changes the DOF topology — Newton solver should handle this well.

## 11. Height Change Mechanism — V2 Theory

**Physical sequence for segment contraction:**

1. Muscle tendons contract segment `seg_k`, pulling `plate_k` and `plate_{k+1}` toward each other axially.

2. Cable strips bow outward (spring steel elasticity). Bottom cables contact ground, pushing upward.

3. Ground reaction force at bottom cables resolves into:
   - Upward force on `c{k}s{bot}_B_first` (welded to `plate_k`) → lifts `plate_k`
   - Upward force on `c{k}s{bot}_B_last` (welded to `plate_{k+1}`) → lifts `plate_{k+1}`

4. **V2 advantage**: `plate_{k+1}`'s lift is transmitted to `plate_{k+2}` only through the soft weld (solref ≈ 0.008-0.015s). With finite compliance, `plate_{k+1}` can rise by Δz without fully dragging `plate_{k+2}`. The magnitude of Δz per plate depends on the ratio of ground reaction force to weld constraint stiffness.

5. **V1 failure mode (absent in V2)**: In V1, `plate_{k+1}` → `plate_{k+2}` → ... are rigid children, so any lift of `plate_{k+1}` instantly lifts all downstream plates by the same amount.

**Expected V2 signature**: Z position plot should show larger variance between individual plates during locomotion (some plates higher, some lower at any given instant). Segment contraction → higher local plate Z, segment extension → lower Z back to nominal.

## 12. Code Generation Architecture

### 12.1 Module Structure

```
worm_5seg_v2.py
├── Parameters (same as V1, add V2-specific params)
│   ├── connect_solref = "0.008 1.0"
│   ├── connect_solimp = "0.90 0.95 0.002 0.5 2"
├── strip_verts(angle, seg_idx)  — unchanged
├── make_plate_content_v2(p)     — same as V1 but adds front/back connect sites
├── make_bulge_bodies_v2(seg)    — nested inside plate{seg} body (no change from V1 structure)
├── XML Generation
│   ├── plates_xml: 6 independent bodies (NOT nested)
│   ├── cables_xml: unchanged
│   ├── welds_xml: cable-end welds (unchanged)
│   ├── plate_welds_xml: NEW — 5 inter-plate soft welds
│   ├── bulge_eq_xml: REMOVED (Phase 1 — use passive springs)
│   ├── tendons_xml: unchanged
│   └── muscles_xml: unchanged
└── Simulation loop: unchanged
```

### 12.2 Key Difference from V1 Code

V1 plate generation:
```python
# V1: nested
plates_xml += f'    <body name="plate0" pos="0 0 0">\n'
plates_xml += f'      <freejoint/>\n'
for p in range(1, num_plates):
    plates_xml += f'      <body name="plate{p}" pos="0 {seg_length} 0">\n'  # relative!
    plates_xml += f'        <joint name="seg{seg}_slide" .../>\n'
    ...
# Close N nested bodies
for p in range(num_plates-1, -1, -1):
    plates_xml += f'    {"  "*p}</body>\n'
```

V2 plate generation:
```python
# V2: flat list
for p in range(num_plates):
    y_world = p * seg_length  # absolute world position
    plates_xml += f'    <body name="plate{p}" pos="0 {y_world:.5f} 0">\n'
    plates_xml += f'      <freejoint/>\n'
    plates_xml += make_plate_content_v2(p)
    if p < num_plates - 1:
        plates_xml += make_bulge_bodies_v2(p)  # bulge nested inside plate{p}
    plates_xml += f'    </body>\n'  # closes immediately (no nesting)

# V2: inter-plate soft welds
for seg in range(num_segments):
    p = seg
    plate_welds_xml += (
        f'    <weld body1="plate{p}" body2="plate{p+1}"\n'
        f'          relpose="0 {seg_length:.5f} 0  1 0 0 0"\n'
        f'          solref="{connect_solref}" solimp="{connect_solimp}"/>\n'
    )
```

## 13. Collision Handling

No change from V1:
- Plates: `contype="2" conaffinity="2"` (collide with ground only)
- Cables: `contype="1" conaffinity="1"` (collide with ground and other cables)
- Floor: `contype="3" conaffinity="3"` (collides with both plates and cables)
- Exclude contact between welded cable ends and their host plate (auto-excluded by MuJoCo via equality active flag)

Add explicit contact exclusions for the inter-plate welds to prevent plate-plate contact from over-constraining the solver:

```xml
<contact>
  <exclude body1="plate{p}" body2="plate{p+1}"/>  <!-- for all adjacent pairs -->
</contact>
```

## 14. Sensors

```xml
<!-- V2 sensors: no jointpos (no slide joint exists) -->
<!-- Instead: tendon length (segment length proxy) + plate frame positions -->
<sensor>
  <!-- Tendon length (unchanged) -->
  <tendonpos name="len_seg{seg}" tendon="tendon_seg{seg}_m0"/>
  <!-- Plate positions (NEW) -->
  <framepos name="plate{p}_pos" objtype="body" objname="plate{p}"/>
  <!-- Optional: plate orientation for tilt monitoring -->
  <framequat name="plate{p}_quat" objtype="body" objname="plate{p}"/>
</sensor>
```

## 15. Risk Analysis and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Soft weld too compliant → worm collapses axially | Medium | Start with solref="0.005 1" (same as cable welds), tune upward |
| Soft weld too stiff → V1 behavior (chain lifts together) | Medium | Check Z-pos variance across plates; target >0.5mm differential |
| Cable weld conflicts with inter-plate weld over-constraining | Low | Use contact exclusions; solver=Newton handles over-constraint well |
| Bulge bodies drift (no coupling in Phase 1) | Low | Passive stiffness=0.5 N/m keeps them near plate; sufficient for Phase 1 |
| Solver instability from 6 independent freejoints | Low | Newton solver + 100 iterations should handle; add `integrator="implicit"` if needed |
| relpose in weld doesn't allow axial compression | Medium | If relpose weld blocks compression: switch to 2-way `<connect>` anchors at offset (one on each plate face) instead |

## 16. Summary: Minimal Changes from V1

| Component | V1 | V2 |
|---|---|---|
| Plate topology | Nested kinematic chain | Flat list, all independent freejoints |
| Plate joints | slide + hingeX + hingeZ per segment | REMOVED |
| Inter-plate coupling | Rigid kinematic joints | Soft weld, relpose=seg_length |
| Cable strips | Unchanged | Unchanged |
| Cable welds | Unchanged | Unchanged |
| Bulge bodies | Nested in plates, joint-coupled to slide | Nested in plates, passive stiffness only (Phase 1) |
| Tendon/muscle | Unchanged | Unchanged |
| Sensors | jointpos (slide) + tendonpos | tendonpos + framepos |
| Contact excludes | None | Add adjacent plate pairs |

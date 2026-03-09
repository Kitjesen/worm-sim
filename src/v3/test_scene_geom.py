"""
Test script: MuJoCo scene geom injection API (MuJoCo 3.5.0)
============================================================
Demonstrates how to add custom visual-only capsule geoms to MjvScene
at render time, AFTER renderer.update_scene() but BEFORE renderer.render().

This technique can be used to draw visual-only "steel strips" between
plate bodies without adding them to the physics model.

API Summary
-----------
1. mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
   - Initialize MjvGeom fields. Set type, default size/pos/mat/rgba.
   - size: np.zeros(3) if using connector to override
   - mat: np.eye(3).flatten() for identity rotation
   - rgba: np.array([r,g,b,a], dtype=np.float32)

2. mujoco.mjv_connector(geom, type, width, from_, to)
   - Set (type, size, pos, mat) for a connector-type geom between two 3D points.
   - Assumes mjv_initGeom was already called for other fields.
   - type: mjtGeom.mjGEOM_CAPSULE (3), mjGEOM_CYLINDER, mjGEOM_LINE, etc.
   - width: radius for capsule/cylinder, pixel width for LINE
   - from_/to: np.array([x, y, z], dtype=np.float64)

3. Scene access:
   - renderer.scene.geoms[i]  (MjvGeom at index i)
   - renderer.scene.ngeom     (current count, increment after adding)
   - renderer.scene.maxgeom   (capacity, default 10000 for Renderer)

Workflow:
   renderer.update_scene(data, camera)   # populate scene from model
   # --- inject custom geoms here ---
   g = renderer.scene.geoms[renderer.scene.ngeom]
   mjv_initGeom(g, ...)
   mjv_connector(g, ...)
   renderer.scene.ngeom += 1
   # --- end injection ---
   pixels = renderer.render()            # custom geoms appear in frame
"""

import sys
import os
import numpy as np

# Add parent path so we can import build_model_xml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco


def add_capsule_to_scene(scene, p1, p2, width, rgba):
    """Add a visual-only capsule between two 3D points to the scene.

    Args:
        scene: MjvScene (e.g., renderer.scene)
        p1: start point, shape (3,)
        p2: end point, shape (3,)
        width: capsule radius in meters
        rgba: [r, g, b, a] in [0,1] range

    Returns:
        True if geom was added, False if scene is full.
    """
    if scene.ngeom >= scene.maxgeom:
        return False

    g = scene.geoms[scene.ngeom]

    # Init all fields with defaults, set rgba
    mujoco.mjv_initGeom(
        g,
        type=int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )

    # Set connector geometry (overrides type, size, pos, mat)
    mujoco.mjv_connector(
        g,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        width,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )

    scene.ngeom += 1
    return True


def test_basic_injection():
    """Test 1: Basic capsule injection into a simple scene."""
    print("=" * 60)
    print("Test 1: Basic capsule injection")
    print("=" * 60)

    xml = """<mujoco>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0 0 1" diffuse="1 1 1"/>
        <geom type="plane" size="1 1 0.01" rgba="0.7 0.7 0.7 1"/>
        <body pos="0 0.15 0.1">
          <geom type="sphere" size="0.03" rgba="0 0 1 1"/>
        </body>
      </worldbody>
    </mujoco>"""

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    renderer.update_scene(d)

    scene = renderer.scene
    ngeom_before = scene.ngeom
    print(f"  Scene before injection: ngeom={ngeom_before}, maxgeom={scene.maxgeom}")

    # Add a red capsule at a known position
    p1 = np.array([0.0, 0.1, 0.05])
    p2 = np.array([0.0, 0.2, 0.05])
    ok = add_capsule_to_scene(scene, p1, p2, width=0.003, rgba=[1, 0, 0, 1])
    assert ok, "Failed to add capsule (scene full?)"

    print(f"  Scene after injection:  ngeom={scene.ngeom}")
    assert scene.ngeom == ngeom_before + 1

    # Verify geom properties
    g = scene.geoms[ngeom_before]
    print(f"  Geom type: {g.type} (expected {int(mujoco.mjtGeom.mjGEOM_CAPSULE)})")
    print(f"  Geom pos:  {g.pos}")
    print(f"  Geom size: {g.size}")
    print(f"  Geom rgba: {g.rgba}")
    assert g.type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert np.allclose(g.rgba, [1, 0, 0, 1], atol=0.01)

    # Render and check for red pixels
    pixels = renderer.render()
    red_mask = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 50) & (pixels[:, :, 2] < 50)
    red_count = int(np.sum(red_mask))
    print(f"  Red pixels in image: {red_count}")
    assert red_count > 50, f"Expected visible red capsule but only got {red_count} red pixels"

    # Save image
    out_path = os.path.join(os.path.dirname(__file__), "test_scene_geom_basic.png")
    try:
        from PIL import Image
        Image.fromarray(pixels).save(out_path)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  (PIL not available, skipping image save)")

    renderer.close()
    print("  PASSED\n")


def test_multiple_capsules():
    """Test 2: Multiple capsules with different colors."""
    print("=" * 60)
    print("Test 2: Multiple capsules with different colors")
    print("=" * 60)

    xml = """<mujoco>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0 0 2" diffuse="1 1 1"/>
        <geom type="plane" size="2 2 0.01" rgba="0.8 0.8 0.8 1"/>
      </worldbody>
    </mujoco>"""

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)

    renderer = mujoco.Renderer(m, 480, 640)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.05]
    cam.distance = 0.6
    cam.elevation = -40
    cam.azimuth = 30
    renderer.update_scene(d, cam)

    scene = renderer.scene
    ngeom_before = scene.ngeom
    print(f"  Before: ngeom={ngeom_before}")

    # Add 5 colored capsules in a fan pattern
    colors = [
        [1, 0, 0, 1],  # red
        [0, 1, 0, 1],  # green
        [0, 0, 1, 1],  # blue
        [1, 1, 0, 1],  # yellow
        [1, 0, 1, 1],  # magenta
    ]
    for i, rgba in enumerate(colors):
        angle = np.radians(i * 30 - 60)
        x = 0.15 * np.cos(angle)
        y = 0.15 * np.sin(angle)
        p1 = np.array([0, 0, 0.01])
        p2 = np.array([x, y, 0.08])
        ok = add_capsule_to_scene(scene, p1, p2, width=0.004, rgba=rgba)
        assert ok

    print(f"  After:  ngeom={scene.ngeom} (+{scene.ngeom - ngeom_before})")
    assert scene.ngeom == ngeom_before + 5

    pixels = renderer.render()
    print(f"  Image shape: {pixels.shape}")

    out_path = os.path.join(os.path.dirname(__file__), "test_scene_geom_multi.png")
    try:
        from PIL import Image
        Image.fromarray(pixels).save(out_path)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  (PIL not available, skipping image save)")

    renderer.close()
    print("  PASSED\n")


def test_with_worm_model():
    """Test 3: Inject capsules into the worm model scene."""
    print("=" * 60)
    print("Test 3: Capsule injection with worm model")
    print("=" * 60)

    from exp_runner import build_model_xml

    # Build worm model with no_cables mode
    xml_str, P = build_model_xml("test", {"no_cables": 1, "sim_time": 0.1})
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)

    renderer = mujoco.Renderer(m, 480, 640)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0.16, 0.025]
    cam.distance = 0.5
    cam.elevation = -25
    cam.azimuth = 60
    renderer.update_scene(d, cam)

    scene = renderer.scene
    ngeom_before = scene.ngeom
    print(f"  Worm scene ngeom: {ngeom_before}, maxgeom: {scene.maxgeom}")

    # Add visual "steel strip" capsules between consecutive plates
    num_plates = P["num_segments"] + 1
    num_strips = 8
    strip_circle_r = P["strip_circle_r"]
    seg_length = P["seg_length"]
    z_center = P["plate_radius"] + 0.001

    strips_added = 0
    for seg_idx in range(P["num_segments"]):
        for s in range(num_strips):
            angle = 2 * np.pi * s / num_strips
            cx = strip_circle_r * np.cos(angle)
            cz = z_center + strip_circle_r * np.sin(angle)

            # Get plate positions from model data
            p_start_name = f"plate{seg_idx}"
            p_end_name = f"plate{seg_idx + 1}"
            try:
                bid0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p_start_name)
                bid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p_end_name)
                pos0 = d.xpos[bid0].copy()
                pos1 = d.xpos[bid1].copy()
            except Exception:
                # Fallback to nominal positions
                pos0 = np.array([0, seg_idx * seg_length, 0])
                pos1 = np.array([0, (seg_idx + 1) * seg_length, 0])

            # Strip endpoints (offset from plate center by strip circle position)
            p1 = pos0 + np.array([cx, 0, cz])
            p2 = pos1 + np.array([cx, 0, cz])

            # Silver/steel color with slight variation
            brightness = 0.6 + 0.2 * np.sin(angle)
            rgba = [brightness, brightness, brightness + 0.05, 0.9]

            ok = add_capsule_to_scene(scene, p1, p2, width=0.002, rgba=rgba)
            if ok:
                strips_added += 1

    print(f"  Added {strips_added} strip capsules (ngeom: {ngeom_before} -> {scene.ngeom})")
    assert strips_added == P["num_segments"] * num_strips

    pixels = renderer.render()

    out_path = os.path.join(os.path.dirname(__file__), "test_scene_geom_worm.png")
    try:
        from PIL import Image
        Image.fromarray(pixels).save(out_path)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  (PIL not available, skipping image save)")

    renderer.close()
    print("  PASSED\n")


def test_scene_capacity():
    """Test 4: Verify maxgeom limit is respected."""
    print("=" * 60)
    print("Test 4: Scene capacity limit")
    print("=" * 60)

    xml = """<mujoco>
      <worldbody><geom type="sphere" size="0.1"/></worldbody>
    </mujoco>"""

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    # Create scene with very small maxgeom
    scene = mujoco.MjvScene(m, maxgeom=5)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

    ngeom_after_update = scene.ngeom
    print(f"  After updateScene: ngeom={ngeom_after_update}, maxgeom={scene.maxgeom}")

    # Try to fill up remaining slots
    added = 0
    for i in range(20):  # try more than maxgeom
        ok = add_capsule_to_scene(
            scene,
            np.array([0, 0, 0]),
            np.array([0.1, 0, 0]),
            width=0.001,
            rgba=[1, 0, 0, 1],
        )
        if ok:
            added += 1
        else:
            break

    print(f"  Added {added} geoms before hitting limit")
    print(f"  Final ngeom={scene.ngeom}, maxgeom={scene.maxgeom}")
    assert scene.ngeom <= scene.maxgeom
    assert added == scene.maxgeom - ngeom_after_update

    print("  PASSED\n")


def print_api_summary():
    """Print a summary of key API findings."""
    print("=" * 60)
    print("API Summary (MuJoCo " + mujoco.__version__ + ")")
    print("=" * 60)
    print("""
  Functions:
    mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
      - geom: MjvGeom (from scene.geoms[idx])
      - type: int (mujoco.mjtGeom.mjGEOM_CAPSULE = 3)
      - size: np.zeros(3) or np.array([...])
      - pos:  np.zeros(3) or np.array([x,y,z])
      - mat:  np.eye(3).flatten() (9-element rotation matrix)
      - rgba: np.array([r,g,b,a], dtype=np.float32)

    mujoco.mjv_connector(geom, type, width, from_, to)
      - Overrides type, size, pos, mat for connector geometry
      - geom: MjvGeom (must call mjv_initGeom first!)
      - type: int (mjGEOM_CAPSULE, mjGEOM_CYLINDER, mjGEOM_LINE)
      - width: float (radius for capsule/cylinder, pixels for LINE)
      - from_: np.array([x,y,z], dtype=np.float64)
      - to:    np.array([x,y,z], dtype=np.float64)

  Scene access:
    renderer.scene              -> MjvScene
    renderer.scene.geoms[i]     -> MjvGeom at index i
    renderer.scene.ngeom        -> current geom count (read/write)
    renderer.scene.maxgeom      -> capacity (default 10000 for Renderer)

  MjvGeom fields:
    .type, .size, .pos, .mat, .rgba, .category, .dataid,
    .objtype, .objid, .matid, .emission, .specular, .shininess,
    .reflectance, .label, .camdist, .modelrbound, .transparent,
    .segid, .texcoord

  Geom type constants:
    mjtGeom.mjGEOM_CAPSULE  = 3
    mjtGeom.mjGEOM_CYLINDER = 5
    mjtGeom.mjGEOM_LINE     = 104
    mjtGeom.mjGEOM_SPHERE   = 2
    mjtGeom.mjGEOM_BOX      = 6

  Workflow:
    renderer.update_scene(data, camera)
    # inject geoms into renderer.scene here
    pixels = renderer.render()
""")


if __name__ == "__main__":
    print(f"MuJoCo version: {mujoco.__version__}\n")

    print_api_summary()
    test_basic_injection()
    test_multiple_capsules()
    test_with_worm_model()
    test_scene_capacity()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

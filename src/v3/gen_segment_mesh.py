#!/usr/bin/env python3
"""
gen_segment_mesh.py — Generate custom STL meshes for middle body segments.
==========================================================================
Creates open-frame body segments matching real hardware reference:
  - Two ring-shaped end plates (隔板) with bolt hole cutouts
  - Wide curved structural bands connecting the plates
  - Central actuator housing box between segments
  - Open structure lets inject_strips visual show through

Coordinate frame: MuJoCo sim frame (Y=forward, Z=up, X=lateral)
Center at body origin (joint position).

Usage:
    python gen_segment_mesh.py              # generate default mesh
    python gen_segment_mesh.py --preview    # generate + render preview
"""

import numpy as np
import struct
import os
import math
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Mesh generation primitives
# ─────────────────────────────────────────────────────────────────────────────

def _make_frame(axis):
    """Create orthonormal frame (u, v) perpendicular to axis."""
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1, 0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def ring_triangles(center, axis, outer_r, inner_r, thickness, n_sides=48):
    """Generate triangles for a ring (annular disc) with thickness.

    Parameters
    ----------
    center : (3,) array — disc center position
    axis : (3,) array — disc normal direction
    outer_r, inner_r : float — outer/inner radii
    thickness : float — disc thickness
    n_sides : int — polygon resolution
    """
    axis = np.asarray(axis, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis_n = axis / np.linalg.norm(axis)
    u, v = _make_frame(axis_n)
    half_t = thickness / 2.0

    triangles = []

    for face_sign in [-1.0, 1.0]:
        face_center = center + face_sign * half_t * axis_n
        for i in range(n_sides):
            a0 = 2 * math.pi * i / n_sides
            a1 = 2 * math.pi * (i + 1) / n_sides
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)

            d0 = c0 * u + s0 * v
            d1 = c1 * u + s1 * v

            po0 = face_center + outer_r * d0
            po1 = face_center + outer_r * d1
            pi0 = face_center + inner_r * d0
            pi1 = face_center + inner_r * d1

            if face_sign > 0:
                triangles.append((po0.copy(), po1.copy(), pi0.copy()))
                triangles.append((pi0.copy(), po1.copy(), pi1.copy()))
            else:
                triangles.append((po0.copy(), pi0.copy(), po1.copy()))
                triangles.append((pi0.copy(), pi1.copy(), po1.copy()))

    # Outer rim (side faces)
    for i in range(n_sides):
        a0 = 2 * math.pi * i / n_sides
        a1 = 2 * math.pi * (i + 1) / n_sides
        d0 = math.cos(a0) * u + math.sin(a0) * v
        d1 = math.cos(a1) * u + math.sin(a1) * v

        p0b = center - half_t * axis_n + outer_r * d0
        p0t = center + half_t * axis_n + outer_r * d0
        p1b = center - half_t * axis_n + outer_r * d1
        p1t = center + half_t * axis_n + outer_r * d1
        triangles.append((p0b.copy(), p1b.copy(), p0t.copy()))
        triangles.append((p0t.copy(), p1b.copy(), p1t.copy()))

        # Inner rim
        p0b = center - half_t * axis_n + inner_r * d0
        p0t = center + half_t * axis_n + inner_r * d0
        p1b = center - half_t * axis_n + inner_r * d1
        p1t = center + half_t * axis_n + inner_r * d1
        triangles.append((p0b.copy(), p0t.copy(), p1b.copy()))
        triangles.append((p0t.copy(), p1t.copy(), p1b.copy()))

    return triangles


def ring_with_holes_triangles(center, axis, outer_r, inner_r, thickness,
                               hole_r, hole_circle_r, n_holes=8, n_sides=48,
                               hole_sides=12):
    """Ring (annular disc) with circular bolt hole cutouts.

    The bolt holes are circular cutouts arranged evenly around a circle of
    radius hole_circle_r. Each hole is approximated as an n-gon. The ring
    face is tessellated using angular sectors, and sectors that overlap a
    hole are replaced with geometry that routes around the hole edge.

    Parameters
    ----------
    center : (3,) array — disc center
    axis : (3,) array — disc normal
    outer_r, inner_r : float — ring outer/inner radii (m)
    thickness : float — plate thickness (m)
    hole_r : float — bolt hole radius (m)
    hole_circle_r : float — radius of bolt hole circle (m)
    n_holes : int — number of bolt holes
    n_sides : int — ring polygon resolution
    hole_sides : int — polygon resolution per hole
    """
    axis = np.asarray(axis, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis_n = axis / np.linalg.norm(axis)
    u, v = _make_frame(axis_n)
    half_t = thickness / 2.0

    # Precompute hole centers in the (u, v) plane
    hole_angles = [2 * math.pi * k / n_holes for k in range(n_holes)]
    hole_centers_uv = [(hole_circle_r * math.cos(a),
                        hole_circle_r * math.sin(a)) for a in hole_angles]

    def _point_in_any_hole(pu, pv):
        """Check if a point in (u,v) coords falls inside any bolt hole."""
        for hcu, hcv in hole_centers_uv:
            if (pu - hcu) ** 2 + (pv - hcv) ** 2 < hole_r ** 2:
                return True
        return False

    def _segment_crosses_hole(pu0, pv0, pu1, pv1):
        """Check if a radial segment might cross a hole (conservative)."""
        # Check midpoint and endpoints
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pu = pu0 + t * (pu1 - pu0)
            pv = pv0 + t * (pv1 - pv0)
            if _point_in_any_hole(pu, pv):
                return True
        return False

    triangles = []

    # Strategy: generate the ring face sectors, skipping any triangle whose
    # centroid falls inside a bolt hole. Then separately generate hole rim
    # cylinders (the inner wall of each bolt hole through the plate).

    for face_sign in [-1.0, 1.0]:
        face_center = center + face_sign * half_t * axis_n
        for i in range(n_sides):
            a0 = 2 * math.pi * i / n_sides
            a1 = 2 * math.pi * (i + 1) / n_sides
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)

            d0 = c0 * u + s0 * v
            d1 = c1 * u + s1 * v

            po0 = face_center + outer_r * d0
            po1 = face_center + outer_r * d1
            pi0 = face_center + inner_r * d0
            pi1 = face_center + inner_r * d1

            # Check centroids of the two triangles in this sector
            for tri_verts in [(po0, po1, pi0), (pi0, po1, pi1)]:
                cx_uv = sum(np.dot(tv - face_center, u) for tv in tri_verts) / 3.0
                cy_uv = sum(np.dot(tv - face_center, v) for tv in tri_verts) / 3.0
                if not _point_in_any_hole(cx_uv, cy_uv):
                    if face_sign > 0:
                        triangles.append((tri_verts[0].copy(), tri_verts[1].copy(), tri_verts[2].copy()))
                    else:
                        triangles.append((tri_verts[0].copy(), tri_verts[2].copy(), tri_verts[1].copy()))

    # Outer rim (side faces) — same as plain ring
    for i in range(n_sides):
        a0 = 2 * math.pi * i / n_sides
        a1 = 2 * math.pi * (i + 1) / n_sides
        d0 = math.cos(a0) * u + math.sin(a0) * v
        d1 = math.cos(a1) * u + math.sin(a1) * v

        p0b = center - half_t * axis_n + outer_r * d0
        p0t = center + half_t * axis_n + outer_r * d0
        p1b = center - half_t * axis_n + outer_r * d1
        p1t = center + half_t * axis_n + outer_r * d1
        triangles.append((p0b.copy(), p1b.copy(), p0t.copy()))
        triangles.append((p0t.copy(), p1b.copy(), p1t.copy()))

        # Inner rim (center hole)
        p0b = center - half_t * axis_n + inner_r * d0
        p0t = center + half_t * axis_n + inner_r * d0
        p1b = center - half_t * axis_n + inner_r * d1
        p1t = center + half_t * axis_n + inner_r * d1
        triangles.append((p0b.copy(), p0t.copy(), p1b.copy()))
        triangles.append((p0t.copy(), p1t.copy(), p1b.copy()))

    # Bolt hole rims — cylindrical walls through the plate thickness
    for hcu, hcv in hole_centers_uv:
        hole_center_3d = center + hcu * u + hcv * v
        for i in range(hole_sides):
            a0 = 2 * math.pi * i / hole_sides
            a1 = 2 * math.pi * (i + 1) / hole_sides
            d0 = math.cos(a0) * u + math.sin(a0) * v
            d1 = math.cos(a1) * u + math.sin(a1) * v

            p0b = hole_center_3d - half_t * axis_n + hole_r * d0
            p0t = hole_center_3d + half_t * axis_n + hole_r * d0
            p1b = hole_center_3d - half_t * axis_n + hole_r * d1
            p1t = hole_center_3d + half_t * axis_n + hole_r * d1
            # Winding: inner hole walls face inward (toward hole center)
            triangles.append((p0b.copy(), p0t.copy(), p1b.copy()))
            triangles.append((p0t.copy(), p1t.copy(), p1b.copy()))

    return triangles


def rod_triangles(start, end, radius, n_sides=8):
    """Generate triangles for a cylindrical rod between two points."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return []
    axis_n = axis / length
    u, v = _make_frame(axis_n)

    triangles = []
    for i in range(n_sides):
        a0 = 2 * math.pi * i / n_sides
        a1 = 2 * math.pi * (i + 1) / n_sides
        d0 = math.cos(a0) * u + math.sin(a0) * v
        d1 = math.cos(a1) * u + math.sin(a1) * v

        # Side quad
        ps0 = start + radius * d0
        ps1 = start + radius * d1
        pe0 = end + radius * d0
        pe1 = end + radius * d1
        triangles.append((ps0.copy(), ps1.copy(), pe0.copy()))
        triangles.append((pe0.copy(), ps1.copy(), pe1.copy()))

        # Start cap
        triangles.append((start.copy(), ps1.copy(), ps0.copy()))
        # End cap
        triangles.append((end.copy(), pe0.copy(), pe1.copy()))

    return triangles


def curved_band_triangles(center, axis, radius, half_len, band_width,
                           band_thickness, arc_start, arc_span,
                           n_length=16, n_arc=24):
    """Generate a wide curved band (sheet metal strip) wrapping around the body.

    The band is a rectangular sheet bent into a cylindrical arc at the given
    radius. It connects the two end plates along the body axis (Y).

    Parameters
    ----------
    center : (3,) array — body center
    axis : (3,) array — body forward axis (Y)
    radius : float — cylindrical radius of the band center surface
    half_len : float — half the body length (band extends from -half_len to +half_len)
    band_width : float — angular width of the band (in radians)
    band_thickness : float — radial thickness of the band sheet (m)
    arc_start : float — starting angle of the band center (radians, in u-v plane)
    arc_span : float — angular span the band covers (radians) — for partial wraps
    n_length : int — subdivisions along the length
    n_arc : int — subdivisions along the arc width
    """
    center = np.asarray(center, dtype=np.float64)
    axis_n = np.asarray(axis, dtype=np.float64)
    axis_n = axis_n / np.linalg.norm(axis_n)
    u, v = _make_frame(axis_n)

    triangles = []
    half_w = band_width / 2.0
    half_t = band_thickness / 2.0

    # The band wraps around angle arc_start with angular half-width half_w
    a_min = arc_start - half_w
    a_max = arc_start + half_w

    # Build the outer and inner surfaces as grids
    for r_sign, r_offset in [(1.0, half_t), (-1.0, -half_t)]:
        r = radius + r_offset
        for i in range(n_length):
            y0 = -half_len + (2.0 * half_len) * i / n_length
            y1 = -half_len + (2.0 * half_len) * (i + 1) / n_length
            for j in range(n_arc):
                a0 = a_min + (a_max - a_min) * j / n_arc
                a1 = a_min + (a_max - a_min) * (j + 1) / n_arc

                p00 = center + y0 * axis_n + r * (math.cos(a0) * u + math.sin(a0) * v)
                p01 = center + y0 * axis_n + r * (math.cos(a1) * u + math.sin(a1) * v)
                p10 = center + y1 * axis_n + r * (math.cos(a0) * u + math.sin(a0) * v)
                p11 = center + y1 * axis_n + r * (math.cos(a1) * u + math.sin(a1) * v)

                if r_sign > 0:
                    # Outer face — normal points outward
                    triangles.append((p00.copy(), p01.copy(), p10.copy()))
                    triangles.append((p10.copy(), p01.copy(), p11.copy()))
                else:
                    # Inner face — normal points inward
                    triangles.append((p00.copy(), p10.copy(), p01.copy()))
                    triangles.append((p10.copy(), p11.copy(), p01.copy()))

    # Side edges (along the length at a_min and a_max)
    for a_edge in [a_min, a_max]:
        d = math.cos(a_edge) * u + math.sin(a_edge) * v
        for i in range(n_length):
            y0 = -half_len + (2.0 * half_len) * i / n_length
            y1 = -half_len + (2.0 * half_len) * (i + 1) / n_length

            po0 = center + y0 * axis_n + (radius + half_t) * d
            po1 = center + y1 * axis_n + (radius + half_t) * d
            pi0 = center + y0 * axis_n + (radius - half_t) * d
            pi1 = center + y1 * axis_n + (radius - half_t) * d

            if a_edge == a_min:
                triangles.append((po0.copy(), pi0.copy(), po1.copy()))
                triangles.append((po1.copy(), pi0.copy(), pi1.copy()))
            else:
                triangles.append((po0.copy(), po1.copy(), pi0.copy()))
                triangles.append((po1.copy(), pi1.copy(), pi0.copy()))

    # End caps (at y = -half_len and y = +half_len)
    for y_end, flip in [(-half_len, False), (half_len, True)]:
        for j in range(n_arc):
            a0 = a_min + (a_max - a_min) * j / n_arc
            a1 = a_min + (a_max - a_min) * (j + 1) / n_arc

            d0 = math.cos(a0) * u + math.sin(a0) * v
            d1 = math.cos(a1) * u + math.sin(a1) * v

            po0 = center + y_end * axis_n + (radius + half_t) * d0
            po1 = center + y_end * axis_n + (radius + half_t) * d1
            pi0 = center + y_end * axis_n + (radius - half_t) * d0
            pi1 = center + y_end * axis_n + (radius - half_t) * d1

            if flip:
                triangles.append((po0.copy(), po1.copy(), pi0.copy()))
                triangles.append((pi0.copy(), po1.copy(), pi1.copy()))
            else:
                triangles.append((po0.copy(), pi0.copy(), po1.copy()))
                triangles.append((pi0.copy(), pi1.copy(), po1.copy()))

    return triangles


def box_triangles(center, half_extents):
    """Generate triangles for an axis-aligned box.

    Parameters
    ----------
    center : (3,) array — box center [x, y, z]
    half_extents : (3,) array — half-sizes [hx, hy, hz]

    Returns list of triangles. Faces have outward-pointing normals.
    """
    center = np.asarray(center, dtype=np.float64)
    hx, hy, hz = half_extents

    # 8 corners
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(center + np.array([sx * hx, sy * hy, sz * hz]))

    # corners indexed as: 0=(-,-,-) 1=(-,-,+) 2=(-,+,-) 3=(-,+,+)
    #                      4=(+,-,-) 5=(+,-,+) 6=(+,+,-) 7=(+,+,+)
    c = corners
    triangles = []

    # 6 faces, 2 triangles each, outward normals
    faces = [
        # -X face (0,1,2,3)
        (0, 2, 1), (1, 2, 3),
        # +X face (4,5,6,7)
        (4, 5, 6), (5, 7, 6),
        # -Y face (0,1,4,5)
        (0, 1, 4), (1, 5, 4),
        # +Y face (2,3,6,7)
        (2, 6, 3), (3, 6, 7),
        # -Z face (0,2,4,6)
        (0, 4, 2), (2, 4, 6),
        # +Z face (1,3,5,7)
        (1, 3, 5), (3, 7, 5),
    ]
    for f in faces:
        triangles.append((c[f[0]].copy(), c[f[1]].copy(), c[f[2]].copy()))

    return triangles


def ring_with_tabs_triangles(center, axis, outer_r, inner_r, thickness,
                             tab_angles, tab_width, tab_depth, n_sides=48):
    """Ring with inward tabs at specified angles (for strip attachment points).

    Tabs are small rectangular protrusions on the inner edge of the ring,
    pointing radially inward. They mark where steel strips attach to the
    end plates.
    """
    triangles = ring_triangles(center, axis, outer_r, inner_r, thickness, n_sides)

    axis_n = np.asarray(axis, dtype=np.float64)
    axis_n = axis_n / np.linalg.norm(axis_n)
    center = np.asarray(center, dtype=np.float64)
    u, v = _make_frame(axis_n)
    half_t = thickness / 2.0

    for angle in tab_angles:
        ca, sa = math.cos(angle), math.sin(angle)
        radial = ca * u + sa * v
        tangent = -sa * u + ca * v

        # Tab extends from inner_r inward by tab_depth
        r_outer = inner_r
        r_inner = inner_r - tab_depth
        hw = tab_width / 2.0

        # 8 corners of the tab box
        corners = []
        for dy in [-half_t, half_t]:
            for r in [r_inner, r_outer]:
                for tw in [-hw, hw]:
                    p = center + dy * axis_n + r * radial + tw * tangent
                    corners.append(p.copy())

        # corners: [0](-t,inner,-w) [1](-t,inner,+w) [2](-t,outer,-w) [3](-t,outer,+w)
        #          [4](+t,inner,-w) [5](+t,inner,+w) [6](+t,outer,-w) [7](+t,outer,+w)
        c = corners
        # 6 faces x 2 triangles
        faces = [
            (0, 1, 4), (4, 1, 5),  # inner face (radially inward)
            (2, 6, 3), (3, 6, 7),  # outer face (radially outward)
            (0, 4, 2), (2, 4, 6),  # side -w
            (1, 3, 5), (5, 3, 7),  # side +w
            (0, 2, 1), (1, 2, 3),  # bottom (-t)
            (4, 5, 6), (6, 5, 7),  # top (+t)
        ]
        for f in faces:
            triangles.append((c[f[0]], c[f[1]], c[f[2]]))

    return triangles


# ─────────────────────────────────────────────────────────────────────────────
# Segment mesh generator
# ─────────────────────────────────────────────────────────────────────────────

# Default dimensions matching worm_v5_1.py constants
DEFAULT_HALF_LEN     = 0.080    # 80mm (= SEG_HALF_LEN)
DEFAULT_PLATE_OUTER  = 0.060    # 60mm outer radius (body shell)
DEFAULT_PLATE_INNER  = 0.018    # 18mm inner hole (wiring passage)
DEFAULT_PLATE_THICK  = 0.004    # 4mm end plate thickness

# Bolt holes on end plates
DEFAULT_N_BOLT_HOLES = 8        # 8 bolt holes evenly spaced
DEFAULT_BOLT_HOLE_R  = 0.0035   # 3.5mm bolt hole radius (M6 clearance)
DEFAULT_BOLT_CIRCLE_R = 0.042   # 42mm bolt circle radius

# Wide structural bands (replacing thin rails)
DEFAULT_N_BANDS      = 3        # 3 wide bands
DEFAULT_BAND_RADIUS  = 0.052    # 52mm — between strips@45 and shell@60
DEFAULT_BAND_WIDTH   = 0.35     # angular width in radians (~20deg, ~18mm arc at R=52)
DEFAULT_BAND_THICK   = 0.0015   # 1.5mm sheet metal thickness

# Actuator housing
DEFAULT_ACTUATOR_BOX = True     # add rectangular actuator housing
DEFAULT_ACTUATOR_HX  = 0.020    # 20mm half-width (X)
DEFAULT_ACTUATOR_HY  = 0.035    # 35mm half-length (Y) — sits in the middle
DEFAULT_ACTUATOR_HZ  = 0.015    # 15mm half-height (Z)
DEFAULT_ACTUATOR_Z   = 0.000    # centered vertically

# Legacy parameters (kept for backward compatibility)
DEFAULT_N_RAILS      = 4
DEFAULT_RAIL_RADIUS  = 0.055
DEFAULT_RAIL_CROSS   = 0.003
DEFAULT_MID_RING     = False    # disabled by default — not in reference
DEFAULT_STRIP_TABS   = False


def generate_mid_segment(
    half_len=DEFAULT_HALF_LEN,
    plate_outer_r=DEFAULT_PLATE_OUTER,
    plate_inner_r=DEFAULT_PLATE_INNER,
    plate_thick=DEFAULT_PLATE_THICK,
    n_rails=DEFAULT_N_RAILS,
    rail_radius=DEFAULT_RAIL_RADIUS,
    rail_cross_r=DEFAULT_RAIL_CROSS,
    add_mid_ring=DEFAULT_MID_RING,
    add_strip_tabs=DEFAULT_STRIP_TABS,
    # New parameters
    n_bolt_holes=DEFAULT_N_BOLT_HOLES,
    bolt_hole_r=DEFAULT_BOLT_HOLE_R,
    bolt_circle_r=DEFAULT_BOLT_CIRCLE_R,
    n_bands=DEFAULT_N_BANDS,
    band_radius=DEFAULT_BAND_RADIUS,
    band_width=DEFAULT_BAND_WIDTH,
    band_thick=DEFAULT_BAND_THICK,
    add_actuator_box=DEFAULT_ACTUATOR_BOX,
    use_wide_bands=True,
    n_sides=48,
):
    """Generate open-frame middle segment mesh in sim frame.

    Returns list of (v0, v1, v2) triangles (numpy arrays, float64, meters).

    The mesh consists of:
      - Front end plate at y = +half_len (with bolt hole cutouts)
      - Rear end plate at y = -half_len (with bolt hole cutouts)
      - Wide curved structural bands connecting the plates
      - Optional central actuator housing box
      - Legacy: thin connecting rails if use_wide_bands=False

    Coordinate frame: Y=forward (body axis), Z=up, X=lateral.
    """
    triangles = []
    y_axis = np.array([0, 1, 0], dtype=np.float64)
    body_center = np.array([0, 0, 0], dtype=np.float64)

    # ── Front end plate (y = +half_len) with bolt holes ──
    front_center = np.array([0, half_len, 0], dtype=np.float64)
    if n_bolt_holes > 0:
        triangles += ring_with_holes_triangles(
            front_center, y_axis, plate_outer_r, plate_inner_r, plate_thick,
            hole_r=bolt_hole_r,
            hole_circle_r=bolt_circle_r,
            n_holes=n_bolt_holes,
            n_sides=n_sides,
            hole_sides=12,
        )
    elif add_strip_tabs:
        strip_angles = [2.0 * math.pi * k / 6 for k in range(6)]
        triangles += ring_with_tabs_triangles(
            front_center, y_axis, plate_outer_r, plate_inner_r, plate_thick,
            tab_angles=strip_angles,
            tab_width=0.010,
            tab_depth=0.008,
            n_sides=n_sides,
        )
    else:
        triangles += ring_triangles(
            front_center, y_axis, plate_outer_r, plate_inner_r, plate_thick,
            n_sides=n_sides,
        )

    # ── Rear end plate (y = -half_len) with bolt holes ──
    rear_center = np.array([0, -half_len, 0], dtype=np.float64)
    if n_bolt_holes > 0:
        triangles += ring_with_holes_triangles(
            rear_center, y_axis, plate_outer_r, plate_inner_r, plate_thick,
            hole_r=bolt_hole_r,
            hole_circle_r=bolt_circle_r,
            n_holes=n_bolt_holes,
            n_sides=n_sides,
            hole_sides=12,
        )
    elif add_strip_tabs:
        strip_angles = [2.0 * math.pi * k / 6 for k in range(6)]
        triangles += ring_with_tabs_triangles(
            rear_center, y_axis, plate_outer_r, plate_inner_r, plate_thick,
            tab_angles=strip_angles,
            tab_width=0.010,
            tab_depth=0.008,
            n_sides=n_sides,
        )
    else:
        triangles += ring_triangles(
            rear_center, y_axis, plate_outer_r, plate_inner_r, plate_thick,
            n_sides=n_sides,
        )

    # ── Wide structural bands (replacing thin rails) ──
    if use_wide_bands:
        plate_margin = plate_thick / 2.0
        band_half_len = half_len - plate_margin  # bands span between end plates
        for i in range(n_bands):
            # Distribute bands evenly, offset from bolt hole positions
            # 3 bands at 0deg, 120deg, 240deg (offset by 60deg from standard)
            angle = 2 * math.pi * i / n_bands + math.pi / 6.0
            triangles += curved_band_triangles(
                body_center, y_axis,
                radius=band_radius,
                half_len=band_half_len,
                band_width=band_width,
                band_thickness=band_thick,
                arc_start=angle,
                arc_span=band_width,
                n_length=12,
                n_arc=16,
            )
    else:
        # Legacy thin rails
        plate_margin = plate_thick / 2.0
        for i in range(n_rails):
            angle = 2 * math.pi * i / n_rails + math.pi / n_rails
            rx = rail_radius * math.cos(angle)
            rz = rail_radius * math.sin(angle)

            start = np.array([rx, -half_len + plate_margin, rz])
            end = np.array([rx, half_len - plate_margin, rz])
            triangles += rod_triangles(start, end, rail_cross_r, n_sides=8)

    # ── Optional mid-ring (legacy — not in reference design) ──
    if add_mid_ring:
        mid_center = np.array([0, 0, 0], dtype=np.float64)
        triangles += ring_triangles(
            mid_center, y_axis,
            outer_r=plate_outer_r,
            inner_r=plate_outer_r - 0.004,
            thickness=0.002,
            n_sides=n_sides,
        )

    # ── Actuator housing box (centered between end plates) ──
    if add_actuator_box:
        actuator_center = np.array([0, 0, DEFAULT_ACTUATOR_Z], dtype=np.float64)
        triangles += box_triangles(
            actuator_center,
            half_extents=(DEFAULT_ACTUATOR_HX, DEFAULT_ACTUATOR_HY, DEFAULT_ACTUATOR_HZ),
        )

    return triangles


# ─────────────────────────────────────────────────────────────────────────────
# STL I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_binary_stl(filepath, triangles):
    """Write triangles to binary STL file."""
    with open(filepath, 'wb') as f:
        # 80-byte header
        header = b"gen_segment_mesh.py - custom worm body segment"
        f.write(header.ljust(80, b'\0'))
        # Triangle count
        f.write(struct.pack('<I', len(triangles)))
        # Triangles
        for v0, v1, v2 in triangles:
            v0 = np.asarray(v0, dtype=np.float32)
            v1 = np.asarray(v1, dtype=np.float32)
            v2 = np.asarray(v2, dtype=np.float32)
            normal = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(normal)
            if nl > 1e-10:
                normal /= nl
            f.write(struct.pack('<3f', *normal.astype(np.float32)))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

    print(f"  Written: {filepath} ({len(triangles)} triangles, "
          f"{os.path.getsize(filepath) / 1024:.1f} KB)")


def read_stl_info(filepath):
    """Read binary STL and report basic stats."""
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        n_tri = struct.unpack('<I', f.read(4))[0]
        verts = []
        for _ in range(n_tri):
            f.read(12)  # normal
            for _ in range(3):
                x, y, z = struct.unpack('<3f', f.read(12))
                verts.append([x, y, z])
            f.read(2)  # attrib
    verts = np.array(verts)
    lo = verts.min(axis=0) * 1000  # mm
    hi = verts.max(axis=0) * 1000
    print(f"  {os.path.basename(filepath)}: {n_tri} triangles")
    print(f"    Bounds: X[{lo[0]:.1f}, {hi[0]:.1f}] "
          f"Y[{lo[1]:.1f}, {hi[1]:.1f}] "
          f"Z[{lo[2]:.1f}, {hi[2]:.1f}] mm")
    print(f"    Size:   {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm")


# ─────────────────────────────────────────────────────────────────────────────
# Preview (optional — requires mujoco)
# ─────────────────────────────────────────────────────────────────────────────

def preview_mesh(stl_path):
    """Quick MuJoCo render of the generated mesh."""
    import mujoco

    mesh_dir = os.path.dirname(stl_path)
    mesh_name = os.path.splitext(os.path.basename(stl_path))[0]

    xml = f"""<?xml version="1.0"?>
<mujoco model="mesh_preview">
  <compiler meshdir="{mesh_dir}" angle="radian"/>
  <option gravity="0 0 0"/>
  <asset>
    <mesh name="{mesh_name}" file="{os.path.basename(stl_path)}"/>
  </asset>
  <worldbody>
    <light pos="0 -0.5 0.5" dir="0 1 -0.5"/>
    <light pos="0.3 0 0.5" dir="-0.3 0 -1"/>
    <body pos="0 0 0.1">
      <geom type="mesh" mesh="{mesh_name}" rgba="0.85 0.85 0.88 1"/>
    </body>
    <geom type="plane" size="0.5 0.5 0.01" rgba="0.9 0.9 0.88 1"/>
  </worldbody>
</mujoco>"""

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 720, 1280)

    views = [
        ("3qtr", -25, 45, 0.35),
        ("front", 0, 0, 0.25),
        ("side", 0, 90, 0.25),
        ("top", -89, 0, 0.30),
    ]

    preview_dir = os.path.join(mesh_dir, "..", "record", "v5_1", "frames")
    os.makedirs(preview_dir, exist_ok=True)

    for name, elev, azim, dist in views:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = dist
        cam.elevation = elev
        cam.azimuth = azim
        cam.lookat[:] = [0, 0, 0.1]
        renderer.update_scene(d, cam)
        frame = renderer.render()

        try:
            from PIL import Image
            img = Image.fromarray(frame)
            path = os.path.join(preview_dir, f"seg_mid_frame_{name}.png")
            img.save(path)
            print(f"  Preview: {path}")
        except ImportError:
            print(f"  PIL not found — preview image not saved")
            break

    renderer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate custom segment mesh")
    ap.add_argument("--preview", action="store_true",
                    help="Render preview images after generation")
    ap.add_argument("--half-len", type=float, default=DEFAULT_HALF_LEN,
                    help=f"Body half-length in meters (default {DEFAULT_HALF_LEN})")
    ap.add_argument("--outer-r", type=float, default=DEFAULT_PLATE_OUTER,
                    help=f"End plate outer radius (default {DEFAULT_PLATE_OUTER})")
    ap.add_argument("--no-bolt-holes", action="store_true",
                    help="Omit bolt hole cutouts on end plates")
    ap.add_argument("--n-bolt-holes", type=int, default=DEFAULT_N_BOLT_HOLES,
                    help=f"Number of bolt holes (default {DEFAULT_N_BOLT_HOLES})")
    ap.add_argument("--legacy-rails", action="store_true",
                    help="Use legacy thin rails instead of wide bands")
    ap.add_argument("--n-bands", type=int, default=DEFAULT_N_BANDS,
                    help=f"Number of wide structural bands (default {DEFAULT_N_BANDS})")
    ap.add_argument("--no-actuator", action="store_true",
                    help="Omit central actuator housing box")
    ap.add_argument("--no-mid-ring", action="store_true",
                    help="Omit midpoint structural ring (default: omitted)")
    ap.add_argument("--add-mid-ring", action="store_true",
                    help="Add midpoint structural ring (legacy)")
    ap.add_argument("--no-tabs", action="store_true",
                    help="Omit strip attachment tabs")
    args = ap.parse_args()

    use_bands = not args.legacy_rails
    use_bolt_holes = not args.no_bolt_holes
    use_actuator = not args.no_actuator
    use_mid_ring = args.add_mid_ring and not args.no_mid_ring

    print("gen_segment_mesh — Custom Worm Body Segment (v2)")
    print(f"  half_len:     {args.half_len*1000:.0f}mm")
    print(f"  outer_r:      {args.outer_r*1000:.0f}mm")
    print(f"  inner_r:      {DEFAULT_PLATE_INNER*1000:.0f}mm")
    print(f"  plate:        {DEFAULT_PLATE_THICK*1000:.0f}mm thick")
    if use_bolt_holes:
        print(f"  bolt holes:   {args.n_bolt_holes} @ R={DEFAULT_BOLT_CIRCLE_R*1000:.0f}mm, "
              f"hole_r={DEFAULT_BOLT_HOLE_R*1000:.1f}mm")
    else:
        print(f"  bolt holes:   disabled")
    if use_bands:
        print(f"  bands:        {args.n_bands} wide @ R={DEFAULT_BAND_RADIUS*1000:.0f}mm, "
              f"width={math.degrees(DEFAULT_BAND_WIDTH):.0f}deg, "
              f"thick={DEFAULT_BAND_THICK*1000:.1f}mm")
    else:
        print(f"  rails:        {DEFAULT_N_RAILS} thin @ R={DEFAULT_RAIL_RADIUS*1000:.0f}mm (legacy)")
    print(f"  actuator box: {use_actuator}")
    print(f"  mid_ring:     {use_mid_ring}")

    triangles = generate_mid_segment(
        half_len=args.half_len,
        plate_outer_r=args.outer_r,
        n_bolt_holes=args.n_bolt_holes if use_bolt_holes else 0,
        n_bands=args.n_bands,
        use_wide_bands=use_bands,
        add_actuator_box=use_actuator,
        add_mid_ring=use_mid_ring,
        add_strip_tabs=not args.no_tabs and not use_bolt_holes,
    )

    # Output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    mesh_dir = os.path.join(project_root, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    out_path = os.path.join(mesh_dir, "seg_mid_frame.STL")

    write_binary_stl(out_path, triangles)
    read_stl_info(out_path)

    if args.preview:
        print("\nRendering preview...")
        preview_mesh(out_path)

    print("\nDone.")

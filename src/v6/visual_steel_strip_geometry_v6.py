"""
Procedural visual steel-strip geometry for Worm V6.

This module generates box segments for visual-only steel strips. The generated
boxes can be drawn by MuJoCo, Isaac Sim USD prims, matplotlib previews, or any
other renderer. They do not create forces or contacts.
"""

from dataclasses import dataclass
import math

import numpy as np


NUM_STRIPS = 8
STRIP_CIRCLE_R = 0.068
STRIP_WIDTH_M = 0.018
STRIP_THICKNESS_M = 0.002
VIS_BOW_MIN_M = 0.012
VIS_BOW_MAX_M = 0.040
DEFAULT_NORMAL_GAP_M = 0.050
DEFAULT_PREBEND_COMPRESSION_M = 0.050
DEFAULT_EXTRA_COMPRESSION_RANGE_M = 0.050
ARC_SEGS = 16
ARC_OVERLAP = 1.12
STRIP_RGBA = (0.05, 0.05, 0.05, 1.0)
STRIP_ANGLES = tuple(2.0 * math.pi * k / NUM_STRIPS
                     for k in range(NUM_STRIPS))


@dataclass(frozen=True)
class SteelStripBox:
    """One visual cuboid segment of a curved spring-steel strip."""

    center_m: np.ndarray
    rotation: np.ndarray
    half_size_m: np.ndarray
    rgba: tuple
    joint_index: int
    strip_index: int
    arc_index: int


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def compute_visual_bow_m(
    distance_m,
    natural_spacing_m,
    prebend_compression_m=DEFAULT_PREBEND_COMPRESSION_M,
    extra_compression_range_m=DEFAULT_EXTRA_COMPRESSION_RANGE_M,
):
    """Return outward visual bow from pre-bend plus additional compression."""
    if natural_spacing_m <= 1e-9:
        raise ValueError("natural_spacing_m must be positive")
    if prebend_compression_m < 0.0:
        raise ValueError("prebend_compression_m must be non-negative")
    if extra_compression_range_m <= 1e-9:
        raise ValueError("extra_compression_range_m must be positive")
    additional_compression = max(
        0.0, float(natural_spacing_m) - float(distance_m))
    total_compression = float(prebend_compression_m) + additional_compression
    full_compression = (
        float(prebend_compression_m) + float(extra_compression_range_m))
    bow_alpha = min(1.0, total_compression / full_compression)
    return VIS_BOW_MIN_M + (VIS_BOW_MAX_M - VIS_BOW_MIN_M) * bow_alpha


def generate_steel_strip_boxes(
    parent_pos_m,
    child_pos_m,
    parent_rot,
    natural_spacing_m,
    joint_index=0,
    num_strips=NUM_STRIPS,
    arc_segments=ARC_SEGS,
    strip_radius_m=STRIP_CIRCLE_R,
    strip_width_m=STRIP_WIDTH_M,
    strip_thickness_m=STRIP_THICKNESS_M,
    bow_min_m=VIS_BOW_MIN_M,
    bow_max_m=VIS_BOW_MAX_M,
    prebend_compression_m=DEFAULT_PREBEND_COMPRESSION_M,
    extra_compression_range_m=DEFAULT_EXTRA_COMPRESSION_RANGE_M,
    arc_overlap=ARC_OVERLAP,
    rgba=STRIP_RGBA,
):
    """Generate visual box segments for one slide joint.

    Args:
        parent_pos_m: world position of the parent segment center.
        child_pos_m: world position of the child segment center.
        parent_rot: 3x3 parent body rotation matrix, columns are local axes.
        natural_spacing_m: relaxed parent-child spacing for this slide.
        joint_index: slide-joint index used only for metadata.

    Returns:
        List of SteelStripBox objects. Count is num_strips * arc_segments.
    """
    parent_pos = np.asarray(parent_pos_m, dtype=np.float64)
    child_pos = np.asarray(child_pos_m, dtype=np.float64)
    rot = np.asarray(parent_rot, dtype=np.float64).reshape(3, 3)

    link = child_pos - parent_pos
    distance_m = float(np.linalg.norm(link))
    if distance_m < 0.005:
        return []
    e_ax = _normalize(link)

    if natural_spacing_m <= 1e-9:
        raise ValueError("natural_spacing_m must be positive")
    default_bow = compute_visual_bow_m(
        distance_m,
        natural_spacing_m,
        prebend_compression_m=prebend_compression_m,
        extra_compression_range_m=extra_compression_range_m,
    )
    default_alpha = (
        (default_bow - VIS_BOW_MIN_M)
        / (VIS_BOW_MAX_M - VIS_BOW_MIN_M))
    vis_bow = bow_min_m + (bow_max_m - bow_min_m) * default_alpha

    span = distance_m * 0.92
    half_span = span * 0.5
    body_y = rot[:, 1]
    body_z = rot[:, 2]
    mid = (parent_pos + child_pos) * 0.5

    arc_t = np.linspace(0.0, 1.0, int(arc_segments) + 1)
    arc_r = strip_radius_m + vis_bow * 4.0 * arc_t * (1.0 - arc_t)
    arc_ax = -half_span + span * arc_t

    boxes = []
    for strip_index in range(int(num_strips)):
        angle = 2.0 * math.pi * strip_index / int(num_strips)
        e_r = math.cos(angle) * body_y + math.sin(angle) * body_z
        e_r = _normalize(e_r)

        e_tang = np.cross(e_ax, e_r)
        tang_n = float(np.linalg.norm(e_tang))
        if tang_n < 1e-9:
            continue
        e_tang = e_tang / tang_n

        pts = [
            mid + arc_ax[idx] * e_ax + arc_r[idx] * e_r
            for idx in range(int(arc_segments) + 1)
        ]

        for arc_index in range(int(arc_segments)):
            pa = pts[arc_index]
            pb = pts[arc_index + 1]
            seg_mid = (pa + pb) * 0.5
            dv = pb - pa
            seg_len = float(np.linalg.norm(dv))
            if seg_len < 1e-9:
                continue
            z_dir = dv / seg_len
            radial = np.cross(e_tang, z_dir)
            radial = _normalize(radial)
            rotation = np.column_stack([e_tang, radial, z_dir])
            half_size = np.array([
                strip_width_m * 0.5,
                strip_thickness_m * 0.5,
                seg_len * arc_overlap * 0.5,
            ], dtype=np.float64)
            boxes.append(SteelStripBox(
                center_m=seg_mid.astype(np.float64),
                rotation=rotation.astype(np.float64),
                half_size_m=half_size,
                rgba=tuple(float(v) for v in rgba),
                joint_index=int(joint_index),
                strip_index=int(strip_index),
                arc_index=int(arc_index),
            ))
    return boxes


def box_corners(box):
    """Return the 8 world-space corners of a SteelStripBox."""
    signs = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=np.float64)
    local = signs * box.half_size_m
    return box.center_m + local @ box.rotation.T


def generate_demo_chain_boxes(compressions_m, natural_spacing_m=0.151):
    """Generate visual-strip boxes for a straight chain preview."""
    all_boxes = []
    x = 0.0
    parent_rot = np.eye(3)
    for idx, compression in enumerate(compressions_m):
        distance = natural_spacing_m - float(compression)
        parent_pos = np.array([x, 0.0, 0.0], dtype=np.float64)
        child_pos = np.array([x + distance, 0.0, 0.0], dtype=np.float64)
        all_boxes.extend(generate_steel_strip_boxes(
            parent_pos,
            child_pos,
            parent_rot,
            natural_spacing_m,
            joint_index=idx,
        ))
        x += distance
    return all_boxes

"""
Smoke tests for visual steel-strip box geometry.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from visual_steel_strip_geometry_v6 import (  # noqa: E402
    ARC_SEGS,
    NUM_STRIPS,
    box_corners,
    compute_visual_bow_m,
    generate_demo_chain_boxes,
    generate_steel_strip_boxes,
)


def main():
    parent = np.array([0.0, 0.0, 0.0])
    child_relaxed = np.array([0.151, 0.0, 0.0])
    child_compressed = np.array([0.101, 0.0, 0.0])
    rot = np.eye(3)

    relaxed = generate_steel_strip_boxes(parent, child_relaxed, rot, 0.151)
    compressed = generate_steel_strip_boxes(
        parent, child_compressed, rot, 0.151)
    assert len(relaxed) == NUM_STRIPS * ARC_SEGS
    assert len(compressed) == NUM_STRIPS * ARC_SEGS

    relaxed_bow = compute_visual_bow_m(0.151, 0.151)
    compressed_bow = compute_visual_bow_m(0.101, 0.151)
    assert compressed_bow > relaxed_bow

    first = compressed[0]
    np.testing.assert_allclose(
        first.rotation.T @ first.rotation, np.eye(3), atol=1e-6)
    assert np.all(first.half_size_m > 0.0)
    assert box_corners(first).shape == (8, 3)

    rot_z = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    turned = generate_steel_strip_boxes(
        np.array([1.0, 2.0, 0.0]),
        np.array([1.0, 2.12, 0.0]),
        rot_z,
        0.151,
    )
    assert len(turned) == NUM_STRIPS * ARC_SEGS

    chain = generate_demo_chain_boxes([0.0, 0.025, 0.05])
    assert len(chain) == 3 * NUM_STRIPS * ARC_SEGS

    print("visual steel strip geometry passed")


if __name__ == "__main__":
    main()

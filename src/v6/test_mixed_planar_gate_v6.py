"""Checks for the mixed-planar gait-gate center used by V75."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from action_adapter_v6 import (  # noqa: E402
    COMMAND_GATE_LATERAL_CENTER,
    COMMAND_GATE_MIXED_CENTER,
    command_conditioned_gate_center,
)


def main():
    lateral = command_conditioned_gate_center((0.0, 1.0, 0.0))
    slow_forward_left = command_conditioned_gate_center((0.5, 1.0, 0.0))
    balanced_forward_left = command_conditioned_gate_center((1.0, 1.0, 0.0))
    axial_dominant = command_conditioned_gate_center((1.0, 0.5, 0.0))

    assert lateral == COMMAND_GATE_LATERAL_CENTER
    assert COMMAND_GATE_LATERAL_CENTER < slow_forward_left
    assert slow_forward_left < balanced_forward_left
    assert balanced_forward_left <= COMMAND_GATE_MIXED_CENTER
    assert axial_dominant > balanced_forward_left

    print("mixed planar gait gate checks passed")


if __name__ == "__main__":
    main()

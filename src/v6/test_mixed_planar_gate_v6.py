"""Checks for the optional mixed-planar gait-gate experiment."""

import importlib
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import action_adapter_v6 as action_adapter  # noqa: E402


def main():
    os.environ.pop("WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE", None)
    module = importlib.reload(action_adapter)
    lateral = module.command_conditioned_gate_center((0.0, 1.0, 0.0))
    default_slow_forward_left = module.command_conditioned_gate_center(
        (0.5, 1.0, 0.0))

    assert lateral == module.COMMAND_GATE_LATERAL_CENTER
    assert default_slow_forward_left == module.COMMAND_GATE_LATERAL_CENTER
    assert not module.MIXED_PLANAR_CONTINUOUS_GATE_ENABLED

    os.environ["WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE"] = "1"
    module = importlib.reload(module)
    assert module.MIXED_PLANAR_CONTINUOUS_GATE_ENABLED

    slow_forward_left = module.command_conditioned_gate_center(
        (0.5, 1.0, 0.0))
    balanced_forward_left = module.command_conditioned_gate_center(
        (1.0, 1.0, 0.0))
    axial_dominant = module.command_conditioned_gate_center((1.0, 0.5, 0.0))

    assert module.COMMAND_GATE_LATERAL_CENTER < slow_forward_left
    assert slow_forward_left < balanced_forward_left
    assert balanced_forward_left <= module.COMMAND_GATE_MIXED_CENTER
    assert axial_dominant > balanced_forward_left

    os.environ.pop("WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE", None)
    importlib.reload(module)

    print("mixed planar gait gate checks passed")


if __name__ == "__main__":
    main()

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from record_eval_video_v6 import command_for_time  # noqa: E402
from worm_env_v6 import CMD_VX_RANGE, CMD_VY_RANGE, CMD_YAW_RANGE  # noqa: E402


def main():
    fixed = command_for_time(
        "fixed", 0.5, 2.0, 0.33, -0.22, 0.8)
    assert fixed == (
        float(CMD_VX_RANGE[1]),
        float(CMD_VY_RANGE[0]),
        float(CMD_YAW_RANGE[1]),
    )

    times = np.linspace(0.0, 20.0, 201)
    cmds = np.asarray([
        command_for_time("continuous_sweep", t, 20.0, 0.0, 0.0, 0.0)
        for t in times
    ], dtype=np.float64)
    low_yaw_cmds = np.asarray([
        command_for_time(
            "continuous_sweep", t, 20.0, 0.0, 0.0, 0.0,
            dynamic_yaw_scale=0.25)
        for t in times
    ], dtype=np.float64)
    assert np.all(cmds[:, 0] <= CMD_VX_RANGE[1] + 1e-9)
    assert np.all(cmds[:, 0] >= CMD_VX_RANGE[0] - 1e-9)
    assert np.all(cmds[:, 1] <= CMD_VY_RANGE[1] + 1e-9)
    assert np.all(cmds[:, 1] >= CMD_VY_RANGE[0] - 1e-9)
    assert np.all(cmds[:, 2] <= CMD_YAW_RANGE[1] + 1e-9)
    assert np.all(cmds[:, 2] >= CMD_YAW_RANGE[0] - 1e-9)
    assert np.any(cmds[:, 0] > 0.05)
    assert np.any(cmds[:, 0] < -0.05)
    assert np.any(cmds[:, 1] > 0.03)
    assert np.any(cmds[:, 1] < -0.03)
    assert np.any(cmds[:, 2] > 0.03)
    assert np.any(cmds[:, 2] < -0.03)
    assert np.max(np.abs(np.diff(cmds, axis=0))) < 0.04
    assert np.allclose(low_yaw_cmds[:, :2], cmds[:, :2])
    assert np.max(np.abs(low_yaw_cmds[:, 2])) < (
        0.30 * np.max(np.abs(cmds[:, 2])))

    print("dynamic record command schedule passed")


if __name__ == "__main__":
    main()

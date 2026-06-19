import math

from train_v6 import best_eval_schedule
from worm_env_v6 import CMD_VX_RANGE, CMD_VY_RANGE


def _case_by_name(schedule, name):
    matches = [case for case in schedule if case["case_name"] == name]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def _assert_command(case, vx, vy, yaw):
    assert math.isclose(case["cmd_vx_m_s"], vx, abs_tol=1e-12), case
    assert math.isclose(case["cmd_vy_m_s"], vy, abs_tol=1e-12), case
    assert math.isclose(case["cmd_yaw_rad_s"], yaw, abs_tol=1e-12), case


def test_best_eval_schedule_includes_v74_hard_mixed_cases():
    schedule = best_eval_schedule("random")
    _assert_command(
        _case_by_name(schedule, "mixed_slow_forward_full_left"),
        CMD_VX_RANGE[1] * 0.5,
        CMD_VY_RANGE[1],
        0.0,
    )
    _assert_command(
        _case_by_name(schedule, "mixed_slow_forward_full_right"),
        CMD_VX_RANGE[1] * 0.5,
        CMD_VY_RANGE[0],
        0.0,
    )
    _assert_command(
        _case_by_name(schedule, "mixed_slow_reverse_full_left"),
        CMD_VX_RANGE[0] * 0.5,
        CMD_VY_RANGE[1],
        0.0,
    )
    _assert_command(
        _case_by_name(schedule, "mixed_slow_reverse_full_right"),
        CMD_VX_RANGE[0] * 0.5,
        CMD_VY_RANGE[0],
        0.0,
    )


if __name__ == "__main__":
    test_best_eval_schedule_includes_v74_hard_mixed_cases()
    print("best eval hardcase schedule checks passed")

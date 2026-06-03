"""Checks for the optional mixed-planar gait-gate experiment."""

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import action_adapter_v6 as action_adapter  # noqa: E402


EXPERIMENT_FLAGS = (
    "WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE",
    "WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE",
    "WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE",
    "WORM_V6_ENABLE_MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR",
    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE",
    "WORM_V6_SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN",
    "WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN",
    "WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN",
    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW",
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN",
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT",
    "WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE",
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD",
    "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN",
    "WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT",
    "WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT",
    "WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE",
    "WORM_V6_SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT",
    "WORM_V6_SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR",
    "WORM_V6_SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD",
)


def _reload_without_flags():
    for name in EXPERIMENT_FLAGS:
        os.environ.pop(name, None)
    return importlib.reload(action_adapter)


def test_mixed_planar_gate_experiment_flags():
    os.environ.pop("WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE", None)
    os.environ.pop("WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE", None)
    module = _reload_without_flags()
    lateral = module.command_conditioned_gate_center((0.0, 1.0, 0.0))
    default_slow_forward_left = module.command_conditioned_gate_center(
        (0.5, 1.0, 0.0))
    default_slow_reverse_left = module.command_conditioned_gate_center(
        (-0.5, 1.0, 0.0))

    assert lateral == module.COMMAND_GATE_LATERAL_CENTER
    assert default_slow_forward_left == module.COMMAND_GATE_LATERAL_CENTER
    assert default_slow_reverse_left == module.COMMAND_GATE_LATERAL_CENTER
    assert not module.MIXED_PLANAR_CONTINUOUS_GATE_ENABLED
    assert not module.MIXED_PLANAR_HARDCASE_GATE_ENABLED

    os.environ["WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE"] = "1"
    module = importlib.reload(module)
    assert module.MIXED_PLANAR_HARDCASE_GATE_ENABLED
    assert module.command_conditioned_gate_center(
        (0.5, 1.0, 0.0)) == module.MIXED_PLANAR_HARDCASE_GATE_CENTER
    assert module.command_conditioned_gate_center(
        (0.0, 1.0, 0.0)) == module.COMMAND_GATE_LATERAL_CENTER
    assert module.command_conditioned_gate_center(
        (-0.5, 1.0, 0.0)) == module.COMMAND_GATE_LATERAL_CENTER
    os.environ.pop("WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE", None)

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
    os.environ.pop("WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE", None)
    importlib.reload(module)


def test_mixed_planar_full_channel_split_prior_is_opt_in():
    module = _reload_without_flags()
    assert not module.MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED

    phase = 0.37
    gait_blend = 0.5
    command = (0.5, 1.0, 0.0)
    legacy = module.split_channel_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, command)
    axial = module.directional_gait_prior_from_phase(
        phase, gait_blend, (1.0, 0.0, 0.0))
    lateral = module.lateral_primitive_action_from_phase("left", phase)

    np.testing.assert_allclose(
        legacy[:module.NUM_SLIDES],
        0.5 * axial[:module.NUM_SLIDES],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        legacy[module.NUM_SLIDES:],
        lateral[module.NUM_SLIDES:],
        atol=1e-6,
    )

    os.environ["WORM_V6_ENABLE_MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR"] = "1"
    module = importlib.reload(module)
    assert module.MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR_ENABLED
    full = module.split_channel_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, command)
    axis_norm = np.sqrt(0.5 ** 2 + 1.0 ** 2)
    expected = np.zeros(module.NUM_ACTUATORS, dtype=np.float32)
    expected[:module.NUM_SLIDES] = (
        0.5 * axial[:module.NUM_SLIDES]
        + module.MIXED_LATERAL_SLIDE_GAIN * lateral[:module.NUM_SLIDES]
    ) / axis_norm
    expected[module.NUM_SLIDES:] = (
        module.MIXED_AXIAL_YAW_GAIN
        * 0.5
        * axial[module.NUM_SLIDES:]
        + module.MIXED_LATERAL_YAW_GAIN
        * lateral[module.NUM_SLIDES:]
    ) / axis_norm
    expected = np.clip(expected, -1.0, 1.0)

    np.testing.assert_allclose(full, expected, atol=1e-6)
    assert np.linalg.norm(full[:module.NUM_SLIDES]
                          - legacy[:module.NUM_SLIDES]) > 1e-3
    assert np.linalg.norm(full[module.NUM_SLIDES:]
                          - legacy[module.NUM_SLIDES:]) > 1e-3
    _reload_without_flags()


def test_mixed_planar_authority_profile_overrides_are_opt_in():
    module = _reload_without_flags()
    command = (0.5, 0.5, 0.0)
    assert not module.MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED
    assert module.command_conditioned_prior_authority_scale(*command) == 1.0
    assert module.command_conditioned_residual_scale(
        *command) == module.MIXED_PLANAR_RESIDUAL_SCALE_MULT

    os.environ["WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE"] = "1"
    os.environ["WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT"] = "0.42"
    os.environ["WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT"] = "3.25"
    module = importlib.reload(module)

    assert module.MIXED_PLANAR_AUTHORITY_REBALANCE_ENABLED
    assert module.MIXED_PLANAR_PROFILE_PRIOR_AUTHORITY_MULT == 0.42
    assert module.MIXED_PLANAR_PROFILE_RESIDUAL_SCALE_MULT == 3.25
    assert module.command_conditioned_prior_authority_scale(*command) == 0.42
    assert module.command_conditioned_residual_scale(*command) == 3.25
    assert module.command_conditioned_prior_authority_scale(
        1.0, 0.0, 0.0) == 1.0
    assert module.command_conditioned_residual_scale(1.0, 0.0, 0.0) == 1.0

    _reload_without_flags()


def test_slope_forward_axis_profile_is_opt_in():
    module = _reload_without_flags()
    forward_command = (0.5, 0.0, 0.0)
    mixed_command = (0.5, -1.0, 0.0)
    yaw_mixed_command = (0.5, 0.0, 0.5)

    assert not module.SLOPE_FORWARD_AXIS_PROFILE_ENABLED
    assert module.command_conditioned_prior_authority_scale(
        *forward_command) == 1.0
    assert module.command_conditioned_prior_authority_scale(
        *mixed_command) == 1.0
    assert module.command_conditioned_gait_blend(
        0.0, forward_command) < 0.55

    os.environ["WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE"] = "1"
    os.environ["WORM_V6_SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT"] = "1.45"
    os.environ["WORM_V6_SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR"] = "0.55"
    os.environ["WORM_V6_SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD"] = "1.25"
    module = importlib.reload(module)

    assert module.SLOPE_FORWARD_AXIS_PROFILE_ENABLED
    assert module.SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT == 1.45
    assert module.SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR == 0.55
    assert module.SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD == 1.25
    assert module.command_conditioned_prior_authority_scale(
        *forward_command) == 1.45
    assert module.command_conditioned_prior_authority_scale(
        *mixed_command) == 1.45
    assert module.command_conditioned_prior_authority_scale(
        -0.5, 0.0, 0.0) == 1.0
    assert module.command_conditioned_prior_authority_scale(
        *yaw_mixed_command) == 1.0
    assert module.command_conditioned_gait_blend(
        0.0, forward_command) == 0.55
    assert module.command_conditioned_gait_blend(
        0.0, mixed_command) == 0.55
    assert module.command_directional_prior_transform(
        *forward_command)["phase_offset_rad"] == 1.25

    _reload_without_flags()


def test_deploy_full_channel_split_prior_matches_adapter():
    os.environ["WORM_V6_ENABLE_MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR"] = "1"
    module = importlib.reload(action_adapter)
    import deploy_policy_v6 as deploy_policy  # noqa: E402
    deploy_policy = importlib.reload(deploy_policy)

    dummy_policy = SimpleNamespace(
        features_extractor=torch.nn.Identity(),
        mlp_extractor=torch.nn.Identity(),
        action_net=torch.nn.Identity(),
    )
    actor = deploy_policy.DeployablePPOActor(
        dummy_policy,
        obs_mean=np.zeros(80, dtype=np.float32),
        obs_var=np.ones(80, dtype=np.float32),
        epsilon=1e-8,
        clip_obs=10.0,
    )
    phase = 0.37
    gait_blend = 0.5
    command = (0.5, -1.0, 0.0)
    with torch.no_grad():
        deploy_prior = actor._split_channel_mixed_planar_prior(
            torch.tensor([[phase]], dtype=torch.float32),
            torch.tensor([[gait_blend]], dtype=torch.float32),
            torch.tensor([[command[0]]], dtype=torch.float32),
            torch.tensor([[command[1]]], dtype=torch.float32),
        ).cpu().numpy()[0]
    adapter_prior = module.split_channel_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, command)

    np.testing.assert_allclose(deploy_prior, adapter_prior, atol=1e-6)
    _reload_without_flags()
    importlib.reload(deploy_policy)


def test_slope_mixed_planar_primitive_is_opt_in():
    module = _reload_without_flags()
    assert not module.SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED

    phase = 0.37
    gait_blend = 0.5
    command = (0.5, -1.0, 0.0)
    default_prior = module.directional_gait_prior_from_phase(
        phase, gait_blend, command)

    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE"] = "1"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN"] = "0.9"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN"] = "0.8"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN"] = "1.2"
    module = importlib.reload(module)

    assert module.SLOPE_MIXED_PLANAR_PRIMITIVE_ENABLED
    assert module.SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN == 0.9
    assert module.SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN == 0.8
    assert module.SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN == 1.2
    assert not module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED
    assert not module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED

    slope_prior = module.directional_gait_prior_from_phase(
        phase, gait_blend, command)
    axial = module.directional_gait_prior_from_phase(
        phase, gait_blend, (1.0, 0.0, 0.0))
    lateral = module.lateral_primitive_action_from_phase("right", phase)
    axis_norm = np.sqrt(0.5 ** 2 + 1.0 ** 2)
    expected = np.zeros(module.NUM_ACTUATORS, dtype=np.float32)
    expected[:module.NUM_SLIDES] = (
        0.9 * 0.5 * axial[:module.NUM_SLIDES]
        + 0.8 * lateral[:module.NUM_SLIDES]
    ) / axis_norm
    expected[module.NUM_SLIDES:] = (
        1.2 * lateral[module.NUM_SLIDES:]
    ) / axis_norm
    expected = np.clip(expected, -1.0, 1.0)

    np.testing.assert_allclose(slope_prior, expected, atol=1e-6)
    assert np.linalg.norm(slope_prior - default_prior) > 1e-3
    _reload_without_flags()


def test_slope_mixed_positive_vx_axial_yaw_is_opt_in():
    module = _reload_without_flags()
    phase = 0.37
    gait_blend = 0.5
    positive_command = (0.5, -1.0, 0.0)
    negative_command = (-0.5, -1.0, 0.0)

    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE"] = "1"
    module = importlib.reload(module)
    default_positive = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, positive_command)
    default_negative = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, negative_command)
    assert not module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED

    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW"] = "1"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN"] = "0.35"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT"] = "0.75"
    module = importlib.reload(module)

    assert module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_ENABLED
    assert module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN == 0.35
    assert module.SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT == 0.75
    positive = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, positive_command)
    negative = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, negative_command)

    axial = module.directional_gait_prior_from_phase(
        phase, gait_blend, (1.0, 0.0, 0.0))
    lateral = module.lateral_primitive_action_from_phase("right", phase)
    axis_norm = np.sqrt(0.5 ** 2 + 1.0 ** 2)
    expected_yaws = (
        0.75
        * module.SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN
        * lateral[module.NUM_SLIDES:]
        + 0.35 * 0.5 * axial[module.NUM_SLIDES:]
    ) / axis_norm

    np.testing.assert_allclose(
        positive[module.NUM_SLIDES:],
        np.clip(expected_yaws, -1.0, 1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(negative, default_negative, atol=1e-6)
    assert np.linalg.norm(positive[module.NUM_SLIDES:]
                          - default_positive[module.NUM_SLIDES:]) > 1e-3
    _reload_without_flags()


def test_slope_mixed_positive_vx_axial_phase_is_opt_in():
    module = _reload_without_flags()
    phase = 0.37
    gait_blend = 0.5
    positive_command = (0.5, -1.0, 0.0)
    negative_command = (-0.5, -1.0, 0.0)

    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE"] = "1"
    module = importlib.reload(module)
    default_positive = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, positive_command)
    default_negative = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, negative_command)
    assert not module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED

    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE"] = "1"
    os.environ[
        "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD"
    ] = "1.75"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN"] = "-1"
    module = importlib.reload(module)

    assert module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_ENABLED
    assert module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD == 1.75
    assert module.SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN == -1.0
    positive = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, positive_command)
    negative = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, negative_command)

    axial = module.gait_prior_from_phase(phase + 1.75, gait_blend)
    lateral = module.lateral_primitive_action_from_phase("right", phase)
    axis_norm = np.sqrt(0.5 ** 2 + 1.0 ** 2)
    expected = np.zeros(module.NUM_ACTUATORS, dtype=np.float32)
    expected[:module.NUM_SLIDES] = (
        module.SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN
        * 0.5
        * -1.0
        * axial[:module.NUM_SLIDES]
        + module.SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN
        * lateral[:module.NUM_SLIDES]
    ) / axis_norm
    expected[module.NUM_SLIDES:] = (
        module.SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN
        * lateral[module.NUM_SLIDES:]
    ) / axis_norm
    expected = np.clip(expected, -1.0, 1.0)

    np.testing.assert_allclose(positive, expected, atol=1e-6)
    np.testing.assert_allclose(negative, default_negative, atol=1e-6)
    assert np.linalg.norm(positive[:module.NUM_SLIDES]
                          - default_positive[:module.NUM_SLIDES]) > 1e-3
    _reload_without_flags()


def test_deploy_slope_forward_axis_profile_matches_adapter():
    os.environ["WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE"] = "1"
    os.environ["WORM_V6_SLOPE_FORWARD_AXIS_PRIOR_AUTHORITY_MULT"] = "1.45"
    os.environ["WORM_V6_SLOPE_FORWARD_AXIS_GAIT_BLEND_FLOOR"] = "0.55"
    os.environ["WORM_V6_SLOPE_FORWARD_AXIS_PHASE_OFFSET_RAD"] = "1.25"
    module = importlib.reload(action_adapter)
    import deploy_policy_v6 as deploy_policy  # noqa: E402
    deploy_policy = importlib.reload(deploy_policy)

    class DummyMlp(torch.nn.Module):
        def forward_actor(self, features):
            return features

    class ZeroActionNet(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(
                (features.shape[0], module.POLICY_ACTION_DIM),
                dtype=features.dtype,
                device=features.device,
            )

    dummy_policy = SimpleNamespace(
        features_extractor=torch.nn.Identity(),
        mlp_extractor=DummyMlp(),
        action_net=ZeroActionNet(),
    )
    actor = deploy_policy.DeployablePPOActor(
        dummy_policy,
        obs_mean=np.zeros(80, dtype=np.float32),
        obs_var=np.ones(80, dtype=np.float32),
        epsilon=1e-8,
        clip_obs=10.0,
    )
    phase = 0.37
    for command in ((0.5, 0.0, 0.0), (0.5, -1.0, 0.0)):
        raw_obs = np.zeros((1, 80), dtype=np.float32)
        raw_obs[0, 0:3] = np.asarray(command, dtype=np.float32)
        raw_obs[0, 78] = np.sin(phase)
        raw_obs[0, 79] = np.cos(phase)
        with torch.no_grad():
            deploy_action = actor(
                torch.as_tensor(raw_obs, dtype=torch.float32)).cpu().numpy()[0]
        expected = module.compose_deployable_action(
            np.zeros(module.NUM_ACTUATORS, dtype=np.float32),
            phase=phase,
            gait_blend=module.command_conditioned_gait_blend(0.5, command),
            command=command,
        )
        np.testing.assert_allclose(deploy_action, expected, atol=1e-6)

    _reload_without_flags()
    importlib.reload(deploy_policy)


def test_deploy_slope_mixed_planar_primitive_matches_adapter():
    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE"] = "1"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_AXIAL_SLIDE_GAIN"] = "0.9"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_SLIDE_GAIN"] = "0.8"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_LATERAL_YAW_GAIN"] = "1.2"
    module = importlib.reload(action_adapter)
    import deploy_policy_v6 as deploy_policy  # noqa: E402
    deploy_policy = importlib.reload(deploy_policy)

    dummy_policy = SimpleNamespace(
        features_extractor=torch.nn.Identity(),
        mlp_extractor=torch.nn.Identity(),
        action_net=torch.nn.Identity(),
    )
    actor = deploy_policy.DeployablePPOActor(
        dummy_policy,
        obs_mean=np.zeros(80, dtype=np.float32),
        obs_var=np.ones(80, dtype=np.float32),
        epsilon=1e-8,
        clip_obs=10.0,
    )
    phase = 0.37
    gait_blend = 0.5
    command = (0.5, -1.0, 0.0)
    with torch.no_grad():
        deploy_prior = actor._slope_mixed_planar_prior(
            torch.tensor([[phase]], dtype=torch.float32),
            torch.tensor([[gait_blend]], dtype=torch.float32),
            torch.tensor([[command[0]]], dtype=torch.float32),
            torch.tensor([[command[1]]], dtype=torch.float32),
        ).cpu().numpy()[0]
    adapter_prior = module.slope_mixed_planar_gait_prior_from_phase(
        phase, gait_blend, command)

    np.testing.assert_allclose(deploy_prior, adapter_prior, atol=1e-6)
    _reload_without_flags()
    importlib.reload(deploy_policy)


def test_deploy_slope_mixed_positive_vx_axial_yaw_matches_adapter():
    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE"] = "1"
    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW"] = "1"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_YAW_GAIN"] = "0.35"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_LATERAL_YAW_MULT"] = "0.75"
    module = importlib.reload(action_adapter)
    import deploy_policy_v6 as deploy_policy  # noqa: E402
    deploy_policy = importlib.reload(deploy_policy)

    dummy_policy = SimpleNamespace(
        features_extractor=torch.nn.Identity(),
        mlp_extractor=torch.nn.Identity(),
        action_net=torch.nn.Identity(),
    )
    actor = deploy_policy.DeployablePPOActor(
        dummy_policy,
        obs_mean=np.zeros(80, dtype=np.float32),
        obs_var=np.ones(80, dtype=np.float32),
        epsilon=1e-8,
        clip_obs=10.0,
    )
    phase = 0.37
    gait_blend = 0.5
    for command in ((0.5, -1.0, 0.0), (-0.5, -1.0, 0.0)):
        with torch.no_grad():
            deploy_prior = actor._slope_mixed_planar_prior(
                torch.tensor([[phase]], dtype=torch.float32),
                torch.tensor([[gait_blend]], dtype=torch.float32),
                torch.tensor([[command[0]]], dtype=torch.float32),
                torch.tensor([[command[1]]], dtype=torch.float32),
            ).cpu().numpy()[0]
        adapter_prior = module.slope_mixed_planar_gait_prior_from_phase(
            phase, gait_blend, command)
        np.testing.assert_allclose(deploy_prior, adapter_prior, atol=1e-6)

    _reload_without_flags()
    importlib.reload(deploy_policy)


def test_deploy_slope_mixed_positive_vx_axial_phase_matches_adapter():
    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE"] = "1"
    os.environ["WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE"] = "1"
    os.environ[
        "WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_PHASE_OFFSET_RAD"
    ] = "1.75"
    os.environ["WORM_V6_SLOPE_MIXED_PLANAR_POSITIVE_VX_AXIAL_SLIDE_SIGN"] = "-1"
    module = importlib.reload(action_adapter)
    import deploy_policy_v6 as deploy_policy  # noqa: E402
    deploy_policy = importlib.reload(deploy_policy)

    dummy_policy = SimpleNamespace(
        features_extractor=torch.nn.Identity(),
        mlp_extractor=torch.nn.Identity(),
        action_net=torch.nn.Identity(),
    )
    actor = deploy_policy.DeployablePPOActor(
        dummy_policy,
        obs_mean=np.zeros(80, dtype=np.float32),
        obs_var=np.ones(80, dtype=np.float32),
        epsilon=1e-8,
        clip_obs=10.0,
    )
    phase = 0.37
    gait_blend = 0.5
    for command in ((0.5, -1.0, 0.0), (-0.5, -1.0, 0.0)):
        with torch.no_grad():
            deploy_prior = actor._slope_mixed_planar_prior(
                torch.tensor([[phase]], dtype=torch.float32),
                torch.tensor([[gait_blend]], dtype=torch.float32),
                torch.tensor([[command[0]]], dtype=torch.float32),
                torch.tensor([[command[1]]], dtype=torch.float32),
            ).cpu().numpy()[0]
        adapter_prior = module.slope_mixed_planar_gait_prior_from_phase(
            phase, gait_blend, command)
        np.testing.assert_allclose(deploy_prior, adapter_prior, atol=1e-6)

    _reload_without_flags()
    importlib.reload(deploy_policy)


def main():
    test_mixed_planar_gate_experiment_flags()
    test_mixed_planar_full_channel_split_prior_is_opt_in()
    test_mixed_planar_authority_profile_overrides_are_opt_in()
    test_slope_forward_axis_profile_is_opt_in()
    test_slope_mixed_planar_primitive_is_opt_in()
    test_slope_mixed_positive_vx_axial_yaw_is_opt_in()
    test_slope_mixed_positive_vx_axial_phase_is_opt_in()
    test_deploy_full_channel_split_prior_matches_adapter()
    test_deploy_slope_forward_axis_profile_matches_adapter()
    test_deploy_slope_mixed_planar_primitive_matches_adapter()
    test_deploy_slope_mixed_positive_vx_axial_yaw_matches_adapter()
    test_deploy_slope_mixed_positive_vx_axial_phase_matches_adapter()
    print("mixed planar gait gate checks passed")


if __name__ == "__main__":
    main()

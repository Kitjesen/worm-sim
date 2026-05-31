# V55 Mixed Planar Split Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Worm V6 flat mixed `vx/vy` tracking without changing the 80D observation or 12D action ABI.

**Architecture:** Current mixed planar commands fail because the default prior picks one dominant primitive, so either axial or lateral motion disappears. V55 adds an experimental split-channel mixed planar prior: slide actuators come from the signed axial primitive, yaw actuators come from the signed lateral primitive, and PPO residual authority stays unchanged unless an explicit experiment flag is enabled.

**Tech Stack:** Python, MuJoCo, Stable-Baselines3 PPO, TorchScript export parity tests.

---

### Task 1: Lock the Desired V55 Prior Behavior

**Files:**
- Modify: `src/v6/test_reward_contract_v6.py`

- [x] **Step 1: Add a failing unit test**

Add a test that imports `split_channel_mixed_planar_gait_prior_from_phase()` and asserts:

```python
phase = 0.37
gait_blend = 0.5
prior = split_channel_mixed_planar_gait_prior_from_phase(
    phase, gait_blend, (1.0, 1.0, 0.0))
axial = directional_gait_prior_from_phase(phase, gait_blend, (1.0, 0.0, 0.0))
lateral = lateral_primitive_action_from_phase("left", phase)
np.testing.assert_allclose(prior[:NUM_SLIDES], axial[:NUM_SLIDES])
np.testing.assert_allclose(prior[NUM_SLIDES:], lateral[NUM_SLIDES:])
```

- [x] **Step 2: Run test to verify it fails**

Run: `python src/v6/test_reward_contract_v6.py`

Expected: FAIL because the split-channel helper is not implemented yet.

### Task 2: Implement the Experimental Split Prior

**Files:**
- Modify: `src/v6/action_adapter_v6.py`
- Modify: `src/v6/deploy_policy_v6.py`
- Modify: `src/v6/test_reward_contract_v6.py`
- Modify: `src/v6/test_deploy_policy_v6.py`

- [x] **Step 1: Add V55 constants and contract fields**

Add default-off flag:

```python
MIXED_PLANAR_SPLIT_PRIOR_EXPERIMENTAL_AVAILABLE = True
MIXED_PLANAR_SPLIT_PRIOR_ENABLED = _env_flag(
    "WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR", default=False)
```

- [x] **Step 2: Add NumPy split-channel prior**

For mixed `vx/vy` with zero yaw:

```python
slides = axial_prior[:NUM_SLIDES] * vx_gain
yaws = lateral_prior[NUM_SLIDES:] * vy_gain
```

Normalize each channel group by its own active-axis gain floor so pure command behavior is not reused for inactive axes.

- [x] **Step 3: Wire the prior behind the flag**

In `directional_gait_prior_from_phase()`, if the split flag is enabled and command is mixed planar, return the split prior before falling back to the dominant prior.

- [x] **Step 4: Mirror the same logic in TorchScript export**

Add `_split_mixed_planar_prior()` to `DeployablePPOActor` and select it under the same flag.

- [x] **Step 5: Run tests**

Run:

```powershell
python src\v6\test_reward_contract_v6.py
$env:WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR='1'; python src\v6\test_deploy_policy_v6.py; Remove-Item Env:\WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR
```

Expected: both pass.

### Task 3: Evaluate Before Promoting

**Files:**
- Create: `record/current/flat_omni_v55_split_prior_no_retrain_v53final_scan/*`
- Modify: `README.md`
- Modify: `docs/progress.md`
- Modify: `docs/omni_v41_v46_motion_summary.md`

- [x] **Step 1: Run a no-retrain scan**

Run the V53 final model with `WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR=1` and the normal 35-command strict scan.

- [x] **Step 2: Analyze acceptance**

Accept the V55 split prior only if it reduces mixed `vx/vy` planar RMSE without reintroducing wrong signs or increasing yaw-only drift above the V53 default guard.

- [x] **Step 3: Record the result**

Update docs with measured metrics. If the scan fails, keep the split prior default-off and record it as an ablation, not a paper result.

## Execution Result

V55 was rejected. The no-retrain scan reported `planar_rmse_m_s=0.2012` and
`wrong_planar_sign_count=8`; the short continuation still reported
`planar_rmse_m_s≈0.202` with `7-10` wrong planar signs. The split prior remains
default-off and should be treated as an ablation.

V56 then added full-speed diagonal commands to `best_eval_schedule`. It
preserved zero wrong planar and yaw signs, with best scan
`planar_rmse_m_s=0.1509`, `yaw_rmse_rad_s=0.1646`, and
`yaw_only_mean_planar_speed_m_s=0.0873`, but it is still rejected by the strict
continuous-tracking gate.

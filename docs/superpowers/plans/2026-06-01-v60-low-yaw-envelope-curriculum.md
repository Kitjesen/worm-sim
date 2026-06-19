# V60 Low-Yaw Envelope Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-stage low-yaw feasible-envelope curriculum so Worm V6 trains pure yaw stability and slow command tracking before returning to full omni commands.

**Architecture:** Keep the 80D observation and 12D action ABI unchanged. Add one new command curriculum in `worm_env_v6.py` that samples stop, low pure-yaw, slow axial, and slow forward+yaw commands. Lock the curriculum distribution and command bounds with reward-contract tests before using it for V60/V61 training.

**Tech Stack:** Python, MuJoCo, Stable-Baselines3 PPO, existing V6 reward-contract tests.

---

### Task 1: Lock Low-Yaw Curriculum Sampling

**Files:**
- Modify: `src/v6/test_reward_contract_v6.py`

- [x] **Step 1: Add a failing unit test**

Add assertions that `low_yaw_envelope` is in `COMMAND_CURRICULA`, samples exact stop commands, samples pure yaw commands bounded to `0.08..0.12 rad/s`, samples slow axial commands bounded to `0.05..0.10 m/s`, samples forward+yaw mixed commands, and never emits lateral commands.

- [x] **Step 2: Run test to verify it fails**

Run: `python src/v6/test_reward_contract_v6.py`
Expected: FAIL because `low_yaw_envelope` is not implemented yet.

### Task 2: Implement Curriculum

**Files:**
- Modify: `src/v6/worm_env_v6.py`

- [x] **Step 1: Add constants**

Add low-yaw envelope constants near command ranges: yaw absolute range `0.08..0.12`, slow axial absolute range `0.05..0.10`.

- [x] **Step 2: Add sampling branch**

Add `_sample_command()` branch:
- 30% stop `(0,0,0)`;
- 40% pure yaw `(0,0,±0.08..±0.12)`;
- 20% slow axial `(±0.05..±0.10,0,0)`;
- 10% slow forward plus yaw `(+0.05..+0.10,0,±0.08..±0.12)`.

- [x] **Step 3: Run tests**

Run:
`python src/v6/test_reward_contract_v6.py`
`python src/v6/test_deployable_obs_v6.py`

### Task 3: Document V60 Command

**Files:**
- Modify: `docs/omni_locomotion_training_research_report.md`

- [x] **Step 1: Update V61 section**

Replace the proposed `low_yaw_envelope` pseudo-curriculum with the exact implemented curriculum name and train command.

- [x] **Step 2: Verify docs diff**

Run: `git diff --check -- docs/omni_locomotion_training_research_report.md src/v6/worm_env_v6.py src/v6/test_reward_contract_v6.py`

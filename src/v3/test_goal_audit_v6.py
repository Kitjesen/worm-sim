"""
Smoke tests for formal artifact audit rules.
"""

import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import audit_paper_goal_v6 as audit  # noqa: E402


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def training_config(timesteps):
    return {
        "obs_dim": audit.OBS_DIM,
        "terrain": "flat",
        "gait_mode": "worm",
        "num_actuators": 11,
        "num_imus": 7,
        "control_timing": dict(audit.CONTROL_TIMING_DEFAULTS),
        "actuator_contract_fingerprint": (
            audit.ACTUATOR_CONTRACT_FINGERPRINT),
        "sensor_robustness": dict(audit.ROBUST_SENSOR_DEFAULTS),
        "training": {"timesteps": timesteps},
    }


def eval_metrics(episodes, time_s, condition="nominal"):
    sensor = (
        audit.ROBUST_EVAL_DEFAULTS if condition == "robust"
        else audit.NOMINAL_SENSOR_DEFAULTS)
    return {
        "obs_dim": audit.OBS_DIM,
        "terrain": "flat",
        "gait_mode": "worm",
        "gait_blend": 0.0,
        "eval_condition": condition,
        "control_timing": dict(audit.CONTROL_TIMING_DEFAULTS),
        "actuator_contract_fingerprint": (
            audit.ACTUATOR_CONTRACT_FINGERPRINT),
        "episodes": episodes,
        "time_s": time_s,
        "sensor_noise": dict(sensor),
        "mean_speed_mm_s": 1.0,
        "mean_action_l2_per_m": 1.0,
        "mean_lateral_drift_mm": 0.0,
        "success_rate": 1.0,
        "termination_rate": 0.0,
    }


def cmaes_metrics(generations):
    return {
        "best_speed_mm_s": 1.0,
        "best_displacement_mm": 10.0,
        "sim_time_s": 10.0,
        "best_params": [0.0] * 14,
        "generations": generations,
        "total_evals": generations * 8,
        "history": [],
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "runs", "worm_v6_ppo_flat_worm")
        os.makedirs(run_dir, exist_ok=True)
        open(os.path.join(run_dir, "best_model.zip"), "wb").close()
        open(os.path.join(run_dir, "best_model_vecnormalize.pkl"), "wb").close()

        write_json(
            os.path.join(run_dir, "training_config.json"),
            training_config(10_000))
        write_json(
            os.path.join(run_dir, "training_result.json"),
            {"completed_timesteps": 10_000})
        ok, info = audit.valid_training_run(run_dir, "flat", "worm")
        assert not ok
        assert "completed_timesteps below formal threshold" in info["reasons"]

        write_json(
            os.path.join(run_dir, "training_config.json"),
            training_config(audit.REQUIRED_TRAIN_TIMESTEPS))
        write_json(
            os.path.join(run_dir, "training_result.json"),
            {"completed_timesteps": audit.REQUIRED_TRAIN_TIMESTEPS})
        ok, _ = audit.valid_training_run(run_dir, "flat", "worm")
        assert ok

        cmaes_path = os.path.join(
            tmp, "runs", "cmaes_flat_peristaltic", "best_gait.json")
        write_json(cmaes_path, cmaes_metrics(1))
        ok, reasons = audit.valid_cmaes_baseline(cmaes_path)
        assert not ok and "generations" in reasons
        write_json(cmaes_path, cmaes_metrics(
            audit.REQUIRED_CMAES_MIN_GENERATIONS))
        ok, reasons = audit.valid_cmaes_baseline(cmaes_path)
        assert ok, reasons

    ok, reasons = audit.valid_eval_metrics(
        eval_metrics(1, audit.REQUIRED_EVAL_TIME_S), "flat", "worm",
        "nominal")
    assert not ok and "episodes" in reasons

    ok, reasons = audit.valid_eval_metrics(
        eval_metrics(audit.REQUIRED_EVAL_EPISODES,
                     audit.REQUIRED_EVAL_TIME_S,
                     condition="robust"),
        "flat", "worm", "robust")
    assert ok, reasons

    missing_saturation = eval_metrics(
        audit.REQUIRED_EVAL_EPISODES,
        audit.REQUIRED_EVAL_TIME_S,
        condition="robust")
    missing_saturation["sensor_noise"] = dict(audit.ROBUST_SENSOR_DEFAULTS)
    ok, reasons = audit.valid_eval_metrics(
        missing_saturation, "flat", "worm", "robust")
    assert not ok and "sensor_noise" in reasons
    print("goal audit contract passed")


if __name__ == "__main__":
    main()

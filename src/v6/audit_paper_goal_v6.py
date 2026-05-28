"""
Audit completion of the deployable multi-modal Worm V6 paper goal.

This script does not train or evaluate. It inspects the current artifacts and
reports which requirements are proven, missing, or incomplete.
"""

import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from worm_env_v6 import (  # noqa: E402
    CMD_VEL_RANGE,
    CTRL_DT,
    OBS_DIM,
    PERISTALTIC_ACTUATION_PERIOD_S,
    PHASE_FREQ,
    reward_contract,
)
from observation_contract_v6 import observation_contract, write_contract  # noqa: E402
from motor_contract_v6 import (  # noqa: E402
    action_mapping_matches,
    motor_contract,
)
from action_adapter_v6 import action_adapter_contract  # noqa: E402
from training_contract_v6 import residual_exploration_contract  # noqa: E402
from validate_hardware_log_v6 import (  # noqa: E402
    observation_columns,
    validate_csv,
)

PAPER_TERRAINS = ("flat", "sand", "slope")
FIXED_MODES = ("worm", "snake", "mixed")
TRAIN_MODES = ("worm", "snake", "mixed", "random")
MODE_TO_CMAES = {
    "worm": "peristaltic",
    "snake": "serpentine",
    "mixed": "full",
}
REQUIRED_TRAIN_TIMESTEPS = 1_000_000
REQUIRED_EVAL_EPISODES = 5
REQUIRED_EVAL_TIME_S = 20.0
REQUIRED_SCAN_BLENDS = {0.0, 0.25, 0.5, 0.75, 1.0}
REQUIRED_CMAES_MIN_GENERATIONS = 20
REQUIRED_HARDWARE_MIN_ROWS = 5
REQUIRED_HARDWARE_MIN_DURATION_S = 0.1
DEFAULT_RESULTS_INDEX = os.path.join(
    PROJECT_ROOT, "record", "v6", "paper_results", "results_index.md")
ROBUST_SENSOR_DEFAULTS = {
    "encoder_pos_noise_std": 0.01,
    "encoder_vel_noise_std": 0.02,
    "imu_gravity_noise_std": 0.01,
    "imu_gyro_noise_std": 0.01,
    "action_delay_steps": 1,
}
ROBUST_EVAL_DEFAULTS = {
    **ROBUST_SENSOR_DEFAULTS,
    "action_saturation": 0.8,
}
CONTROL_TIMING_DEFAULTS = {
    "control_dt_s": CTRL_DT,
    "control_rate_hz": 1.0 / CTRL_DT,
    "peristaltic_actuation_period_s": PERISTALTIC_ACTUATION_PERIOD_S,
    "phase_freq_hz": PHASE_FREQ,
}
ACTUATOR_CONTRACT_FINGERPRINT = motor_contract()["contract_fingerprint"]
REWARD_CONTRACT = reward_contract()
ACTION_ADAPTER_CONTRACT = action_adapter_contract()
RESIDUAL_EXPLORATION_CONTRACT = residual_exploration_contract()
EVAL_COMMAND_DEFAULTS = {
    "cmd_vel_m_s": CMD_VEL_RANGE[1],
    "cmd_yaw_rad_s": 0.0,
    "command_resample_prob": 0.0,
}
NOMINAL_SENSOR_DEFAULTS = {
    "encoder_pos_noise_std": 0.0,
    "encoder_vel_noise_std": 0.0,
    "imu_gravity_noise_std": 0.0,
    "imu_gyro_noise_std": 0.0,
    "action_delay_steps": 0,
}


def exists(path):
    return os.path.exists(path)


def read_json(path):
    if not exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def run_dir(terrain, mode):
    return os.path.join(PROJECT_ROOT, "runs", f"worm_v6_ppo_{terrain}_{mode}")


def cmaes_path(terrain, mode):
    return os.path.join(PROJECT_ROOT, "runs",
                        f"cmaes_{terrain}_{MODE_TO_CMAES[mode]}",
                        "best_gait.json")


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def item(name, ok, evidence=None, missing=None, detail=None):
    result = {
        "name": name,
        "status": "ok" if ok else "missing",
    }
    if evidence:
        result["evidence"] = evidence
    if missing:
        result["missing"] = missing
    if detail:
        result["detail"] = detail
    return result


def refresh_results_index():
    from audit_observation_sources_v6 import write_audit
    from build_results_index_v6 import write_index
    from summarize_hardware_validation_v6 import write_summary

    write_contract()
    write_audit()
    write_summary()
    write_index(DEFAULT_RESULTS_INDEX)


def get_nested(data, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def close_float(value, expected, tol=1e-6):
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def expected_gait_blend(mode):
    if mode == "worm":
        return 0.0
    if mode == "snake":
        return 1.0
    if mode == "mixed":
        return 0.5
    return None


def sensor_settings_match(actual, expected):
    if not isinstance(actual, dict):
        return False
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if isinstance(expected_value, float):
            if not close_float(actual[key], expected_value):
                return False
        elif actual[key] != expected_value:
            return False
    return True


def control_timing_match(actual):
    return sensor_settings_match(actual, CONTROL_TIMING_DEFAULTS)


def model_vecnormalize_pair(run_path):
    pairs = [
        ("best_model.zip", "best_model_vecnormalize.pkl"),
        ("final_model.zip", "final_model_vecnormalize.pkl"),
    ]
    for model_name, norm_name in pairs:
        model_path = os.path.join(run_path, model_name)
        norm_path = os.path.join(run_path, norm_name)
        if exists(model_path) and exists(norm_path):
            return model_path, norm_path
    return None, None


def valid_training_run(run_path, terrain, mode):
    model_path, norm_path = model_vecnormalize_pair(run_path)
    config = read_json(os.path.join(run_path, "training_config.json"))
    result = read_json(os.path.join(run_path, "training_result.json"))
    reasons = []
    if model_path is None:
        reasons.append("missing model/VecNormalize pair")
    if config is None:
        reasons.append("missing training_config.json")
    if result is None:
        reasons.append("missing training_result.json")
    if config is not None:
        checks = {
            "obs_dim": config.get("obs_dim") == OBS_DIM,
            "terrain": config.get("terrain") == terrain,
            "gait_mode": config.get("gait_mode") == mode,
            "num_actuators": config.get("num_actuators") == 11,
            "num_imus": config.get("num_imus") == 7,
            "control_timing": control_timing_match(
                config.get("control_timing")),
            "actuator_contract": (
                config.get("actuator_contract_fingerprint")
                == ACTUATOR_CONTRACT_FINGERPRINT),
            "reward_contract": (
                config.get("reward_contract") == REWARD_CONTRACT),
            "action_adapter": (
                config.get("action_adapter") == ACTION_ADAPTER_CONTRACT),
            "residual_exploration": (
                config.get("residual_exploration")
                == RESIDUAL_EXPLORATION_CONTRACT),
            "eval_command": (
                config.get("eval_command") == EVAL_COMMAND_DEFAULTS),
            "planned_timesteps": (
                get_nested(config, "training", "timesteps", default=0)
                >= REQUIRED_TRAIN_TIMESTEPS),
            "robust_sensor_training": sensor_settings_match(
                config.get("sensor_robustness"), ROBUST_SENSOR_DEFAULTS),
        }
        reasons.extend([name for name, ok in checks.items() if not ok])
    if result is not None:
        completed = result.get("completed_timesteps", 0)
        if completed < REQUIRED_TRAIN_TIMESTEPS:
            reasons.append("completed_timesteps below formal threshold")
    return not reasons, {
        "model": model_path,
        "vecnormalize": norm_path,
        "config": os.path.join(run_path, "training_config.json"),
        "result": os.path.join(run_path, "training_result.json"),
        "reasons": reasons,
    }


def valid_eval_metrics(data, terrain, mode, expected_condition):
    if not isinstance(data, dict):
        return False, ["missing metrics JSON"]
    reasons = []
    if data.get("obs_dim") != OBS_DIM:
        reasons.append("obs_dim")
    if data.get("terrain") != terrain:
        reasons.append("terrain")
    if data.get("gait_mode") != mode:
        reasons.append("gait_mode")
    blend = expected_gait_blend(mode)
    if blend is not None and not close_float(data.get("gait_blend"), blend):
        reasons.append("gait_blend")
    if data.get("eval_condition", "nominal") != expected_condition:
        reasons.append("eval_condition")
    if not control_timing_match(data.get("control_timing")):
        reasons.append("control_timing")
    if data.get("actuator_contract_fingerprint") != ACTUATOR_CONTRACT_FINGERPRINT:
        reasons.append("actuator_contract")
    if data.get("reward_contract") != REWARD_CONTRACT:
        reasons.append("reward_contract")
    if data.get("action_adapter") != ACTION_ADAPTER_CONTRACT:
        reasons.append("action_adapter")
    if data.get("eval_command") != EVAL_COMMAND_DEFAULTS:
        reasons.append("eval_command")
    if data.get("episodes", 0) < REQUIRED_EVAL_EPISODES:
        reasons.append("episodes")
    if float(data.get("time_s", 0.0)) < REQUIRED_EVAL_TIME_S:
        reasons.append("time_s")
    expected_sensor = (
        ROBUST_EVAL_DEFAULTS if expected_condition == "robust"
        else NOMINAL_SENSOR_DEFAULTS)
    if not sensor_settings_match(data.get("sensor_noise"), expected_sensor):
        reasons.append("sensor_noise")
    required_metrics = [
        "mean_speed_mm_s",
        "mean_action_l2_per_m",
        "mean_lateral_drift_mm",
        "success_rate",
        "termination_rate",
    ]
    for metric in required_metrics:
        if metric not in data:
            reasons.append(metric)
    return not reasons, reasons


def valid_cmaes_baseline(path):
    data = read_json(path)
    if not isinstance(data, dict):
        return False, ["missing baseline JSON"]
    reasons = []
    try:
        speed = float(data.get("best_speed_mm_s"))
    except (TypeError, ValueError):
        speed = None
    if speed is None:
        reasons.append("best_speed_mm_s")
    params = data.get("best_params")
    if not isinstance(params, list) or len(params) != 14:
        reasons.append("best_params")
    if int(data.get("generations", 0) or 0) < REQUIRED_CMAES_MIN_GENERATIONS:
        reasons.append("generations")
    if int(data.get("total_evals", 0) or 0) <= 0:
        reasons.append("total_evals")
    try:
        sim_time = float(data.get("sim_time_s"))
    except (TypeError, ValueError):
        sim_time = 0.0
    if sim_time <= 0.0:
        reasons.append("sim_time_s")
    return not reasons, reasons


def check_training_models():
    missing = []
    evidence = []
    for terrain in PAPER_TERRAINS:
        for mode in TRAIN_MODES:
            rd = run_dir(terrain, mode)
            ok, info = valid_training_run(rd, terrain, mode)
            if ok:
                evidence.append(rel(info["model"]))
            else:
                missing.append(rel(rd))
    return item("12 PPO training model artifacts", not missing,
                evidence=evidence, missing=missing)


def check_fixed_eval(filename, label, expected_condition):
    missing = []
    evidence = []
    for terrain in PAPER_TERRAINS:
        for mode in FIXED_MODES:
            path = os.path.join(run_dir(terrain, mode), filename)
            data = read_json(path)
            ok, _ = valid_eval_metrics(
                data, terrain, mode, expected_condition)
            if ok:
                evidence.append(rel(path))
            else:
                missing.append(rel(path))
    return item(label, not missing, evidence=evidence, missing=missing)


def check_cmaes_baselines():
    missing = []
    evidence = []
    detail = {
        "mapped_modes": MODE_TO_CMAES,
        "min_generations": REQUIRED_CMAES_MIN_GENERATIONS,
    }
    for terrain in PAPER_TERRAINS:
        for mode in FIXED_MODES:
            path = cmaes_path(terrain, mode)
            ok, reasons = valid_cmaes_baseline(path)
            if ok:
                evidence.append(rel(path))
            else:
                reason_text = ", ".join(reasons)
                missing.append(f"{rel(path)} ({reason_text})")
    return item("9 CMA-ES open-loop baseline JSON files", not missing,
                evidence=evidence, missing=missing, detail=detail)


def check_blend_scans():
    missing = []
    evidence = []
    for terrain in PAPER_TERRAINS:
        path = os.path.join(
            PROJECT_ROOT, "runs", f"worm_v6_blend_scan_{terrain}_random",
            "scan_results.json")
        data = read_json(path)
        if isinstance(data, list) and data:
            by_blend = {
                round(float(row.get("gait_blend")), 2): row
                for row in data
                if row.get("gait_blend") is not None
            }
            ok_blends = {
                round(blend, 2) for blend in REQUIRED_SCAN_BLENDS
            }.issubset(by_blend.keys())
            ok_rows = True
            for blend in REQUIRED_SCAN_BLENDS:
                row = by_blend.get(round(blend, 2))
                if row is None:
                    ok_rows = False
                    continue
                row_ok, _ = valid_eval_metrics(
                    row, terrain, "random", "nominal")
                if not (row_ok and row.get("policy_mode") == "random"):
                    ok_rows = False
            if ok_blends and ok_rows:
                evidence.append(rel(path))
                continue
        missing.append(rel(path))
    return item("3 continuous gait_blend scan result files", not missing,
                evidence=evidence, missing=missing)


def check_deploy_bundles():
    missing = []
    evidence = []
    contract = observation_contract()
    for terrain in PAPER_TERRAINS:
        bundle_dir = os.path.join(
            PROJECT_ROOT, "record", "v6", "deploy_bundles",
            f"{terrain}_random")
        actor = os.path.join(bundle_dir, "policy_actor.pt")
        config_path = os.path.join(bundle_dir, "deploy_config.json")
        config = read_json(config_path)
        ok = (
            exists(actor)
            and config is not None
            and config.get("obs_dim") == OBS_DIM
            and config.get("action_dim") == 11
            and action_mapping_matches(config.get("action_mapping"))
            and config.get("actuator_contract_fingerprint")
            == ACTUATOR_CONTRACT_FINGERPRINT
            and config.get("actuator_contract", {}).get(
                "contract_fingerprint") == ACTUATOR_CONTRACT_FINGERPRINT
            and control_timing_match(config.get("control_timing"))
            and config.get("observation_columns") == observation_columns()
            and config.get("observation_contract_fingerprint")
            == contract["abi_fingerprint"]
            and config.get("observation_contract", {}).get("obs_dim") == OBS_DIM
        )
        if ok:
            evidence.append(rel(bundle_dir))
        else:
            missing.append(rel(bundle_dir))
    return item("3 deployable random-policy bundles", not missing,
                evidence=evidence, missing=missing)


def check_summary_outputs():
    paths = [
        "record/v6/paper_results/summary.md",
        "record/v6/paper_results/fixed_mode_summary.csv",
        "record/v6/paper_results/blend_scan_summary.csv",
        "record/v6/paper_results/best_blend_by_terrain.csv",
        "record/v6/paper_results/paper_claims.json",
        "record/v6/paper_results/paper_claims.md",
        "record/v6/paper_results/results_index.md",
        "record/v6/paper_results/observation_contract.json",
        "record/v6/paper_results/observation_contract.md",
        "record/v6/paper_results/observation_source_audit.json",
        "record/v6/paper_results/observation_source_audit.md",
        "record/v6/paper_results/paper_video_manifest.json",
        "record/v6/paper_results/paper_video_manifest.md",
        "record/v6/paper_results/hardware_validation_summary.json",
        "record/v6/paper_results/hardware_validation_summary.md",
        "record/v6/paper_results/hardware_validation_summary.csv",
        "record/v6/paper_results/fixed_mode_speed.svg",
        "record/v6/paper_results/blend_scan_speed.svg",
    ]
    missing = [p for p in paths if not exists(os.path.join(PROJECT_ROOT, p))]
    claims = read_json(os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "paper_claims.json"))
    if not isinstance(claims, dict):
        missing.append("record/v6/paper_results/paper_claims.json content")
    else:
        terrains = claims.get("terrain_mode_selection")
        assessments = claims.get("claim_assessments")
        if not isinstance(terrains, list) or len(terrains) != len(PAPER_TERRAINS):
            missing.append("paper_claims terrain_mode_selection")
        if not isinstance(assessments, list) or not assessments:
            missing.append("paper_claims claim_assessments")
    observation_audit = read_json(os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "observation_source_audit.json"))
    if not isinstance(observation_audit, dict):
        missing.append("observation_source_audit.json content")
    else:
        if observation_audit.get("complete") is not True:
            missing.append("observation_source_audit complete")
        if observation_audit.get("forbidden_policy_observation_hits") != {}:
            missing.append("observation_source_audit forbidden obs hits")
        if observation_audit.get("obs_dim") != OBS_DIM:
            missing.append("observation_source_audit obs_dim")
    hardware_summary = read_json(os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "hardware_validation_summary.json"))
    if not isinstance(hardware_summary, dict):
        missing.append("hardware_validation_summary.json content")
    else:
        rows = hardware_summary.get("rows")
        if not isinstance(rows, list) or len(rows) != len(PAPER_TERRAINS):
            missing.append("hardware_validation_summary flat/sand/slope rows")
    video_manifest = read_json(os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "paper_video_manifest.json"))
    if not valid_video_manifest(video_manifest):
        missing.append("paper_video_manifest complete flat/sand/slope videos")
    evidence = [p for p in paths if p not in missing]
    return item("paper summary tables, figures, and claim analysis", not missing,
                evidence=evidence, missing=missing)


def valid_video_manifest(manifest):
    if not isinstance(manifest, dict) or not manifest.get("complete"):
        return False
    records = manifest.get("records")
    if not isinstance(records, list):
        return False
    by_terrain = {row.get("terrain"): row for row in records}
    for terrain in PAPER_TERRAINS:
        row = by_terrain.get(terrain)
        if not isinstance(row, dict) or row.get("status") != "ok":
            return False
        video = row.get("video", {})
        metrics = row.get("metrics", {})
        if not video.get("exists") or not metrics.get("exists"):
            return False
        video_path = os.path.join(PROJECT_ROOT, video.get("path", ""))
        metrics_path = os.path.join(PROJECT_ROOT, metrics.get("path", ""))
        if not exists(video_path) or os.path.getsize(video_path) <= 0:
            return False
        if not exists(metrics_path):
            return False
    return True


def hardware_csv_candidates(terrain):
    hw_dir = os.path.join(PROJECT_ROOT, "record", "v6", "hardware")
    if not exists(hw_dir):
        return []
    result = []
    for name in os.listdir(hw_dir):
        if not name.endswith(".csv"):
            continue
        if "template" in name or name.endswith("_actions.csv"):
            continue
        if name.startswith(f"{terrain}_"):
            result.append(os.path.join(hw_dir, name))
    return sorted(result)


def valid_hardware_log(path, terrain):
    try:
        validate_csv(
            path,
            expected_terrain=terrain,
            require_video=True,
            min_rows=REQUIRED_HARDWARE_MIN_ROWS,
            min_duration_s=REQUIRED_HARDWARE_MIN_DURATION_S,
            verbose=False)
        return True, []
    except Exception as exc:
        return False, [str(exc)]


def check_hardware_logs():
    missing = []
    evidence = []
    for terrain in PAPER_TERRAINS:
        candidates = hardware_csv_candidates(terrain)
        valid = []
        invalid_reasons = []
        for path in candidates:
            ok, reasons = valid_hardware_log(path, terrain)
            if ok:
                valid.append(path)
            else:
                invalid_reasons.append(f"{rel(path)}: {'; '.join(reasons)}")
        if valid:
            evidence.append(rel(valid[0]))
        else:
            if invalid_reasons:
                missing.append(invalid_reasons[0])
            else:
                missing.append(f"record/v6/hardware/{terrain}_*.csv")
    return item("flat/sand/slope hardware logs with video references",
                not missing, evidence=evidence, missing=missing)


def check_templates():
    paths = [
        "record/v6/hardware/raw_hardware_log_template.csv",
        "record/v6/hardware/raw_hardware_log_template_example.csv",
        "record/v6/hardware/hardware_log_template.csv",
        "record/v6/hardware/hardware_log_template_example.csv",
        "record/v6/hardware/hardware_log_template_schema.json",
    ]
    missing = [p for p in paths if not exists(os.path.join(PROJECT_ROOT, p))]
    schema = read_json(os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        "hardware_log_template_schema.json"))
    ok_schema = bool(schema and schema.get("obs_dim") == OBS_DIM)
    if not ok_schema:
        missing.append("hardware schema obs_dim/layout")
    elif schema.get("observation_contract_fingerprint") not in (
            None, observation_contract()["abi_fingerprint"]):
        missing.append("hardware schema observation_contract_fingerprint")
    evidence = [p for p in paths if p not in missing]
    return item("hardware template/schema", not missing,
                evidence=evidence, missing=missing)


def check_deploy_runtime_tools():
    paths = [
        "src/v6/build_hardware_obs_v6.py",
        "src/v6/audit_observation_sources_v6.py",
        "src/v6/check_controller_stream_v6.py",
        "src/v6/deploy_policy_v6.py",
        "src/v6/motor_contract_v6.py",
        "src/v6/hardware_policy_runtime_v6.py",
        "src/v6/preflight_hardware_deploy_v6.py",
        "src/v6/process_hardware_trial_v6.py",
        "src/v6/summarize_hardware_validation_v6.py",
        "src/v6/test_observation_source_audit_v6.py",
        "src/v6/test_controller_stream_check_v6.py",
        "src/v6/test_hardware_validation_summary_v6.py",
        "src/v6/test_hardware_preflight_v6.py",
        "src/v6/test_hardware_policy_runtime_v6.py",
        "src/v6/test_motor_contract_v6.py",
    ]
    missing = [p for p in paths if not exists(os.path.join(PROJECT_ROOT, p))]
    return item("online deploy runtime tools", not missing,
                evidence=[p for p in paths if p not in missing],
                missing=missing)


def check_hardware_preflight():
    paths = [
        "record/v6/hardware/hardware_deploy_preflight.json",
        "record/v6/hardware/hardware_deploy_preflight.md",
    ]
    missing = [p for p in paths if not exists(os.path.join(PROJECT_ROOT, p))]
    payload = read_json(os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        "hardware_deploy_preflight.json"))
    if not isinstance(payload, dict):
        missing.append("hardware_deploy_preflight.json content")
    else:
        terrains = payload.get("terrains")
        terrain_status = {
            row.get("terrain"): row.get("status")
            for row in terrains or []
            if isinstance(row, dict)
        }
        if not payload.get("complete"):
            missing.append("hardware deploy preflight complete")
        if payload.get("obs_dim") != OBS_DIM:
            missing.append("hardware deploy preflight obs_dim")
        if payload.get("observation_contract_fingerprint") != (
                observation_contract()["abi_fingerprint"]):
            missing.append("hardware deploy preflight ABI fingerprint")
        if payload.get("actuator_contract_fingerprint") != (
                ACTUATOR_CONTRACT_FINGERPRINT):
            missing.append("hardware deploy preflight actuator contract")
        if set(terrain_status) != set(PAPER_TERRAINS):
            missing.append("hardware deploy preflight flat/sand/slope")
        elif any(terrain_status[t] != "ok" for t in PAPER_TERRAINS):
            missing.append("hardware deploy preflight terrain status")
    return item("hardware deploy preflight report", not missing,
                evidence=[p for p in paths if p not in missing],
                missing=missing)


def csv_row_count_and_header(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0, []
        rows = sum(1 for _ in reader)
        return rows, reader.fieldnames


def check_sim_hardware_bridge():
    raw_path = os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware", "sim_flat_mixed_raw.csv")
    policy_path = os.path.join(
        PROJECT_ROOT, "record", "v6", "hardware",
        "sim_flat_mixed_raw_policy.csv")
    missing = []
    evidence = []
    if not exists(raw_path):
        missing.append(rel(raw_path))
    else:
        rows, _ = csv_row_count_and_header(raw_path)
        if rows > 0:
            evidence.append(rel(raw_path))
        else:
            missing.append(f"{rel(raw_path)} rows")

    if not exists(policy_path):
        missing.append(rel(policy_path))
    else:
        rows, header = csv_row_count_and_header(policy_path)
        obs_cols = observation_columns()
        if rows > 0 and all(col in header for col in obs_cols):
            evidence.append(rel(policy_path))
        else:
            missing.append(f"{rel(policy_path)} obs columns/rows")

    return item("sim sensor to hardware-policy bridge", not missing,
                evidence=evidence, missing=missing)


def audit():
    checks = [
        check_templates(),
        check_deploy_runtime_tools(),
        check_hardware_preflight(),
        check_sim_hardware_bridge(),
        check_training_models(),
        check_fixed_eval(
            "eval_metrics.json", "9 fixed-mode eval JSON files", "nominal"),
        check_fixed_eval(
            "eval_metrics_robust.json", "9 robust fixed-mode eval JSON files",
            "robust"),
        check_cmaes_baselines(),
        check_blend_scans(),
        check_deploy_bundles(),
        check_summary_outputs(),
        check_hardware_logs(),
    ]
    complete = all(check["status"] == "ok" for check in checks)
    return {
        "complete": complete,
        "obs_dim": OBS_DIM,
        "terrains": list(PAPER_TERRAINS),
        "fixed_modes": list(FIXED_MODES),
        "train_modes": list(TRAIN_MODES),
        "checks": checks,
    }


def write_markdown(report, path):
    lines = [
        "# Worm V6 Goal Completion Audit",
        "",
        f"Complete: `{str(report['complete']).lower()}`",
        "",
        "| Requirement | Status | Missing |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        missing = ", ".join(check.get("missing", []))
        lines.append(
            f"| {check['name']} | {check['status']} | {missing} |")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Audit Worm V6 paper goal artifact completion")
    ap.add_argument("--json-out", default=os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "completion_audit.json"))
    ap.add_argument("--md-out", default=os.path.join(
        PROJECT_ROOT, "record", "v6", "paper_results",
        "completion_audit.md"))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Exit 0 even when requirements are missing")
    args = ap.parse_args()

    refresh_results_index()
    report = audit()
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, args.md_out)
    refresh_results_index()
    print(json.dumps(report, indent=2))
    if not report["complete"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

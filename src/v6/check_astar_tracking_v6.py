"""Independently replay and verify one successful V6 A* tracking rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def hash_matches(source_bytes, expected):
    lf_bytes = source_bytes.replace(b"\r\n", b"\n")
    variants = {"Exact bytes": source_bytes, "LF serialization": lf_bytes,
                "CRLF serialization": lf_bytes.replace(b"\n", b"\r\n")}
    return [mode for mode, value in variants.items() if hashlib.sha256(value).hexdigest() == expected]


def check(run, source_revision=None):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tree = ET.fromstring((run / "scene.xml").read_bytes())
    tree.find("compiler").set("meshdir", str(ROOT / "meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    result = {"passed": False, "mujoco_version": mujoco.__version__, "checks": {},
              "contact_scope": "Replay checks saved frames only; 2 ms contact count is the runner's report."}

    def require(name, condition):
        result["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def same(name, actual, expected, tolerance=1e-9):
        require(name, bool(np.allclose(actual, expected, atol=tolerance, rtol=0)))

    try:
        revision = None
        if source_revision is not None:
            revision = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options",
                                       f"{source_revision}^{{commit}}"], cwd=ROOT,
                                      check=True, capture_output=True, text=True).stdout.strip()
        with np.load(run / "trajectory.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        times, path, coms = arrays["time_s"], arrays["path_m"], arrays["com_m"]
        require("finite_saved_states", all(np.all(np.isfinite(arrays[key])) for key in
                ("qpos", "qvel", "ctrl", "com_m", "body_pos_m", "time_s", "path_m", "raw_astar_m")))
        require("increasing_times", len(times) > 1 and times[0] == 0 and np.all(np.diff(times) > 0))
        same("final_time", times[-1], summary["sim_time_s"])
        same("physics_timestep", model.opt.timestep, summary["physics_dt_s"])
        same("body_masses", model.body_mass, arrays["body_mass_kg"])
        head, tail = int(arrays["head_id"]), int(arrays["tail_id"])
        require("head_and_tail_names", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, head) == "base_link"
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, tail) == "back6_Link")
        geoms = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_contype != 0))
        same("collision_geom_ids", geoms, arrays["collision_geom_ids"])
        require("complete_robot_collision_geoms", len(geoms) == summary["collision_geom_count"] == 36)
        scene = summary["scene"]
        obstacle_ids = []
        for index, obstacle in enumerate(scene["obstacles"]):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{index}")
            require(f"obstacle_{index}_present", geom >= 0)
            obstacle_ids.append(geom)
            same(f"obstacle_{index}_center", model.geom_pos[geom, :2], obstacle["center_m"])
            same(f"obstacle_{index}_size", model.geom_size[geom, :2], obstacle["half_size_m"])

        maximum_com_error = maximum_body_error = maximum_envelope = 0.
        minimum_clearance, replay_contacts, sample_torque = np.inf, 0, np.zeros(model.nu)
        for i, qpos in enumerate(arrays["qpos"]):
            data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, arrays["qvel"][i], arrays["ctrl"][i]
            mujoco.mj_forward(model, data)
            independent_com = np.sum(model.body_mass[:, None] * data.xipos, axis=0) / np.sum(model.body_mass)
            maximum_com_error = max(maximum_com_error, float(np.max(np.abs(independent_com - coms[i]))),
                                    float(np.max(np.abs(data.subtree_com[head] - coms[i]))))
            maximum_body_error = max(maximum_body_error, float(np.max(np.abs(data.xpos - arrays["body_pos_m"][i]))))
            maximum_envelope = max(maximum_envelope, float(np.max(
                np.linalg.norm(data.geom_xpos[geoms, :2] - independent_com[:2], axis=1) + model.geom_rbound[geoms])))
            replay_contacts += int(any(c.geom1 in obstacle_ids or c.geom2 in obstacle_ids for c in data.contact))
            for obstacle in obstacle_ids:
                for geom in geoms:
                    minimum_clearance = min(minimum_clearance, float(mujoco.mj_geomDistance(model, data, int(geom), obstacle, 10., None)))
            sample_torque = np.maximum(sample_torque, np.abs(data.actuator_force))
        require("independent_com_and_subtree_com", maximum_com_error < 1e-10)
        require("saved_body_positions", maximum_body_error < 1e-10)
        require("replay_samples_clear", replay_contacts == 0 and minimum_clearance > 0)
        require("replay_envelope_within_planner", maximum_envelope <= scene["collision_envelope_m"])
        # The runner's 50 ms extrema and saved 100 ms frames have different phases.

        original_com, original_geoms = data.subtree_com[head].copy(), data.geom_xpos[geoms].copy()
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        offset = model.jnt_qposadr[root]
        shift = np.array([0.37, -0.43, 0.19])
        data.qpos[offset:offset+3] += shift
        mujoco.mj_forward(model, data)
        same("root_translation_com", data.subtree_com[head], original_com + shift)
        same("root_translation_all_collision_geoms", data.geom_xpos[geoms], original_geoms + shift)

        same("path_start", path[0], coms[0, :2])
        same("path_goal", path[-1], scene["goal_m"])
        minimum_path_distance = np.inf
        radius = scene["collision_envelope_m"] + scene["tracking_margin_m"]
        for planned in (path, arrays["raw_astar_m"]):
            for a, b in zip(planned[:-1], planned[1:]):
                length = np.linalg.norm(b-a)
                require("nonzero_path_edges", length > 0)
                points = np.linspace(a, b, max(2, int(np.ceil(length / 0.002)) + 1))
                for obstacle in scene["obstacles"]:
                    distances = np.linalg.norm(np.maximum(np.abs(points - obstacle["center_m"])
                                                         - obstacle["half_size_m"], 0), axis=1)
                    minimum_path_distance = min(minimum_path_distance, float(distances.min()))
        require("dense_path_clearance_certificate", minimum_path_distance > radius + 0.001)
        vectors = np.diff(path, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        fractions = np.clip(np.sum((coms[:, None, :2] - path[:-1]) * vectors, axis=-1) / lengths**2, 0, 1)
        projections = path[:-1] + fractions[..., None] * vectors
        errors = np.min(np.linalg.norm(coms[:, None, :2] - projections, axis=-1), axis=1)
        same("stored_cross_track", errors, arrays["cross_track_m"])
        metrics = {"planned_length_m": float(lengths.sum()),
                   "travelled_com_length_m": float(np.linalg.norm(np.diff(coms[:, :2], axis=0), axis=1).sum()),
                   "net_displacement_m": float(np.linalg.norm(coms[-1, :2]-coms[0, :2])),
                   "goal_error_m": float(np.linalg.norm(coms[-1, :2]-path[-1])),
                   "cross_track_rms_m": float(np.sqrt(np.mean(errors**2))),
                   "cross_track_max_m": float(errors.max())}
        for name, value in metrics.items():
            same(f"summary_{name}", value, summary[name])
        require("metre_scale_path", metrics["planned_length_m"] > 4)
        require("successful_closed_loop", summary["success"] and summary["stop_reason"] == "goal_reached"
                and not summary["controller"]["open_loop"] and metrics["goal_error_m"] <= scene["goal_tolerance_m"])
        require("runtime_reports_no_obstacle_contacts", summary["obstacle_contact_steps"] == 0)
        same("runtime_contact_check_timestep", summary["collision_check_dt_s"], model.opt.timestep)
        require("runtime_envelope_within_planner", summary["max_collision_envelope_m"] <= scene["collision_envelope_m"])
        # Final COM envelope lies wholly left of every obstacle in this -X scene.
        require("entire_robot_passed_obstacles", all(coms[-1, 0] + scene["collision_envelope_m"]
                < obstacle["center_m"][0] - obstacle["half_size_m"][0] for obstacle in scene["obstacles"]))
        yaw = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_front{i}") for i in range(2, 7)]
        slide = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_back{i}") for i in range(1, 7)]
        require("complete_yaw_actuators", all(i >= 0 for i in yaw))
        yaw_joints = model.actuator_trnid[yaw, 0]
        require("five_unique_yaw_hinges", np.all(model.actuator_trntype[yaw] == mujoco.mjtTrn.mjTRN_JOINT)
                and np.all(yaw_joints >= 0) and len(np.unique(yaw_joints)) == 5
                and np.all(model.jnt_type[yaw_joints] == mujoco.mjtJoint.mjJNT_HINGE))
        yaw_angles = arrays["qpos"][:, model.jnt_qposadr[yaw_joints]]
        yaw_speeds = arrays["qvel"][:, model.jnt_dofadr[yaw_joints]]
        sampled_motion = {
            "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in yaw_joints],
            "peak_abs_yaw_deg": np.rad2deg(np.max(np.abs(yaw_angles), axis=0)).tolist(),
            "peak_abs_yaw_speed_deg_s": np.rad2deg(np.max(np.abs(yaw_speeds), axis=0)).tolist(),
            "max_abs_cumulative_yaw_deg": float(np.rad2deg(np.max(np.abs(np.cumsum(yaw_angles, axis=1))))),
            "scope": "Actual saved qpos/qvel, normally sampled every 0.1 s; sampled peaks can underestimate continuous peaks. Cumulative yaw sums the five serial hinge angles relative to the root."}
        for name, ids, key in (("yaw", yaw, "peak_yaw_torque_nm"), ("slide", slide, "peak_slide_force_n")):
            require(f"{name}_force_limit", np.isfinite(summary[key]) and summary[key] <= np.max(np.abs(model.actuator_forcerange[ids])) + 1e-9)
            require(f"{name}_sampled_force_consistency", np.max(sample_torque[ids]) <= summary[key] + 1e-9)

        provenance = "Legacy end hashes only; source capture at startup is unverified."
        scene_bytes = (run / "scene.xml").read_bytes()
        scene_bytes_hash = hashlib.sha256(scene_bytes).hexdigest()
        scene_lf_hash = hashlib.sha256(scene_bytes.replace(b"\r\n", b"\n")).hexdigest()
        scene_hash_mode = "Legacy schema has no scene hash."
        source_hash_modes = {}
        historical_sources = []
        if "sources_unchanged_during_run" in summary:
            require("sources_unchanged_during_run", summary["sources_unchanged_during_run"] is True)
            for name, expected in summary["source_sha256"].items():
                repo_name = name.replace("\\", "/")
                source = ROOT / repo_name
                source_bytes = source.read_bytes()
                matched = hash_matches(source_bytes, expected)
                if not matched and revision is not None:
                    historical_bytes = subprocess.run(["git", "show", f"{revision}:{repo_name}"],
                                                      cwd=ROOT, check=True, capture_output=True).stdout
                    matched = hash_matches(historical_bytes, expected)
                    if matched:
                        historical_sources.append(repo_name)
                require(f"source_hash_{name}", bool(matched))
                source_hash_modes[name] = matched[0]
            expected_scene = summary["scene_xml_sha256"]
            require("scene_xml_hash_exact_or_CRLF_serialization", expected_scene in (scene_bytes_hash, scene_lf_hash))
            scene_hash_mode = "Exact bytes" if expected_scene == scene_bytes_hash else "Windows CRLF serialization of matching LF XML string"
            provenance = "Startup source hashes match current files, allowing only LF/CRLF text serialization; runner confirmed no changes during rollout."
            if historical_sources:
                provenance = "Startup source hashes match current files or the specified Git commit, allowing only LF/CRLF text serialization; historical sources were hashed, not executed; runner confirmed no changes during rollout."
        result.update(passed=True, recomputed_metrics=metrics, maximum_com_error_m=maximum_com_error,
                      maximum_body_position_error_m=maximum_body_error, replay_sample_count=len(times),
                      replay_sample_minimum_clearance_m=minimum_clearance, replay_sample_maximum_envelope_m=maximum_envelope,
                      replay_obstacle_contact_frames=replay_contacts, certified_path_minimum_distance_m=minimum_path_distance,
                      provenance=provenance,
                      replay_minus_runtime_clearance_m=minimum_clearance-summary["min_geom_clearance_m"],
                      replay_minus_runtime_envelope_m=maximum_envelope-summary["max_collision_envelope_m"],
                      replay_resource_mapping="Only compiler.meshdir remapped to current ROOT/meshes; physical XML unchanged.",
                      scene_xml_file_bytes_sha256=scene_bytes_hash, scene_xml_normalized_lf_sha256=scene_lf_hash,
                      scene_xml_hash_match_mode=scene_hash_mode, source_hash_match_modes=source_hash_modes,
                      source_revision=revision, historical_source_paths=historical_sources,
                      sampled_joint_motion=sampled_motion,
                      length_scope="COM arc length from stored frames, not continuous 2 ms trajectory.")
    except (AssertionError, KeyError, ValueError, IndexError, OSError, subprocess.CalledProcessError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    (run / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing scene.xml, trajectory.npz and summary.json")
    parser.add_argument("--source-revision", help="Optional Git commit used only to hash historical sources when current source hashes differ")
    args = parser.parse_args()
    raise SystemExit(0 if check(args.run, args.source_revision) else 1)

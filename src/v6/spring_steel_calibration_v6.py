"""
Single-segment spring-steel strip calibration for Worm V6.

This script builds a small MuJoCo model with two rigid plates connected by
elastic cable strips. It sweeps axial compression and records the holding force
needed to keep the moving plate at each compression. The result is a
force-displacement curve that can be used to tune the fast V6 equivalent spring
model.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
from datetime import datetime

import numpy as np


PROJECT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_OUT_DIR = os.path.join(
    PROJECT_ROOT, "record", "v6", "spring_steel_calibration")


@dataclasses.dataclass
class CalibrationConfig:
    strip_count: int = 8
    strip_vertices: int = 7
    rest_length_m: float = 0.1175
    strip_circle_radius_m: float = 0.068
    strip_bow_m: float = 0.012
    strip_capsule_radius_m: float = 0.0015
    strip_density_kg_m3: float = 7850.0
    cable_bend: float = 1.0e8
    cable_twist: float = 4.0e7
    plate_half_extent_m: float = 0.070
    plate_thickness_m: float = 0.008
    moving_plate_mass_kg: float = 0.08
    fixed_plate_mass_kg: float = 0.08
    actuator_kp_n_m: float = 4000.0
    actuator_force_limit_n: float = 500.0
    timestep_s: float = 0.0005
    settle_steps: int = 1200
    sample_steps: int = 300


def parse_compressions_mm(text):
    values = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        value_mm = float(item)
        if value_mm < 0.0:
            raise ValueError("compression values must be non-negative")
        values.append(value_mm / 1000.0)
    if not values:
        raise ValueError("at least one compression value is required")
    return values


def _strip_vertices(angle, cfg):
    ca = math.cos(angle)
    sa = math.sin(angle)
    points = []
    count = int(cfg.strip_vertices)
    for idx in range(count):
        t = idx / max(count - 1, 1)
        bow = cfg.strip_bow_m * 4.0 * t * (1.0 - t)
        radius = cfg.strip_circle_radius_m + bow
        x = cfg.rest_length_m * t
        y = radius * ca
        z = radius * sa
        points.append(f"{x:.6f} {y:.6f} {z:.6f}")
    return "  ".join(points)


def build_calibration_xml(cfg):
    if cfg.strip_count < 1:
        raise ValueError("strip_count must be positive")
    if cfg.strip_vertices < 3:
        raise ValueError("strip_vertices must be at least 3")
    if cfg.rest_length_m <= 0.0:
        raise ValueError("rest_length_m must be positive")

    cable_blocks = []
    weld_blocks = []
    for strip_idx in range(cfg.strip_count):
        angle = 2.0 * math.pi * strip_idx / cfg.strip_count
        verts = _strip_vertices(angle, cfg)
        cable_blocks.append(f"""
    <body name="strip{strip_idx}_root">
      <freejoint/>
      <composite type="cable" prefix="strip{strip_idx}" initial="none"
                 vertex="{verts}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="bend" value="{cfg.cable_bend:g}"/>
          <config key="twist" value="{cfg.cable_twist:g}"/>
          <config key="vmax" value="0"/>
        </plugin>
        <joint armature="0.001" damping="0.02" kind="main"/>
        <geom type="capsule" size="{cfg.strip_capsule_radius_m:g}"
              density="{cfg.strip_density_kg_m3:g}" material="MatSteel"
              contype="0" conaffinity="0"/>
      </composite>
    </body>""")
        weld_blocks.append(
            f'    <weld body1="fixed_plate" '
            f'body2="strip{strip_idx}B_first" solref="0.002 1"/>\n'
            f'    <weld body1="moving_plate" '
            f'body2="strip{strip_idx}B_last" solref="0.002 1"/>')

    return f"""<mujoco model="worm_v6_single_segment_steel_strip_calibration">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <compiler angle="radian"/>
  <option timestep="{cfg.timestep_s:g}" gravity="0 0 0"
          solver="Newton" iterations="100" tolerance="1e-9"/>

  <default>
    <geom friction="0.8 0.01 0.001"/>
  </default>

  <asset>
    <material name="MatSteel" rgba="0.12 0.12 0.12 1"
              specular="0.4" shininess="0.5"/>
    <material name="MatPlate" rgba="0.78 0.76 0.72 1"/>
  </asset>

  <worldbody>
    <light pos="0 -0.5 0.7" dir="0 0 -1" directional="true"/>
    <camera name="main" pos="0.08 -0.45 0.20"
            xyaxes="1 0 0 0 0.35 1"/>

    <body name="fixed_plate" pos="0 0 0">
      <geom type="box"
            size="{cfg.plate_thickness_m / 2:g} {cfg.plate_half_extent_m:g} {cfg.plate_half_extent_m:g}"
            mass="{cfg.fixed_plate_mass_kg:g}" material="MatPlate"/>
      <site name="fixed_center" pos="0 0 0" size="0.003"/>
    </body>

    <body name="moving_plate" pos="{cfg.rest_length_m:g} 0 0">
      <joint name="moving_slide" type="slide" axis="-1 0 0"
             limited="true" range="0 0.06" damping="0.2"/>
      <geom type="box"
            size="{cfg.plate_thickness_m / 2:g} {cfg.plate_half_extent_m:g} {cfg.plate_half_extent_m:g}"
            mass="{cfg.moving_plate_mass_kg:g}" material="MatPlate"/>
      <site name="moving_center" pos="0 0 0" size="0.003"/>
    </body>
{''.join(cable_blocks)}
  </worldbody>

  <equality>
{os.linesep.join(weld_blocks)}
  </equality>

  <actuator>
    <position name="slide_servo" joint="moving_slide"
              kp="{cfg.actuator_kp_n_m:g}" ctrllimited="true"
              ctrlrange="0 0.06" forcelimited="true"
              forcerange="{-cfg.actuator_force_limit_n:g} {cfg.actuator_force_limit_n:g}"/>
  </actuator>

  <sensor>
    <jointpos name="slide_position" joint="moving_slide"/>
    <jointvel name="slide_velocity" joint="moving_slide"/>
    <actuatorfrc name="slide_force" actuator="slide_servo"/>
  </sensor>
</mujoco>
"""


def _actual_compression_series(usable_rows):
    for key in ("actual_compression_m", "mean_qpos_m"):
        if not all(key in row for row in usable_rows):
            continue
        values = np.asarray(
            [max(0.0, float(row[key])) for row in usable_rows],
            dtype=np.float64,
        )
        if float(np.max(values)) > 1e-9:
            return key, values
    return "compression_m", np.asarray(
        [float(row["compression_m"]) for row in usable_rows],
        dtype=np.float64,
    )


def fit_equivalent_stiffness(records):
    usable_rows = [
        row
        for row in records
        if float(row["compression_m"]) >= 0.0
    ]
    if len(usable_rows) < 2:
        raise ValueError("at least two records are required for stiffness fit")
    compression_source, x = _actual_compression_series(usable_rows)
    usable = [
        (float(x[idx]), float(row["mean_force_n"]))
        for idx, row in enumerate(usable_rows)
    ]
    if len(usable) < 2:
        raise ValueError("at least two records are required for stiffness fit")
    y = np.asarray([item[1] for item in usable], dtype=np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    zero_den = float(np.dot(x, x))
    zero_slope = 0.0 if zero_den < 1e-12 else float(np.dot(x, y) / zero_den)
    max_idx = int(np.argmax(x))
    max_x = float(x[max_idx])
    force_at_max = float(y[max_idx])
    secant_stiffness = 0.0
    if max_x > 1e-12:
        secant_stiffness = force_at_max / max_x
    if abs(intercept) < 1e-12:
        intercept = 0.0
    return {
        "stiffness_n_per_m": float(slope),
        "zero_intercept_stiffness_n_per_m": zero_slope,
        "secant_stiffness_at_max_n_per_m": float(secant_stiffness),
        "force_intercept_n": float(intercept),
        "r2": float(r2),
        "sample_count": len(usable),
        "compression_source": compression_source,
        "max_compression_m": max_x,
        "force_at_max_compression_n": force_at_max,
        "max_force_n": float(np.max(y)),
    }


def _import_mujoco():
    try:
        import mujoco  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "MuJoCo is required for simulation. Install mujoco or run "
            "with --xml-only to just generate the model XML.") from exc
    return mujoco


def run_calibration(cfg, compressions_m, record_video=False):
    mujoco = _import_mujoco()
    xml = build_calibration_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_servo")
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "moving_slide")
    qpos_addr = model.jnt_qposadr[joint_id]

    renderer = None
    frames = []
    if record_video:
        renderer = mujoco.Renderer(model, 540, 960)

    records = []
    for compression_m in compressions_m:
        target = float(compression_m)
        data.ctrl[actuator_id] = target
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)
            if renderer is not None and len(frames) < data.time * 30:
                renderer.update_scene(data, camera="main")
                frames.append(renderer.render().copy())

        forces = []
        qpos_values = []
        for _ in range(cfg.sample_steps):
            mujoco.mj_step(model, data)
            forces.append(abs(float(data.actuator_force[actuator_id])))
            qpos_values.append(float(data.qpos[qpos_addr]))
            if renderer is not None and len(frames) < data.time * 30:
                renderer.update_scene(data, camera="main")
                frames.append(renderer.render().copy())

        records.append({
            "compression_m": target,
            "compression_mm": target * 1000.0,
            "target_compression_m": target,
            "target_compression_mm": target * 1000.0,
            "mean_qpos_m": float(np.mean(qpos_values)),
            "actual_compression_m": max(0.0, float(np.mean(qpos_values))),
            "actual_compression_mm": max(0.0, float(np.mean(qpos_values))) * 1000.0,
            "mean_force_n": float(np.mean(forces)),
            "std_force_n": float(np.std(forces)),
            "max_force_n": float(np.max(forces)),
        })

    if renderer is not None:
        renderer.close()

    return xml, records, frames


def write_result_bundle(out_dir, config, records, fit, xml_text, video_path=None):
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, "single_segment_steel_strip.xml")
    csv_path = os.path.join(out_dir, "force_displacement.csv")
    json_path = os.path.join(out_dir, "calibration_summary.json")
    report_path = os.path.join(out_dir, "calibration_report.md")

    with open(xml_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml_text)

    fieldnames = [
        "target_compression_mm",
        "target_compression_m",
        "actual_compression_mm",
        "actual_compression_m",
        "compression_mm",
        "compression_m",
        "mean_qpos_m",
        "mean_force_n",
        "std_force_n",
        "max_force_n",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            target_m = float(row.get("target_compression_m", row["compression_m"]))
            actual_m = float(row.get(
                "actual_compression_m",
                row.get("mean_qpos_m", target_m),
            ))
            writer.writerow({
                "target_compression_mm": f"{target_m * 1000.0:.3f}",
                "target_compression_m": f"{target_m:.6f}",
                "actual_compression_mm": f"{actual_m * 1000.0:.3f}",
                "actual_compression_m": f"{actual_m:.6f}",
                "compression_mm": f"{target_m * 1000.0:.3f}",
                "compression_m": f"{target_m:.6f}",
                "mean_qpos_m": f"{float(row.get('mean_qpos_m', 0.0)):.6f}",
                "mean_force_n": f"{float(row['mean_force_n']):.6f}",
                "std_force_n": f"{float(row.get('std_force_n', 0.0)):.6f}",
                "max_force_n": f"{float(row.get('max_force_n', row['mean_force_n'])):.6f}",
            })

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "single_segment_spring_steel_force_displacement_calibration",
        "config": dataclasses.asdict(config),
        "fit": fit,
        "records": records,
        "video_path": video_path,
    }
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    tracking_errors = []
    for row in records:
        if "mean_qpos_m" not in row:
            continue
        target_m = float(row.get("target_compression_m", row["compression_m"]))
        actual_m = float(row.get("actual_compression_m", row["mean_qpos_m"]))
        tracking_errors.append(abs(target_m - actual_m))

    lines = [
        "# V6 Spring-Steel Calibration Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Strip count: `{config.strip_count}`",
        f"- Strip vertices: `{config.strip_vertices}`",
        f"- Rest length: `{config.rest_length_m:.4f} m`",
        f"- Cable bend: `{config.cable_bend:g}`",
        f"- Cable twist: `{config.cable_twist:g}`",
        f"- Compression source for fit: `{fit['compression_source']}`",
        f"- Fitted stiffness: `{fit['stiffness_n_per_m']:.3f} N/m`",
        "- Zero-intercept equivalent stiffness: "
        f"`{fit['zero_intercept_stiffness_n_per_m']:.3f} N/m`",
        "- Secant stiffness at max compression: "
        f"`{fit['secant_stiffness_at_max_n_per_m']:.3f} N/m`",
        f"- Force intercept: `{fit['force_intercept_n']:.3f} N`",
        f"- Fit R2: `{fit['r2']:.5f}`",
    ]
    if tracking_errors:
        lines.append(
            "- Max target/actual compression mismatch: "
            f"`{max(tracking_errors) * 1000.0:.3f} mm`")
    lines.extend([
        "",
        "## Files",
        "",
        f"- XML: `{os.path.basename(xml_path)}`",
        f"- CSV: `{os.path.basename(csv_path)}`",
        f"- JSON: `{os.path.basename(json_path)}`",
    ])
    if video_path:
        lines.append(f"- Video: `{os.path.basename(video_path)}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use this fitted stiffness as a candidate equivalent slide-joint "
        "spring constant for the fast whole-body V6 model after checking "
        "that the single-segment deformation looks physically plausible.",
        "",
    ])
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    return {
        "xml": xml_path,
        "csv": csv_path,
        "json": json_path,
        "report": report_path,
    }


def _write_video(path, frames):
    if not frames:
        return None
    try:
        import mediapy as media  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "mediapy is required for --record-video") from exc
    media.write_video(path, frames, fps=30)
    return path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate equivalent spring stiffness from elastic strips.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--compressions-mm",
        default="0,5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--quick", action="store_true",
                        help="Use a short 4-point sweep and fewer steps.")
    parser.add_argument("--xml-only", action="store_true",
                        help="Only write XML/report scaffolding, no simulation.")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--strip-count", type=int, default=8)
    parser.add_argument("--strip-vertices", type=int, default=7)
    parser.add_argument("--rest-length-m", type=float, default=0.1175)
    parser.add_argument("--strip-radius-m", type=float, default=0.068)
    parser.add_argument("--strip-bow-m", type=float, default=0.012)
    parser.add_argument("--cable-bend", type=float, default=1.0e8)
    parser.add_argument("--cable-twist", type=float, default=4.0e7)
    parser.add_argument("--actuator-kp-n-m", type=float, default=4000.0)
    parser.add_argument("--actuator-force-limit-n", type=float, default=500.0)
    parser.add_argument("--timestep-s", type=float, default=0.0005)
    parser.add_argument("--settle-steps", type=int, default=1200)
    parser.add_argument("--sample-steps", type=int, default=300)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = CalibrationConfig(
        strip_count=args.strip_count,
        strip_vertices=args.strip_vertices,
        rest_length_m=args.rest_length_m,
        strip_circle_radius_m=args.strip_radius_m,
        strip_bow_m=args.strip_bow_m,
        cable_bend=args.cable_bend,
        cable_twist=args.cable_twist,
        actuator_kp_n_m=args.actuator_kp_n_m,
        actuator_force_limit_n=args.actuator_force_limit_n,
        timestep_s=args.timestep_s,
        settle_steps=args.settle_steps,
        sample_steps=args.sample_steps,
    )
    compressions = parse_compressions_mm(args.compressions_mm)
    if args.quick:
        compressions = parse_compressions_mm("0,10,25,50")
        cfg.settle_steps = min(cfg.settle_steps, 250)
        cfg.sample_steps = min(cfg.sample_steps, 80)

    xml_text = build_calibration_xml(cfg)
    video_path = None
    if args.xml_only:
        records = [
            {
                "compression_m": value,
                "compression_mm": value * 1000.0,
                "target_compression_m": value,
                "target_compression_mm": value * 1000.0,
                "mean_qpos_m": 0.0,
                "mean_force_n": 0.0,
                "std_force_n": 0.0,
                "max_force_n": 0.0,
            }
            for value in compressions
        ]
        fit = fit_equivalent_stiffness([
            {"compression_m": 0.0, "mean_force_n": 0.0},
            {"compression_m": 1.0, "mean_force_n": 0.0},
        ])
    else:
        xml_text, records, frames = run_calibration(
            cfg, compressions, record_video=args.record_video)
        fit = fit_equivalent_stiffness(records)
        if args.record_video:
            video_path = os.path.join(args.out_dir, "calibration_video.mp4")
            _write_video(video_path, frames)

    paths = write_result_bundle(
        args.out_dir,
        config=cfg,
        records=records,
        fit=fit,
        xml_text=xml_text,
        video_path=video_path,
    )
    print("Spring-steel calibration complete")
    print(f"  stiffness_n_per_m: {fit['stiffness_n_per_m']:.3f}")
    print("  zero_intercept_stiffness_n_per_m: "
          f"{fit['zero_intercept_stiffness_n_per_m']:.3f}")
    print(f"  compression_source: {fit['compression_source']}")
    print(f"  csv: {paths['csv']}")
    print(f"  report: {paths['report']}")


if __name__ == "__main__":
    main()

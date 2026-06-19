"""
Hardware parameter-identification templates and fitters for Worm V6.

The existing hardware validation logs prove the deployable observation/action
contract. This script is narrower: it defines bench-test CSV templates for
physical parameters that must be measured before claiming a calibrated real
robot simulation.
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUT_DIR = os.path.join(
    PROJECT_ROOT, "record", "v6", "hardware", "parameter_id")

sys.path.insert(0, SCRIPT_DIR)

from spring_steel_calibration_v6 import fit_equivalent_stiffness  # noqa: E402


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    key: str
    filename: str
    description: str
    columns: tuple
    example: dict


TEMPLATE_SPECS = (
    TemplateSpec(
        key="spring_steel_force_displacement",
        filename="spring_steel_force_displacement_template.csv",
        description=(
            "Bench compression test for one spring-steel strip assembly. "
            "Use actual measured displacement, not only commanded motion."),
        columns=(
            "trial_id",
            "segment_id",
            "repeat_id",
            "target_compression_mm",
            "actual_compression_mm",
            "hold_time_s",
            "force_n",
            "force_sensor_id",
            "direction",
            "note",
        ),
        example={
            "trial_id": "spring_001",
            "segment_id": "0",
            "repeat_id": "0",
            "target_compression_mm": "10.0",
            "actual_compression_mm": "10.0",
            "hold_time_s": "2.0",
            "force_n": "3.0",
            "force_sensor_id": "loadcell_01",
            "direction": "loading",
            "note": "quasi_static_hold",
        },
    ),
    TemplateSpec(
        key="motor_step_response",
        filename="motor_step_response_template.csv",
        description=(
            "Actuator response test for each slide and yaw motor. This is used "
            "to identify speed limits, delay, saturation, and controller gain."),
        columns=(
            "time_s",
            "joint_id",
            "joint_type",
            "command_norm",
            "target_si",
            "position_si",
            "velocity_si",
            "current_a",
            "voltage_v",
            "pwm",
            "load_n_or_nm",
            "note",
        ),
        example={
            "time_s": "0.020",
            "joint_id": "slide_00",
            "joint_type": "slide",
            "command_norm": "-0.5",
            "target_si": "-0.025",
            "position_si": "-0.003",
            "velocity_si": "-0.12",
            "current_a": "0.8",
            "voltage_v": "12.0",
            "pwm": "0.35",
            "load_n_or_nm": "4.0",
            "note": "loaded_step",
        },
    ),
    TemplateSpec(
        key="imu_static",
        filename="imu_static_template.csv",
        description=(
            "Static IMU orientation/noise test. Record every segment in several "
            "known poses to fit sensor axes, bias, and noise."),
        columns=(
            "segment_id",
            "pose_label",
            "sample_id",
            "acc_x_m_s2",
            "acc_y_m_s2",
            "acc_z_m_s2",
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
            "temperature_c",
            "note",
        ),
        example={
            "segment_id": "0",
            "pose_label": "level",
            "sample_id": "0",
            "acc_x_m_s2": "0.02",
            "acc_y_m_s2": "-0.01",
            "acc_z_m_s2": "-9.81",
            "gyro_x_rad_s": "0.001",
            "gyro_y_rad_s": "0.000",
            "gyro_z_rad_s": "-0.001",
            "temperature_c": "25.0",
            "note": "stationary",
        },
    ),
    TemplateSpec(
        key="contact_drag",
        filename="contact_drag_template.csv",
        description=(
            "Terrain/contact drag test for flat, sand, and slope materials. Pull "
            "a segment or body at known normal load and speed."),
        columns=(
            "terrain",
            "trial_id",
            "normal_load_n",
            "pull_speed_m_s",
            "pull_force_n",
            "slope_deg",
            "contact_part",
            "note",
        ),
        example={
            "terrain": "sand",
            "trial_id": "drag_001",
            "normal_load_n": "12.0",
            "pull_speed_m_s": "0.05",
            "pull_force_n": "5.8",
            "slope_deg": "0.0",
            "contact_part": "body_shell",
            "note": "steady_pull",
        },
    ),
    TemplateSpec(
        key="mass_geometry",
        filename="mass_geometry_template.csv",
        description=(
            "Measured mass and geometry for each segment or assembly. Needed for "
            "body mass, inertia, and center-of-mass updates."),
        columns=(
            "part_id",
            "part_type",
            "mass_kg",
            "length_m",
            "width_m",
            "height_m",
            "com_x_m",
            "com_y_m",
            "com_z_m",
            "note",
        ),
        example={
            "part_id": "segment_00",
            "part_type": "segment",
            "mass_kg": "0.25",
            "length_m": "0.1175",
            "width_m": "0.14",
            "height_m": "0.14",
            "com_x_m": "0.05875",
            "com_y_m": "0.0",
            "com_z_m": "0.0",
            "note": "assembled_segment",
        },
    ),
)


def _write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_parameter_templates(out_dir=DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    readme_lines = [
        "# Worm V6 Hardware Parameter Identification",
        "",
        "These files are bench-test templates for calibrating simulation "
        "parameters from the real robot. They are separate from the field "
        "trial logs used to validate policy deployment.",
        "",
        "Minimum useful dataset:",
        "",
        "1. spring-steel force-displacement points for several compression "
        "values and repeats;",
        "2. actuator step responses for all 6 slide and 5 yaw motors;",
        "3. static IMU samples for all 7 segments in known poses;",
        "4. contact drag tests for flat, sand, and slope-relevant surfaces;",
        "5. measured mass/geometry for every segment assembly.",
        "",
        "Use actual measured values. Commanded values can be logged too, but "
        "the fitters use measured displacement, force, velocity, and current.",
        "",
        "## Files",
        "",
    ]
    for spec in TEMPLATE_SPECS:
        template_path = os.path.join(out_dir, spec.filename)
        example_path = template_path.replace("_template.csv", "_example.csv")
        _write_csv(template_path, spec.columns, rows=[])
        _write_csv(example_path, spec.columns, rows=[spec.example])
        paths[spec.key] = template_path
        paths[f"{spec.key}_example"] = example_path
        readme_lines.append(f"- `{spec.filename}`: {spec.description}")

    readme_lines.extend([
        "",
        "## Spring-Steel Fit",
        "",
        "After filling the spring-steel CSV, run:",
        "",
        "```powershell",
        "python src\\v6\\hardware_parameter_id_v6.py --spring-csv "
        "record\\v6\\hardware\\parameter_id\\spring_steel_force_displacement_REAL.csv",
        "```",
        "",
    ])
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(readme_lines))
    paths["readme"] = readme_path
    return paths


def _first_present(row, names):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return name, row[name]
    raise ValueError(f"Missing required column; expected one of {names}")


def _compression_m(row):
    name, value = _first_present(row, (
        "actual_compression_m",
        "compression_m",
        "actual_compression_mm",
        "compression_mm",
        "displacement_mm",
    ))
    numeric = float(value)
    if name.endswith("_mm") or name == "displacement_mm":
        numeric /= 1000.0
    if numeric < 0.0:
        raise ValueError("compression must be non-negative")
    return numeric


def _force_n(row):
    _, value = _first_present(row, (
        "force_n",
        "measured_force_n",
        "mean_force_n",
        "load_n",
        "pull_force_n",
    ))
    return float(value)


def read_spring_steel_measurements(path):
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        for row_idx, row in enumerate(reader, start=2):
            try:
                compression_m = _compression_m(row)
                force_n = _force_n(row)
            except ValueError as exc:
                raise ValueError(f"Invalid spring row {row_idx}: {exc}") from exc
            records.append({
                "compression_m": compression_m,
                "compression_mm": compression_m * 1000.0,
                "actual_compression_m": compression_m,
                "actual_compression_mm": compression_m * 1000.0,
                "mean_qpos_m": compression_m,
                "mean_force_n": force_n,
                "force_n": force_n,
                "trial_id": row.get("trial_id", ""),
                "segment_id": row.get("segment_id", ""),
                "repeat_id": row.get("repeat_id", ""),
                "direction": row.get("direction", ""),
                "note": row.get("note", ""),
            })
    if len(records) < 2:
        raise ValueError("Need at least two spring measurement rows")
    return records


def fit_spring_steel_measurements(records):
    fit = fit_equivalent_stiffness(records)
    fit["recommended_parameter"] = "SLIDE_JOINT_STIFFNESS"
    fit["recommended_value_source"] = "zero_intercept_stiffness_n_per_m"
    return fit


def write_spring_fit_bundle(out_dir, input_csv, records, fit):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "spring_steel_fit_summary.json")
    report_path = os.path.join(out_dir, "spring_steel_fit_report.md")
    recommended = float(fit["zero_intercept_stiffness_n_per_m"])
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": input_csv,
        "purpose": "real_spring_steel_to_v6_equivalent_slide_spring",
        "records": records,
        "fit": fit,
        "recommended_v6_slide_stiffness_n_per_m": recommended,
    }
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    lines = [
        "# Spring-Steel Hardware Fit Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Input CSV: `{input_csv}`",
        f"- Samples: `{fit['sample_count']}`",
        f"- Compression source: `{fit['compression_source']}`",
        "- Recommended V6 slide stiffness: "
        f"`{recommended:.3f} N/m`",
        "- Linear stiffness with intercept: "
        f"`{fit['stiffness_n_per_m']:.3f} N/m`",
        "- Secant stiffness at max compression: "
        f"`{fit['secant_stiffness_at_max_n_per_m']:.3f} N/m`",
        f"- Force intercept: `{fit['force_intercept_n']:.3f} N`",
        f"- Fit R2: `{fit['r2']:.5f}`",
        "",
        "## How To Use",
        "",
        "Use the recommended value as the first real-data candidate for "
        "`SLIDE_JOINT_STIFFNESS` in the reduced V6 whole-body model. Keep "
        "the full CSV and this report with the paper artifacts.",
        "",
    ]
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    return {"json": json_path, "report": report_path}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create Worm V6 hardware parameter templates and fits.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--write-templates", action="store_true")
    parser.add_argument("--spring-csv", default=None,
                        help="Spring-steel force-displacement CSV to fit")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    wrote_anything = False
    if args.write_templates:
        paths = write_parameter_templates(args.out_dir)
        wrote_anything = True
        print("Hardware parameter templates written")
        for key in sorted(paths):
            print(f"  {key}: {paths[key]}")
    if args.spring_csv:
        records = read_spring_steel_measurements(args.spring_csv)
        fit = fit_spring_steel_measurements(records)
        paths = write_spring_fit_bundle(
            args.out_dir,
            input_csv=args.spring_csv,
            records=records,
            fit=fit,
        )
        wrote_anything = True
        print("Spring-steel hardware fit complete")
        print("  recommended_v6_slide_stiffness_n_per_m: "
              f"{fit['zero_intercept_stiffness_n_per_m']:.3f}")
        print(f"  json: {paths['json']}")
        print(f"  report: {paths['report']}")
    if not wrote_anything:
        args = build_arg_parser().parse_args(["--help"])
        return args
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

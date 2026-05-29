import csv
import json
import os
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from hardware_parameter_id_v6 import (  # noqa: E402
    fit_spring_steel_measurements,
    read_spring_steel_measurements,
    write_parameter_templates,
    write_spring_fit_bundle,
)


def _write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_parameter_templates(tmp)
        expected = {
            "spring_steel_force_displacement",
            "motor_step_response",
            "imu_static",
            "contact_drag",
            "mass_geometry",
            "readme",
        }
        assert expected.issubset(set(paths)), paths
        for key in expected:
            assert os.path.exists(paths[key]), key

        measurement_csv = os.path.join(tmp, "spring_measurements.csv")
        _write_csv(
            measurement_csv,
            [
                "trial_id",
                "segment_id",
                "repeat_id",
                "actual_compression_mm",
                "force_n",
                "direction",
                "note",
            ],
            [
                {
                    "trial_id": "spring_001",
                    "segment_id": 0,
                    "repeat_id": 0,
                    "actual_compression_mm": 0.0,
                    "force_n": 0.0,
                    "direction": "loading",
                    "note": "zero",
                },
                {
                    "trial_id": "spring_001",
                    "segment_id": 0,
                    "repeat_id": 0,
                    "actual_compression_mm": 10.0,
                    "force_n": 3.0,
                    "direction": "loading",
                    "note": "ten_mm",
                },
                {
                    "trial_id": "spring_001",
                    "segment_id": 0,
                    "repeat_id": 0,
                    "actual_compression_mm": 20.0,
                    "force_n": 6.0,
                    "direction": "loading",
                    "note": "twenty_mm",
                },
            ],
        )
        rows = read_spring_steel_measurements(measurement_csv)
        assert len(rows) == 3
        assert rows[1]["actual_compression_m"] == 0.010
        fit = fit_spring_steel_measurements(rows)
        assert abs(fit["zero_intercept_stiffness_n_per_m"] - 300.0) < 1e-6
        assert fit["sample_count"] == 3
        assert fit["compression_source"] == "actual_compression_m"

        out_paths = write_spring_fit_bundle(
            tmp,
            input_csv=measurement_csv,
            records=rows,
            fit=fit,
        )
        for key in ("json", "report"):
            assert os.path.exists(out_paths[key]), key
        with open(out_paths["json"], encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["recommended_v6_slide_stiffness_n_per_m"] == 300.0
        with open(out_paths["report"], encoding="utf-8") as f:
            report = f.read()
        assert "Recommended V6 slide stiffness" in report

    print("hardware parameter identification contract passed")


if __name__ == "__main__":
    main()

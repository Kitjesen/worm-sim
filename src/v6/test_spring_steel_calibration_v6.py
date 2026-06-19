import csv
import json
import os
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from spring_steel_calibration_v6 import (  # noqa: E402
    CalibrationConfig,
    build_calibration_xml,
    fit_equivalent_stiffness,
    parse_compressions_mm,
    write_result_bundle,
)


def main():
    cfg = CalibrationConfig(
        strip_count=4,
        strip_vertices=5,
        rest_length_m=0.10,
        strip_circle_radius_m=0.03,
        strip_bow_m=0.004,
    )
    xml = build_calibration_xml(cfg)
    assert 'plugin plugin="mujoco.elasticity.cable"' in xml
    assert xml.count('type="cable"') == 4
    assert 'name="moving_slide"' in xml
    assert 'name="slide_position"' in xml
    assert 'name="slide_force"' in xml
    assert 'strip0B_first' in xml
    assert 'strip3B_last' in xml

    compressions = parse_compressions_mm("0, 5,10, 50")
    assert compressions == [0.0, 0.005, 0.010, 0.050]

    records = [
        {"compression_m": 0.000, "mean_force_n": 0.0},
        {"compression_m": 0.010, "mean_force_n": 3.0},
        {"compression_m": 0.020, "mean_force_n": 6.0},
        {"compression_m": 0.030, "mean_force_n": 9.0},
    ]
    fit = fit_equivalent_stiffness(records)
    assert abs(fit["stiffness_n_per_m"] - 300.0) < 1e-6
    assert fit["sample_count"] == 4
    assert fit["force_intercept_n"] == 0.0
    assert fit["compression_source"] == "compression_m"
    assert abs(fit["zero_intercept_stiffness_n_per_m"] - 300.0) < 1e-6
    assert abs(fit["secant_stiffness_at_max_n_per_m"] - 300.0) < 1e-6

    servo_limited_records = [
        {
            "compression_m": 0.050,
            "mean_qpos_m": 0.010,
            "mean_force_n": 3.0,
        },
        {
            "compression_m": 0.100,
            "mean_qpos_m": 0.020,
            "mean_force_n": 6.0,
        },
    ]
    actual_fit = fit_equivalent_stiffness(servo_limited_records)
    assert abs(actual_fit["stiffness_n_per_m"] - 300.0) < 1e-6
    assert actual_fit["compression_source"] == "mean_qpos_m"

    with tempfile.TemporaryDirectory() as tmp:
        paths = write_result_bundle(
            tmp,
            config=cfg,
            records=records,
            fit=fit,
            xml_text=xml,
            video_path=None,
        )
        for key in ("xml", "csv", "json", "report"):
            assert os.path.exists(paths[key]), (key, paths)
        with open(paths["csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[1]["compression_mm"] == "10.000"
        assert rows[1]["target_compression_mm"] == "10.000"
        assert rows[1]["actual_compression_mm"] == "10.000"
        with open(paths["json"], encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["fit"]["stiffness_n_per_m"] == 300.0
        assert payload["fit"]["zero_intercept_stiffness_n_per_m"] == 300.0
        assert payload["config"]["strip_count"] == 4
        with open(paths["report"], encoding="utf-8") as f:
            report_text = f.read()
        assert "Compression source for fit" in report_text

    print("spring steel calibration checks passed")


if __name__ == "__main__":
    main()

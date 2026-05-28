"""
Scan continuous gait_blend values for a trained deployable Worm V6 policy.

This produces the data needed for the paper's mode-selection curves:
performance versus gait_blend on flat, sand, and slope.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

PAPER_TERRAINS = ("flat", "sand", "slope")
DEFAULT_BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_blends(text):
    return [float(x) for x in text.split(",") if x.strip()]


def scan_dir(terrain, policy_mode):
    return os.path.join(PROJECT_ROOT, "runs",
                        f"worm_v6_blend_scan_{terrain}_{policy_mode}")


def metrics_path(out_dir, blend):
    tag = f"{blend:.2f}".replace(".", "p")
    return os.path.join(out_dir, f"blend_{tag}.json")


def run_scan(args):
    blends = parse_blends(args.blends)
    all_rows = []

    for terrain in args.terrain:
        out_dir = args.output_dir or scan_dir(terrain, args.policy_mode)
        os.makedirs(out_dir, exist_ok=True)
        for blend in blends:
            json_out = metrics_path(out_dir, blend)
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "eval_v6.py"),
                "--terrain", terrain,
                "--gait-mode", args.policy_mode,
                "--gait-blend", f"{blend:.4f}",
                "--episodes", str(args.episodes),
                "--time", str(args.time),
                "--success-distance", str(args.success_distance),
                "--json-out", json_out,
            ]
            if args.model:
                cmd.extend(["--model", args.model])
            if args.run_dir:
                cmd.extend(["--run-dir", args.run_dir])
            print(f"\nScan terrain={terrain} policy={args.policy_mode} blend={blend:.2f}")
            print("  " + " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

            if os.path.exists(json_out):
                with open(json_out, "r", encoding="utf-8") as f:
                    row = json.load(f)
                row["policy_mode"] = args.policy_mode
                all_rows.append(row)

        if not args.dry_run:
            write_scan_outputs(out_dir, all_rows_for_terrain(all_rows, terrain))

    if not args.dry_run and len(args.terrain) > 1:
        combined_dir = os.path.join(PROJECT_ROOT, "record", "v6",
                                    "paper_results")
        os.makedirs(combined_dir, exist_ok=True)
        write_scan_outputs(combined_dir, all_rows,
                           json_name="blend_scan_all.json",
                           csv_name="blend_scan_all.csv")


def all_rows_for_terrain(rows, terrain):
    return [row for row in rows if row.get("terrain") == terrain]


def write_scan_outputs(out_dir, rows, json_name="scan_results.json",
                       csv_name="scan_results.csv"):
    rows = sorted(rows, key=lambda r: (r.get("terrain", ""),
                                      float(r.get("gait_blend", 0.0))))
    json_path = os.path.join(out_dir, json_name)
    csv_path = os.path.join(out_dir, csv_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fields = [
        "terrain", "policy_mode", "gait_mode", "gait_blend",
        "mean_speed_mm_s", "std_speed_mm_s", "mean_lateral_drift_mm",
        "mean_action_l2_per_m", "mean_path_efficiency", "mean_slip_proxy",
        "mean_propulsion_efficiency_m_per_action_l2", "success_rate",
        "termination_rate", "model_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved scan JSON: {json_path}")
    print(f"Saved scan CSV:  {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Scan Worm V6 gait_blend values")
    ap.add_argument("--terrain", nargs="+", choices=PAPER_TERRAINS,
                    default=list(PAPER_TERRAINS))
    ap.add_argument("--policy-mode", default="random",
                    choices=["worm", "snake", "mixed", "random"],
                    help="Which trained policy run to load")
    ap.add_argument("--blends", default=",".join(str(x) for x in DEFAULT_BLENDS))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--success-distance", type=float, default=0.05)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()

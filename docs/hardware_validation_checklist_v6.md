# Worm V6 Hardware Validation Checklist

## Purpose

The hardware run is a small deployability validation, not a full replacement
for the simulation matrix. It must prove that the policy input and control
chain can run on the real robot using only joint encoders and one IMU per
segment.

## Required Terrains

- `flat`: flat rigid floor.
- `sand`: sand or high-slip granular surface.
- `slope`: fixed incline; record the slope angle in lab notes.

## Sensor Contract

- 11 actuator encoder positions and velocities.
- 7 segment IMUs with local gravity direction and local angular velocity.
- Command fields: target forward speed, target yaw rate, and `gait_blend`.
- Previous action and phase clock.
- No external localization, global pose, MuJoCo state, or ground-truth
  `base_linvel` as policy input.

## Collection Files

Use the generated field-trial package:

```powershell
python src\v3\prepare_hardware_trials_v6.py --force
```

For each terrain, collect raw hardware data into the terrain-specific
`*_raw.csv` template. The final validated CSV must be copied or written to:

- `record/v6/hardware/flat_random_YYYYMMDD.csv`
- `record/v6/hardware/sand_random_YYYYMMDD.csv`
- `record/v6/hardware/slope_random_YYYYMMDD.csv`

Each CSV row must include a resolvable video reference, typically:

- `record/v6/videos/flat_random_hardware_demo.mp4`
- `record/v6/videos/sand_random_hardware_demo.mp4`
- `record/v6/videos/slope_random_hardware_demo.mp4`

## Commands

Check what is still missing for each terrain:

```powershell
python src\v3\hardware_trial_status_v6.py --write-report
```

Capture a controller JSONL stream into the raw CSV expected by the import
step:

```powershell
python src\v3\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.5 --cmd-vel 0.025 --cmd-yaw 0.0
```

After a terrain run, import the captured raw sensor CSV and video reference into
the paper audit layout:

```powershell
python src\v3\import_hardware_trial_v6.py --terrain flat --mode random --raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.5 --date YYYYMMDD
```

For a live controller bridge, stream one raw sensor JSON object per line to
stdin and read one action JSON object per line from stdout:

```powershell
python src\v3\hardware_policy_runtime_v6.py --bundle-dir record\v6\deploy_bundles\flat_random --input-jsonl - --output-jsonl - --policy-log-csv record\v6\hardware\flat_random_YYYYMMDD.csv --terrain flat --mode random --gait-blend 0.5 --video-file record/v6/videos/flat_random_hardware_demo.mp4 --max-action-delta 0.2
```

Run the deployed TorchScript policy on a raw hardware sensor log and write the
formal 80-D audit CSV plus the physical target log:

```powershell
python src\v3\hardware_policy_runtime_v6.py --bundle-dir record\v6\deploy_bundles\flat_random --input-raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --output-csv record\v6\hardware\field_trials\current\flat\flat_random_actions.csv --policy-log-csv record\v6\hardware\flat_random_YYYYMMDD.csv --terrain flat --mode random --gait-blend 0.5 --video-file record/v6/videos/flat_random_hardware_demo.mp4 --max-action-delta 0.2 --validate-policy-log --require-video --expected-terrain flat --min-rows 5 --min-duration 0.1
```

Validate a final hardware log:

```powershell
python src\v3\validate_hardware_log_v6.py --input record\v6\hardware\flat_random_YYYYMMDD.csv --expected-terrain flat --require-video --min-rows 5 --min-duration 0.1
```

Replay a deploy bundle on logged observations:

```powershell
python src\v3\deploy_policy_v6.py replay --bundle-dir record\v6\deploy_bundles\flat_random --input-csv record\v6\hardware\flat_random_YYYYMMDD.csv
```

Refresh the paper audit:

```powershell
python src\v3\paper_status_v6.py --refresh-audit
```

## Pass Criteria

- One validated CSV per terrain.
- At least 5 rows and at least 0.1 s duration per CSV.
- `gait_blend` remains in `[0, 1]`.
- 7 IMU gravity vectors have reasonable norms.
- Phase `sin/cos` norm is reasonable.
- 11 policy actions are finite and within the normalized action range.
- Each CSV has a resolvable video reference.

## Paper Use

Report hardware validation as evidence that the state design and deployment
interface are real-robot compatible. Use simulation for the full terrain-mode
matrix and use these hardware runs as small-scale feasibility demonstrations.

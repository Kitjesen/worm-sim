# Worm V6 Hardware Trial: flat

- Terrain: `flat`
- Policy bundle: `record/v6/deploy_bundles/flat_random`
- Mode label: `random`
- Recommended gait_blend from scan: `0.500`
- Required video reference in CSV: `record/v6/videos/flat_random_hardware_demo.mp4`
- Controller JSONL example: `record/v6/hardware/field_trials/current/flat/flat_random_controller_stream_example.jsonl`
- Controller JSONL schema: `record/v6/hardware/field_trials/current/flat/flat_random_controller_stream_schema.json`

## Capture Requirements

- Record raw encoder positions and velocities for all 11 actuators.
- Record one IMU per segment: local gravity direction and angular velocity.
- Record normalized policy actions and physical command timing.
- Do not use external pose, MuJoCo state, or ground-truth base velocity as policy input.
- Keep at least 5 rows and at least 0.1 s duration for audit validation.

## Commands After Capture

### preflight deploy bundle and observation ABI

```powershell
python src\v3\preflight_hardware_deploy_v6.py --strict
```

### check controller JSONL stream before import

```powershell
python src\v3\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\flat_random --strict
```

### optional: live JSONL controller bridge

```powershell
python src\v3\hardware_policy_runtime_v6.py --bundle-dir record\v6\deploy_bundles\flat_random --input-jsonl - --output-jsonl - --policy-log-csv record\v6\hardware\flat_random_YYYYMMDD.csv --terrain flat --mode random --gait-blend 0.500 --video-file record/v6/videos/flat_random_hardware_demo.mp4 --max-action-delta 0.2
```

### capture controller JSONL stream into raw CSV

```powershell
python src\v3\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0
```

### one-command post-capture processing

```powershell
python src\v3\process_hardware_trial_v6.py --terrain flat --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

### import captured raw trial into audit files

```powershell
python src\v3\import_hardware_trial_v6.py --terrain flat --mode random --raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --date YYYYMMDD
```

### run deploy policy on raw sensors and write formal audit CSV

```powershell
python src\v3\hardware_policy_runtime_v6.py --bundle-dir record\v6\deploy_bundles\flat_random --input-raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --output-csv record\v6\hardware\field_trials\current\flat\flat_random_actions.csv --policy-log-csv record\v6\hardware\flat_random_YYYYMMDD.csv --terrain flat --mode random --gait-blend 0.500 --video-file record/v6/videos/flat_random_hardware_demo.mp4 --max-action-delta 0.2 --validate-policy-log --require-video --expected-terrain flat --min-rows 5 --min-duration 0.1
```

### validate policy CSV for the paper audit

```powershell
python src\v3\validate_hardware_log_v6.py --input record\v6\hardware\flat_random_YYYYMMDD.csv --expected-terrain flat --require-video --min-rows 5 --min-duration 0.1
```

### optional: replay deploy bundle on logged observations

```powershell
python src\v3\deploy_policy_v6.py replay --bundle-dir record\v6\deploy_bundles\flat_random --input-csv record\v6\hardware\flat_random_YYYYMMDD.csv --output-csv record\v6\hardware\field_trials\current\flat\flat_random_actions.csv
```

### show hardware trial status

```powershell
python src\v3\hardware_trial_status_v6.py --write-report
```

### refresh completion audit

```powershell
python src\v3\paper_status_v6.py --refresh-audit
```

## Final Files Expected By Audit

- `record/v6/hardware/flat_random_YYYYMMDD.csv`
- `record/v6/videos/flat_random_hardware_demo.mp4`

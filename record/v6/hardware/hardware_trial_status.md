# Worm V6 Hardware Trial Status

Complete: `false`

| Terrain | Status | Raw rows | Raw duration s | Video | Policy CSV |
| --- | --- | ---: | ---: | --- | --- |
| flat | `needs_raw_csv` | 0 | 0.000 | no | record/v6/hardware/flat_random_YYYYMMDD.csv |
| sand | `needs_raw_csv` | 0 | 0.000 | no | record/v6/hardware/sand_random_YYYYMMDD.csv |
| slope | `needs_raw_csv` | 0 | 0.000 | no | record/v6/hardware/slope_random_YYYYMMDD.csv |

## Next Commands

### flat

Controller stream self-check before importing the run:

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\flat_random --strict
```

One-command processing after the controller JSONL stream and demo video are available:

```powershell
python src\v6\process_hardware_trial_v6.py --terrain flat --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

Status-specific fallback:

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0
```

### sand

Controller stream self-check before importing the run:

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 1.000 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\sand_random --strict
```

One-command processing after the controller JSONL stream and demo video are available:

```powershell
python src\v6\process_hardware_trial_v6.py --terrain sand --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 1.000 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

Status-specific fallback:

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 1.000 --cmd-vel 0.025 --cmd-yaw 0.0
```

### slope

Controller stream self-check before importing the run:

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.000 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\slope_random --strict
```

One-command processing after the controller JSONL stream and demo video are available:

```powershell
python src\v6\process_hardware_trial_v6.py --terrain slope --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.000 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

Status-specific fallback:

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.000 --cmd-vel 0.025 --cmd-yaw 0.0
```


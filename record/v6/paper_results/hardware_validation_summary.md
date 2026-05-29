# Worm V6 Hardware Validation Summary

Complete: `false`

One validated real-hardware CSV with a resolvable video reference for each terrain: flat, sand, and slope.

## Terrain Evidence

| Terrain | Status | Evidence | gait_blend | Rows | Duration s | Velocity mm/s | Video |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| flat | `needs_raw_csv` | pending_real_run | 0.750 | 0.000 | 0.000 |  | no |
| sand | `needs_raw_csv` | pending_real_run | 0.500 | 0.000 | 0.000 |  | no |
| slope | `needs_raw_csv` | pending_real_run | 0.500 | 0.000 | 0.000 |  | no |

## Aggregate

- Validated terrains: `0/3`
- Mean velocity estimate: `` mm/s

## Not Counted As Hardware Evidence

- hardware templates
- controller stream examples
- simulation-to-hardware bridge logs
- deploy preflight reports

## Remaining Actions

### flat

Status: `needs_raw_csv`

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --cmd-vx 0.025 --cmd-vy 0.0 --cmd-yaw 0.0
```

### sand

Status: `needs_raw_csv`

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --cmd-vx 0.025 --cmd-vy 0.0 --cmd-yaw 0.0
```

### slope

Status: `needs_raw_csv`

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --cmd-vx 0.025 --cmd-vy 0.0 --cmd-yaw 0.0
```

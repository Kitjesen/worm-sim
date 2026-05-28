# Worm V6 Training Result Index

This page indexes the current deployable multimodal snake/worm training results. It only reports artifacts already present on disk.

## Open First

- [paper result summary](summary.md)
- [fixed-mode speed figure](fixed_mode_speed.svg)
- [gait_blend scan figure](blend_scan_speed.svg)
- [paper claim analysis](paper_claims.md)
- [deployable observation contract](observation_contract.md)
- [observation source audit](observation_source_audit.md)
- [representative simulation videos](paper_video_manifest.md)
- [hardware validation summary](hardware_validation_summary.md)
- [hardware deploy preflight](../hardware/hardware_deploy_preflight.md)
- [completion audit](completion_audit.md)

## Fixed-Mode RL Results

| Terrain | Mode | Speed mm/s | Success | Slip proxy | Robust speed mm/s | Robust success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flat | worm | 20.003 | 1.000 | 0.935 | -0.399 | 0.600 |
| flat | snake | 16.612 | 1.000 | 0.918 | 8.665 | 0.400 |
| flat | mixed | 5.445 | 0.800 | 0.983 | -13.509 | 0.200 |
| sand | worm | 2.376 | 0.600 | 0.974 | 3.931 | 0.400 |
| sand | snake | 4.964 | 0.400 | 0.978 | 3.438 | 0.600 |
| sand | mixed | 3.749 | 0.600 | 0.979 | 4.311 | 0.400 |
| slope | worm | -13.208 | 0.000 | 1.000 | -24.403 | 0.000 |
| slope | snake | -10.439 | 0.800 | 0.895 | -42.730 | 0.000 |
| slope | mixed | -11.201 | 0.000 | 1.000 | -12.709 | 0.200 |

## Best Continuous gait_blend Scan

| Terrain | Best gait_blend | Label | Speed mm/s | Success |
| --- | ---: | --- | ---: | ---: |
| flat | 0.500 | mixed | 16.657 | 1.000 |
| sand | 1.000 | snake | 4.880 | 0.600 |
| slope | 0.000 | worm | -2.086 | 0.400 |

## Cross-Terrain Claim Snapshot

- Adaptive best-blend average speed: 6.484 mm/s.
- Adaptive best-blend average success: 0.667.
- Completion audit: incomplete.
- supported: Continuous gait_blend exposes terrain-dependent mode preferences.
- supported: Adaptive best-blend random policy improves cross-terrain average speed over the best single fixed mode.
- not_supported: Adaptive best-blend random policy improves or preserves cross-terrain success rate versus the best single fixed mode.
- supported: Adaptive best-blend random policy improves or preserves cross-terrain termination rate versus the most stable single fixed mode.

## Current Cautions

- flat: best continuous random-policy blend does not beat the best terrain-specific fixed policy speed.
- sand: best continuous random-policy blend does not beat the best terrain-specific fixed policy speed.
- Success rate is not improved; frame Goal 3 around average speed and termination stability, not success.

## Training Runs

| Terrain | Mode | Model | VecNormalize | Completed steps | Run dir |
| --- | --- | --- | --- | ---: | --- |
| flat | worm | ok | ok | 1015808 | [runs/worm_v6_ppo_flat_worm](../../../runs/worm_v6_ppo_flat_worm) |
| flat | snake | ok | ok | 1015808 | [runs/worm_v6_ppo_flat_snake](../../../runs/worm_v6_ppo_flat_snake) |
| flat | mixed | ok | ok | 1015808 | [runs/worm_v6_ppo_flat_mixed](../../../runs/worm_v6_ppo_flat_mixed) |
| flat | random | ok | ok | 606208 | [runs/worm_v6_ppo_flat_random](../../../runs/worm_v6_ppo_flat_random) |
| sand | worm | ok | ok | 1015808 | [runs/worm_v6_ppo_sand_worm](../../../runs/worm_v6_ppo_sand_worm) |
| sand | snake | ok | ok | 1015808 | [runs/worm_v6_ppo_sand_snake](../../../runs/worm_v6_ppo_sand_snake) |
| sand | mixed | ok | ok | 1015808 | [runs/worm_v6_ppo_sand_mixed](../../../runs/worm_v6_ppo_sand_mixed) |
| sand | random | ok | ok | 1015808 | [runs/worm_v6_ppo_sand_random](../../../runs/worm_v6_ppo_sand_random) |
| slope | worm | ok | ok | 1015808 | [runs/worm_v6_ppo_slope_worm](../../../runs/worm_v6_ppo_slope_worm) |
| slope | snake | ok | ok | 1015808 | [runs/worm_v6_ppo_slope_snake](../../../runs/worm_v6_ppo_slope_snake) |
| slope | mixed | ok | ok | 1015808 | [runs/worm_v6_ppo_slope_mixed](../../../runs/worm_v6_ppo_slope_mixed) |
| slope | random | ok | ok | 1015808 | [runs/worm_v6_ppo_slope_random](../../../runs/worm_v6_ppo_slope_random) |

## Deployable Policy Bundles

| Terrain | TorchScript actor | Deploy config | Bundle |
| --- | --- | --- | --- |
| flat | ok | ok | [record/v6/deploy_bundles/flat_random](../deploy_bundles/flat_random) |
| sand | ok | ok | [record/v6/deploy_bundles/sand_random](../deploy_bundles/sand_random) |
| slope | ok | ok | [record/v6/deploy_bundles/slope_random](../deploy_bundles/slope_random) |

## Observation Source Audit

- Complete: true.
- Observation dimension: 80.
- Forbidden policy observation hits: 0.
- Reward-only privileged hits: forward_speed, lateral_speed, termination_height, termination_nan_qpos, yaw_rate.

## Hardware Deploy Preflight

- Complete: true.
- Observation ABI: a2970eeac045b4838117f76b0381d9a4c0b0bf540a7b3834f64d700aa059fe55.

| Terrain | Status | gait_blend | Max abs action |
| --- | --- | ---: | ---: |
| flat | ok | 0.500 | 0.200 |
| sand | ok | 1.000 | 0.200 |
| slope | ok | 0.000 | 0.200 |

## Hardware Validation Summary

- Complete: false.
- Validated terrains: 0/3.

| Terrain | Status | Evidence | gait_blend | Velocity mm/s | Video |
| --- | --- | --- | ---: | ---: | --- |
| flat | needs_raw_csv | pending_real_run | 0.500 |  | False |
| sand | needs_raw_csv | pending_real_run | 1.000 |  | False |
| slope | needs_raw_csv | pending_real_run | 0.000 |  | False |

## Viewable Videos

- [record/v6/videos/eval_flat_random.mp4](../videos/eval_flat_random.mp4) (0.6 MiB, flat best-blend preview)
- [record/v6/videos/eval_sand_random.mp4](../videos/eval_sand_random.mp4) (0.8 MiB, sand best-blend preview)
- [record/v6/videos/eval_slope_random.mp4](../videos/eval_slope_random.mp4) (0.4 MiB, slope best-blend preview)
- [training_arena_1280x720.mp4](../videos/training_arena_1280x720.mp4) (4.8 MiB)
- [gait_comparison_1280x720.mp4](../videos/gait_comparison_1280x720.mp4) (6.3 MiB)
- [worm_v6_combined.mp4](../videos/worm_v6_combined.mp4) (2.3 MiB)
- [worm_v6_worm.mp4](../videos/worm_v6_worm.mp4) (2.0 MiB)
- [eval_straight_fast.mp4](../videos/eval_straight_fast.mp4) (1.8 MiB)
- [eval_turn_left.mp4](../videos/eval_turn_left.mp4) (1.6 MiB)
- [eval_turn_right.mp4](../videos/eval_turn_right.mp4) (1.4 MiB)

## Hardware Trial Status

| Terrain | Status | Raw rows | Duration s | Video | Policy valid |
| --- | --- | ---: | ---: | --- | --- |
| flat | needs_raw_csv | 0 | 0.000 | False | False |
| sand | needs_raw_csv | 0 | 0.000 | False | False |
| slope | needs_raw_csv | 0 | 0.000 | False | False |

Controller stream self-check:

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\flat_random --strict
```

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 1.000 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\sand_random --strict
```

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.000 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\slope_random --strict
```


One-command hardware processing:

```powershell
python src\v6\process_hardware_trial_v6.py --terrain flat --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

```powershell
python src\v6\process_hardware_trial_v6.py --terrain sand --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 1.000 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

```powershell
python src\v6\process_hardware_trial_v6.py --terrain slope --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.000 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```


Status-specific fallback commands:

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0
```

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 1.000 --cmd-vel 0.025 --cmd-yaw 0.0
```

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.000 --cmd-vel 0.025 --cmd-yaw 0.0
```

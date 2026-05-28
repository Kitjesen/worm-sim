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
| flat | worm | 17.196 | 1.000 | 0.940 | 13.770 | 1.000 |
| flat | snake | 17.944 | 1.000 | 0.910 | -14.457 | 0.000 |
| flat | mixed | 10.451 | 0.800 | 0.962 | 31.325 | 1.000 |
| sand | worm |  |  |  |  |  |
| sand | snake |  |  |  |  |  |
| sand | mixed |  |  |  |  |  |
| slope | worm |  |  |  |  |  |
| slope | snake |  |  |  |  |  |
| slope | mixed |  |  |  |  |  |

## Best Continuous gait_blend Scan

| Terrain | Best gait_blend | Label | Speed mm/s | Success |
| --- | ---: | --- | ---: | ---: |
| flat | 0.750 | mixed | 24.832 | 1.000 |

## Cross-Terrain Claim Snapshot

- Adaptive best-blend average speed: 24.832 mm/s.
- Adaptive best-blend average success: 1.000.
- Completion audit: incomplete.
- incomplete: Continuous gait_blend exposes terrain-dependent mode preferences.
- incomplete: Adaptive best-blend random policy improves cross-terrain average speed over the best single fixed mode.
- incomplete: Adaptive best-blend random policy improves or preserves cross-terrain success rate versus the best single fixed mode.
- incomplete: Adaptive best-blend random policy improves or preserves cross-terrain termination rate versus the most stable single fixed mode.

## Current Cautions

- Current-contract cross-terrain claims are incomplete until these terrains have fresh eval and scan evidence: sand, slope.

## Training Runs

| Terrain | Mode | Model | VecNormalize | Completed steps | Run dir |
| --- | --- | --- | --- | ---: | --- |
| flat | worm | ok | ok | 278528 | [runs/worm_v6_ppo_flat_worm](../../../runs/worm_v6_ppo_flat_worm) |
| flat | snake | ok | ok | 1015808 | [runs/worm_v6_ppo_flat_snake](../../../runs/worm_v6_ppo_flat_snake) |
| flat | mixed | ok | ok | 1015808 | [runs/worm_v6_ppo_flat_mixed](../../../runs/worm_v6_ppo_flat_mixed) |
| flat | random | ok | ok | 1015808 | [runs/worm_v6_ppo_flat_random](../../../runs/worm_v6_ppo_flat_random) |
| sand | worm | ok | ok | 1002592 | [runs/worm_v6_ppo_sand_worm](../../../runs/worm_v6_ppo_sand_worm) |
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

- Complete: false.
- Observation ABI: 672933c397be678aeae9067d299f38bfe69c748729beeab948cb339bb727183b.

| Terrain | Status | gait_blend | Max abs action |
| --- | --- | ---: | ---: |
| flat | ok | 0.750 | 0.200 |
| sand | failed | 0.500 | 0.200 |
| slope | failed | 0.500 | 0.200 |

## Hardware Validation Summary

- Complete: false.
- Validated terrains: 0/3.

| Terrain | Status | Evidence | gait_blend | Velocity mm/s | Video |
| --- | --- | --- | ---: | ---: | --- |
| flat | needs_raw_csv | pending_real_run | 0.750 |  | False |
| sand | needs_raw_csv | pending_real_run | 0.500 |  | False |
| slope | needs_raw_csv | pending_real_run | 0.500 |  | False |

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
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.750 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\flat_random --strict
```

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\sand_random --strict
```

```powershell
python src\v6\check_controller_stream_v6.py --input-jsonl controller_stream.jsonl --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --bundle-dir record\v6\deploy_bundles\slope_random --strict
```


One-command hardware processing:

```powershell
python src\v6\process_hardware_trial_v6.py --terrain flat --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.750 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

```powershell
python src\v6\process_hardware_trial_v6.py --terrain sand --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```

```powershell
python src\v6\process_hardware_trial_v6.py --terrain slope --mode random --input-jsonl controller_stream.jsonl --raw-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0 --date YYYYMMDD
```


Status-specific fallback commands:

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\flat\flat_random_raw.csv --terrain flat --mode random --video-file record/v6/videos/flat_random_hardware_demo.mp4 --gait-blend 0.750 --cmd-vel 0.025 --cmd-yaw 0.0
```

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\sand\sand_random_raw.csv --terrain sand --mode random --video-file record/v6/videos/sand_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0
```

```powershell
python src\v6\capture_hardware_stream_v6.py --input-jsonl controller_stream.jsonl --output-csv record\v6\hardware\field_trials\current\slope\slope_random_raw.csv --terrain slope --mode random --video-file record/v6/videos/slope_random_hardware_demo.mp4 --gait-blend 0.500 --cmd-vel 0.025 --cmd-yaw 0.0
```

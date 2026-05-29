# Worm V6 Goal Completion Audit

Complete: `false`

| Requirement | Status | Missing |
| --- | --- | --- |
| hardware template/schema | ok |  |
| online deploy runtime tools | ok |  |
| hardware deploy preflight report | missing | hardware deploy preflight complete, hardware deploy preflight terrain status |
| sim sensor to hardware-policy bridge | ok |  |
| 12 PPO training model artifacts | missing | runs/worm_v6_ppo_flat_worm, runs/worm_v6_ppo_flat_snake, runs/worm_v6_ppo_flat_mixed, runs/worm_v6_ppo_flat_random, runs/worm_v6_ppo_sand_worm, runs/worm_v6_ppo_sand_snake, runs/worm_v6_ppo_sand_mixed, runs/worm_v6_ppo_sand_random, runs/worm_v6_ppo_slope_worm, runs/worm_v6_ppo_slope_snake, runs/worm_v6_ppo_slope_mixed, runs/worm_v6_ppo_slope_random |
| 9 fixed-mode eval JSON files | missing | runs/worm_v6_ppo_flat_worm/eval_metrics.json, runs/worm_v6_ppo_flat_snake/eval_metrics.json, runs/worm_v6_ppo_flat_mixed/eval_metrics.json, runs/worm_v6_ppo_sand_worm/eval_metrics.json, runs/worm_v6_ppo_sand_snake/eval_metrics.json, runs/worm_v6_ppo_sand_mixed/eval_metrics.json, runs/worm_v6_ppo_slope_worm/eval_metrics.json, runs/worm_v6_ppo_slope_snake/eval_metrics.json, runs/worm_v6_ppo_slope_mixed/eval_metrics.json |
| 9 robust fixed-mode eval JSON files | missing | runs/worm_v6_ppo_flat_worm/eval_metrics_robust.json, runs/worm_v6_ppo_flat_snake/eval_metrics_robust.json, runs/worm_v6_ppo_flat_mixed/eval_metrics_robust.json, runs/worm_v6_ppo_sand_worm/eval_metrics_robust.json, runs/worm_v6_ppo_sand_snake/eval_metrics_robust.json, runs/worm_v6_ppo_sand_mixed/eval_metrics_robust.json, runs/worm_v6_ppo_slope_worm/eval_metrics_robust.json, runs/worm_v6_ppo_slope_snake/eval_metrics_robust.json, runs/worm_v6_ppo_slope_mixed/eval_metrics_robust.json |
| 9 CMA-ES open-loop baseline JSON files | ok |  |
| 3 fixed-gate gait_blend ablation scan result files | missing | runs/worm_v6_blend_scan_flat_random/scan_results.json, runs/worm_v6_blend_scan_sand_random/scan_results.json, runs/worm_v6_blend_scan_slope_random/scan_results.json |
| 3 deployable random-policy bundles | missing | record/v6/deploy_bundles/flat_random, record/v6/deploy_bundles/sand_random, record/v6/deploy_bundles/slope_random |
| paper summary tables, figures, and claim analysis | ok |  |
| flat/sand/slope hardware logs with video references | missing | record/v6/hardware/flat_*.csv, record/v6/hardware/sand_*.csv, record/v6/hardware/slope_*.csv |

# Worm V6 V93b mixed-sign-repair acceptance

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Selected sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `v93b_best_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0541 | 0.0217 | 0 | 0 | 2 | 3 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0572, -0.0018, 0.0456)` | pass |
| `v93b_best_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0592 | 0.0201 | 1 | 0 | 2 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0342, 0.0200, 0.0446)` | fail |
| `v93b_final_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0571 | 0.0216 | 0 | 0 | 2 | 3 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0568, -0.0030, 0.0758)` | pass |
| `v93b_final_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0569 | 0.0200 | 1 | 0 | 2 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0316, 0.0152, 0.0221)` | fail |

# Worm V6 V92b low-yaw-envelope acceptance

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Selected sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `v92b_best_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0584 | 0.0191 | 0 | 0 | 2 | 3 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0636, -0.0057, 0.0669)` | pass |
| `v92b_best_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0582 | 0.0202 | 1 | 0 | 1 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0384, 0.0108, 0.0211)` | fail |
| `v92b_final_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0545 | 0.0184 | 0 | 0 | 0 | 2 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0510, -0.0112, 0.0479)` | pass |
| `v92b_final_robust` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0610 | 0.0203 | 0 | 0 | 1 | 1 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0423, 0.0070, 0.0894)` | pass |

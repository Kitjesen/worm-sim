# Worm V6 V94 robust-forward-left acceptance

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Selected sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `v94_best_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0547 | 0.0217 | 0 | 0 | 1 | 3 | true | `(0.0500, 0.0000, -0.1000)` | `(0.0543, 0.0213, 0.0895)` | pass |
| `v94_best_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0570 | 0.0201 | 1 | 0 | 2 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0345, 0.0059, 0.0113)` | fail |
| `v94_final_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0534 | 0.0218 | 0 | 0 | 1 | 3 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0616, 0.0100, 0.0762)` | pass |
| `v94_final_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0553 | 0.0201 | 1 | 0 | 1 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0344, 0.0157, 0.0333)` | fail |

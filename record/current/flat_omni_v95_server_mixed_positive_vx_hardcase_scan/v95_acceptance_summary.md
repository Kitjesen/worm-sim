# Worm V6 V95 mixed positive-vx hardcase strict component scan

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Wrong mixed comp | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Projected sign | Mixed comp sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `v95_best_nominal` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0547 | 0.0217 | 0 | 0 | 7 | 1 | 3 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0229, 0.0578, 0.1136)` | pass | fail |
| `v95_best_robust` | false | `wrong_planar_sign_count,wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0570 | 0.0201 | 1 | 0 | 8 | 2 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0345, 0.0059, 0.0113)` | fail | fail |
| `v95_final_nominal` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0572 | 0.0218 | 0 | 0 | 11 | 2 | 2 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0267, 0.0516, 0.0851)` | pass | fail |
| `v95_final_robust` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0616 | 0.0201 | 0 | 0 | 10 | 2 | 1 | true | `(-0.1000, 0.0750, 0.0000)` | `(-0.0605, -0.0338, -0.0882)` | pass | fail |

# Worm V6 V89a robust-forward-diagonal repair acceptance

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Selected sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `v89a_best_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0609 | 0.0189 | 0 | 0 | 2 | 4 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0620, 0.0063, 0.0740)` | pass |
| `v89a_best_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0618 | 0.0198 | 1 | 0 | 2 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0356, 0.0142, 0.0205)` | fail |
| `v89a_final_nominal` | true | `-` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0532 | 0.0192 | 0 | 0 | 0 | 2 | true | `(0.1000, 0.0000, -0.1000)` | `(0.0647, 0.0135, 0.0780)` | pass |
| `v89a_final_robust` | false | `wrong_planar_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v37` | 0.0632 | 0.0201 | 1 | 0 | 3 | 1 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0405, 0.0046, 0.0163)` | fail |

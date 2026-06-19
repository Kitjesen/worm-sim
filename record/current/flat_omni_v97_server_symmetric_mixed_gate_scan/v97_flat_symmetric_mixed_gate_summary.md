# Worm V6 V97 flat symmetric mixed-gate acceptance

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Wrong mixed comp | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Projected sign | Mixed comp sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `v97_final_model_nominal` | false | `wrong_planar_sign_count,wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0775 | 0.0218 | 1 | 0 | 8 | 4 | 0 | true | `(0.0500, -0.0750, 0.0000)` | `(-0.0870, 0.0097, 0.0808)` | fail | fail |
| `v97_final_model_robust` | false | `wrong_planar_sign_count,wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0844 | 0.0201 | 3 | 0 | 8 | 9 | 2 | true | `(0.0500, -0.0750, 0.0000)` | `(-0.0913, -0.0124, 0.0458)` | fail | fail |
| `v97_progress_best_model_nominal` | false | `wrong_planar_sign_count,wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0741 | 0.0219 | 1 | 0 | 7 | 5 | 0 | true | `(0.0500, 0.0750, 0.0000)` | `(-0.0277, 0.0156, 0.0373)` | fail | fail |
| `v97_progress_best_model_robust` | false | `wrong_planar_sign_count,wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0825 | 0.0201 | 2 | 0 | 7 | 6 | 1 | true | `(0.0500, -0.0750, 0.0000)` | `(-0.1053, 0.0136, 0.0782)` | fail | fail |

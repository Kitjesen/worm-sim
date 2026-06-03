# Worm V6 V96 flat mixed-component sign acceptance

Selected row: sign-failure command when available, otherwise worst strict-analysis command.

| Label | Accepted | Failed | Adapter | Planar RMSE | Yaw RMSE | Wrong planar | Wrong yaw | Wrong mixed comp | Planar exceed | Off-axis exceed | Fixed lateral strict | Selected command | Measured `(vx, vy, yaw)` | Projected sign | Mixed comp sign |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `v96_final_model_nominal` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0597 | 0.0217 | 0 | 0 | 9 | 2 | 1 | true | `(0.1000, -0.0750, 0.0000)` | `(0.0810, 0.0363, 0.1319)` | pass | fail |
| `v96_final_model_robust` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0575 | 0.0200 | 0 | 0 | 8 | 1 | 1 | true | `(-0.1000, 0.0375, 0.0000)` | `(-0.0526, -0.0355, -0.1017)` | pass | fail |
| `v96_progress_best_model_nominal` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0595 | 0.0218 | 0 | 0 | 9 | 2 | 1 | true | `(0.1000, -0.0750, 0.0000)` | `(0.0635, 0.0274, 0.0838)` | pass | fail |
| `v96_progress_best_model_robust` | false | `wrong_mixed_component_sign_count` | `cmaes_tri_anchor_auto_gate_directional_v40` | 0.0595 | 0.0201 | 0 | 0 | 8 | 2 | 1 | true | `(-0.1000, 0.0750, 0.0000)` | `(-0.0615, -0.0288, -0.0729)` | pass | fail |

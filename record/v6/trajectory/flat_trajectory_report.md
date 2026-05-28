# Worm V6 Body-Segment Trajectory Report

- Terrain: `flat`
- Duration: `6.00 s`
- Forward convention: `-X` in the MuJoCo world frame
- Segment order: `head/base_link`, then `back1_Link` ... `back6_Link`

| Mode | Head direction | Head delta XY (mm) | Unit XY | Forward speed (mm/s) | Absolute position plot | Relative motion plot | CSV |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| snake | forward (-X) | (-182.8, -13.7) | (-0.997, -0.075) | 30.47 | record/v6/trajectory/flat_snake_absolute_positions.png | record/v6/trajectory/flat_snake_relative_motion.png | record/v6/trajectory/flat_snake_segment_trajectories.csv |
| worm | forward (-X) | (-121.8, -0.0) | (-1.000, -0.000) | 20.30 | record/v6/trajectory/flat_worm_absolute_positions.png | record/v6/trajectory/flat_worm_relative_motion.png | record/v6/trajectory/flat_worm_segment_trajectories.csv |
| combined | forward (-X) | (-257.0, -22.4) | (-0.996, -0.087) | 42.83 | record/v6/trajectory/flat_combined_absolute_positions.png | record/v6/trajectory/flat_combined_relative_motion.png | record/v6/trajectory/flat_combined_segment_trajectories.csv |

`absolute_positions` plots show each body segment's actual world position time history: world `X(t)`, `Y(t)`, and `Z(t)`. Segment initial positions are different because the robot has physical length.

`relative_motion` plots show each body segment's motion relative to its own initial position, useful for comparing displacement phases.

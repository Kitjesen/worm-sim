# Worm V6 Body-Segment Trajectory Report

- Terrain: `sand`
- Duration: `6.00 s`
- Forward convention: `-X` in the MuJoCo world frame
- Segment order: `head/base_link`, then `back1_Link` ... `back6_Link`

| Mode | Head direction | Head delta XY (mm) | Unit XY | Forward speed (mm/s) | Absolute position plot | Relative motion plot | CSV |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| snake | forward (-X) | (-15.4, 3.5) | (-0.975, 0.221) | 2.56 | record/v6/trajectory/sand_snake_absolute_positions.png | record/v6/trajectory/sand_snake_relative_motion.png | record/v6/trajectory/sand_snake_segment_trajectories.csv |
| worm | forward (-X) | (-78.7, -0.0) | (-1.000, -0.000) | 13.11 | record/v6/trajectory/sand_worm_absolute_positions.png | record/v6/trajectory/sand_worm_relative_motion.png | record/v6/trajectory/sand_worm_segment_trajectories.csv |
| combined | forward (-X) | (-94.9, 9.8) | (-0.995, 0.103) | 15.81 | record/v6/trajectory/sand_combined_absolute_positions.png | record/v6/trajectory/sand_combined_relative_motion.png | record/v6/trajectory/sand_combined_segment_trajectories.csv |

`absolute_positions` plots show each body segment's actual world position time history: world `X(t)`, `Y(t)`, and `Z(t)`. Segment initial positions are different because the robot has physical length.

`relative_motion` plots show each body segment's motion relative to its own initial position, useful for comparing displacement phases.

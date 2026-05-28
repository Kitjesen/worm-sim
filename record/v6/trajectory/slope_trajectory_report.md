# Worm V6 Body-Segment Trajectory Report

- Terrain: `slope`
- Duration: `6.00 s`
- Forward convention: `-X` in the MuJoCo world frame
- Segment order: `head/base_link`, then `back1_Link` ... `back6_Link`

| Mode | Head direction | Head delta XY (mm) | Unit XY | Forward speed (mm/s) | Absolute position plot | Relative motion plot | CSV |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| snake | backward (+X) | (77.1, -41.2) | (0.882, -0.471) | -12.86 | record/v6/trajectory/slope_snake_absolute_positions.png | record/v6/trajectory/slope_snake_relative_motion.png | record/v6/trajectory/slope_snake_segment_trajectories.csv |
| worm | backward (+X) | (201.3, -0.0) | (1.000, -0.000) | -33.55 | record/v6/trajectory/slope_worm_absolute_positions.png | record/v6/trajectory/slope_worm_relative_motion.png | record/v6/trajectory/slope_worm_segment_trajectories.csv |
| combined | backward (+X) | (126.6, -68.7) | (0.879, -0.477) | -21.10 | record/v6/trajectory/slope_combined_absolute_positions.png | record/v6/trajectory/slope_combined_relative_motion.png | record/v6/trajectory/slope_combined_segment_trajectories.csv |

`absolute_positions` plots show each body segment's actual world position time history: world `X(t)`, `Y(t)`, and `Z(t)`. Segment initial positions are different because the robot has physical length.

`relative_motion` plots show each body segment's motion relative to its own initial position, useful for comparing displacement phases.

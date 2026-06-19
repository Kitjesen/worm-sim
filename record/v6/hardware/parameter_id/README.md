# Worm V6 Hardware Parameter Identification

These files are bench-test templates for calibrating simulation parameters from the real robot. They are separate from the field trial logs used to validate policy deployment.

Minimum useful dataset:

1. spring-steel force-displacement points for several compression values and repeats;
2. actuator step responses for all 6 slide and 5 yaw motors;
3. static IMU samples for all 7 segments in known poses;
4. contact drag tests for flat, sand, and slope-relevant surfaces;
5. measured mass/geometry for every segment assembly.

Use actual measured values. Commanded values can be logged too, but the fitters use measured displacement, force, velocity, and current.

## Files

- `spring_steel_force_displacement_template.csv`: Bench compression test for one spring-steel strip assembly. Use actual measured displacement, not only commanded motion.
- `motor_step_response_template.csv`: Actuator response test for each slide and yaw motor. This is used to identify speed limits, delay, saturation, and controller gain.
- `imu_static_template.csv`: Static IMU orientation/noise test. Record every segment in several known poses to fit sensor axes, bias, and noise.
- `contact_drag_template.csv`: Terrain/contact drag test for flat, sand, and slope materials. Pull a segment or body at known normal load and speed.
- `mass_geometry_template.csv`: Measured mass and geometry for each segment or assembly. Needed for body mass, inertia, and center-of-mass updates.

## Spring-Steel Fit

After filling the spring-steel CSV, run:

```powershell
python src\v6\hardware_parameter_id_v6.py --spring-csv record\v6\hardware\parameter_id\spring_steel_force_displacement_REAL.csv
```

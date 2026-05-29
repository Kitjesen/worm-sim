# Worm V6 Real-Robot Experiment TODO

Status: **TODO / bench data not collected yet**

Last updated: 2026-05-29

## 1. Goal

The goal is to turn the current V6 MuJoCo + RL project into a real-robot
deployable system whose simulation parameters are grounded in measurements from
the physical robot.

The real slide mechanism is asymmetric:

- active contraction: servo pulls a rope/cable;
- passive release: spring-steel strips return the segment;
- the motor cannot actively push the segment outward.

Therefore the experiment plan must identify two separate mechanisms:

1. passive spring-steel return force;
2. active servo-rope pulling dynamics.

## 2. Final Package We Want

The expected final deliverable is a hardware-calibrated V6 package:

- calibrated MuJoCo parameters for spring return, motor limits, IMU noise,
  contact friction, mass, and geometry;
- deployable 80-D observation pipeline using only encoders, per-segment IMUs,
  previous action, phase clock, and body-frame velocity commands;
- trained or replayable policy bundle for `flat`, `sand`, and `slope`;
- videos and CSV logs for real hardware validation;
- paper-ready tables and plots comparing worm, snake, mixed, and learned
  auto-gated locomotion.

## 3. Current Scope Marked For GitHub

This TODO marks the real-data side of the project. Code scaffolding exists, but
the real measurements are still missing.

Already added:

- single-segment MuJoCo spring-steel calibration model;
- real-robot parameter CSV templates;
- spring-steel force-displacement fitter;
- documentation for unilateral rope-pull actuation.

Still TODO:

- collect real spring-steel compression data;
- collect servo-rope step-response data;
- collect IMU static calibration data;
- collect contact drag data on flat, sand, and slope surfaces;
- collect mass and geometry data for each segment;
- update MuJoCo parameters from measured data;
- retrain/evaluate policies after parameter update.

## 4. Experiment A: Spring-Steel Passive Return

Purpose: identify passive return force from the spring-steel strip assembly.

This should be tested with a tensile/compression testing machine if available.
The test must compress the actual strip/segment assembly in the robot's axial
direction.

### A.1 Setup

- Use the real spring-steel strip assembly or one representative segment pair.
- Use fixture boundary conditions close to the robot mounting condition.
- Compression axis should match the robot slide direction.
- Record force in Newtons.
- Record actual displacement in millimeters.

### A.2 Data To Record

Template:

```text
record/v6/hardware/parameter_id/spring_steel_force_displacement_template.csv
```

Required columns:

```text
trial_id
segment_id
repeat_id
target_compression_mm
actual_compression_mm
hold_time_s
force_n
force_sensor_id
direction
note
```

### A.3 Test Points

Recommended quasi-static compression points:

```text
0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 mm
```

Recommended speed:

```text
1-5 mm/min
```

Recommended repeats:

```text
3 loading cycles and 3 unloading cycles
```

### A.4 Acceptance Criteria

- CSV has at least 11 compression levels.
- Each level has at least 3 repeats.
- Loading and unloading are both present.
- Force sign convention is documented; compression resistance should be
  positive in the final CSV.
- No plastic deformation or fixture slip is visible in the notes/video.

### A.5 Fit Command

After the real CSV is filled:

```powershell
python src\v6\hardware_parameter_id_v6.py --spring-csv record\v6\hardware\parameter_id\spring_steel_force_displacement_YYYYMMDD.csv
```

Output:

- `spring_steel_fit_summary.json`
- `spring_steel_fit_report.md`
- recommended candidate for `SLIDE_JOINT_STIFFNESS`

## 5. Experiment B: Servo-Rope Active Pulling

Purpose: identify the real contraction actuator, not the passive spring.

The slide motor should be treated as unilateral cable/tendon pulling:

- command increase pulls rope and contracts the segment;
- command decrease reduces cable tension;
- return is caused by spring steel, not motor pushing.

### B.1 Data To Record

Template:

```text
record/v6/hardware/parameter_id/motor_step_response_template.csv
```

Required minimum columns:

```text
time_s
joint_id
joint_type
command_norm
target_si
position_si
velocity_si
current_a
voltage_v
pwm
load_n_or_nm
note
```

If current, voltage, or PWM cannot be recorded yet, leave the field empty and
record command, target, encoder position, and encoder velocity first.

### B.2 Slide Joint Steps

For each of the 6 slide joints:

```text
0 -> 25% contraction
0 -> 50% contraction
0 -> 75% contraction
0 -> 100% contraction
100% -> 0 release
```

Record no-load and typical-load cases.

Sampling rate:

```text
>= 100 Hz preferred, >= 50 Hz minimum
```

### B.3 What We Fit

- maximum contraction speed;
- release speed under passive spring return;
- delay/deadband;
- rope slack region;
- motor force/current limit;
- controller saturation.

### B.4 Acceptance Criteria

- all 6 slide joints have logs;
- every test includes command, encoder position, and time;
- release phase clearly shows passive return behavior;
- notes mark whether rope is taut or slack.

## 6. Experiment C: Yaw Motor Response

Purpose: identify yaw servo limits for snake gait.

Use the same motor step template.

For each of the 5 yaw joints:

```text
0 -> +25%, 0 -> +50%, 0 -> +75%, 0 -> +100%
0 -> -25%, 0 -> -50%, 0 -> -75%, 0 -> -100%
```

Required outputs:

- max yaw speed;
- delay;
- overshoot;
- position tracking error;
- current/torque saturation if measurable.

## 7. Experiment D: IMU Static Calibration

Purpose: verify segment IMU axes, gravity direction, gyro bias, and noise.

Template:

```text
record/v6/hardware/parameter_id/imu_static_template.csv
```

For each of the 7 segment IMUs, record 5-10 seconds in:

```text
level
nose_up
nose_down
left_side
right_side
```

Required columns:

```text
segment_id
pose_label
sample_id
acc_x_m_s2
acc_y_m_s2
acc_z_m_s2
gyro_x_rad_s
gyro_y_rad_s
gyro_z_rad_s
temperature_c
note
```

Acceptance criteria:

- all 7 IMUs have all poses;
- gravity magnitude is near `9.81 m/s^2`;
- gyro mean is near zero in static poses;
- axis/sign convention is documented.

## 8. Experiment E: Contact Drag

Purpose: fit contact/friction behavior for flat, sand, and slope-relevant
surfaces.

Template:

```text
record/v6/hardware/parameter_id/contact_drag_template.csv
```

Setup:

- place one segment or representative shell on the test surface;
- add known normal load;
- pull at controlled speed with a force gauge or test machine;
- record steady pulling force.

Required terrains:

```text
flat
sand
slope material or incline surface
```

Acceptance criteria:

- at least 3 repeats per terrain;
- normal load is recorded;
- pull speed is recorded;
- slope angle is recorded for slope tests.

## 9. Experiment F: Mass And Geometry

Purpose: update body mass, segment dimensions, approximate center of mass, and
inertia estimates.

Template:

```text
record/v6/hardware/parameter_id/mass_geometry_template.csv
```

Required measurements:

- each assembled segment mass;
- segment length, width, height;
- approximate center of mass;
- battery/electronics/bracket mass if separated.

Acceptance criteria:

- all 7 segment assemblies are measured;
- units are SI;
- center of mass assumption is documented when approximate.

## 10. Simulation Update TODO

After the measured CSVs exist:

1. Fit spring-steel `SLIDE_JOINT_STIFFNESS`.
2. Add unilateral cable/tendon pull model for slide actuation.
3. Add rope slack/deadband and motor speed limits.
4. Update yaw actuator speed/torque limits from step responses.
5. Add IMU bias/noise model from static data.
6. Update contact/friction parameters for flat, sand, and slope.
7. Update body masses and approximate inertias.
8. Re-run calibration smoke tests.
9. Re-run V6 policy eval videos.
10. Retrain or fine-tune policies if dynamics changed materially.

## 11. Paper Evidence TODO

The paper evidence should eventually include:

- parameter-identification setup photos or video;
- spring-steel force-displacement curve;
- actuator step-response plots;
- IMU static-noise table;
- friction/contact table for flat, sand, and slope;
- calibrated-vs-uncalibrated simulation comparison;
- worm/snake/mixed/auto-gate locomotion videos;
- segment trajectory time-history plots;
- final hardware deployment logs and videos.

## 12. Current Risk

The current V6 model is useful for algorithm development but is not yet a
fully hardware-calibrated model. The largest known mismatch is slide actuation:
real hardware uses unilateral servo-rope pulling plus passive spring return,
while current MuJoCo still uses an idealized position-servo abstraction.

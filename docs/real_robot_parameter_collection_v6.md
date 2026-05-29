# Real Robot Parameter Collection for Worm V6

## Purpose

This is the data package needed to make the MuJoCo simulation match the real
robot. The field-trial hardware logs prove that the policy can run on real
sensors. The files here are different: they identify physical parameters such
as spring stiffness, motor response, sensor bias, mass, and contact friction.

Generated templates:

```powershell
python src\v6\hardware_parameter_id_v6.py --write-templates
```

Default folder:

```text
record/v6/hardware/parameter_id
```

## 1. Spring-Steel Force-Displacement

This is the most important missing mechanical parameter.

The real robot actuation is asymmetric:

- contraction is active: the servo pulls a rope/cable;
- extension/release is passive: the spring-steel strips push the segment back;
- the actuator should not be treated as a bidirectional linear motor that can
  actively push extension.

Therefore, collect spring-steel data and servo-rope actuation data separately.
The spring-steel test identifies passive return stiffness. The motor step test
identifies active pulling speed, delay, force/current limit, deadband, and rope
slack.

Use a bench setup with one segment or one spring-steel strip assembly:

1. Put a load cell in the axial direction.
2. Measure actual compression with calipers, a linear encoder, or a dial gauge.
3. Record both target compression and actual compression.
4. Hold each point for about 2 seconds and record the steady force.
5. Repeat loading and unloading at least 3 times.

Recommended compression points:

```text
0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 mm
```

Fill:

```text
record/v6/hardware/parameter_id/spring_steel_force_displacement_template.csv
```

Then copy it to a real-data filename, for example:

```text
record/v6/hardware/parameter_id/spring_steel_force_displacement_20260529.csv
```

Run:

```powershell
python src\v6\hardware_parameter_id_v6.py --spring-csv record\v6\hardware\parameter_id\spring_steel_force_displacement_20260529.csv
```

The output gives a recommended candidate for:

```text
SLIDE_JOINT_STIFFNESS
```

## 2. Motor Step Response

For all 6 slide joints and 5 yaw joints, log step commands under no-load and
typical-load conditions.

Minimum log rate: `100 Hz`.

For each joint, record:

- normalized command;
- physical target in meters or radians;
- encoder position;
- encoder velocity;
- motor current;
- voltage;
- PWM or duty cycle if available;
- external load if measured.

The output will later be used for:

- position servo gain;
- speed limit;
- force/torque limit;
- delay;
- deadband;
- backlash;
- current saturation.

For slide joints specifically, the important physical rule is unilateral cable
pulling:

- increasing contraction command should pull the rope and reduce segment
  spacing;
- decreasing contraction command should mainly reduce cable tension;
- return motion should be produced by the spring-steel assembly, not by an
  active pushing force from the motor.

If cable tension cannot be measured directly, record at least command, encoder
position, encoder velocity, motor current, and whether the rope is taut or
slack in the note field.

Template:

```text
record/v6/hardware/parameter_id/motor_step_response_template.csv
```

## 3. IMU Static Calibration

For each of the 7 segment IMUs:

1. Keep the segment still.
2. Record 5-10 seconds in known poses such as level, nose up, nose down, left
   side, right side.
3. Keep raw accelerometer and gyro units converted to SI.

Template:

```text
record/v6/hardware/parameter_id/imu_static_template.csv
```

This identifies:

- sensor axis direction relative to the body segment;
- accelerometer bias;
- gyro bias;
- noise level;
- gravity sign convention.

## 4. Contact Drag

For flat, sand, and slope-relevant surfaces:

1. Put one segment or representative contact shell on the surface.
2. Add known normal load.
3. Pull at a controlled speed.
4. Record steady pull force.
5. Repeat for several speeds and loads.

Template:

```text
record/v6/hardware/parameter_id/contact_drag_template.csv
```

This identifies the friction/contact parameters that matter for flat, sand, and
slope locomotion.

## 5. Mass And Geometry

Measure every assembled segment:

- mass;
- length/width/height;
- approximate center of mass;
- major add-ons such as batteries, PCBs, brackets, and shells.

Template:

```text
record/v6/hardware/parameter_id/mass_geometry_template.csv
```

## What To Send Back

Send the filled CSV files plus short notes:

- robot version and date;
- units used by sensors before conversion;
- load cell calibration method;
- encoder zeroing method;
- controller frequency;
- motor voltage;
- any mechanical stops or soft limits;
- photos or video of the bench setup.

For the spring-steel part, the most useful first file is:

```text
spring_steel_force_displacement_YYYYMMDD.csv
```

With that file, the simulation can be updated from a guessed stiffness to a
real measured equivalent stiffness.

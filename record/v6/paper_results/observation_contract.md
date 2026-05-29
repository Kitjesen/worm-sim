# Worm V6 Deployable Observation Contract

- ABI fingerprint: `27e31e546bb79ba12a70a286add79a8a72323f65ff3dbb58406e8d85c855c932`
- Observation dimension: `80`
- Actuated joints: `11`
- Segment IMUs: `7`
- Control rate: `50.0 Hz`
- Peristaltic actuation period: `1.000 s`
- Phase clock frequency: `1.000 Hz`

## Policy Input Groups

| Group | Range | Count | Hardware source |
| --- | --- | ---: | --- |
| command | [0, 3) | 3 | high-level command interface: cmd_vel_m_s, cmd_yaw_rad_s, and gait_blend |
| joint_pos | [3, 14) | 11 | 11 joint encoders: 6 slide positions and 5 yaw positions |
| joint_vel | [14, 25) | 11 | 11 joint encoder velocity estimates from the actuator controller |
| previous_action | [25, 36) | 11 | controller memory of the previous normalized policy action |
| segment_gravity | [36, 57) | 21 | one IMU per body segment: local gravity direction from attitude/accelerometer fusion |
| segment_gyro | [57, 78) | 21 | one IMU per body segment: local angular velocity |
| phase_clock | [78, 80) | 2 | controller clock, not a simulator state estimate |

## Forbidden Policy Inputs

- MuJoCo freejoint global position
- MuJoCo freejoint global orientation quaternion
- MuJoCo root/base linear velocity
- MuJoCo root/base angular velocity as a privileged state
- global pose from external tracking
- motion-capture state
- ground-truth terrain contact or slip labels
- reward-only forward/lateral/yaw velocity measurements

## Reward-Only Quantities

- forward speed
- yaw rate
- lateral drift
- energy/action cost
- termination and stability diagnostics

## Deployability Claim

The policy input is reconstructable from onboard joint encoders, distributed IMUs, command metadata, controller memory, and a local clock. Training rewards may use simulator truth, but the policy observation ABI may not.

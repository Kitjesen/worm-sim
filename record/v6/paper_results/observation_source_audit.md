# Worm V6 Observation Source Audit

Complete: `true`
Observation dimension: `80`
ABI fingerprint: `27e31e546bb79ba12a70a286add79a8a72323f65ff3dbb58406e8d85c855c932`

MuJoCo qpos/qvel/xmat/objectVelocity are acceptable here only as simulated encoder/IMU readouts matching the hardware ABI. Privileged root velocity/pose terms may appear in reward or termination code, but must not appear in WormEnvV6._get_obs().

## Policy Observation Groups

| Group | Range | Count | Sim source | Real source |
| --- | --- | ---: | --- | --- |
| command | [0, 3) | 3 | self._cmd_vel; self._cmd_yaw; self._gait_blend | controller command velocity; controller command yaw rate; controller-selected gait_blend |
| joint_pos | [3, 14) | 11 | self.data.qpos[self._act_qpos_idx[i]] | 11 joint encoder positions |
| joint_vel | [14, 25) | 11 | self.data.qvel[self._act_qvel_idx[i]] | 11 joint encoder velocity estimates |
| previous_action | [25, 36) | 11 | self._last_action | controller memory of previous normalized action |
| segment_gravity | [36, 57) | 21 | self.data.xmat[body_id] projected gravity for each IMU body | one IMU per segment: gravity direction from attitude fusion |
| segment_gyro | [57, 78) | 21 | mujoco.mj_objectVelocity(..., body_id, local_vel, 1) | one IMU per segment: local angular velocity |
| phase_clock | [78, 80) | 2 | self._step_count * CTRL_DT | controller clock |

## Forbidden Policy Observation Hits

- none

## Reward-Only Privileged Hits

- `forward_speed`: `1`
- `lateral_speed`: `1`
- `yaw_rate`: `1`
- `termination_height`: `2`
- `termination_nan_qpos`: `1`

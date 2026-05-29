# Worm V6 Observation Source Audit

Complete: `true`
Observation dimension: `80`
ABI fingerprint: `83bf1ec48b816810335fabc08800e2e11f551c7ce72e96a94431e6afa1840d59`

MuJoCo qpos/qvel/xmat/objectVelocity are acceptable here only as simulated encoder/IMU readouts matching the hardware ABI. Privileged root velocity/pose terms may appear in reward or termination code, but must not appear in WormEnvV6._get_obs().

## Policy Observation Groups

| Group | Range | Count | Sim source | Real source |
| --- | --- | ---: | --- | --- |
| command | [0, 3) | 3 | self._cmd_vx; self._cmd_vy; self._cmd_yaw | controller body-frame vx command; controller body-frame vy command; controller command yaw rate |
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

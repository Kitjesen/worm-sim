"""Measure V6 steering from joint targets and real MuJoCo state, without root drive."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from worm_v6 import build_xml, setup_terrain

ROOT = Path(__file__).resolve().parents[2]
GAIT = ROOT / 'runs/cmaes_flat_serpentine/best_gait.json'


def yaw_targets(params, t, count, bias, startup_s=1.0):
    phase = (2 * np.pi * (params[4] + params[13] * params[1]) * t
             + 2 * np.pi * params[5] * np.arange(count) / count)
    ramp = .5 * (1 - math.cos(math.pi * min(1., t / startup_s)))
    return ramp * np.clip(params[3] * np.sin(phase) + bias, -1.57, 1.57)


def measure(model, data, head, tail, collision_ids, segment_ids):
    # mj_step leaves some derived arrays at the previous position: sync first.
    mujoco.mj_forward(model, data)
    masses = model.body_mass[1:]
    com = np.sum(data.xipos[1:] * masses[:, None], axis=0) / np.sum(masses)
    forward = data.xpos[tail, :2] - data.xpos[head, :2]
    heading = math.atan2(forward[1], forward[0])
    root_R = data.xmat[head].reshape(3, 3)
    root_yaw = math.atan2(-root_R[1, 0], -root_R[0, 0])
    distances = np.linalg.norm(data.geom_xpos[collision_ids, :2] - com[:2], axis=1)
    radius = float(np.max(distances + model.geom_rbound[collision_ids]))
    up = data.xmat[segment_ids].reshape(-1, 3, 3)[:, 2, 2]
    tilt = float(np.rad2deg(np.arccos(np.clip(up, -1., 1.))).max())
    return com, heading, root_yaw, radius, tilt


def calibrate(duration=20., biases=(-.1, 0., .1), settle_s=1.):
    params = np.asarray(json.loads(GAIT.read_text(encoding='utf-8'))['best_params'])
    urdf = ROOT / 'meshes/longworm2/longworm2.SLDASM.urdf'
    xml = build_xml(str(ROOT / 'meshes'), str(urdf), terrain='flat')
    model = mujoco.MjModel.from_xml_string(xml)
    setup_terrain(model, 'flat')
    data = mujoco.MjData(model)
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    tail = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'back6_Link')
    segments = [head] + [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         f'back{i}_Link') for i in range(1, 7)]
    yaw_ids = [i for i in range(model.nu) if
               mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith('act_front')]
    collision_ids = np.flatnonzero((model.geom_bodyid != 0) &
                                  ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    assert len(yaw_ids) == 5 and model.nu == 11
    dt = float(model.opt.timestep)
    sample_stride = max(1, round(.02 / dt))
    cases = []
    for bias in biases:
        started = time.perf_counter()
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0.
        for _ in range(round(settle_s / dt)):
            mujoco.mj_step(model, data)
        samples = []
        for step in range(round(duration / dt) + 1):
            t = step * dt
            if step:
                data.ctrl[yaw_ids] = yaw_targets(params, t, len(yaw_ids), bias)
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f'Nonfinite physics at bias={bias}, t={t}')
            if step % sample_stride == 0:
                com, heading, root_yaw, radius, tilt = measure(
                    model, data, head, tail, collision_ids, segments)
                samples.append([t, *com, heading, root_yaw, radius, tilt,
                                float(data.xpos[head, 2])])
        trace = np.asarray(samples)
        headings = np.unwrap(trace[:, 4])
        root_yaws = np.unwrap(trace[:, 5])
        steady = trace[:, 0] >= min(5., duration / 3)
        fit = np.polyfit(trace[steady, 0], headings[steady], 1)
        root_fit = np.polyfit(trace[steady, 0], root_yaws[steady], 1)
        velocities = np.diff(trace[:, 1:3], axis=0) / np.diff(trace[:, 0])[:, None]
        axes = np.column_stack((np.cos(headings[1:]), np.sin(headings[1:])))
        speed = float(np.mean(np.sum(velocities[steady[1:]] * axes[steady[1:]], axis=1)))
        case = dict(bias_rad=float(bias), duration_s=duration,
                    wall_time_s=time.perf_counter()-started,
                    net_com_displacement_world_m=(trace[-1, 1:4]-trace[0, 1:4]).tolist(),
                    heading_rate_fit_rad_s=float(fit[0]), root_yaw_rate_fit_rad_s=float(root_fit[0]),
                    heading_change_rad=float(headings[-1]-headings[0]),
                    steady_mean_forward_speed_m_s=speed,
                    steady_mean_velocity_world_m_s=np.mean(velocities[steady[1:]], axis=0).tolist(),
                    turning_radius_estimate_m=abs(speed/fit[0]) if abs(fit[0]) > 1e-6 else None,
                    collision_envelope_radius_m=float(trace[:, 6].max()),
                    max_segment_tilt_deg=float(trace[:, 7].max()),
                    root_height_min_m=float(trace[:, 8].min()), root_height_max_m=float(trace[:, 8].max()),
                    stable=bool(trace[:, 8].min() >= .01 and trace[:, 8].max() <= .28 and trace[:, 7].max() < 90.),
                    columns=['time_s', 'com_x_m', 'com_y_m', 'com_z_m', 'heading_rad',
                             'forward_root_yaw_rad', 'collision_envelope_radius_m', 'max_segment_tilt_deg', 'root_z_m'],
                    samples=trace.tolist())
        cases.append(case)
        print(json.dumps({k: v for k, v in case.items() if k not in ('samples', 'columns')}), flush=True)
    zero = next((c for c in cases if abs(c['bias_rad']) < 1e-12), None)
    for case in cases:
        case['yaw_gain_vs_zero_rad_s_per_rad'] = (
            (case['heading_rate_fit_rad_s']-zero['heading_rate_fit_rad_s'])/case['bias_rad']
            if zero is not None and abs(case['bias_rad']) > 1e-12 else None)
    return dict(status='completed', model='V6 MuJoCo rigid bodies, passive rolling wheels, slide spring; steel visuals only',
                terrain='flat', physics_dt_s=dt, sample_dt_s=sample_stride*dt, settle_s=settle_s,
                gait_source=str(GAIT.relative_to(ROOT)), gait_sha256=hashlib.sha256(GAIT.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                slide_targets_m=0., continuous_time=True, yaw_wave='2*pi*(yaw_freq+coupling*slide_freq)*t + 2*pi*yaw_wave_n*j/5',
                yaw_amplitude_rad=float(params[3]), effective_yaw_frequency_hz=float(params[4]+params[13]*params[1]),
                startup_s=1., heading_definition='atan2(tail_y-head_y, tail_x-head_x); initial -world X',
                com_definition='all MuJoCo body masses at xipos after mj_forward',
                envelope_definition='max distance(COM_xy, collision geom xy)+geom_rbound; conservative whole-robot disk',
                root_pose_drive=False, cases=cases)


def self_check():
    params = np.arange(14, dtype=float) / 10
    assert np.array_equal(yaw_targets(params, 0., 5, .1), np.zeros(5))
    phase = 2*np.pi*(params[4]+params[13]*params[1])*2 + 2*np.pi*params[5]*np.arange(5)/5
    np.testing.assert_allclose(yaw_targets(params, 2., 5, .1), params[3]*np.sin(phase)+.1)
    assert np.max(abs(yaw_targets(params, 2., 5, 5.))) <= 1.57
    print('yaw waveform/startup/clipping self-check passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time', type=float, default=20.)
    parser.add_argument('--biases', type=float, nargs='+', default=[-.1, 0., .1])
    parser.add_argument('--settle', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=ROOT/'record/v6/astar_tracking_20261002/steering_calibration.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if (not np.isfinite([args.time, args.settle, *args.biases]).all() or args.time < 6
            or args.settle < .5 or max(abs(v) for v in args.biases) > .5):
        parser.error('Need finite duration >= 6s, settle >= .5s, biases within +/-.5 rad')
    payload = calibrate(args.time, args.biases, args.settle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()

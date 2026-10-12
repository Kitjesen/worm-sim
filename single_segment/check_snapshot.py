"""Check imported provenance, saved topology and the complete 1x replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
RUN = HERE / 'sofa_dr_runs/retrograde_wave_20261002T050058Z'


def main():
    manifest = json.loads((HERE / 'upstream_snapshot.json').read_text())
    for name, item in manifest['files'].items():
        content = (HERE / name).read_bytes()
        assert len(content) == item['bytes'], name
        assert hashlib.sha256(content).hexdigest() == item['sha256'], name
        if name.endswith('.py'):
            compile(content, name, 'exec')
    with np.load(RUN / 'whole_rollout.npz') as data:
        poses = data['poses']
        assert poses.shape == (800, 290, 7)
        for name in data.files:
            if np.issubdtype(data[name].dtype, np.number):
                assert np.isfinite(data[name]).all(), name
        assert np.allclose(np.linalg.norm(poses[:, :, 3:], axis=-1), 1., atol=1e-6)
        ids = data['strip_nodes']
        assert ids.shape == (40, 9)
        assert np.array_equal(ids[:, 0], np.repeat(np.arange(0, 10, 2), 8))
        assert np.array_equal(ids[:, -1], np.repeat(np.arange(1, 10, 2), 8))
        assert np.array_equal(np.sort(ids[:, 1:-1].ravel()), np.arange(10, 290))
        assert data['wheel_angle'].shape == (800, 20)
        assert data['anchors'].shape == data['guides'].shape == (4, 3)
        assert data['actions'].shape == (800, 14)
        assert data['joint_angle_rad'].shape == (800, 4)
        gap = np.linalg.norm(poses[:, 1:10:2, :3]-poses[:, :10:2, :3], axis=-1)
        assert np.allclose(gap, data['segment_gap_m'])
        assert np.allclose(data['head_xy'], poses[:, 9, :2])
        report = json.loads((RUN / 'path_report.json').read_text())
        assert report['failed'] is False
        forward = float(data['head_xy'][0, 0]-data['head_xy'][-1, 0])
    frames = imageio_ffmpeg.read_frames(str(RUN / 'whole_sofa_follow.mp4'))
    meta = next(frames)
    count = sum(1 for _ in frames)
    assert count == 400 and meta['fps'] == 25
    assert tuple(meta['size']) == (1600, 900)
    assert count/meta['fps'] == report['physical_s'] == 16.
    result = dict(passed=True, source_files_verified=len(manifest['files']),
                  saved_states=800, modules=5, partitions=10, strips=40,
                  nodes_per_strip=9, elements_per_strip=8, cable_spans=20,
                  wheels=20, intermodule_joints=4, frames=count, fps=meta['fps'],
                  playback_s=count/meta['fps'], physical_s=report['physical_s'],
                  sampled_head_forward_m=forward,
                  min_gap_mm=float(gap.min()*1000), max_gap_mm=float(gap.max()*1000),
                  scope='Provenance/topology/replay audit; not new dynamics or physical calibration.')
    (RUN / 'snapshot_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

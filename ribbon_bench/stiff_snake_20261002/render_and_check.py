"""Replay the completed t020 run with the existing renderer and publication QA.

Run from ribbon_bench with its usual vendor PYTHONPATH. No physics is re-solved.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CASE = Path(__file__).resolve().parent
BASE = CASE.parent
RUN = CASE/'t020_n9_cuda'
PARAMETERS = CASE/'parameters_t020mm.json'


def main():
    summary = json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed', 'Wait for the completed physics run'
    assert hashlib.sha256(PARAMETERS.read_bytes()).hexdigest() == summary['metadata']['parameters_sha256']
    assert len(summary['frames']) == 201 and summary['frames'][85]['time_s'] == 1.7
    source = BASE/'render_full_robot.py'
    snapshot = RUN/'render_full_robot.snapshot.py'
    snapshot.write_bytes(source.read_bytes())
    source_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    result = subprocess.run([sys.executable, str(BASE/'render_full_robot.py'),
        '--input', str(RUN), '--output', str(RUN/'robot_motion.png'),
        '--parameters', str(PARAMETERS), '--cad-meshes', '--mesh-lod-m', '.001',
        '--frame-stride', '4', '--frame-index', '85'],
        check=True, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    (RUN/'render.log').write_text(result.stdout, encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, 'Renderer changed during replay'
    paths = [snapshot, RUN/'summary.json', RUN/'trajectory.npz', PARAMETERS]
    record = dict(renderer_original_path=str(source),
        files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        role='Original CAD front/back STL visual replay; no added STL collision or wheel visuals. '
             'GIF frame stride changes display only; all mechanical trajectory data remain unchanged.')
    (RUN/'render_provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    for item in record['files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    render = json.loads(result.stdout)
    indices = render['animation_source_indices']
    assert indices[0] == 0 and indices[-1] == render['frames']-1
    assert all(a < b for a, b in zip(indices, indices[1:]))
    for item in render['cad_mesh_sources']:
        assert hashlib.sha256(Path(item['source_path']).read_bytes()).hexdigest() == item['sha256']
    results = {}
    for name in ('robot_motion.gif', 'robot_motion_top.gif'):
        margins, durations = [], []
        with Image.open(RUN/name) as gif:
            size = gif.size
            for i in range(gif.n_frames):
                gif.seek(i)
                rgb = np.asarray(gif.convert('RGB'))
                y, x = np.nonzero(rgb.min(axis=2) < 200)
                margin = [int(x.min()), int(y.min()), int(size[0]-1-x.max()), int(size[1]-1-y.max())]
                assert min(margin) >= 12, (name, i, margin)
                margins.append(margin)
                durations.append(gif.info['duration'])
            assert gif.n_frames == render['animation_source_frames']
            assert sum(durations) == round(1000*render['playback_duration_s'])
            results[name] = dict(size_px=size, encoded_frames=gif.n_frames,
                playback_duration_ms=sum(durations), minimum_subject_margins_px=np.min(margins, axis=0).tolist())
    results.update(status='passed', physical_frames=render['frames'], static_frame=render['static_frame'],
        static_time_s=render['static_time_s'], simulated_duration_s=render['simulated_duration_s'],
        frame_stride=render['frame_stride'], animation_source_indices=indices, source_sha256_verified=True,
        view_cropping_check='Each GIF frame: object pixels RGB minimum < 200, excluding the pale floor',
        original_triangles=sum(s['original_triangles'] for s in render['cad_mesh_sources']),
        rendered_triangles=sum(s['rendered_triangles'] for s in render['cad_mesh_sources']),
        maximum_lod_vertex_displacement_mm=1000*max(s['max_vertex_displacement_m'] for s in render['cad_mesh_sources']))
    (RUN/'visual_validation.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

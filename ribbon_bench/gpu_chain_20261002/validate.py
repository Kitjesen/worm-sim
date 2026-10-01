"""Audit the saved five-segment N33 trajectory and its rigid attachments."""
import hashlib
import json
from pathlib import Path
import numpy as np

out = Path(__file__).resolve().parent/'chain_n33_cuda_motion'
summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
data = np.load(out/'trajectory.npz')
rows, metadata = summary['frames'], summary['metadata']
q, c, R = (data[key] for key in ('q', 'body_com', 'body_R'))
assert q.shape == (41, 40, 131) and c.shape == (41, 6, 3)
assert all(np.isfinite(data[key]).all() for key in data.files)
assert metadata['source_sha256'] == hashlib.sha256((out/'full_robot.snapshot.py').read_bytes()).hexdigest()
assert np.allclose(np.diff([row['time_s'] for row in rows]), .005, atol=1e-15)
rotation_error = float(np.max(np.abs(R@R.swapaxes(-1, -2)-np.eye(3))))
assert rotation_error < 1e-12
nodes = q[..., :99].reshape(41, 40, 33, 3)
attachment_error = 0.
for strip, bodies in enumerate(data['strip_bodies']):
    for body, indices in zip(bodies, ([0, 1], [31, 32])):
        local = (nodes[0, strip, indices]-c[0, body])@R[0, body]
        expected = c[:, body, None]+np.einsum('tij,nj->tni', R[:, body], local)
        attachment_error = max(attachment_error, float(np.max(abs(expected-nodes[:, strip][:, indices]))))
assert attachment_error < 1e-12
pb = data['plate_body_indices']
plate_centers = c[:, pb]+np.einsum('tpij,pj->tpi', R[:, pb], data['plate_centers_local_m']-data['body_com_local_m'][pb])
lengths = np.linalg.norm(plate_centers[:, 1::2]-plate_centers[:, ::2], axis=-1)
np.testing.assert_allclose(lengths, [row['segment_lengths_m'] for row in rows], atol=1e-12)
max_residual = max(row['residual_max'] for row in rows)
assert max_residual <= 1e-5
result = dict(status='passed', frames=41, segments=5, ribbons=40, nodes_per_ribbon=33,
              rigid_attachment_max_error_m=attachment_error, rotation_orthogonality_max_error=rotation_error,
              saved_frame_max_residual=max_residual, simulated_time_s=rows[-1]['time_s'],
              max_segment_contraction_mm=np.max((lengths[0]-lengths)*1000, axis=0).tolist(),
              peak_tension_all_accepted_substeps_n=max(row.get('substep_peak_tension_n', 0.) for row in rows),
              max_penetration_all_accepted_substeps_um=1e6*max(row.get('substep_max_penetration_m', 0.) for row in rows),
              accepted_substeps=sum(row.get('command_substeps', 0) for row in rows),
              recovery_substeps=sum(row.get('recovery_substeps', 0) for row in rows), wall_seconds=summary['wall_seconds'])
(out/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

"""Archive the verified paper figures and their existing numerical sources.

This does not rerun physics. Re-render with the commands in REPORT.zh-CN.md first.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parent
FIGURES = [
    ('fig01_baseline17', 'output', 'results.json', 'simulation.gif'),
    ('fig02_mesh33', 'mesh_demo/n33', 'results.json', 'simulation.gif'),
    ('fig03_mesh65', 'mesh_demo/n65', 'results.json', 'simulation.gif'),
    ('fig04_mesh129', 'mesh_demo/n129', 'results.json', 'simulation.gif'),
    ('fig05_mesh_comparison', 'mesh_demo', None, 'comparison.gif'),
    ('fig06_large_motion15', 'large_motion_demo', 'results.json', 'simulation.gif'),
    ('fig07_prescribed30_loading', 'prescribed_30_refined', 'loading_results.json', 'simulation.gif'),
    ('fig08_cable_driven', 'actuated_demo', 'results.json', 'simulation.gif'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_markdown(source, target, prefix):
    """Keep local links valid when a root document is copied into the archive."""
    text = source.read_text(encoding='utf-8')
    text = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m.group(0) if
                  re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:|^#', m[2]) else
                  m[1] + prefix + m[2] + m[3], text)
    target.write_text(text, encoding='utf-8')


def main():
    destination = ROOT/'publication'
    expected = json.loads((destination/'results_summary.json').read_text(encoding='utf-8'))
    # Numeric results must remain byte-identical throughout the figure restyling.
    for row in expected:
        assert digest(ROOT/row['source']) == row['sha256'], f'Numerical result changed: {row["source"]}'
        metadata = json.loads((ROOT/row['source']).read_text(encoding='utf-8'))['metadata']
        assert digest(destination/'data/cad_reference.urdf') == metadata['urdf_sha256']
    entries = []
    for stem, folder, result_name, animation_name in FIGURES:
        directory = ROOT/folder
        entry = dict(figure=stem, source_directory=folder, files=[])
        expected_frames = 13 if result_name == 'loading_results.json' else 25
        with Image.open(directory/animation_name) as movie:
            assert movie.n_frames == expected_frames, stem
            duration = 0
            for frame in range(movie.n_frames):
                movie.seek(frame)
                movie.load()
                duration += movie.info.get('duration', 0)
            entry.update(gif_frames=movie.n_frames, gif_duration_ms=duration, gif_size_px=list(movie.size))
        assert duration == expected_frames*200, (stem, duration)
        with Image.open(directory/'preview.png') as preview:
            preview.load()
            assert min(preview.info.get('dpi', (0, 0))) >= 299, stem
            entry['png_size_px'] = list(preview.size)
        svg = ET.parse(directory/'preview.svg')
        assert svg.getroot().tag.endswith('svg')
        assert svg.findall('.//{http://www.w3.org/2000/svg}text'), 'Keep vector figure labels editable'
        for source_name, suffix in [('preview.png', '.png'), ('preview.svg', '.svg'), (animation_name, '.gif')]:
            target = destination/'figures'/(stem+suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory/source_name, target)
            entry['files'].append(dict(path=target.relative_to(destination).as_posix(), sha256=digest(target)))
        source_names = [result_name, 'parameters.snapshot.json', 'validation.json', 'wrench_feasibility.json'] if result_name else ['timings.json']
        entry['data'] = []
        for source_name in source_names:
            source = directory/source_name
            if not source.exists():
                continue
            target = destination/'data'/folder/source_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entry['data'].append(dict(path=target.relative_to(destination).as_posix(), sha256=digest(target)))
        if not result_name:
            entry['source_figures'] = ['fig02_mesh33', 'fig03_mesh65', 'fig04_mesh129']
        entries.append(entry)
    code = ['run.py', 'actuate.py', 'cable_loads.py', 'render.py', 'render_actuated.py',
            'compare_mesh.py', 'check.py', 'check_actuated.py', 'check_target_wrench.py',
            'predictor_probe.py', 'scaling_probe.py', 'publish_results.py', 'README.md']
    code_entries = []
    for name in code:
        target = destination/'code'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.md'):
            archive_markdown(ROOT/name, target, '../../')
        else:
            shutil.copy2(ROOT/name, target)
        code_entries.append(dict(path=target.relative_to(destination).as_posix(), sha256=digest(target)))
    for name in ('check.json', 'check_actuated.json'):
        if (ROOT/name).exists():
            target = destination/'data'/name
            shutil.copy2(ROOT/name, target)
    references = []
    for source, name in [(destination/'data/cad_reference.urdf', 'cad_reference.urdf'),
                         (ROOT/'vendor/discrete-elastic-ribbon/LICENSE', 'upstream_LICENSE')]:
        target = destination/'data'/name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        references.append(dict(path=target.relative_to(destination).as_posix(), sha256=digest(target)))
    report = ROOT/'REPORT.zh-CN.md'
    assert report.exists(), 'Write the report before finalizing the archive'
    archive_markdown(report, destination/report.name, '../')
    manifest = dict(status='verified figure archive; numerical sources unchanged',
        report=dict(path=report.name, sha256=digest(destination/report.name)),
        environment='environment.json', numerical_summary='results_summary.json',
        upstream_commit='c9d341164e2927fc24b2c43dff97fcfb492cf700',
        code_snapshot_note='Source snapshot for traceability; execute the matching scripts in the ribbon_bench root after installing dependencies and initializing the pinned vendor submodule. See README.md.',
        limitations='All sequences are quasi-static. Figure 7 is loading only; unloading failed. Mesh comparison is not a convergence proof. See report captions.',
        figures=entries, code=code_entries, references=references)
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=len(entries), files=sum(len(e['files']) for e in entries),
                         numerical_sources_unchanged=len(expected), manifest=str(destination/'manifest.json')), indent=2))


if __name__ == '__main__':
    main()

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh

SOURCE = Path(os.environ.get('PSHUMAN_TEST_MESH_UTIL', Path(__file__).resolve().parents[1] / 'lib/dataset/mesh_util.py'))


def setup_namespace(monkeypatch, capture_dir=None):
    tree = ast.parse(SOURCE.read_text())
    chosen = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in {'_PoissonStageCapture', 'poisson'}]
    for x in chosen:
        x.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[]))
    events = []
    original = trimesh.creation.box()
    original.vertices = original.vertices.copy()
    def make_raw(points, normals, depth):
        events.append('raw')
        return original.copy()
    def attract(mesh, depth):
        events.append('attract')
        mesh.vertices = np.asarray(mesh.vertices) + [0.25, 0., 0.]
        return mesh
    def largest(mesh, min_fraction):
        events.append('filter')
        return mesh
    ns = {'np': np, 'os': os, 'json': json, '__file__': str(SOURCE), 'POISSON_KEEP_COMPONENT_MIN_FRACTION': .01,
          '_poisson_points': make_raw, 'attract_parts_to_body_plane': attract, 'keep_largest': largest,
          'rebuild_arms_from_front_mask': lambda mesh, front, scale: (np.zeros((1,3)), np.zeros((1,3)), {'_join_points': None})}
    exec(compile(module, str(SOURCE), 'exec'), ns)
    monkeypatch.setenv('PSHUMAN_POISSON_FOLD_NORMALS', '0')
    monkeypatch.setenv('PSHUMAN_ARM_FRONTAL_REBUILD', '1')
    monkeypatch.setenv('PSHUMAN_POISSON_ATTRACT_PARTS', '1')
    if capture_dir is None:
        monkeypatch.delenv('PSHUMAN_POISSON_CAPTURE_DIR', raising=False)
    else:
        monkeypatch.setenv('PSHUMAN_POISSON_CAPTURE_DIR', str(capture_dir))
    return ns, events, original


def test_disabled_does_not_create_capture(monkeypatch):
    ns, events, original = setup_namespace(monkeypatch)
    ns['_PoissonStageCapture'] = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('capture ran'))
    result = ns['poisson'](original.copy(), 'unused.obj', decimation=False, front_mask=True, ortho_scale=1)
    assert events == ['raw', 'attract', 'filter']
    assert np.allclose(result.vertices, original.vertices + [0.25, 0, 0])


def test_stages_capture_same_raw_before_mutation_and_unique_runs(monkeypatch, tmp_path):
    ns, events, original = setup_namespace(monkeypatch, tmp_path)
    for _ in range(2):
        result = ns['poisson'](original.copy(), 'unused.obj', decimation=False, front_mask=True, ortho_scale=1)
        assert np.allclose(result.vertices, original.vertices + [0.25, 0, 0])
    assert events == ['raw', 'attract', 'filter'] * 2
    dirs = sorted(tmp_path.iterdir())
    assert len(dirs) == 2 and dirs[0] != dirs[1]
    for directory in dirs:
        report = json.loads((directory / 'manifest.json').read_text())
        assert report['status'] == 'complete'
        assert report['front_rebuild_used'] is True
        assert report['arm_join_points_present'] is False
        packet = directory / report['poisson_input']['file']
        assert hashlib.sha256(packet.read_bytes()).hexdigest() == report['poisson_input']['sha256']
        assert np.array_equal(np.load(packet)['points'], np.zeros((1, 3)))
        assert [s['stage'] for s in report['stages']] == ['raw-before-attract', 'post-attract-before-filter', 'selected-output']
        for s in report['stages']:
            assert hashlib.sha256((directory / s['file']).read_bytes()).hexdigest() == s['sha256']
            assert (s['vertices'], s['faces'], s['components']) == (8, 12, 1)
        before = trimesh.load(directory / 'raw-before-attract.ply', process=False)
        after = trimesh.load(directory / 'post-attract-before-filter.ply', process=False)
        selected = trimesh.load(directory / 'selected-output.ply', process=False)
        assert np.allclose(before.vertices, original.vertices)
        assert np.allclose(after.vertices, original.vertices + [0.25, 0, 0])
        assert np.allclose(selected.vertices, after.vertices)


def test_no_attraction_snapshots_same_raw(monkeypatch, tmp_path):
    ns, events, original = setup_namespace(monkeypatch, tmp_path)
    monkeypatch.setenv('PSHUMAN_POISSON_ATTRACT_PARTS', '0')
    result = ns['poisson'](original.copy(), 'unused.obj', decimation=False, front_mask=True, ortho_scale=1)
    assert events == ['raw', 'filter']
    report = json.loads((next(tmp_path.iterdir()) / 'manifest.json').read_text())
    assert report['attract'] == '0'
    assert report['stages'][0]['sha256'] == report['stages'][1]['sha256']
    assert np.array_equal(result.vertices, original.vertices)


def test_no_front_records_exact_point_cloud_passed_to_poisson(monkeypatch, tmp_path):
    from types import SimpleNamespace
    ns, events, original = setup_namespace(monkeypatch, tmp_path / 'captures')
    monkeypatch.setenv('PSHUMAN_POISSON_ATTRACT_PARTS', '0')
    points = np.asarray(original.vertices).copy()
    normals = np.asarray(original.vertex_normals).copy()
    pcl = SimpleNamespace(points=points, normals=normals)
    class Context:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    observed = []
    def solve(passed, depth, n_threads):
        observed.append(passed)
        return SimpleNamespace(vertices=original.vertices, triangles=original.faces), None
    ns['o3d'] = SimpleNamespace(
        io=SimpleNamespace(read_point_cloud=lambda path: pcl),
        utility=SimpleNamespace(VerbosityContextManager=lambda level: Context(), VerbosityLevel=SimpleNamespace(Error=0)),
        geometry=SimpleNamespace(TriangleMesh=SimpleNamespace(create_from_point_cloud_poisson=solve)))
    ns['trimesh'] = trimesh
    ns['poisson'](original.copy(), str(tmp_path / 'input.obj'), decimation=False)
    assert observed == [pcl]
    directory = next((tmp_path / 'captures').iterdir())
    report = json.loads((directory / 'manifest.json').read_text())
    assert report['front_rebuild_used'] is False
    packet = np.load(directory / report['poisson_input']['file'])
    assert np.array_equal(packet['points'], observed[0].points)
    assert np.array_equal(packet['normals'], observed[0].normals)

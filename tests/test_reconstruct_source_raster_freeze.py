import ast
from pathlib import Path
import os
from types import SimpleNamespace

import pytest
import torch


def _projection_method():
    path = Path(__file__).parents[1] / 'reconstruct.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ReMesh')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'proj_texture')
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return module


class Mesh:
    def to(self, device):
        return self


def test_both_source_passes_use_frozen_policy_despite_mutation(monkeypatch):
    from utils import texture_raster_budget
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE', 'all_faces')
    monkeypatch.setenv('PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES', '123456')
    seen = []
    def projection(mesh, images, **kwargs):
        env = kwargs['source_raster_env']
        seen.append(dict(env))
        with pytest.raises(TypeError):
            env['PSHUMAN_TEXTURE_RASTER_MODE'] = 'default'
        monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE', 'default')
        monkeypatch.setenv('PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES', '1')
        return mesh
    namespace = {'os':os, 'torch':torch, 'to_py3d_mesh':lambda *a:Mesh(),
                 'get_cameras_list':lambda *a,**k: [None]*6,
                 '_parse_view_indices':lambda _v,default:default,
                 'multiview_color_projection':projection}
    exec(compile(_projection_method(), 'frozen-projection', 'exec'), namespace)
    obj=SimpleNamespace(device='cpu', resolution=9, weights=torch.ones(6))
    vertices=torch.zeros((3,3)); faces=torch.tensor([[0,1,2]])
    namespace['proj_texture'](obj, [None]*6, vertices, faces, fallback_images=[None]*6,
                              source_only=True, support_silhouettes=[None]*6)
    assert seen == [{'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces',
                     'PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':'123456'}] * 2


def test_non_source_default_does_not_consult_raster_policy(monkeypatch):
    from utils import texture_raster_budget
    monkeypatch.setattr(texture_raster_budget, 'source_raster_policy',
                        lambda env: pytest.fail('non-source policy must remain unchanged'))
    seen=[]
    def projection(mesh, images, **kwargs):
        seen.append(kwargs['source_raster_env']); return mesh
    namespace={'os':os,'torch':torch,'to_py3d_mesh':lambda *a:Mesh(),
               'get_cameras_list':lambda *a,**k:[None]*6,
               '_parse_view_indices':lambda _v,default:default,
               'multiview_color_projection':projection}
    exec(compile(_projection_method(), 'non-source-projection', 'exec'), namespace)
    obj=SimpleNamespace(device='cpu',resolution=9,weights=torch.ones(6))
    namespace['proj_texture'](obj,[None]*6,torch.zeros((3,3)),torch.tensor([[0,1,2]]))
    assert seen == [None]

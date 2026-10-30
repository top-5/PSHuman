from types import SimpleNamespace
import pytest
import torch
from utils import project_mesh


def test_visibility_selects_unbinned_only_when_opted_in(monkeypatch):
    calls=[]
    def raster(meshes,cameras,**kwargs):
        calls.append(kwargs)
        return {'pix_to_face':'faces','zbuf':'depth'}
    monkeypatch.setattr(project_mesh,'render_pix2faces_py3d',raster)
    monkeypatch.delenv('PSHUMAN_TEXTURE_RASTER_MODE',raising=False)
    assert project_mesh.get_visibility_buffers(None,None,resolution=64)=={'pix_to_face':'faces','zbuf':'depth'}
    assert calls[-1]['bin_size'] is None
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','naive')
    project_mesh.get_visibility_buffers(None,None,resolution=64)
    assert calls[-1]['bin_size']==0 and calls[-1]['H']==64 and calls[-1]['W']==64
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','bogus')
    with pytest.raises(ValueError,match='default or naive'):
        project_mesh.get_visibility_buffers(None,None)


def test_naive_preserves_nearest_face_and_depth_output(monkeypatch):
    seen={}
    class Rasterizer:
        def __init__(self,cameras,raster_settings): seen['settings']=raster_settings
        def __call__(self,meshes,cameras):
            return SimpleNamespace(pix_to_face=torch.tensor([[[[7]]]]),zbuf=torch.tensor([[[[2.5]]]]))
    monkeypatch.setattr(project_mesh,'MeshRasterizer',Rasterizer)
    result=project_mesh.render_pix2faces_py3d(None,None,H=1,W=1,bin_size=0)
    assert seen['settings'].bin_size==0 and seen['settings'].faces_per_pixel==1 and seen['settings'].blur_radius==0
    assert result['pix_to_face'].item()==7 and result['zbuf'].item()==2.5

import json
import numpy as np
import pytest
from PIL import Image
import trimesh
import texture_only_source_color as driver


def fixture(tmp_path):
    def item(path): return {'path':str(path),'sha256':driver.digest(path)}
    mesh=trimesh.Trimesh(vertices=np.array([[0,0,1],[1,0,1],[0,1,1]],np.float32),faces=[[0,1,2]],vertex_colors=[[10,20,30,255],[40,50,60,255],[70,80,90,255]],process=False)
    source=tmp_path/'mesh.ply';source.write_bytes(mesh.export(file_type='ply'))
    camera=tmp_path/'camera.json';camera.write_text(json.dumps({'schema':'pshuman.canonical-camera/1','azimuth_degrees':[0,45,90,180,270,315],'focal':.5,'distance':1.1}))
    views=[]
    for name in driver.VIEW_NAMES:
        color=tmp_path/f'{name}-color.png';normal=tmp_path/f'{name}-normal.png'
        Image.fromarray(np.full((9,9,4),255,np.uint8)).save(color)
        Image.fromarray(np.full((9,9,4),255,np.uint8)).save(normal)
        views.append({'name':name,'color':item(color),'normal':item(normal)})
    support=tmp_path/'support.npz';obs=np.zeros((6,9,9),bool);obs[:,4,4]=1
    np.savez(support,color_observed=obs,parent_color_inferred=np.zeros_like(obs),baseline_foreground=np.ones_like(obs))
    manifest={'schema':'pshuman.texture-only-source/1','coordinate_frame':'PSHuman-shared','mesh':item(source),'camera':item(camera),'support':item(support),'views':views}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));return path,manifest


def test_manifest_pins_every_view_and_camera(tmp_path):
    p,m=fixture(tmp_path);inputs,camera=driver.validate_manifest(m)
    assert len(inputs['colors'])==6 and camera['focal']==.5
    (tmp_path/'front_face-color.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'): driver.validate_manifest(m)


@pytest.mark.parametrize('mode,budget,expected', [('default','ignored',None), ('all_faces','123456',123456)])
def test_texture_only_output_retains_baseline_and_rejects_reuse(tmp_path,monkeypatch,mode,budget,expected):
    import torch
    from utils import project_mesh
    p,m=fixture(tmp_path)
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE',mode)
    monkeypatch.setenv('PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES',budget)
    monkeypatch.setattr(project_mesh,'get_cameras_list',lambda *a,**kw: [None]*6)
    def project(mesh,*a,support_counts_out,**kw):
        assert kw['min_source_support']==.95
        support_counts_out['measured_mask']=torch.tensor([False,False,False])
        return mesh
    monkeypatch.setattr(project_mesh,'multiview_color_projection',project)
    output=tmp_path/'out'
    report=driver.run(p,driver.digest(p),output,device='cpu',resolution=9)
    assert report['measured_vertices']==0 and report['baseline_vertices']==3
    assert report['source_raster']=={'mode':mode,'bin_buffer_limit_bytes':expected}
    assert np.array_equal(driver.load_mesh(output/'texture-only.ply').visual.vertex_colors[:,:3],np.array([[10,20,30],[40,50,60],[70,80,90]],np.uint8))
    with pytest.raises(FileExistsError): driver.run(p,driver.digest(p),output,device='cpu',resolution=9)


def test_normal_color_source_size_mismatch_rejected(tmp_path,monkeypatch):
    from utils import project_mesh
    p,m=fixture(tmp_path)
    normal=tmp_path/'front_face-normal.png'
    Image.fromarray(np.full((7,9,4),255,np.uint8)).save(normal)
    m['views'][0]['normal']['sha256']=driver.digest(normal)
    p.write_text(json.dumps(m))
    monkeypatch.setattr(project_mesh,'get_cameras_list',lambda *a,**kw: [None]*6)
    with pytest.raises(ValueError,match='normal image.*paired color'):
        driver.run(p,driver.digest(p),tmp_path/'out',device='cpu',resolution=9)
    assert not (tmp_path/'out').exists()


def test_nonrepresentable_geometry_rejected_before_projection(tmp_path):
    p,m=fixture(tmp_path)
    obj=tmp_path/'mesh.obj'
    obj.write_text('v 0.1 0 1 0.2 0.3 0.4\nv 1 0 1 0.3 0.4 0.5\nv 0 1 1 0.4 0.5 0.6\nf 1 2 3\n')
    m['mesh']={'path':str(obj),'sha256':driver.digest(obj)}
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='round-trip exactly'):
        driver.run(p,driver.digest(p),tmp_path/'out',device='cpu',resolution=9)
    assert not (tmp_path/'out').exists()


def test_raster_policy_is_frozen_despite_transient_environment_change(tmp_path,monkeypatch):
    import torch
    from utils import project_mesh
    p,_=fixture(tmp_path)
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','default')
    monkeypatch.setattr(project_mesh,'get_cameras_list',lambda *a,**kw: [None]*6)
    def project(mesh,*a,support_counts_out,**kw):
        frozen=kw['source_raster_env']
        assert frozen['PSHUMAN_TEXTURE_RASTER_MODE']=='default'
        monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','all_faces')
        assert frozen['PSHUMAN_TEXTURE_RASTER_MODE']=='default'
        with pytest.raises(TypeError): frozen['PSHUMAN_TEXTURE_RASTER_MODE']='all_faces'
        monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','default')
        support_counts_out['measured_mask']=torch.tensor([False,False,False])
        return mesh
    monkeypatch.setattr(project_mesh,'multiview_color_projection',project)
    report=driver.run(p,driver.digest(p),tmp_path/'out',device='cpu',resolution=9)
    assert report['source_raster']=={'mode':'default','bin_buffer_limit_bytes':None}

import pytest
from utils.texture_raster_budget import source_raster_options

class Tensor:
    is_cuda = True
    shape = (296202, 3)
class Mesh:
    def verts_packed(self): return Tensor()
    def faces_packed(self): return Tensor()
    def __len__(self): return 1
class Ortho:
    def is_perspective(self): return False
class Perspective:
    def is_perspective(self): return True

def test_default_returns_none_and_ignores_cap():
    assert source_raster_options(Mesh(), Perspective(), 1024, 1024, {'PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':'bad'}) is None

def test_all_faces_capacity_and_exact_memory_limit():
    required = 16 * 16 * 296202 * 4
    env={'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces','PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':str(required)}
    assert source_raster_options(Mesh(),Ortho(),1024,1024,env) == {'bin_size':64,'max_faces_per_bin':296202}
    env['PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES']=str(required-1)
    with pytest.raises(ValueError, match='exceeds configured limit'): source_raster_options(Mesh(),Ortho(),1024,1024,env)

@pytest.mark.parametrize('value',['0','-1','bad','1.0','', ' 12'])
def test_invalid_budget_is_refused(value):
    with pytest.raises(ValueError,match='positive integer'):
        source_raster_options(Mesh(),Ortho(),1024,1024,{'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces','PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':value})

def test_unsupported_mode_and_perspective_fail_closed():
    with pytest.raises(ValueError,match='must be default or all_faces'): source_raster_options(Mesh(),Ortho(),1024,1024,{'PSHUMAN_TEXTURE_RASTER_MODE':'typo'})
    with pytest.raises(ValueError,match='without face clipping'): source_raster_options(Mesh(),Perspective(),1024,1024,{'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces'})

def test_batched_rectangular_buffer_is_conservatively_bound():
    class Two(Mesh):
        def __len__(self): return 2
    required = 2 * 8 * 16 * 296202 * 4
    env={'PSHUMAN_TEXTURE_RASTER_MODE':'all_faces','PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES':str(required)}
    assert source_raster_options(Two(),Ortho(),512,1024,env)['bin_size'] == 64
    env['PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES']=str(required-1)
    with pytest.raises(ValueError,match='exceeds'): source_raster_options(Two(),Ortho(),512,1024,env)

def test_only_separate_source_support_activates_options(monkeypatch):
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from utils import project_mesh, texture_raster_budget
    vertices=torch.tensor([[-.02,-.02,1.],[.02,-.02,1.],[0.,.02,1.]])
    mesh=Meshes(verts=[vertices],faces=[torch.tensor([[0,1,2]])],textures=TexturesVertex(verts_features=[torch.zeros((3,3))]))
    class Camera:
        def to(self,device): return self
        def unproject_points(self,p): return p
        def transform_points(self,p): return p
        def get_world_to_view_transform(self): return self
    calls=[]
    def buffers(*args,**kwargs):
        calls.append(kwargs.get('raster_options'))
        return {'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':torch.ones((1,9,9))}
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',buffers)
    monkeypatch.setattr(texture_raster_budget,'source_raster_options',lambda *args:{'bin_size':8,'max_faces_per_bin':3})
    image=torch.ones((4,9,9))
    project_mesh.project_color(mesh,Camera(),image,resolution=9,device='cpu',depth_edge_tol=0.)
    project_mesh.project_color(mesh,Camera(),image,resolution=9,device='cpu',depth_edge_tol=0.,support_silhouette=image)
    assert calls==[None,{'bin_size':8,'max_faces_per_bin':3}]


def test_explicit_policy_reaches_each_source_projection(monkeypatch):
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from utils import project_mesh,texture_raster_budget
    vertices=torch.tensor([[-.02,-.02,1.],[.02,-.02,1.],[0.,.02,1.]])
    mesh=Meshes(verts=[vertices],faces=[torch.tensor([[0,1,2]])],textures=TexturesVertex(verts_features=[torch.zeros((3,3))]))
    class Camera:
        def to(self,device): return self
        def unproject_points(self,p): return p
        def transform_points(self,p): return p
        def get_world_to_view_transform(self): return self
    calls=[]
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**k:{'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':torch.ones((1,9,9))})
    def policy(mesh,camera,h,w,env):
        calls.append(dict(env));return None
    monkeypatch.setattr(texture_raster_budget,'source_raster_options',policy)
    image=torch.ones((4,9,9));frozen={'PSHUMAN_TEXTURE_RASTER_MODE':'default'}
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','all_faces')
    project_mesh.project_color(mesh,Camera(),image,resolution=9,device='cpu',depth_edge_tol=0.,support_silhouette=image,source_raster_env=frozen)
    monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','default')
    project_mesh.project_color(mesh,Camera(),image,resolution=9,device='cpu',depth_edge_tol=0.,support_silhouette=image,source_raster_env=frozen)
    assert calls==[frozen,frozen]


def test_multiview_forwards_frozen_policy_and_default_is_unchanged(monkeypatch):
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from utils import project_mesh
    mesh=Meshes(verts=[torch.tensor([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])],faces=[torch.tensor([[0,1,2]])],textures=TexturesVertex(verts_features=[torch.zeros((3,3))]))
    seen=[]
    def project(*args,**kwargs):
        seen.append(kwargs['source_raster_env'])
        monkeypatch.setenv('PSHUMAN_TEXTURE_RASTER_MODE','all_faces')
        return {'valid_verts':torch.empty((0,),dtype=torch.long)}
    monkeypatch.setattr(project_mesh,'project_color',project)
    images=[torch.ones((4,9,9)),torch.ones((4,9,9))]
    frozen={'PSHUMAN_TEXTURE_RASTER_MODE':'default'}
    project_mesh.multiview_color_projection(mesh,images,cameras_list=[None,None],weights=[1.,1.],device='cpu',support_silhouettes=images,source_raster_env=frozen)
    assert seen==[frozen,frozen]
    seen.clear()
    project_mesh.multiview_color_projection(mesh,images,cameras_list=[None,None],weights=[1.,1.],device='cpu')
    assert seen==[None,None]

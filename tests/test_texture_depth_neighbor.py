import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils import project_mesh


def test_depth_neighborhood_inspects_all_nine_offsets(monkeypatch):
    vertices=torch.tensor([[-.02,-.02,1.],[.02,-.02,1.],[0.,.02,1.]])
    faces=torch.tensor([[0,1,2]])
    mesh=Meshes(verts=[vertices],faces=[faces],textures=TexturesVertex(verts_features=[torch.zeros((3,3))]))
    class Camera:
        def to(self,device): return self
        def unproject_points(self,p): return p
        def transform_points(self,p): return p
        def get_world_to_view_transform(self): return self
    z=torch.ones((1,9,9));z[0,4,5]=1.03
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**kw:{'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':z})
    image=torch.ones((4,9,9))
    result=project_mesh.project_color(mesh,Camera(),image,resolution=9,device='cpu',depth_tol=.01,depth_edge_tol=.006,depth_edge_dilate_px=0,support_silhouette=image)
    assert result['valid_verts'].numel()==0


def test_legacy_color_path_keeps_historical_single_offset(monkeypatch):
    vertices=torch.tensor([[-.02,-.02,1.],[.02,-.02,1.],[0.,.02,1.]])
    mesh=Meshes(verts=[vertices],faces=[torch.tensor([[0,1,2]])],textures=TexturesVertex(verts_features=[torch.zeros((3,3))]))
    class Camera:
        def to(self,device): return self
        def unproject_points(self,p): return p
        def transform_points(self,p): return p
        def get_world_to_view_transform(self): return self
    z=torch.ones((1,9,9));z[0,4,5]=1.03
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**kw:{'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':z})
    result=project_mesh.project_color(mesh,Camera(),torch.ones((4,9,9)),resolution=9,device='cpu',depth_tol=.01,depth_edge_tol=.006,depth_edge_dilate_px=0)
    assert result['valid_verts'].numel()==3

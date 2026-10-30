import torch
import pytest
from utils.source_color_sampling import sample_source_color


def test_sparse_supported_color_survives_separate_silhouette_erosion():
    image=torch.zeros((4,9,9));image[:3,4,4]=torch.tensor([.8,.2,.1]);image[3,4,4]=1
    silhouette=torch.ones((4,9,9));points=torch.tensor([[0.,0.]])
    rgb,alpha=sample_source_color(image,silhouette,points,alpha_erode_px=4)
    torch.testing.assert_close(rgb,torch.tensor([[.8,.2,.1]]))
    torch.testing.assert_close(alpha,torch.ones((1,1)))
    # The same sparse alpha incorrectly treated as the silhouette is eroded away.
    _,old_alpha=sample_source_color(image,image,points,alpha_erode_px=4)
    assert old_alpha.item()==0


def test_cutout_edge_remains_rejected_and_unsupported_never_contributes():
    image=torch.zeros((4,9,9));image[:3,4,4]=1;image[3,4,4]=1
    silhouette=torch.ones((4,9,9));silhouette[3,4,3]=0
    points=torch.tensor([[0.,0.]])
    _,alpha=sample_source_color(image,silhouette,points,alpha_erode_px=1)
    assert alpha.item()==0
    image[3,4,4]=0;silhouette[3,4,3]=1
    rgb,alpha=sample_source_color(image,silhouette,points,alpha_erode_px=1)
    assert alpha.item()==0 and torch.equal(rgb,torch.zeros_like(rgb))


def test_partial_support_uses_premultiplied_color_without_unobserved_bleed():
    image=torch.tensor([[[1.,0.]], [[0.,1.]], [[0.,0.]], [[1.,0.]]])
    silhouette=torch.ones_like(image)
    rgb,alpha=sample_source_color(image,silhouette,torch.tensor([[0.,0.]]))
    torch.testing.assert_close(alpha,torch.tensor([[.5]]))
    torch.testing.assert_close(rgb,torch.tensor([[1.,0.,0.]]))


def test_reject_misaligned_silhouette_and_invalid_erosion():
    image=torch.ones((4,3,3));points=torch.tensor([[0.,0.]])
    with pytest.raises(ValueError,match='matching'):
        sample_source_color(image,torch.ones((4,4,4)),points)
    with pytest.raises(ValueError,match='nonnegative integer'):
        sample_source_color(image,image,points,alpha_erode_px=True)


def test_real_projection_keeps_visibility_gate_and_sparse_support(monkeypatch):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from utils import project_mesh
    vertices=torch.tensor([[-.02,-.02,1.],[.02,-.02,1.],[0.,.02,1.]])
    faces=torch.tensor([[0,1,2]])
    mesh=Meshes(verts=[vertices],faces=[faces],textures=TexturesVertex(verts_features=[torch.zeros((3,3))]))
    class Camera:
        def to(self,device): return self
        def unproject_points(self,p): return p
        def transform_points(self,p): return p
        def get_world_to_view_transform(self): return self
    camera=Camera()
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**kw:{
        'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':torch.ones((1,9,9))})
    image=torch.zeros((4,9,9));image[:3,4,4]=torch.tensor([.9,.3,.1]);image[3,4,4]=1
    silhouette=torch.ones_like(image)
    common=dict(meshes=mesh,cameras=camera,image=image,resolution=9,device='cpu',
                alpha_erode_px=4,depth_edge_tol=0.)
    separate=project_mesh.project_color(**common,support_silhouette=silhouette)
    old=project_mesh.project_color(**common)
    assert len(separate['valid_verts'])==3 and separate['vertex_visible'].all()
    assert separate['valid_alpha'].min()>0 and old['valid_alpha'].max()==0
    torch.testing.assert_close(separate['valid_colors'],torch.tensor([[.9,.3,.1]]).expand(3,3),atol=1e-5,rtol=0)
    assert torch.equal(separate['normal_gate'],old['normal_gate'])
    # A depth mismatch must still reject every projected vertex.
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**kw:{
        'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':torch.zeros((1,9,9))})
    hidden=project_mesh.project_color(**common,support_silhouette=silhouette)
    assert hidden['valid_verts'].numel()==0


def test_baseline_requires_exact_geometry_and_finite_colors():
    from utils.source_color_sampling import validate_baseline_colors
    vertices=torch.tensor([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    faces=torch.tensor([[0,1,2]])
    colors=torch.tensor([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]])
    baseline={'vertices':vertices.clone(),'faces':faces.clone(),'colors':colors.clone()}
    torch.testing.assert_close(validate_baseline_colors(vertices,faces,baseline),colors)
    baseline['faces']=torch.tensor([[0,2,1]])
    with pytest.raises(ValueError,match='exactly match'):
        validate_baseline_colors(vertices,faces,baseline)
    baseline['faces']=faces.clone(); baseline['colors'][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite RGB'):
        validate_baseline_colors(vertices,faces,baseline)


def test_real_multiview_projection_preserves_baseline_for_unsupported_and_hidden(monkeypatch):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from utils import project_mesh
    vertices=torch.tensor([[-.02,-.02,1.],[.02,-.02,1.],[0.,.02,1.]])
    faces=torch.tensor([[0,1,2]])
    colors=torch.tensor([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]])
    mesh=Meshes(verts=[vertices],faces=[faces],textures=TexturesVertex(verts_features=[colors]))
    class Camera:
        def to(self,device): return self
        def unproject_points(self,p): return p
        def transform_points(self,p): return p
        def get_world_to_view_transform(self): return self
    image=torch.zeros((4,9,9)); image[:3,4,4]=torch.tensor([.9,.3,.1]); image[3,4,4]=1
    silhouette=torch.ones_like(image)
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**kw:{'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':torch.ones((1,9,9))})
    def run(image):
        return project_mesh.multiview_color_projection(mesh,[image,image],cameras_list=[Camera(),Camera()],weights=[1.,1.],device='cpu',resolution=9,support_silhouettes=[silhouette,silhouette],alpha_erode_px=4,depth_edge_tol=0.,below_confidence_strategy='original').textures.verts_features_packed()
    result=run(image)
    torch.testing.assert_close(result,torch.tensor([[.9,.3,.1]]).expand(3,3),atol=1e-5,rtol=0)
    absent=image.clone(); absent[3]=0
    torch.testing.assert_close(run(absent),colors)
    monkeypatch.setattr(project_mesh,'get_visibility_buffers',lambda *a,**kw:{'pix_to_face':torch.zeros((1,9,9),dtype=torch.long),'zbuf':torch.zeros((1,9,9))})
    torch.testing.assert_close(run(image),colors)

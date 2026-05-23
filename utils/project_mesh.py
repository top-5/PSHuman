from typing import List
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pytorch3d.renderer.cameras import look_at_view_transform, OrthographicCameras, CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    TexturesVertex,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
)
from pytorch3d.renderer import MeshRasterizer

def render_pix2faces_py3d(meshes, cameras, H=512, W=512, blur_radius=0.0, faces_per_pixel=1):
    """
    Renders pix2face of visible faces.

    :param mesh: Pytorch3d.structures.Meshes
    :param cameras: pytorch3d.renderer.Cameras
    :param H: target image height
    :param W: target image width
    :param blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
    :param faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
    """
    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    fragments: Fragments = rasterizer(meshes, cameras=cameras)
    return {
        "pix_to_face": fragments.pix_to_face[..., 0],
        "zbuf": fragments.zbuf[..., 0],
    }

import nvdiffrast.torch as dr

def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class Pix2FacesRenderer:
    def __init__(self, device="cuda"):
        self._glctx = dr.RasterizeCudaContext(device=device)
        self.device = device
        _warmup(self._glctx, device)

    def transform_vertices(self, meshes: Meshes, cameras: CamerasBase):
        vertices = cameras.transform_points_ndc(meshes.verts_padded())

        perspective_correct = cameras.is_perspective()
        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if not perspective_correct or znear is None else znear / 2

        if z_clip:
            vertices = vertices[vertices[..., 2] >= cameras.get_znear()][None]    # clip
        vertices = vertices * torch.tensor([-1, -1, 1]).to(vertices)
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1).to(torch.float32)
        return vertices

    def render_pix2faces_nvdiff(self, meshes: Meshes, cameras: CamerasBase, H=512, W=512):
        meshes = meshes.to(self.device)
        cameras = cameras.to(self.device)
        vertices = self.transform_vertices(meshes, cameras)
        faces = meshes.faces_packed().to(torch.int32)
        rast_out,_ = dr.rasterize(self._glctx, vertices, faces, resolution=(H, W), grad_db=False) #C,H,W,4
        pix_to_face = rast_out[..., -1].to(torch.int32) - 1
        return pix_to_face

pix2faces_renderer = None

def get_visible_faces(meshes: Meshes, cameras: CamerasBase, resolution=1024):
    # global pix2faces_renderer
    # if pix2faces_renderer is None:
    #     pix2faces_renderer = Pix2FacesRenderer()
    pix_to_face = render_pix2faces_py3d(meshes, cameras, H=resolution, W=resolution)['pix_to_face']
    # pix_to_face = pix2faces_renderer.render_pix2faces_nvdiff(meshes, cameras, H=resolution, W=resolution)

    unique_faces = torch.unique(pix_to_face.flatten())
    unique_faces = unique_faces[unique_faces != -1]
    return unique_faces

def get_visibility_buffers(meshes: Meshes, cameras: CamerasBase, resolution=1024):
    return render_pix2faces_py3d(meshes, cameras, H=resolution, W=resolution)

def project_color(
    meshes: Meshes,
    cameras: CamerasBase,
    image: torch.Tensor,
    normal_image: torch.Tensor = None,
    use_alpha=True,
    eps=0.05,
    resolution=1024,
    device="cuda",
    normal_gate_threshold=0.35,
    depth_tol=None,
    depth_edge_tol=None,
    depth_edge_dilate_px=0,
    alpha_erode_px=0,
    alpha_edge_threshold=0.95,
) -> dict:
    """
    Projects color from a given image onto a 3D mesh.

    Args:
        meshes (pytorch3d.structures.Meshes): The 3D mesh object.
        cameras (pytorch3d.renderer.cameras.CamerasBase): The camera object.
        pil_image (PIL.Image.Image): The input image.
        use_alpha (bool, optional): Whether to use the alpha channel of the image. Defaults to True.
        eps (float, optional): The threshold for selecting visible faces. Defaults to 0.05.
        resolution (int, optional): The resolution of the projection. Defaults to 1024.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        debug (bool, optional): Whether to save debug images. Defaults to False.

    Returns:
        dict: A dictionary containing the following keys:
            - "new_texture" (TexturesVertex): The updated texture with interpolated colors.
            - "valid_verts" (Tensor of [M,3]): The indices of the vertices being projected.
            - "valid_colors" (Tensor of [M,3]): The interpolated colors for the valid vertices.
    """
    meshes = meshes.to(device)
    cameras = cameras.to(device)

    visibility = get_visibility_buffers(meshes, cameras, resolution=resolution)
    pix_to_face = visibility["pix_to_face"]
    zbuf = visibility["zbuf"]
    unique_faces = torch.unique(pix_to_face.flatten())
    unique_faces = unique_faces[unique_faces != -1]

    # visible faces
    faces_normals = meshes.faces_normals_packed()[unique_faces]
    faces_normals = faces_normals / faces_normals.norm(dim=1, keepdim=True)
    world_points = cameras.unproject_points(torch.tensor([[[0., 0., 0.1], [0., 0., 0.2]]]).to(device))[0]
    view_direction = world_points[1] - world_points[0]
    view_direction = view_direction / view_direction.norm(dim=0, keepdim=True)

    # find invalid faces
    cos_angles = (faces_normals * view_direction).sum(dim=1)
    if cos_angles.mean() >= 0:
        # Poisson reconstruction can produce inside-out meshes; auto-correct view direction
        view_direction = -view_direction
        cos_angles = -cos_angles
    selected_faces = unique_faces[cos_angles < -eps]

    # find verts
    faces = meshes.faces_packed()[selected_faces]   # [N, 3]
    verts = torch.unique(faces.flatten())   # [N, 1]
    verts_coordinates = meshes.verts_packed()[verts]   # [N, 3]

    # compute color
    projected = cameras.transform_points(verts_coordinates)
    view_z = cameras.get_world_to_view_transform().transform_points(verts_coordinates)[..., 2:3]
    pt_tensor = projected[..., :2] # NDC space points
    valid = ~(projected.isnan().any(dim=1) | (pt_tensor < -1).any(dim=1) | (1 < pt_tensor).any(dim=1))  # checked, correct
    valid_pt = pt_tensor[valid, :]
    valid_idx = verts[valid]
    valid_z = view_z[valid]
    # Face-level visibility is not sufficient: a side-view dress triangle can
    # be partly visible while one of its vertices projects behind the arm.
    # Sampling color at that hidden vertex burns an arm-shaped silhouette into
    # the dress. Gate each vertex by the rasterized nearest depth at its own
    # projected pixel before sampling source color.
    zbuf_sample = zbuf[0][None, None].flip((-1, -2))
    sampled_z = torch.nn.functional.grid_sample(
        zbuf_sample,
        valid_pt[None, :, None, :],
        align_corners=False,
        padding_mode="zeros",
        mode="nearest",
    )[0, :, :, 0].T
    if depth_tol is None:
        depth_tol = float(os.environ.get("PSHUMAN_TEXTURE_DEPTH_TOL", "0.01"))
    if depth_edge_tol is None:
        depth_edge_tol = float(os.environ.get("PSHUMAN_TEXTURE_DEPTH_EDGE_TOL", "0.006"))
    z_tol = float(depth_tol)
    edge_tol = float(depth_edge_tol)
    vertex_visible = (sampled_z >= 0) & ((sampled_z - valid_z).abs() <= z_tol)
    if edge_tol > 0 and valid_pt.numel() > 0:
        # The side-view arm bleed sits exactly on a depth discontinuity: the
        # vertex itself can sample the dress depth while adjacent source pixels
        # still belong to the foreground arm/antialiased cutout. Reject samples
        # whose 3x3 depth neighborhood is not locally flat before color lookup.
        ndc_step = 2.0 / float(resolution)
        offsets = torch.tensor(
            [
                [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
                [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0],
            ],
            device=device,
            dtype=valid_pt.dtype,
        ) * ndc_step
        neigh_pt = (valid_pt[:, None, :] + offsets[None, :, :]).clamp(-1.0, 1.0)
        neigh_z = torch.nn.functional.grid_sample(
            zbuf_sample,
            neigh_pt[None, :, :, :],
            align_corners=False,
            padding_mode="zeros",
            mode="nearest",
        )[0, :, :, 0].T
        neigh_valid = neigh_z >= 0
        z_delta = torch.where(neigh_valid, (neigh_z - sampled_z).abs(), torch.zeros_like(neigh_z))
        local_depth_edge = z_delta.max(dim=1, keepdim=True).values > edge_tol
        vertex_visible = vertex_visible & ~local_depth_edge
        if int(depth_edge_dilate_px) > 0:
            # For side/oblique views, the generated colour image can smear the
            # foreground arm contour onto the surface just behind it. Reject a
            # configurable corridor around any rasterized depth discontinuity,
            # independent of material colour.
            dilate = int(depth_edge_dilate_px)
            finite = zbuf_sample >= 0
            z_for_max = torch.where(finite, zbuf_sample, torch.full_like(zbuf_sample, -1e6))
            z_for_min = torch.where(finite, zbuf_sample, torch.full_like(zbuf_sample, 1e6))
            k_edge = 3
            max_z = F.max_pool2d(z_for_max, kernel_size=k_edge, stride=1, padding=k_edge // 2)
            min_z = -F.max_pool2d(-z_for_min, kernel_size=k_edge, stride=1, padding=k_edge // 2)
            depth_edge = ((max_z - min_z) > edge_tol) & finite
            k_dilate = 2 * dilate + 1
            depth_edge = F.max_pool2d(
                depth_edge.to(zbuf_sample.dtype),
                kernel_size=k_dilate,
                stride=1,
                padding=dilate,
            )
            sampled_depth_edge = torch.nn.functional.grid_sample(
                depth_edge,
                valid_pt[None, :, None, :],
                align_corners=False,
                padding_mode="zeros",
                mode="nearest",
            )[0, :, :, 0].T
            vertex_visible = vertex_visible & (sampled_depth_edge < 0.5)
    valid_pt = valid_pt[vertex_visible[:, 0], :]
    valid_idx = valid_idx[vertex_visible[:, 0]]
    if valid_idx.numel() == 0:
        return {
            "new_texture": meshes.textures,
            "valid_verts": valid_idx,
            "valid_colors": torch.empty((0, 3), device=device, dtype=meshes.verts_packed().dtype),
            "valid_alpha": torch.empty((0, 1), device=device, dtype=meshes.verts_packed().dtype),
            "cos_angles": torch.empty((0,), device=device, dtype=meshes.verts_packed().dtype),
            "normal_gate": torch.empty((0, 1), device=device, dtype=meshes.verts_packed().dtype),
            "vertex_visible": vertex_visible,
        }
    valid_color = torch.nn.functional.grid_sample(image[None].flip((-1, -2)), valid_pt[None, :, None, :], align_corners=False, padding_mode="border", mode="bilinear")[0, :, :, 0].T.clamp(0, 1)   # [N, 4], note that bicubic may give invalid value
    alpha, valid_color = valid_color[:, 3:], valid_color[:, :3]
    if not use_alpha:
        alpha = torch.ones_like(alpha)
    elif alpha_erode_px > 0:
        # Reject samples close to source RGBA cutout edges. Those pixels are
        # where side-view arm/background antialiasing most often bleeds into
        # dark clothing vertices.
        edge_alpha = image[3:4][None]
        for _ in range(int(alpha_erode_px)):
            edge_alpha = -F.max_pool2d(-edge_alpha, kernel_size=3, stride=1, padding=1)
        safe_alpha = torch.nn.functional.grid_sample(
            edge_alpha.flip((-1, -2)),
            valid_pt[None, :, None, :],
            align_corners=False,
            padding_mode="zeros",
            mode="bilinear",
        )[0, :, :, 0].T.clamp(0, 1)
        alpha = alpha * (safe_alpha >= float(alpha_edge_threshold)).to(alpha.dtype)

    # Do not mutate mesh vertex colors here. multiview_color_projection()
    # accumulates all view samples below; mutating during sampling pollutes the
    # "original color" fallback and can preserve arm-shaped side-view stains in
    # low-confidence dress vertices.
    new_texture = meshes.textures

    valid_verts_normals = meshes.verts_normals_packed()[valid_idx]
    valid_verts_normals = valid_verts_normals / valid_verts_normals.norm(dim=1, keepdim=True).clamp_min(0.001)
    cos_angles = (valid_verts_normals * view_direction).sum(dim=1)
    normal_gate = torch.ones_like(alpha)
    if normal_image is not None:
        valid_normal = torch.nn.functional.grid_sample(
            normal_image[None].flip((-1, -2)),
            valid_pt[None, :, None, :],
            align_corners=False,
            padding_mode="reflection",
            mode="bilinear",
        )[0, :, :, 0].T.clamp(0, 1)
        normal_alpha = valid_normal[:, 3:]
        sampled_normals = valid_normal[:, :3] * 2.0 - 1.0
        sampled_normals = sampled_normals / sampled_normals.norm(dim=1, keepdim=True).clamp_min(0.001)
        normal_agreement = (sampled_normals * valid_verts_normals).sum(dim=1, keepdim=True)
        # Source-side occlusion guard: reject pixels whose generated normal map
        # belongs to a different surface than the vertex being colored. This
        # prevents arm-color pixels from being accepted by dress vertices when
        # the multiview color image contains a self-occlusion imprint.
        normal_gate = ((normal_agreement - normal_gate_threshold) / (1.0 - normal_gate_threshold)).clamp(0, 1)
        normal_gate = normal_gate * normal_alpha
    return {
        "new_texture": new_texture,
        "valid_verts": valid_idx,
        "valid_colors": valid_color,
        "valid_alpha": alpha,
        "cos_angles": cos_angles,
        "normal_gate": normal_gate,
        "vertex_visible": vertex_visible,
    }

def complete_unseen_vertex_color(meshes: Meshes, valid_index: torch.Tensor) -> dict:
    """
    meshes: the mesh with vertex color to be completed.
    valid_index: the index of the valid vertices, where valid means colors are fixed. [V, 1]
    """
    valid_index = valid_index.to(meshes.device)
    colors = meshes.textures.verts_features_packed()    # [V, 3]
    V = colors.shape[0]
    
    invalid_index = torch.ones_like(colors[:, 0]).bool()    # [V]
    invalid_index[valid_index] = False
    invalid_index = torch.arange(V).to(meshes.device)[invalid_index]
    
    L = meshes.laplacian_packed()
    E = torch.sparse_coo_tensor(torch.tensor([list(range(V))] * 2), torch.ones((V,)), size=(V, V)).to(meshes.device)
    L = L + E
    # E = torch.eye(V, layout=torch.sparse_coo, device=meshes.device)
    # L = L + E
    colored_count = torch.ones_like(colors[:, 0])   # [V]
    colored_count[invalid_index] = 0
    L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]
    
    total_colored = colored_count.sum()
    coloring_round = 0
    stage = "uncolored"
    from tqdm import tqdm
    pbar = tqdm(miniters=100)
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, colors * colored_count[:, None])    # [IV, 3]
        new_count = torch.matmul(L_invalid, colored_count)[:, None]             # [IV, 1]
        colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
        colored_count[invalid_index] = (new_count[:, 0] > 0).float()
        
        new_total_colored = colored_count.sum()
        if new_total_colored > total_colored:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        pbar.update(1)
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break
    assert not torch.isnan(colors).any()
    meshes.textures = TexturesVertex(verts_features=[colors])
    return meshes

def multiview_color_projection(
    meshes: Meshes,
    image_list: torch.Tensor,
    normal_image_list: torch.Tensor = None,
    cameras_list: List[CamerasBase]=None,
    camera_focal: float = 2 / 1.35,
    weights=None,
    eps=0.05,
    resolution=1024,
    device="cuda",
    reweight_with_cosangle="square",
    use_alpha=True,
    confidence_threshold=0.1,
    complete_unseen=False,
    below_confidence_strategy="smooth",
    normal_gate_threshold=0.35,
    depth_tol=None,
    depth_edge_tol=None,
    depth_edge_dilate_px=0,
    alpha_erode_px=0,
    alpha_edge_threshold=0.95,
    side_occlusion_indices=None,
    side_occlusion_guard=True,
) -> Meshes:
    """
    Projects color from a given image onto a 3D mesh.

    Args:
        meshes (pytorch3d.structures.Meshes): The 3D mesh object, only one mesh.
        image_list (PIL.Image.Image): List of images.
        cameras_list (list): List of cameras.
        camera_focal (float, optional): The focal length of the camera, if cameras_list is not passed. Defaults to 2 / 1.35.
        weights (list, optional): List of weights for each image, for ['front', 'front_right', 'right', 'back', 'left', 'front_left']. Defaults to None.
        eps (float, optional): The threshold for selecting visible faces. Defaults to 0.05.
        resolution (int, optional): The resolution of the projection. Defaults to 1024.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        reweight_with_cosangle (str, optional): Whether to reweight the color with the angle between the view direction and the vertex normal. Defaults to None.
        use_alpha (bool, optional): Whether to use the alpha channel of the image. Defaults to True.
        confidence_threshold (float, optional): The threshold for the confidence of the projected color, if final projection weight is less than this, we will use the original color. Defaults to 0.1.
        complete_unseen (bool, optional): Whether to complete the unseen vertex color using laplacian. Defaults to False.

    Returns:
        Meshes: the colored mesh
    """
    # 1. preprocess inputs
    if image_list is None:
        raise ValueError("image_list is None")
    if normal_image_list is not None and len(normal_image_list) != len(image_list):
        raise ValueError("normal_image_list must have the same length as image_list")
    if cameras_list is None:
        if len(image_list) == 8:
            cameras_list = get_8view_cameras(device, focal=camera_focal)
        elif len(image_list) == 6:
            cameras_list = get_6view_cameras(device, focal=camera_focal)
        elif len(image_list) == 4:
            cameras_list = get_4view_cameras(device, focal=camera_focal)
        elif len(image_list) == 2:
            cameras_list = get_2view_cameras(device, focal=camera_focal)
        else:
            raise ValueError("cameras_list is None, and can not be guessed from image_list")
    if weights is None:
        if len(image_list) == 8:
            weights = [2.0, 0.05, 0.2, 0.02, 1.0, 0.02, 0.2, 0.05]
        elif len(image_list) == 6:
            weights = [2.0, 0.05, 0.2, 1.0, 0.2, 0.05]
            # weights = [2.0, 0.2, 1.5, 1.0, 1.5, 0.2]
        elif len(image_list) == 4:
            weights = [2.0, 0.2, 1.0, 0.2]
        elif len(image_list) == 2:
            weights = [1.0, 1.0]
        else:
            raise ValueError("weights is None, and can not be guessed from image_list")
    if side_occlusion_indices is None:
        side_occlusion_indices = []
    side_occlusion_indices = set(int(x) for x in side_occlusion_indices)
    
    # 2. run projection
    meshes = meshes.clone().to(device)
    assert len(cameras_list) == len(image_list) == len(weights)
    original_color = meshes.textures.verts_features_packed().clone()
    assert not torch.isnan(original_color).any()
    texture_counts = torch.zeros_like(original_color[..., :1])
    texture_values = torch.zeros_like(original_color)
    max_texture_counts = torch.zeros_like(original_color[..., :1])
    max_texture_values = torch.zeros_like(original_color)
    normal_iter = normal_image_list if normal_image_list is not None else [None] * len(image_list)
    gated_samples = 0
    total_samples = 0
    vertex_occluded_samples = 0
    vertex_total_samples = 0

    for view_i, (camera, image, normal_image, weight) in enumerate(zip(cameras_list, image_list, normal_iter, weights)):
        view_depth_edge_dilate_px = (
            int(depth_edge_dilate_px)
            if side_occlusion_guard and int(view_i) in side_occlusion_indices
            else 0
        )
        ret = project_color(
            meshes,
            camera,
            image,
            normal_image=normal_image,
            eps=eps,
            resolution=resolution,
            device=device,
            use_alpha=use_alpha,
            normal_gate_threshold=normal_gate_threshold,
            depth_tol=depth_tol,
            depth_edge_tol=depth_edge_tol,
            depth_edge_dilate_px=view_depth_edge_dilate_px,
            alpha_erode_px=alpha_erode_px,
            alpha_edge_threshold=alpha_edge_threshold,
        )
        if "vertex_visible" in ret:
            vertex_visible = ret["vertex_visible"]
            vertex_occluded_samples += int((~vertex_visible).sum().detach().cpu())
            vertex_total_samples += int(vertex_visible.numel())
        if ret["valid_verts"].numel() == 0:
            continue
        if reweight_with_cosangle == "linear":
            weight = (ret['cos_angles'].abs() * weight)[:, None]
        elif reweight_with_cosangle == "square":
            weight = (ret['cos_angles'].abs() ** 2 * weight)[:, None]
        if use_alpha:
            weight = weight * ret['valid_alpha']
        if normal_image is not None:
            normal_gate = ret["normal_gate"]
            gated_samples += int((normal_gate <= 0.05).sum().detach().cpu())
            total_samples += int(normal_gate.numel())
            weight = weight * normal_gate
        assert weight.min() > -0.0001
        texture_counts[ret['valid_verts']] += weight
        texture_values[ret['valid_verts']] += ret['valid_colors'] * weight
        max_texture_values[ret['valid_verts']] = torch.where(weight > max_texture_counts[ret['valid_verts']], ret['valid_colors'], max_texture_values[ret['valid_verts']])
        max_texture_counts[ret['valid_verts']] = torch.max(max_texture_counts[ret['valid_verts']], weight)

    # Normalize every filtered observation, not just high-confidence ones. The
    # previous confidence-threshold fallback treated glancing but valid dress
    # samples as "unseen", then Laplacian completion diffused arm/skin colors
    # into the under-arm dress area. After source-side visibility/normal gates,
    # a low-weight sample is still better evidence than global mesh smoothing.
    observed = texture_counts > 1e-6
    texture_values = torch.where(
        observed,
        texture_values / texture_counts.clamp_min(1e-6),
        original_color,
    )
    if below_confidence_strategy not in ("smooth", "original"):
        raise ValueError(f"below_confidence_strategy={below_confidence_strategy} is not supported")
    assert not torch.isnan(texture_values).any()
    meshes.textures = TexturesVertex(verts_features=[texture_values])
    if total_samples:
        print(
            f"[project_mesh] normal-gated color samples: "
            f"{gated_samples:,}/{total_samples:,} "
            f"({100.0 * gated_samples / max(total_samples, 1):.1f}%)",
            flush=True,
        )
    if vertex_total_samples:
        print(
            f"[project_mesh] vertex-occluded color samples: "
            f"{vertex_occluded_samples:,}/{vertex_total_samples:,} "
            f"({100.0 * vertex_occluded_samples / max(vertex_total_samples, 1):.1f}%)",
            flush=True,
        )
    if complete_unseen:
        meshes = complete_unseen_vertex_color(
            meshes,
            torch.arange(texture_values.shape[0]).to(device)[observed[:, 0]],
        )
    ret_mesh = meshes.detach()
    del meshes
    return ret_mesh

def get_camera(R, T, fov_in_degrees=60, focal_length=1 / (2**0.5), cam_type='fov'):
    if cam_type == 'fov':
        camera = FoVPerspectiveCameras(device=R.device, R=R, T=T, fov=fov_in_degrees, degrees=True)
    else:
        focal_length = 1 / focal_length
        camera = FoVOrthographicCameras(device=R.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length)
    return camera

def get_cameras_list(azim_list, device, focal=2/1.35, dist=1.1):
    ret = []
    for azim in azim_list:
        R, T = look_at_view_transform(dist, 0, azim)
        cameras: OrthographicCameras = get_camera(R, T, focal_length=focal, cam_type='orthogonal').to(device)
        ret.append(cameras)
    return ret

def get_8view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 225, 270, 315, 0, 45, 90, 135], device=device, focal=focal)

def get_6view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 225, 270, 0, 90, 135], device=device, focal=focal)

def get_4view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 270, 0, 90], device=device, focal=focal)

def get_2view_cameras(device, focal=2/1.35):
    return get_cameras_list(azim_list = [180, 0], device=device, focal=focal)

def get_multiple_view_cameras(device, focal=2/1.35, offset=180, num_views=8, dist=1.1):
    return get_cameras_list(azim_list = (np.linspace(0, 360, num_views+1)[:-1] + offset) % 360, device=device, focal=focal, dist=dist)

def align_with_alpha_bbox(source_img, target_img, final_size=1024):
    # align source_img with target_img using alpha channel
    # source_img and target_img are PIL.Image.Image
    source_img = source_img.convert("RGBA")
    target_img = target_img.convert("RGBA").resize((final_size, final_size))
    source_np = np.array(source_img)
    target_np = np.array(target_img)
    source_alpha = source_np[:, :, 3]
    target_alpha = target_np[:, :, 3]
    bbox_source_min, bbox_source_max = np.argwhere(source_alpha > 0).min(axis=0), np.argwhere(source_alpha > 0).max(axis=0)
    bbox_target_min, bbox_target_max = np.argwhere(target_alpha > 0).min(axis=0), np.argwhere(target_alpha > 0).max(axis=0)
    source_content = source_np[bbox_source_min[0]:bbox_source_max[0]+1, bbox_source_min[1]:bbox_source_max[1]+1, :]
    # resize source_content to fit in the position of target_content
    source_content = Image.fromarray(source_content).resize((bbox_target_max[1]-bbox_target_min[1]+1, bbox_target_max[0]-bbox_target_min[0]+1), resample=Image.BICUBIC)
    target_np[bbox_target_min[0]:bbox_target_max[0]+1, bbox_target_min[1]:bbox_target_max[1]+1, :] = np.array(source_content)
    return Image.fromarray(target_np)
    
def load_image_list_from_mvdiffusion(mvdiffusion_path, front_from_pil_or_path=None):
    import os
    image_list = []
    for dir in ['front', 'front_right', 'right', 'back', 'left', 'front_left']:
        image_path = os.path.join(mvdiffusion_path, f"rgb_000_{dir}.png")
        pil = Image.open(image_path)
        if dir == 'front':
            if front_from_pil_or_path is not None:
                if isinstance(front_from_pil_or_path, str):
                    replace_pil = Image.open(front_from_pil_or_path)
                else:
                    replace_pil = front_from_pil_or_path
                # align replace_pil with pil using bounding box in alpha channel
                pil = align_with_alpha_bbox(replace_pil, pil, final_size=1024)
        image_list.append(pil)
    return image_list

def load_image_list_from_img_grid(img_grid_path, resolution = 1024):
    img_list = []
    grid = Image.open(img_grid_path)
    w, h = grid.size
    for row in range(0, h, resolution):
        for col in range(0, w, resolution):
            img_list.append(grid.crop((col, row, col + resolution, row + resolution)))
    return img_list

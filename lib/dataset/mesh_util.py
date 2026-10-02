# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import cv2
import pymeshlab
import torch
import torchvision
import trimesh
import json
from pytorch3d.io import load_obj
import os
from termcolor import colored
import os.path as osp
from scipy.spatial import cKDTree
import _pickle as cPickle
import open3d as o3d

from pytorch3d.structures import Meshes
import torch.nn.functional as F
from lib.common.render_utils import Pytorch3dRasterizer, face_vertices

from pytorch3d.renderer.mesh import rasterize_meshes
from PIL import Image, ImageFont, ImageDraw
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance

from pytorch3d.loss import (mesh_laplacian_smoothing, mesh_normal_consistency)

# import tinyobjloader


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def obj_loader(path):
    # Create reader.
    reader = tinyobjloader.ObjReader()

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(path)

    if ret == False:
        print("Failed to load : ", path)
        return None

    # note here for wavefront obj, #v might not equal to #vt, same as #vn.
    attrib = reader.GetAttrib()
    verts = np.array(attrib.vertices).reshape(-1, 3)

    shapes = reader.GetShapes()
    tri = shapes[0].mesh.numpy_indices().reshape(-1, 9)
    faces = tri[:, [0, 3, 6]]

    return verts, faces


class HoppeMesh:

    def __init__(self, verts, faces):
        '''
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        :param points: pts
        :param normals: normals
        '''
        self.trimesh = trimesh.Trimesh(verts, faces, process=True)
        self.verts = np.array(self.trimesh.vertices)
        self.faces = np.array(self.trimesh.faces)
        self.vert_normals, self.faces_normals = compute_normal(
            self.verts, self.faces)

    def contains(self, points):

        labels = check_sign(
            torch.as_tensor(self.verts).unsqueeze(0),
            torch.as_tensor(self.faces),
            torch.as_tensor(points).unsqueeze(0))
        return labels.squeeze(0).numpy()

    def triangles(self):
        return self.verts[self.faces]  # [n, 3, 3]


def tensor2variable(tensor, device):
    # [1,23,3,3]
    return torch.tensor(tensor, device=device, requires_grad=True)


class GMoF(torch.nn.Module):

    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        dist = torch.div(residual, residual + self.rho**2)
        return self.rho**2 * dist


def mesh_edge_loss(meshes, target_length: float = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor([0.0],
                            dtype=torch.float32,
                            device=meshes.device,
                            requires_grad=True)

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length)**2.0
    loss_vertex = loss * weights
    # loss_outlier = torch.topk(loss, 100)[0].mean()
    # loss_all = (loss_vertex.sum() + loss_outlier.mean()) / N
    loss_all = loss_vertex.sum() / N

    return loss_all


def remesh(obj_path, perc, device):

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_coord_laplacian_smoothing()
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(perc), adaptive=True)    
    # ms.remeshing_isotropic_explicit_remeshing(
    #     targetlen=pymeshlab.Percentage(perc), adaptive=True)
    ms.save_current_mesh(obj_path.replace("recon", "remesh"))
    polished_mesh = trimesh.load_mesh(obj_path.replace("recon", "remesh"))
    verts_pr = torch.tensor(
        polished_mesh.vertices).float().unsqueeze(0).to(device)
    faces_pr = torch.tensor(polished_mesh.faces).long().unsqueeze(0).to(device)

    return verts_pr, faces_pr


def possion(mesh, obj_path):

    mesh.export(obj_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.surface_reconstruction_screened_poisson(depth=10)
    ms.set_current_mesh(1)
    ms.save_current_mesh(obj_path)

    return trimesh.load(obj_path)


def get_mask(tensor, dim):

    mask = torch.abs(tensor).sum(dim=dim, keepdims=True) > 0.0
    mask = mask.type_as(tensor)

    return mask


def blend_rgb_norm(rgb, norm, mask):

    # [0,0,0] or [127,127,127] should be marked as mask
    final = rgb * (1 - mask) + norm * (mask)

    return final.astype(np.uint8)


def unwrap(image, data):

    img_uncrop = uncrop(
        np.array(
            Image.fromarray(image).resize(
                data['uncrop_param']['box_shape'][:2])),
        data['uncrop_param']['center'], data['uncrop_param']['scale'],
        data['uncrop_param']['crop_shape'])

    img_orig = cv2.warpAffine(img_uncrop,
                              np.linalg.inv(data['uncrop_param']['M'])[:2, :],
                              data['uncrop_param']['ori_shape'][::-1][1:],
                              flags=cv2.INTER_CUBIC)

    return img_orig


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, losses):

    # and (b) the edge length of the predicted mesh
    losses["edge"]['value'] = mesh_edge_loss(mesh)
    # mesh normal consistency
    losses["nc"]['value'] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    losses["laplacian"]['value'] = mesh_laplacian_smoothing(mesh,
                                                            method="uniform")


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def load_checkpoint(model, cfg):

    model_dict = model.state_dict()
    main_dict = {}
    normal_dict = {}

    device = torch.device(f"cuda:{cfg['test_gpus'][0]}")

    if os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
        main_dict = torch.load(cfg.resume_path,
                               map_location=device)['state_dict']

        main_dict = {
            k: v
            for k, v in main_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape and (
                'reconEngine' not in k) and ("normal_filter" not in k) and (
                    'voxelization' not in k)
        }
        print(colored(f"Resume MLP weights from {cfg.resume_path}", 'green'))

    if os.path.exists(cfg.normal_path) and cfg.normal_path.endswith("ckpt"):
        normal_dict = torch.load(cfg.normal_path,
                                 map_location=device)['state_dict']

        for key in normal_dict.keys():
            normal_dict = rename(normal_dict, key,
                                 key.replace("netG", "netG.normal_filter"))

        normal_dict = {
            k: v
            for k, v in normal_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(colored(f"Resume normal model from {cfg.normal_path}", 'green'))

    model_dict.update(main_dict)
    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    model.netG = model.netG.to(device)
    model.reconEngine = model.reconEngine.to(device)

    model.netG.training = False
    model.netG.eval()

    del main_dict
    del normal_dict
    del model_dict

    return model


def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(os.path.join(folder, 'vertices.txt'))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))
    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(os.path.join(folder, 'faces.txt'),
                            dtype=np.int32) - 1
    smpl_face_code = (smpl_vertex_code[smpl_faces[:, 0]] +
                      smpl_vertex_code[smpl_faces[:, 1]] +
                      smpl_vertex_code[smpl_faces[:, 2]]) / 3.0
    smpl_tetras = np.loadtxt(os.path.join(folder, 'tetrahedrons.txt'),
                             dtype=np.int32) - 1

    return smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras


def surface_field_deformation(xyz,  de_nn_verts, de_nn_normals, ori_nn_verts, ori_nn_normals):
    '''
    xyz: [B, N, 3]
    de_nn_verts: [B, N, 3]
    de_nn_normals: [B, N, 3]
    ori_nn_verts: [B, N, 3]
    ori_nn_normals: [B, N, 3]
    '''
    vector=xyz-de_nn_verts # [B, N, 3]
    delta=torch.sum(vector*de_nn_normals, dim=-1, keepdim=True)*ori_nn_normals
    ori_xyz=ori_nn_verts+delta
    
    return ori_xyz # the deformed xyz
    

def feat_select(feat, select):

    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1] // 2
    idx = torch.tile((1-select), (1, dim, 1))*dim + \
        torch.arange(0, dim).unsqueeze(0).unsqueeze(2).type_as(select)
    feat_select = torch.gather(feat, 1, idx.long())

    return feat_select

def get_visibility_color(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # 新增的部分: 检测边缘像素
    edge_mask = torch.zeros_like(pix_to_face)
    offset=1
    for i in range(-1-offset, 2+offset):
        for j in range(-1-offset, 2+offset):
            if i == 0 and j == 0:
                continue
            shifted = torch.roll(pix_to_face, shifts=(i,j), dims=(0,1))
            edge_mask = torch.logical_or(edge_mask, shifted == -1)

    # 更新可见性掩码
    edge_faces = torch.unique(pix_to_face[edge_mask])
    edge_vertices = torch.unique(faces[edge_faces])
    vis_mask[edge_vertices] = 0.0

    return vis_mask


def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def cal_sdf_batch(verts, faces, cmaps, vis, points):

    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    Bsize = points.shape[0]

    normals = Meshes(verts, faces).verts_normals_padded()

    # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
    # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
    # 2. fill mouth holes with 30 more faces

    if verts.shape[1] == 10475:
        faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
        mouth_faces = torch.as_tensor(
            SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(Bsize, 1,
                                                         1).to(faces.device)
        faces = torch.cat([faces, mouth_faces], dim=1)

    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)
    cmaps = face_vertices(cmaps, faces)
    vis = face_vertices(vis, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                       3)).view(-1, 3, 3)
    closest_normals = torch.gather(
        normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                     3)).view(-1, 3, 3)
    closest_cmaps = torch.gather(
        cmaps, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                   3)).view(-1, 3, 3)
    closest_vis = torch.gather(vis, 1, pts_ind[:, :, None,
                                               None].expand(-1, -1, 3,
                                                            1)).view(-1, 3, 1)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles)

    pts_cmap = (closest_cmaps * bary_weights[:, :, None]).sum(1).unsqueeze(0)
    pts_vis = (closest_vis *
               bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1)
    pts_norm = (closest_normals *
                bary_weights[:, :, None]).sum(1).unsqueeze(0) * torch.tensor(
                    [-1.0, 1.0, -1.0]).type_as(normals)
    pts_norm = F.normalize(pts_norm, dim=2)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1,
                        1), pts_norm.view(Bsize, -1, 3), pts_cmap.view(
                            Bsize, -1, 3), pts_vis.view(Bsize, -1, 1)


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def load_obj_mesh_for_Hoppe(mesh_file):
    vertex_data = []
    face_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    normals, _ = compute_normal(vertices, faces)

    return vertices, normals, faces


def load_obj_mesh_with_color(mesh_file):
    vertex_data = []
    color_data = []
    face_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            c = list(map(float, values[4:7]))
            color_data.append(c)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

    vertices = np.array(vertex_data)
    colors = np.array(color_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    return vertices, colors, faces


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[1]),
                            [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[2]),
                            [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data)
        face_uvs[face_uvs > 0] -= 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms, _ = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data)
            face_normals[face_normals > 0] -= 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    vert_norms = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    face_norms = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(face_norms)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    vert_norms[faces[:, 0]] += face_norms
    vert_norms[faces[:, 1]] += face_norms
    vert_norms[faces[:, 2]] += face_norms
    normalize_v3(vert_norms)

    return vert_norms, face_norms


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                   (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def calculate_mIoU(outputs, labels):

    SMOOTH = 1e-6

    outputs = outputs.int()
    labels = labels.int()

    intersection = (
        outputs
        & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH
                                     )  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(
        20 * (iou - 0.5), 0,
        10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean().detach().cpu().numpy(
    )  # Or thresholded.mean() if you are interested in average across the batch


def mask_filter(mask, number=1000):
    """only keep {number} True items within a mask

    Args:
        mask (bool array): [N, ]
        number (int, optional): total True item. Defaults to 1000.
    """
    true_ids = np.where(mask)[0]
    keep_ids = np.random.choice(true_ids, size=number)
    filter_mask = np.isin(np.arange(len(mask)), keep_ids)

    return filter_mask


def query_mesh(path):

    verts, faces_idx, _ = load_obj(path)

    return verts, faces_idx.verts_idx


def add_alpha(colors, alpha=0.7):

    colors_pad = np.pad(colors, ((0, 0), (0, 1)),
                        mode='constant',
                        constant_values=alpha)

    return colors_pad


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type='smpl'):

    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0),
                                           nrow=nrow)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 *
         255.0).astype(np.uint8))

    # add text
    draw = ImageDraw.Draw(grid_img)
    grid_size = 512
    if loss is not None:
        draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)

    if type == 'smpl':
        for col_id, col_txt in enumerate([
                'image', 'smpl-norm(render)', 'cloth-norm(pred)', 'diff-norm',
                'diff-mask'
        ]):
            draw.text((10 + (col_id * grid_size), 5),
                      col_txt, (255, 0, 0),
                      font=font)
    elif type == 'cloth':
        for col_id, col_txt in enumerate(
            ['cloth-norm(recon)']):
            draw.text((10 + (col_id * grid_size), 5),
                      col_txt, (255, 0, 0),
                      font=font)
        for col_id, col_txt in enumerate(['0', '90', '180', '270']):
            draw.text((10 + (col_id * grid_size), grid_size * 2 + 5),
                      col_txt, (255, 0, 0),
                      font=font)
    else:
        print(f"{type} should be 'smpl' or 'cloth'")

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]),
                               Image.LANCZOS)

    return grid_img


def clean_mesh(verts, faces):

    device = verts.device

    mesh_lst = trimesh.Trimesh(verts.detach().cpu().numpy(),
                               faces.detach().cpu().numpy())
    mesh_lst = mesh_lst.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh_clean = mesh_lst[comp_num.index(max(comp_num))]

    final_verts = torch.as_tensor(mesh_clean.vertices).float().to(device)
    final_faces = torch.as_tensor(mesh_clean.faces).int().to(device)

    return final_verts, final_faces


def merge_mesh(verts_A, faces_A, verts_B, faces_B, color=False):

    sep_mesh = trimesh.Trimesh(np.concatenate([verts_A, verts_B], axis=0),
                               np.concatenate(
                                   [faces_A, faces_B + faces_A.max() + 1],
                                   axis=0),
                               maintain_order=True,
                               process=False)
    if color:
        colors = np.ones_like(sep_mesh.vertices)
        colors[:verts_A.shape[0]] *= np.array([255.0, 0.0, 0.0])
        colors[verts_A.shape[0]:] *= np.array([0.0, 255.0, 0.0])
        sep_mesh.visual.vertex_colors = colors

    # union_mesh = trimesh.boolean.union([trimesh.Trimesh(verts_A, faces_A),
    #                                     trimesh.Trimesh(verts_B, faces_B)], engine='blender')

    return sep_mesh


def mesh_move(mesh_lst, step, scale=1.0):

    trans = np.array([1.0, 0.0, 0.0]) * step

    resize_matrix = trimesh.transformations.scale_and_translate(
        scale=(scale), translate=trans)

    results = []

    for mesh in mesh_lst:
        mesh.apply_transform(resize_matrix)
        results.append(mesh)

    return results


def rescale_smpl(fitted_path, scale=100, translate=(0, 0, 0)):

    fitted_body = trimesh.load(fitted_path,
                               process=False,
                               maintain_order=True,
                               skip_materials=True)
    resize_matrix = trimesh.transformations.scale_and_translate(
        scale=(scale), translate=translate)

    fitted_body.apply_transform(resize_matrix)

    return np.array(fitted_body.vertices)


class SMPLX():

    def __init__(self):

        self.current_dir = "smpl_related"  # new smplx file in ECON folder

        self.smpl_verts_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_verts.npy")
        self.smpl_faces_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_faces.npy")
        self.smplx_verts_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_verts.npy")
        self.smplx_faces_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir,
                                       "smpl_data/smplx_cmap.npy")

        self.smplx_to_smplx_path = osp.join(self.current_dir,
                                            "smpl_data/smplx_to_smpl.pkl")

        self.smplx_eyeball_fid = osp.join(self.current_dir,
                                          "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid = osp.join(self.current_dir,
                                             "smpl_data/fill_mouth_fid.npy")

        self.smplx_faces = np.load(self.smplx_faces_path)
        self.smplx_verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)
        self.smpl_faces = np.load(self.smpl_faces_path)

        self.smplx_eyeball_fid_mask = np.load(self.smplx_eyeball_fid)
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid)

        self.smplx_to_smpl = cPickle.load(open(self.smplx_to_smplx_path, 'rb'))

        self.model_dir = osp.join(self.current_dir, "models")
        # self.tedra_dir = osp.join(self.current_dir, "../tedra_data")



        # copy from econ
        self.smplx_flame_vid_path = osp.join(
            self.current_dir, "smpl_data/FLAME_SMPLX_vertex_ids.npy"
        )
        self.smplx_mano_vid_path = osp.join(self.current_dir, "smpl_data/MANO_SMPLX_vertex_ids.pkl")
        self.smpl_vert_seg_path = osp.join(
            self.current_dir, "smpl_vert_segmentation.json"
        )
        self.front_flame_path = osp.join(self.current_dir, "smpl_data/FLAME_face_mask_ids.npy")
        self.smplx_vertex_lmkid_path = osp.join(
            self.current_dir, "smpl_data/smplx_vertex_lmkid.npy"
        )

        self.smplx_vertex_lmkid = np.load(self.smplx_vertex_lmkid_path)
        self.smpl_vert_seg = json.load(open(self.smpl_vert_seg_path))
        self.smpl_mano_vid = np.concatenate(
            [
                self.smpl_vert_seg["rightHand"], self.smpl_vert_seg["rightHandIndex1"],
                self.smpl_vert_seg["leftHand"], self.smpl_vert_seg["leftHandIndex1"]
            ]
        )

        self.smplx_mano_vid_dict = np.load(self.smplx_mano_vid_path, allow_pickle=True)
        self.smplx_mano_vid = np.concatenate(
            [self.smplx_mano_vid_dict["left_hand"], self.smplx_mano_vid_dict["right_hand"]]
        )
        self.smplx_flame_vid = np.load(self.smplx_flame_vid_path, allow_pickle=True)
        self.smplx_front_flame_vid = self.smplx_flame_vid[np.load(self.front_flame_path)]


        # hands
        self.smplx_mano_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_mano_vid), 1.0
        )
        self.smpl_mano_vertex_mask = torch.zeros(self.smpl_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smpl_mano_vid), 1.0
        )

         # face
        self.front_flame_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_front_flame_vid), 1.0
        )
        self.eyeball_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_faces[self.smplx_eyeball_fid_mask].flatten()), 1.0
        )


        self.ghum_smpl_pairs = torch.tensor(
            [
                (0, 24), (2, 26), (5, 25), (7, 28), (8, 27), (11, 16), (12, 17), (13, 18), (14, 19),
                (15, 20), (16, 21), (17, 39), (18, 44), (19, 36), (20, 41), (21, 35), (22, 40),
                (23, 1), (24, 2), (25, 4), (26, 5), (27, 7), (28, 8), (29, 31), (30, 34), (31, 29),
                (32, 32)
            ]
        ).long()

        # smpl-smplx correspondence
        self.smpl_joint_ids_24 = np.arange(22).tolist() + [68, 73]
        self.smpl_joint_ids_24_pixie = np.arange(22).tolist() + [61 + 68, 72 + 68]
        self.smpl_joint_ids_45 = np.arange(22).tolist() + [68, 73] + np.arange(55, 76).tolist()

        self.extra_joint_ids = np.array(
            [
                61, 72, 66, 69, 58, 68, 57, 56, 64, 59, 67, 75, 70, 65, 60, 61, 63, 62, 76, 71, 72,
                74, 73
            ]
        )

        self.extra_joint_ids += 68

        self.smpl_joint_ids_45_pixie = (np.arange(22).tolist() + self.extra_joint_ids.tolist())


    def cmap_smpl_vids(self, type):

        # keys:
        # closest_faces -   [6890, 3] with smplx vert_idx
        # bc            -   [6890, 3] with barycentric weights

        cmap_smplx = torch.as_tensor(np.load(self.cmap_vert_path)).float()
        if type == 'smplx':
            return cmap_smplx
        elif type == 'smpl':
            bc = torch.as_tensor(self.smplx_to_smpl['bc'].astype(np.float32))
            closest_faces = self.smplx_to_smpl['closest_faces'].astype(
                np.int32)

            cmap_smpl = torch.einsum('bij, bi->bj', cmap_smplx[closest_faces],
                                     bc)

            return cmap_smpl



# copy from ECON

def apply_face_mask(mesh, face_mask):

    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def apply_vertex_mask(mesh, vertex_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1)
    mesh = apply_face_mask(mesh, faces_mask)

    return mesh


def apply_vertex_face_mask(mesh, vertex_mask, face_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1) * torch.tensor(face_mask)
    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def clean_floats(mesh):
    thres = mesh.vertices.shape[0] * 1e-2
    mesh_lst = mesh.split(only_watertight=False)
    clean_mesh_lst = [mesh for mesh in mesh_lst if mesh.vertices.shape[0] > thres]
    return sum(clean_mesh_lst)

def isin(input, test_elements):
    # 扩展输入和测试元素的维度以进行广播
    input = input.unsqueeze(-1)
    test_elements = test_elements.unsqueeze(0)
    
    # 比较两个张量的元素
    comparison_result = torch.eq(input, test_elements)
    
    # 沿着新添加的维度进行求和，以检查每个输入元素是否在测试元素中
    isin_result = comparison_result.sum(-1).bool()
    
    return isin_result



def part_removal(full_mesh, part_mesh, thres, device, smpl_obj, region, clean=True):

    smpl_tree = cKDTree(smpl_obj.vertices)
    SMPL_container = SMPLX()

    from lib.dataset.PointFeat import ECON_PointFeat

    part_extractor = ECON_PointFeat(
        torch.tensor(part_mesh.vertices).unsqueeze(0).to(device),
        torch.tensor(part_mesh.faces).unsqueeze(0).to(device)
    )

    (part_dist, _) = part_extractor.query(torch.tensor(full_mesh.vertices).unsqueeze(0).to(device))

    remove_mask = part_dist < thres

    if region == "hand":
        _, idx = smpl_tree.query(full_mesh.vertices, k=1)
        full_lmkid = SMPL_container.smplx_vertex_lmkid[idx]
        remove_mask = torch.logical_and(
            remove_mask,
            torch.tensor(full_lmkid >= 20).type_as(remove_mask).unsqueeze(0)
        )

    elif region == "face":
        _, idx = smpl_tree.query(full_mesh.vertices, k=5)
        face_space_mask = isin(
            torch.tensor(idx), torch.tensor(SMPL_container.smplx_front_flame_vid)
        )
        remove_mask = torch.logical_and(
            remove_mask,
            face_space_mask.any(dim=1).type_as(remove_mask).unsqueeze(0)
        )

    BNI_part_mask = ~(remove_mask).flatten()[full_mesh.faces].any(dim=1)
    full_mesh.update_faces(BNI_part_mask.detach().cpu())
    full_mesh.remove_unreferenced_vertices()

    if clean:
        full_mesh = clean_floats(full_mesh)

    return full_mesh

#: A Poisson component this large relative to the body is anatomy, not noise. Measured on
#: alena-bikini, whose two hands come back as separate components after Poisson pinches the
#: wrists: body 143,504 verts, hands 2,596 (1.81%) and 2,512 (1.75%), and the only other
#: components are hair specks at 132, 90 and 47 verts (0.09%, 0.06%, 0.03%). A 1% threshold
#: separates those two populations by a factor of twenty.
POISSON_KEEP_COMPONENT_MIN_FRACTION = 0.01

#: ...AND it must be ADJACENT to the body, as a fraction of the body's longest axis. A hand
#: that Poisson pinched off at the wrist is still touching: measured 0.0103 and 0.0460 on
#: alena-bikini's two hands. Genuinely detached debris is nowhere near: on a badly fragmented
#: reconstruction the eight largest stray components sit at 0.138 to 0.706. Size alone is NOT
#: sufficient -- those strays are 11% to 98% of the body by vertex count, so a size-only rule
#: readmits exactly the bridge artifacts that got PSHUMAN_POISSON_MERGE_COMPONENTS disabled.
POISSON_KEEP_COMPONENT_MAX_GAP_FRACTION = 0.05


def keep_largest(
    mesh,
    min_fraction: float = 0.0,
    max_gap_fraction: float = POISSON_KEEP_COMPONENT_MAX_GAP_FRACTION,
):
    """Keep the body component; optionally also keep adjacent body-part-sized components.

    DEFAULT IS LARGEST-ONLY, AGAIN. Keeping the pinched-off hands (pass
    `min_fraction=POISSON_KEEP_COMPONENT_MIN_FRACTION`) restores arm SPAN, but what it keeps
    is a pancake at the wrong depth: on alena-bikini the kept hand sat 0.052 deeper than its
    arm stump, 0.023 thick by 0.123 wide, floating 0.045 from the body. Concatenating it is
    not fusing it. The arm is now re-attached BEFORE Poisson by `rebuild_arms_from_front_mask`
    (called from `poisson()` when reconstruct.py passes the front mask), so it comes out as one
    connected component and there is nothing to keep. The option remains for diagnostics.

    WHY THIS IS NO LONGER LARGEST-ONLY. Poisson reconstructs from a POINT CLOUD -- `poisson()`
    exports the mesh and reads it back with `o3d.io.read_point_cloud`, discarding connectivity
    -- and it pinches thin structures. On a subject with arms held out, it pinches both wrists,
    so each hand comes back as its own connected component. Taking only the largest then
    DELETES BOTH HANDS.

    Measured on alena-bikini, arm span as width/height and hand depth/width, against a Meshy
    ground truth of 0.725 / 0.986:

        SMPL-X init                 0.745 / 0.927
        after MeshOptimizer loop    0.692 / 0.716
        Poisson raw output          0.690 / 0.751     <- arms still here
        after largest-only          0.515 / 0.482     <- hands deleted

    Poisson is not the problem and the depth is not the problem: 10, 11 and 12 give identical
    results. One line was removing 26% of the arm span.

    The discarded components are unmistakably hands -- 2,596 and 2,512 verts at lateral offset
    +0.301 and -0.294, both at 0.59-0.60 of body height, which is where an A-pose hand is.

    THIS IS NOT `PSHUMAN_POISSON_MERGE_COMPONENTS`, which is permanently disabled in this fork
    because merging EVERY component stitches hair specks, foreground scratches and clothing
    fragments into the body. The distinction is size: hands are 1.8% of the body, the specks
    are 0.09% and below. This keeps the first population and still rejects the second, and it
    welds nothing -- components are concatenated, not bridged.
    """
    mesh_lst = mesh.split(only_watertight=False)
    if not len(mesh_lst):
        return mesh
    keep_mesh = mesh_lst[0]
    for candidate in mesh_lst:
        if candidate.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = candidate
    if min_fraction <= 0.0:
        return keep_mesh

    import numpy as _np
    import trimesh as _trimesh

    body = _np.asarray(keep_mesh.vertices)
    scale = float((body.max(axis=0) - body.min(axis=0)).max())
    if scale <= 0.0:
        return keep_mesh
    threshold = keep_mesh.vertices.shape[0] * float(min_fraction)
    max_gap = scale * float(max_gap_fraction)
    try:
        from scipy.spatial import cKDTree as _KDTree

        tree = _KDTree(body)
    except Exception:
        tree = None

    parts = [keep_mesh]
    for part in mesh_lst:
        if part is keep_mesh or part.vertices.shape[0] < threshold:
            continue
        if tree is not None:
            gap = float(tree.query(_np.asarray(part.vertices))[0].min())
            if gap > max_gap:
                # Big, but detached: debris, not a pinched-off body part.
                continue
        parts.append(part)
    if len(parts) <= 1:
        return keep_mesh
    return _trimesh.util.concatenate(parts)


def _orthonormal_basis(axis):
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(axis, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis, helper)
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-9)
    return u, v


def _ordered_ring(ids, verts, center, axis, ring_n):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0:
        return np.array([], dtype=np.int64)
    u, v = _orthonormal_basis(axis)
    rel = verts[ids] - center[None, :]
    ang = np.arctan2(rel.dot(v), rel.dot(u))
    order = np.argsort(ang)
    ordered = ids[order]
    if len(ordered) <= ring_n:
        return ordered
    pick = np.linspace(0, len(ordered) - 1, ring_n).astype(np.int64)
    return ordered[pick]


def _stitch_components(main_mesh, comp_mesh, ring_vertices=48):
    main_v = np.asarray(main_mesh.vertices)
    comp_v = np.asarray(comp_mesh.vertices)
    main_tree = cKDTree(main_v)
    dists, nn = main_tree.query(comp_v, k=1)
    min_idx = int(np.argmin(dists))
    gap = float(dists[min_idx])

    p_comp = comp_v[min_idx]
    p_main = main_v[int(nn[min_idx])]
    axis = p_comp - p_main
    if np.linalg.norm(axis) < 1e-6:
        axis = comp_v.mean(axis=0) - main_v.mean(axis=0)
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    main_tree_local = cKDTree(main_v)
    comp_tree_local = cKDTree(comp_v)
    base_radius = max(0.02, min(0.12, 1.5 * gap + 0.01))
    main_seed = np.array([], dtype=np.int64)
    comp_seed = np.array([], dtype=np.int64)

    for mul in [1.0, 1.5, 2.0, 2.5, 3.0]:
        r = base_radius * mul
        m_ids = np.asarray(main_tree_local.query_ball_point(p_main, r), dtype=np.int64)
        c_ids = np.asarray(comp_tree_local.query_ball_point(p_comp, r), dtype=np.int64)
        if len(m_ids) == 0 or len(c_ids) == 0:
            continue
        m_keep = np.abs((main_v[m_ids] - p_main[None, :]).dot(axis)) < (0.45 * r)
        c_keep = np.abs((comp_v[c_ids] - p_comp[None, :]).dot(axis)) < (0.45 * r)
        m_ids = m_ids[m_keep]
        c_ids = c_ids[c_keep]
        if len(m_ids) >= 12 and len(c_ids) >= 12:
            main_seed = m_ids
            comp_seed = c_ids
            break

    if len(main_seed) < 12 or len(comp_seed) < 12:
        k = int(min(max(ring_vertices * 6, 72), len(main_v), len(comp_v)))
        _, m_knn = main_tree_local.query(p_main, k=k)
        _, c_knn = comp_tree_local.query(p_comp, k=k)
        main_seed = np.asarray(m_knn, dtype=np.int64).reshape(-1)
        comp_seed = np.asarray(c_knn, dtype=np.int64).reshape(-1)

    ring_n = int(min(ring_vertices, len(main_seed), len(comp_seed)))
    if ring_n < 12:
        return trimesh.util.concatenate([main_mesh, comp_mesh]), False, gap

    main_ring = _ordered_ring(main_seed, main_v, main_v[main_seed].mean(axis=0), axis, ring_n)
    comp_ring = _ordered_ring(comp_seed, comp_v, comp_v[comp_seed].mean(axis=0), axis, ring_n)
    if len(main_ring) != len(comp_ring) or len(main_ring) < 12:
        return trimesh.util.concatenate([main_mesh, comp_mesh]), False, gap

    m_pts = main_v[main_ring]
    c_pts = comp_v[comp_ring]
    shift_cost = []
    for s in range(len(comp_ring)):
        shift_cost.append(np.mean(np.linalg.norm(m_pts - np.roll(c_pts, s, axis=0), axis=1)))
    comp_ring = np.roll(comp_ring, int(np.argmin(shift_cost)))

    main_f = np.asarray(main_mesh.faces, dtype=np.int64)
    comp_f = np.asarray(comp_mesh.faces, dtype=np.int64)
    v_out = np.vstack([main_v, comp_v])
    off = len(main_v)
    bridge = []
    for i in range(len(main_ring)):
        j = (i + 1) % len(main_ring)
        m0 = int(main_ring[i])
        m1 = int(main_ring[j])
        c0 = int(off + comp_ring[i])
        c1 = int(off + comp_ring[j])
        bridge.append([m0, c0, m1])
        bridge.append([m1, c0, c1])

    f_out = np.vstack([main_f, comp_f + off, np.asarray(bridge, dtype=np.int64)])
    stitched = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    return stitched, True, gap


def merge_close_components(mesh, max_gap=0.28, min_faces=1200, ring_vertices=48):
    comps = sorted(mesh.split(only_watertight=False), key=lambda x: len(x.faces), reverse=True)
    if len(comps) <= 1:
        return comps[0] if len(comps) == 1 else mesh

    main = comps[0]
    passthrough = []
    merged = 0
    for comp in comps[1:]:
        if len(comp.faces) < min_faces:
            passthrough.append(comp)
            continue
        d, _ = cKDTree(main.vertices).query(comp.vertices, k=1)
        gap = float(np.min(d))
        if gap > max_gap:
            passthrough.append(comp)
            continue
        stitched, ok, used_gap = _stitch_components(main, comp, ring_vertices=ring_vertices)
        if ok:
            main = stitched
            merged += 1
            print(f"[poisson-merge] stitched comp faces={len(comp.faces)} gap={used_gap:.4f} m")
        else:
            passthrough.append(comp)
            print(f"[poisson-merge] stitch-fallback keep comp faces={len(comp.faces)} gap={used_gap:.4f} m")

    if merged == 0:
        return keep_largest(mesh)
    if len(passthrough) == 0:
        return main
    return trimesh.util.concatenate([main] + passthrough)


def _poisson_raw(mesh, depth):
    """Open3D Poisson on a mesh's oriented vertices, without any component filtering."""
    import tempfile

    mesh = mesh.copy()
    _ = mesh.vertex_normals
    with tempfile.TemporaryDirectory() as tmp:
        ply = os.path.join(tmp, "soups.ply")
        mesh.export(ply)
        pcl = o3d.io.read_point_cloud(ply)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=depth, n_threads=6)
    return trimesh.Trimesh(np.array(out.vertices), np.array(out.triangles), process=False)


#: Below this fraction of the body a component is a speck (hair strands measured at 0.03-0.09%)
#: and is left to be dropped.
ATTRACT_MIN_PART_FRACTION = 0.002
#: Body vertices consulted, nearest in the frontal (X,Y) plane, to find the depth a part belongs at.
ATTRACT_NEIGHBOURS = 64
#: A part whose closest points are within this fraction of body extent already touches.
ATTRACT_TOUCH_FRACTION = 0.004


#: Radius of the healed band around each join, as a fraction of the body's longest extent.
HEAL_JOIN_RADIUS_FRACTION = 0.03
#: Taubin smoothing: shrink-free (lambda > 0, mu < -lambda), so a heal cannot thin the limb.
HEAL_TAUBIN_LAMBDA = 0.5
HEAL_TAUBIN_MU = -0.53
HEAL_TAUBIN_ITERATIONS = 10


def heal_joins(mesh, contact_points, radius_fraction=HEAL_JOIN_RADIUS_FRACTION):
    """Restore a natural surface across the welds where displaced parts were re-fused.

    Poisson welds a moved part to its stump at a narrow contact, which can leave a pinch or a
    crease. This smooths ONLY a band of `radius_fraction` around those contacts, with Taubin
    (alternating lambda/mu) rather than plain Laplacian so the band is not shrunk -- plain
    Laplacian thins exactly the limbs this is meant to restore. Vertices outside the band are
    held fixed by construction: their rows of the operator are the identity.
    """
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    scale = float((verts.max(axis=0) - verts.min(axis=0)).max())
    if scale <= 0 or len(contact_points) == 0:
        return mesh
    dist, _ = cKDTree(np.asarray(contact_points)).query(verts)
    band = dist <= radius_fraction * scale
    if not band.any():
        return mesh

    lap = trimesh.smoothing.laplacian_calculation(mesh)          # row-normalised neighbour mean
    n = len(verts)
    keep = sp.diags((~band).astype(np.float64))
    move = sp.diags(band.astype(np.float64))
    eye = sp.identity(n, format="csr")
    v = verts.copy()
    for _ in range(int(HEAL_TAUBIN_ITERATIONS)):
        for weight in (HEAL_TAUBIN_LAMBDA, HEAL_TAUBIN_MU):
            delta = (lap @ v) - v
            v = v + move @ (weight * delta)                       # only band vertices move
            v = keep @ verts + move @ v                           # and the rest stay exactly put
    healed = mesh.copy()
    healed.vertices = v
    return healed


def attract_parts_to_body_plane(raw_mesh, depth=10, passes=2):
    """Move displaced body parts back onto the body's depth plane, then re-fuse. Do not drop them.

    A HUMAN IS ONE CONNECTED SURFACE. When Poisson returns a hand or forearm as its own
    component, that piece is not noise -- it is the subject's arm, reconstructed at the wrong
    DEPTH, so it no longer touches the stump. Measured on alena-bikini the detached arm chunks
    sat at dz -0.131, -0.195, -0.249 (mesh units, body ~2 tall) from where the body is behind
    them, and the hand that was kept separately sat 0.052 off its own stump.

    For an A-pose subject the body is close to planar in depth, so each chunk is translated
    along Z only -- its X,Y are what the views agree on -- to the median depth of the body
    surface it projects onto, then the whole set is re-run through Poisson so the contacts fuse.
    Specks below `ATTRACT_MIN_PART_FRACTION` are left for `keep_largest` to drop.

    On alena-bikini: Poisson alone gave 30 components and an arm span of 0.488 (width/height);
    after attraction and re-fusion it is one body with span 0.692, against the reconstruction's
    own 0.694 before Poisson and a Meshy ground truth of 0.725.

    SUPERSEDED as the primary fix by `rebuild_arms_from_front_mask`, which runs before Poisson.
    This pass is the fallback for callers without a front mask. It is not sufficient on its own:
    on alena-bikini the moved right forearm re-separated after the second Poisson (gap 0.011) and
    `keep_largest` dropped it.

    PSHUMAN_POISSON_ATTRACT_PARTS=0 disables it.
    """
    from scipy.spatial import cKDTree

    mesh = raw_mesh
    contacts = []
    for _ in range(int(passes)):
        parts = sorted(mesh.split(only_watertight=False), key=lambda m: -len(m.vertices))
        if len(parts) <= 1:
            break
        body = parts[0]
        body_v = np.asarray(body.vertices)
        tree = cKDTree(body_v[:, :2])
        moved = [body]
        changed = False
        for part in parts[1:]:
            if len(part.vertices) < ATTRACT_MIN_PART_FRACTION * len(body_v):
                continue
            pv = np.asarray(part.vertices).copy()
            k = min(ATTRACT_NEIGHBOURS, len(body_v))
            _, idx = tree.query(pv[:, :2], k=k)
            dz = float(np.median(body_v[np.asarray(idx).ravel(), 2]) - np.median(pv[:, 2]))
            pv[:, 2] += dz
            # ATTRACT TO CONTACT, not only to the plane. Depth alignment is enough when the
            # break was a depth error, but where Poisson pinched a thin limb there is also a
            # lateral gap, and a part moved only in Z still floats. Measured on alena-bikini:
            # depth alignment re-fused the left forearm (reach 68% -> 91% of the SMPL-X arm) but
            # left the right one detached (71%). So translate the part rigidly by the mean offset
            # of its closest point pairs to the body, which closes the gap to contact while
            # preserving the part's own shape.
            body_tree = cKDTree(body_v)
            gap, nearest = body_tree.query(pv)
            near = np.argsort(gap)[: max(8, len(pv) // 50)]
            touch = ATTRACT_TOUCH_FRACTION * float((body_v.max(axis=0) - body_v.min(axis=0)).max())
            if float(gap[near].mean()) > touch:
                pv += (body_v[nearest[near]] - pv[near]).mean(axis=0)
                gap, _ = body_tree.query(pv)
                near = np.argsort(gap)[: max(8, len(pv) // 20)]
            # Where this part meets the body after moving: its points closest to the body,
            # in 3-D. That is where the weld will form and where the heal is applied.
            contacts.append(pv[near])
            shifted = part.copy()
            shifted.vertices = pv
            moved.append(shifted)
            changed = True
        if not changed:
            break
        mesh = _poisson_raw(trimesh.util.concatenate(moved), depth)
    if contacts and os.environ.get("PSHUMAN_POISSON_HEAL_JOINS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        mesh = heal_joins(mesh, np.concatenate(contacts))
    return mesh


#: Arm stations are this many front-mask pixels apart along the arm.
ARM_STATION_STEP_PX = 2.0
#: Mesh vertices within this many pixels outside the mask half-width still belong to a station.
ARM_BAND_MARGIN_PX = 3.0
#: Depth statistics at a station pool the vertices of this many neighbouring stations each side.
ARM_WINDOW_STATIONS = 2
#: Arm seed: front-mask pixels beyond this fraction of the side's lateral extent are arm, not torso.
ARM_SEED_LATERAL_FRACTION = 0.6
#: Walking inward, a cross-section this many times wider than the recent median has hit the torso.
ARM_TORSO_MERGE_RATIO = 2.2
#: Two depth groups at one station are separate surfaces when this far apart, as a fraction of the
#: local half-width (floored at ARM_DEPTH_GAP_SCALE_FRACTION of the body extent).
ARM_DEPTH_GAP_FRACTION = 0.3
ARM_DEPTH_GAP_SCALE_FRACTION = 0.006
#: A depth-centre jump between neighbouring stations above this fraction of the half-width is a step.
ARM_DEPTH_JUMP_FRACTION = 0.6
#: Thinner than this fraction of the arm's own healthy depth/width ratio is a strangled neck.
ARM_NECK_FRACTION = 0.35
#: Healthy stations the rebuilt tube overlaps before the first defect, so the two blend in Poisson.
ARM_RING_OVERLAP_STATIONS = 6
#: Defects are searched for on the ARM only: the first this fraction of the stations from the
#: armpit. The hand is the distal ~25% and naturally shows several surfaces (fingers, thumb);
#: on flighter that read as "2 surfaces" at station 98/115 and "depth jump" at 114/115, healthy
#: hands were flagged and 425 hand vertices moved. alena-bikini's real breaks were at 46-54%.
ARM_DEFECT_SEARCH_FRACTION = 0.75
#: Healthy stations the depth line is fitted over, to extrapolate across the defect.
ARM_DEPTH_FIT_STATIONS = 30
#: Rings per station. Kept at ONE: interpolating three per station with a half-point stagger
#: between successive rings turned the arm into a twisted spiral in Poisson (alena-bikini,
#: 2026-10-02 02:41), visibly worse than the faint banding of one ring per station.
ARM_RING_SUBSTEPS = 1


def _front_mask_bool(front_mask):
    m = front_mask
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    return m > (127 if m.dtype == np.uint8 else 0.5)


def _arm_stations(mask, ortho_scale, side, cx):
    """Walk one arm of the FRONT silhouette from fingertip to armpit.

    Returns origin, direction, normal (2-D, mesh units) and a (S, 3) array of stations
    (t along the arm, w centre across it, r half-width), ordered proximal -> distal. Every
    quantity here is read off the front mask, which is the ground truth for where the arm is.
    """
    H, W = mask.shape
    px = 2.0 * ortho_scale / W
    ys, xs = np.nonzero(mask)
    X = (xs + 0.5) * px - ortho_scale
    Y = ortho_scale - (ys + 0.5) * px
    lat = side * (X - cx)
    if lat.size == 0 or lat.max() <= 0:
        return None
    seed = lat > ARM_SEED_LATERAL_FRACTION * lat.max()
    if seed.sum() < 50:
        return None
    pts = np.c_[X[seed], Y[seed]]
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    d = vt[0]
    if side * d[0] < 0:
        d = -d
    n = np.array([-d[1], d[0]])
    step = ARM_STATION_STEP_PX * px
    wgrid = np.arange(-0.6, 0.6, px)
    t = float(((pts - origin) @ d).max())
    t_min = float(((np.c_[X, Y] - origin) @ d).min())
    stations, w_prev = [], None
    while t > t_min:
        q = origin[None, :] + t * d[None, :] + wgrid[:, None] * n[None, :]
        cols = ((q[:, 0] + ortho_scale) / px).astype(int)
        rows = ((ortho_scale - q[:, 1]) / px).astype(int)
        inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        on = np.zeros(len(wgrid), dtype=bool)
        on[inside] = mask[rows[inside], cols[inside]]
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
        runs = list(zip(edges[0::2], edges[1::2] - 1))
        if not runs:
            if stations:
                break
            t -= step
            continue
        ref = 0.0 if w_prev is None else w_prev
        a, b = min(runs, key=lambda r: abs(0.5 * (wgrid[r[0]] + wgrid[r[1]]) - ref))
        half = 0.5 * (wgrid[b] - wgrid[a] + px)
        if len(stations) > 15:
            recent = np.median([s[2] for s in stations[-15:]])
            if half > ARM_TORSO_MERGE_RATIO * recent or a == 0 or b == len(wgrid) - 1:
                break
        w_prev = 0.5 * (wgrid[a] + wgrid[b])
        stations.append((t, w_prev, half))
        t -= step
    if len(stations) < 20:
        return None
    return origin, d, n, np.asarray(stations[::-1]), step


#: Normal z-component that counts as facing the front (or, negated, the back) when deciding
#: whether a depth gap is empty space between two surfaces or just the sides of one tube.
ARM_FACING_NZ = 0.2


def _depth_groups(z, labels, gap, nz):
    """Split one station's vertex depths into separate surfaces: by component, then by depth gaps.

    A gap only separates two surfaces when it is OUTSIDE the body: the vertices just below it face
    the front (+z) and those just above face the back (-z). Inside one tube it is the other way
    round, and a sparse tube has gaps at its sides that are not a second surface.
    """
    groups = []
    for lab in np.unique(labels):
        sel = labels == lab
        order = np.argsort(z[sel])
        zl, nl = z[sel][order], nz[sel][order]
        cuts = []
        for i in np.flatnonzero(np.diff(zl) > gap):
            below, above = nl[max(0, i - 4): i + 1], nl[i + 1: i + 6]
            if below.mean() > ARM_FACING_NZ and above.mean() < -ARM_FACING_NZ:
                cuts.append(i + 1)
        for part in np.split(zl, cuts):
            lo, hi = np.percentile(part, 5), np.percentile(part, 95)
            groups.append({"label": int(lab), "n": len(part), "c": 0.5 * (lo + hi), "h": 0.5 * (hi - lo)})
    return groups


def rebuild_arms_from_front_mask(mesh, front_mask, ortho_scale):
    """Re-attach every arm piece at the depth the FRONT VIEW implies, and rebuild the tissue between.

    THE DEFECT. The MeshOptimizer returns arms whose distal part sits at a different depth from the
    upper arm. On alena-bikini the left forearm is its own component 0.15 in front of its stump,
    and the right forearm is 0.18 in front, joined to the upper arm only by a 0.019-wide sliver.
    Poisson cuts the sliver, and `keep_largest` then deleted the whole right forearm: the remeshed
    mesh had both arms, coarse.glb had one. Moving the piece afterwards and re-running Poisson did
    not hold either -- it re-separated at 0.011 and was dropped again.

    THE GROUND TRUTH. The front view is the one view every pipeline stage agrees on: rasterised
    from the front, this mesh matches the front mask at IoU 0.967, including both arms. So the arm
    is walked ALONG THE FRONT SILHOUETTE from the fingertip to the armpit, and every station's
    position and width come from the mask. Depth is the only thing taken from the mesh, and it is
    made continuous:

      1. Proximal to distal, the first station where the arm stops being one surface (a second
         component, a second depth layer, a depth jump, or a strangled neck) is the defect.
      2. Distal of it, the piece is tracked inward from the fingertip and SHIFTED IN DEPTH so its
         centre continues the upper arm's depth line -- a human is one connected surface, and an
         A-pose arm continues from its shoulder. The piece's own depth slope is kept.
      3. The tissue from the defect to the fingertip is rebuilt as elliptical rings: half-width
         from the front mask, half-depth from the moved piece, never thinner than the subject's
         own upper-arm depth/width ratio. Mesh vertices in that span are replaced, not kept beside
         the rings, so no pancake survives inside the tube.

    Nothing is dropped. A healthy arm has no defect and is returned untouched. The join where the
    rebuilt tube meets the upper arm is returned as `report["_join_points"]` for `heal_joins`.

    Returns (points, normals, report): the Poisson input.
    """
    import trimesh as _trimesh

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    vnormals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    labels = _trimesh.graph.connected_component_labels(mesh.edges_unique, node_count=len(verts))
    body_label = int(np.bincount(labels).argmax())
    scale = float((verts.max(axis=0) - verts.min(axis=0)).max())
    mask = _front_mask_bool(front_mask)
    px = 2.0 * float(ortho_scale) / mask.shape[1]
    cx = float(np.median(verts[:, 0]))

    keep = np.ones(len(verts), dtype=bool)
    verts_out = verts.copy()
    ring_pts, ring_nrm, joins = [], [], []
    report = {"contract": "pshuman.arm_frontal_rebuild.v1", "sides": {}}

    for side, name in ((1, "left"), (-1, "right")):
        walk = _arm_stations(mask, float(ortho_scale), side, cx)
        if walk is None:
            report["sides"][name] = {"status": "no_arm_in_front_mask"}
            continue
        origin, d, n, st, step = walk
        S = len(st)
        tv = (verts[:, :2] - origin) @ d
        wv = (verts[:, :2] - origin) @ n
        idx = np.rint((tv - st[0, 0]) / step).astype(np.int64)
        valid = (idx >= 0) & (idx < S)
        ci = np.clip(idx, 0, S - 1)
        band = valid & (np.abs(wv - st[ci, 1]) <= st[ci, 2] + ARM_BAND_MARGIN_PX * px)
        bi = np.flatnonzero(band)
        b_idx, b_z, b_lab, b_nz = idx[bi], verts[bi, 2], labels[bi], vnormals[bi, 2]

        def groups_at(s):
            m = np.abs(b_idx - s) <= ARM_WINDOW_STATIONS
            if not m.any():
                return []
            gap = max(ARM_DEPTH_GAP_FRACTION * st[s, 2], ARM_DEPTH_GAP_SCALE_FRACTION * scale)
            return _depth_groups(b_z[m], b_lab[m], gap, b_nz[m])

        # 1. proximal -> distal: healthy while one body surface, continuous, not strangled
        c = np.full(S, np.nan)
        h = np.full(S, np.nan)
        defect, reason = None, None
        k_ref = None
        search_end = int(ARM_DEFECT_SEARCH_FRACTION * S)
        for s in range(S):
            if s >= search_end:
                break
            g = groups_at(s)
            if not g:
                defect, reason = s, "empty"
                break
            if len(g) > 1 or g[0]["label"] != body_label:
                defect, reason = s, f"{len(g)} surfaces" if len(g) > 1 else "detached component"
                break
            if s > 0 and abs(g[0]["c"] - c[s - 1]) > ARM_DEPTH_JUMP_FRACTION * st[s, 2]:
                defect, reason = s, "depth jump"
                break
            if k_ref is None and s >= 20:
                k_ref = float(np.median(h[:s] / st[:s, 2]))
            if k_ref is not None and g[0]["h"] < ARM_NECK_FRACTION * k_ref * st[s, 2]:
                defect, reason = s, "strangled neck"
                break
            c[s], h[s] = g[0]["c"], g[0]["h"]
        if k_ref is None:
            healthy = np.isfinite(h[: max(defect or 0, 1)])
            k_ref = float(np.median(h[: max(defect or 0, 1)][healthy] / st[: max(defect or 0, 1), 2][healthy])) if healthy.any() else 0.8
        if defect is None:
            report["sides"][name] = {"status": "healthy", "stations": S, "depth_over_width": k_ref}
            continue
        if defect < 5:
            report["sides"][name] = {"status": "defect_at_armpit_not_rebuilt", "stations": S, "reason": reason}
            continue

        # 2. fingertip -> defect: track the distal piece, then shift it onto the upper arm's depth line
        cd = np.full(S, np.nan)
        hd = np.full(S, np.nan)
        prev = None
        for s in range(S - 1, defect - 1, -1):
            g = groups_at(s)
            if not g:
                continue
            pick = max(g, key=lambda q: q["n"]) if prev is None else min(g, key=lambda q: abs(q["c"] - prev))
            cd[s], hd[s], prev = pick["c"], pick["h"], pick["c"]
        dist = np.flatnonzero(np.isfinite(cd))
        if dist.size == 0:
            report["sides"][name] = {"status": "no_distal_surface", "stations": S, "defect_station": defect}
            continue
        fit = np.arange(max(0, defect - ARM_DEPTH_FIT_STATIONS), defect)
        fit = fit[np.isfinite(c[fit])]
        slope, icpt = np.polyfit(fit, c[fit], 1) if fit.size >= 3 else (0.0, float(np.nanmean(c[:defect])))
        first = dist[:10]
        offset = float(np.median(slope * first + icpt - cd[first]))
        all_s = np.arange(S)
        cd = np.interp(all_s, dist, cd[dist]) + offset
        hd = np.interp(all_s, dist, hd[dist])

        start = max(0, defect - ARM_RING_OVERLAP_STATIONS)
        cc = np.where(all_s < defect, c, cd)
        hh = np.where(all_s < defect, h, np.maximum(hd, k_ref * st[:, 2]))
        span = all_s >= start
        kern = np.ones(5) / 5.0
        for arr in (cc, hh):
            seg = np.pad(arr[span], 2, mode="edge")
            arr[span] = np.convolve(seg, kern, mode="valid")

        # 3. MOVE THE DISTAL PIECE, KEEP ITS OWN SURFACE. No tube is generated: an elliptical
        # ring rebuild here read as "circles" / a mummy wrap, and the arm's anatomical shape is
        # supplied later by the SMPL-X donor graft (`seed.graft.arm_graft`). PSHuman's job is only
        # to not lose the piece: each vertex past the defect that belongs to the distal surface
        # (nearer its depth track than the upper arm's line) is shifted by the offset.
        sel = bi[b_idx >= defect]
        si = np.clip(idx[sel], 0, S - 1)
        distal_track = cd[si] - offset
        upper_line = slope * si + icpt
        is_distal = np.abs(verts[sel, 2] - distal_track) < np.abs(verts[sel, 2] - upper_line)
        moved_ids = sel[is_distal]
        verts_out[moved_ids, 2] += offset
        removed = np.zeros(0, dtype=np.int64)
        n_ring = 0
        for s in range(max(0, defect - 4), min(S, defect + 5)):
            joins.append(np.r_[origin + st[s, 0] * d + st[s, 1] * n, cc[s]])

        moved_before = slope * defect + icpt - (cd[defect] - offset)
        report["sides"][name] = {
            "status": "rebuilt",
            "rebuilt_from_station": int(start),
            "reason": reason,
            "stations": S,
            "defect_station": int(defect),
            "defect_fraction_of_arm": round(defect / S, 3),
            "depth_offset_applied": round(offset, 4),
            "depth_step_closed": round(float(moved_before), 4),
            "depth_over_width_floor": round(k_ref, 3),
            "distal_depth_over_width_before": round(float(np.nanmedian(hd[defect:] / st[defect:, 2])), 3),
            "mesh_vertices_moved": int(moved_ids.size),
        }
        print(
            f"[arm-rebuild] {name}: {reason} at {defect}/{S} ({defect / S:.0%} of arm); distal piece moved "
            f"{offset:+.4f} in depth onto the upper arm ({moved_ids.size} verts, own surface kept, no tube)",
            flush=True,
        )

    report["_join_points"] = np.asarray(joins).reshape(-1, 3)
    points = verts_out[keep]
    normals = vnormals[keep]
    if ring_pts:
        points = np.vstack([points] + ring_pts)
        normals = np.vstack([normals] + ring_nrm)
    return points, normals, report


def _poisson_points(points, normals, depth):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    pcl.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=depth, n_threads=6)
    return trimesh.Trimesh(np.array(out.vertices), np.array(out.triangles), process=False)


def poisson(mesh, obj_path, depth=10, decimation=True, front_mask=None, ortho_scale=None, report_path=None):

    assert (mesh.vertex_normals.shape[1] == 3)
    arm_joins = None
    if front_mask is not None and ortho_scale is not None and os.environ.get(
        "PSHUMAN_ARM_FRONTAL_REBUILD", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}:
        # Arms are re-attached and their tissue rebuilt from the FRONT silhouette before the
        # single Poisson; see `rebuild_arms_from_front_mask`.
        points, normals, report = rebuild_arms_from_front_mask(mesh, front_mask, ortho_scale)
        arm_joins = report.pop("_join_points")
        if report_path:
            import json as _json

            with open(report_path, "w") as fh:
                _json.dump(report, fh, indent=2)
        raw_mesh = _poisson_points(points, normals, depth)
    else:
        pcd_path = obj_path[:-4] + "_soups.ply"
        mesh.export(pcd_path)
        pcl = o3d.io.read_point_cloud(pcd_path)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcl, depth=depth, n_threads=6
            )
        os.remove(pcd_path)
        raw_mesh = trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.triangles), process=False)
    # Permanently disabled in this fork.
    #
    # The close-component bridge pass was a local experiment, not upstream
    # PSHuman behavior. It can stitch semi-transparent foreground scratches,
    # arm-side noise, toe islands, and clothing/skin fragments into the body.
    # Preserve largest-component behavior for production output.
    if os.environ.get("PSHUMAN_POISSON_MERGE_COMPONENTS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        print("[poisson-merge] requested but permanently disabled because it creates bridge artifacts")
    if os.environ.get("PSHUMAN_POISSON_ATTRACT_PARTS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        raw_mesh = attract_parts_to_body_plane(raw_mesh, depth=depth)
    # After an arm re-attachment, an arm piece Poisson still left separate is KEPT (adjacent and
    # body-part sized), never dropped; the donor graft replaces the arm anyway.
    largest_mesh = keep_largest(
        raw_mesh, min_fraction=POISSON_KEEP_COMPONENT_MIN_FRACTION if arm_joins is not None else 0.0
    )
    if arm_joins is not None and len(arm_joins) and os.environ.get(
        "PSHUMAN_POISSON_HEAL_JOINS", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}:
        largest_mesh = heal_joins(largest_mesh, arm_joins)
    
    if decimation:
        # mesh decimation for faster rendering
        low_res_mesh = largest_mesh.simplify_quadratic_decimation(50000)
        return low_res_mesh
    else:
        return largest_mesh
    

    

"""Explicit, hash-bound, vertex-colored source-only projection; no reconstruction.

The manifest declares a shared PSHuman coordinate frame; hashes establish byte
identity, not that the declared cross-view registration is physically correct.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import Image

VIEW_NAMES = ('front_face', 'front_right', 'right', 'back', 'left', 'front_left')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bound(item):
    if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
        raise ValueError('each input must declare path and sha256')
    path = Path(item['path'])
    if not path.is_absolute() or digest(path) != item['sha256']:
        raise ValueError(f'input hash mismatch or nonabsolute path: {path}')
    return path


def validate_manifest(manifest):
    if manifest.get('schema') != 'pshuman.texture-only-source/1':
        raise ValueError('unsupported manifest schema')
    if manifest.get('coordinate_frame') != 'PSHuman-shared':
        raise ValueError('explicit PSHuman-shared coordinate frame is required')
    views = manifest.get('views')
    if not isinstance(views, list) or [v.get('name') for v in views] != list(VIEW_NAMES):
        raise ValueError('six ordered canonical views required')
    inputs = {'mesh': bound(manifest['mesh']), 'support': bound(manifest['support']),
              'camera': bound(manifest['camera'])}
    inputs['baseline'] = bound(manifest['baseline']) if manifest.get('baseline') else inputs['mesh']
    inputs['colors'] = [bound(v['color']) for v in views]
    inputs['normals'] = [bound(v['normal']) for v in views]
    camera = json.loads(inputs['camera'].read_text())
    if camera != {'schema': 'pshuman.canonical-camera/1', 'azimuth_degrees': [0,45,90,180,270,315], 'focal': 0.5, 'distance': 1.1}:
        raise ValueError('camera must specify the supported canonical projection exactly')
    return inputs, camera


def load_mesh(path):
    import trimesh
    scene = trimesh.load(path, force='scene', process=False, maintain_order=True)
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) != 1:
        raise ValueError('exactly one mesh instance is required')
    transform, geometry = scene.graph[nodes[0]]
    if not np.array_equal(transform, np.eye(4)):
        raise ValueError('mesh node transform must be identity in the declared shared frame')
    mesh = scene.geometry[geometry]
    if not isinstance(mesh, trimesh.Trimesh) or mesh.visual.kind != 'vertex':
        raise ValueError('only a single vertex-colored indexed mesh is supported; UV/material assets are not')
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('mesh geometry is empty or nonfinite')
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.shape != (len(mesh.vertices), 4) or colors.dtype != np.uint8 or np.any(colors[:, 3] != 255):
        raise ValueError('baseline requires opaque uint8 vertex colors')
    return mesh


def run(manifest_path, manifest_sha256, output, *, min_source_support=.95, resolution=1024, device='cuda'):
    import torch
    import trimesh
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from utils.project_mesh import get_cameras_list, multiview_color_projection
    from utils.source_color_sampling import validate_baseline_colors
    from utils.source_color_support import load_support, masked_images
    from utils.texture_raster_budget import source_raster_policy

    raster_policy = source_raster_policy(os.environ)
    raster_env = {'PSHUMAN_TEXTURE_RASTER_MODE': raster_policy['mode']}
    if raster_policy['bin_buffer_limit_bytes'] is not None:
        raster_env['PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES'] = str(raster_policy['bin_buffer_limit_bytes'])
    raster_env = MappingProxyType(raster_env)
    started = time.perf_counter()
    if Path(output).exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest_path = Path(manifest_path)
    if digest(manifest_path) != manifest_sha256:
        raise ValueError('manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    inputs, camera = validate_manifest(manifest)
    mesh, baseline = load_mesh(inputs['mesh']), load_mesh(inputs['baseline'])
    if not np.array_equal(np.asarray(mesh.vertices), np.asarray(baseline.vertices)) or not np.array_equal(np.asarray(mesh.faces), np.asarray(baseline.faces)):
        raise ValueError('baseline indexed geometry does not exactly match the input mesh')
    source_vertices = np.asarray(mesh.vertices)
    verts_np = np.asarray(source_vertices, dtype=np.float32)
    if not np.isfinite(verts_np).all() or not np.array_equal(verts_np.astype(source_vertices.dtype), source_vertices):
        raise ValueError('mesh coordinates must round-trip exactly through float32')
    faces_np = np.asarray(mesh.faces, dtype=np.int64)
    vertices = torch.as_tensor(verts_np, device=device)
    faces = torch.as_tensor(faces_np, device=device)
    baseline_colors = np.asarray(baseline.visual.vertex_colors)[:, :3].copy()
    baseline_data = {'vertices': torch.as_tensor(np.asarray(baseline.vertices, dtype=np.float32), device=device),
                     'faces': torch.as_tensor(np.asarray(baseline.faces, dtype=np.int64), device=device),
                     'colors': torch.as_tensor(baseline_colors, device=device, dtype=vertices.dtype) / 255.}
    colors = validate_baseline_colors(vertices, faces, baseline_data)
    views = [Image.open(p).convert('RGBA') for p in inputs['colors']]
    normals = [Image.open(p).convert('RGBA') for p in inputs['normals']]
    if any(color.size != normal.size for color, normal in zip(views, normals, strict=True)):
        raise ValueError('each normal image must match its paired color dimensions')
    observed, _inferred = load_support(inputs['support'], manifest['support']['sha256'], views)
    measured = masked_images(views, observed)
    def tensor(image):
        arr = np.asarray(image.resize((resolution, resolution), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        return torch.from_numpy(arr).permute(2,0,1).to(device)
    source_images, silhouette, normal_images = list(map(tensor, measured)), list(map(tensor, views)), list(map(tensor, normals))
    cameras = get_cameras_list(camera['azimuth_degrees'], device=device, focal=camera['focal'], dist=camera['distance'])
    result_counts = {}
    torch_mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=[colors]))
    torch.cuda.synchronize() if device.startswith('cuda') else None
    projection_start = time.perf_counter()
    result = multiview_color_projection(torch_mesh, source_images, normal_image_list=normal_images,
        cameras_list=cameras, weights=[1., .4, .8, 1., .8, .4], resolution=resolution, device=device,
        complete_unseen=False, below_confidence_strategy='original', normal_gate_threshold=.65,
        depth_tol=.01, depth_edge_tol=.006, depth_edge_dilate_px=12,
        alpha_erode_px=4, alpha_edge_threshold=.95, side_occlusion_indices=[1,2,4,5],
        support_silhouettes=silhouette, min_source_support=min_source_support, support_counts_out=result_counts,
        source_raster_env=raster_env)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    projection_seconds = time.perf_counter() - projection_start
    mask = result_counts['measured_mask'].numpy()
    projected = result.textures.verts_features_packed().detach().cpu().numpy()
    rgb = np.clip(np.rint(projected * 255), 0, 255).astype(np.uint8)
    if not np.array_equal(rgb[~mask], baseline_colors[~mask]):
        raise RuntimeError('unsupported baseline colors changed')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=rgb, process=False)
    for name, kind in [('texture-only.ply','ply'), ('texture-only.glb','glb')]:
        with (output / name).open('xb') as f: f.write(result_mesh.export(file_type=kind))
    np.save(output / 'measured-vertex-mask.npy', mask, allow_pickle=False)
    report = {'schema':'pshuman.texture-only-result/1', 'manifest_sha256':manifest_sha256,
        'baseline_sha256':digest(inputs['baseline']), 'measured_vertices':int(mask.sum()),
        'baseline_vertices':int((~mask).sum()), 'inferred_fallback_vertices':0,
        'min_source_support':min_source_support, 'resolution':resolution, 'source_raster':raster_policy, 'geometry_precision':'float32 projection and export',
        'projection_seconds':projection_seconds, 'wall_seconds':time.perf_counter()-started,
        'outputs':{n:digest(output/n) for n in ('texture-only.ply','texture-only.glb','measured-vertex-mask.npy')},
        'limitations':'Vertex colors only; no UV/material preservation. Declared shared-frame registration is not validated. Unknown source stays baseline; inferred views are not used.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--min-source-support', type=float, default=.95)
    parser.add_argument('--resolution', type=int, default=1024); parser.add_argument('--device', default='cuda')
    args=parser.parse_args()
    print(json.dumps(run(args.manifest,args.manifest_sha256,args.output,min_source_support=args.min_source_support,resolution=args.resolution,device=args.device),indent=2))

if __name__ == '__main__':
    main()

"""Opt-in conservative PyTorch3D coarse-bin capacity for source texture only.

The limit bounds the int32 bin-to-face index buffer, not total device memory.
No perspective/clipping expansion is supported by the all-face bound.
"""
import math


def source_raster_options(meshes, cameras, height, width, env):
    mode = env.get('PSHUMAN_TEXTURE_RASTER_MODE', 'default')
    if mode == 'default':
        return None
    if mode != 'all_faces':
        raise ValueError('PSHUMAN_TEXTURE_RASTER_MODE must be default or all_faces')
    if cameras.is_perspective():
        raise ValueError('all_faces source raster requires orthographic camera without face clipping')
    if not meshes.verts_packed().is_cuda:
        raise ValueError('all_faces source raster requires CUDA coarse rasterization')
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError('positive integer raster dimensions required')
    face_count = meshes.faces_packed().shape[0]
    batch = len(meshes)
    if face_count <= 0 or face_count >= 2**31 or batch <= 0:
        raise ValueError('nonempty mesh and int32-safe face count required')
    raw = env.get('PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES', '536870912')
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
        raise ValueError('PSHUMAN_TEXTURE_BIN_BUFFER_MAX_BYTES must be a positive integer')
    max_image = max(height, width)
    bin_size = 8 if max_image <= 64 else 2 ** max(math.ceil(math.log2(max_image)) - 4, 4)
    bins_h = (height + bin_size - 1) // bin_size
    bins_w = (width + bin_size - 1) // bin_size
    bin_bytes = batch * bins_h * bins_w * face_count * 4
    if bin_bytes > int(raw):
        raise ValueError(f'source raster bin buffer {bin_bytes} exceeds configured limit {raw}')
    return {'bin_size': bin_size, 'max_faces_per_bin': face_count}

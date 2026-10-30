"""Optional topology-bound attenuation of projected normal loss around arms."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def topology_digest(faces):
    arr = np.asarray(faces)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError('faces must be integer triangles')
    return hashlib.sha256(np.ascontiguousarray(arr, dtype='<i8').tobytes()).hexdigest()


def load_weights(path, faces, vertex_count):
    if type(vertex_count) is not int or vertex_count <= 0:
        raise ValueError('vertex_count must be a positive integer')
    a = np.asarray(faces)
    digest = topology_digest(a)
    if a.size == 0 or a.min() < 0 or a.max() >= vertex_count:
        raise ValueError('faces out of range')
    with np.load(Path(path), allow_pickle=False) as d:
        if str(d['faces_sha256'].item()) != digest or int(d['vertex_count'].item()) != vertex_count:
            raise ValueError('topology binding mismatch')
        if ('weights' in d) == ('vertex_ids' in d):
            raise ValueError('supply exactly one of weights or vertex_ids')
        if 'vertex_ids' in d:
            ids = np.asarray(d['vertex_ids'])
            if ids.ndim != 1 or not np.issubdtype(ids.dtype,np.integer) or ids.size == 0 or ids.min() < 0 or ids.max() >= vertex_count or len(np.unique(ids)) != len(ids):
                raise ValueError('invalid vertex IDs')
            w = np.zeros(vertex_count,np.float32); w[ids] = 1
        else:
            w = np.asarray(d['weights'])
            if w.shape != (vertex_count,) or not np.issubdtype(w.dtype,np.number) or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)) or not np.any(w > 0):
                raise ValueError('invalid weights')
            w = w.astype(np.float32)
    return w


def attach(optimizer, weights):
    state = optimizer._vertices_etc
    if state.ndim != 2 or state.shape[1] != 9 or len(weights) != len(state):
        raise ValueError('unsupported optimizer state or weight count')
    part = torch.as_tensor(weights, dtype=state.dtype,device=state.device).reshape(-1,1)
    optimizer._vertices_etc = torch.cat((state.detach(),part),dim=1)
    optimizer._split_vertices_etc()
    optimizer._vertices.requires_grad_()


def normal_confidence(renderer, vertices, faces, state, alpha, dilation=8, threshold=.75):
    if state.ndim != 2 or state.shape[1] != 10 or state.shape[0] != vertices.shape[0]:
        raise ValueError('invalid transported optimizer state')
    if type(dilation) is not int or dilation < 0:
        raise ValueError('invalid dilation')
    with torch.no_grad():
        part = state[:,9].detach().clamp(0,1)
        projection = renderer.render(vertices.detach(),faces,colors=part[:,None].expand(-1,3).contiguous())
        active = (projection[...,:1] > threshold) & (alpha.detach() > .5)
        if dilation:
            active = F.max_pool2d(active.float().permute(0,3,1,2),2*dilation+1,stride=1,padding=dilation).permute(0,2,3,1) > .5
            active &= alpha.detach() > .5
        return (~active).to(alpha.dtype)


def remap_weights(path, source_faces, source_vertex_count, kept_ids, compact_faces):
    """Map an exact source-topology binding through an indexed outer-surface compaction."""
    source = np.asarray(source_faces)
    kept = np.asarray(kept_ids)
    compact = np.asarray(compact_faces)
    if kept.ndim != 1 or not np.issubdtype(kept.dtype,np.integer) or not len(kept) or kept.min() < 0 or kept.max() >= source_vertex_count or not np.all(kept[1:] > kept[:-1]):
        raise ValueError('invalid kept vertex mapping')
    weights = load_weights(path, source, source_vertex_count)
    remap = np.full(source_vertex_count,-1,dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    mapped = remap[source]
    mapped = mapped[(mapped >= 0).all(axis=1)]
    if not np.array_equal(mapped, compact):
        raise ValueError('compacted topology does not match kept vertex mapping')
    result = weights[kept]
    if not np.any(result > 0):
        raise ValueError('no bound arm vertices survived compaction')
    return result

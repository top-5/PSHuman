"""Optional topology-bound attenuation of projected normal loss around arms."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def capture_initial_state(path, **tensors):
    """Save exact loaded tensors before optimization; refuse output replacement."""
    arrays = {name: value.detach().cpu().contiguous().numpy() for name, value in tensors.items()}
    output = Path(path)
    with output.open('xb') as stream:
        np.savez(stream, **arrays)
    return {'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'arrays': {name: {'shape': list(value.shape), 'dtype': value.dtype.str,
                              'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
                       for name, value in arrays.items()}}


def topology_digest(faces):
    arr = np.asarray(faces)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.dtype.kind not in 'iu' or (arr.size and arr.dtype.kind == 'u' and arr.max() > np.iinfo(np.int64).max):
        raise ValueError('faces must be integer triangles')
    return hashlib.sha256(np.ascontiguousarray(arr, dtype='<i8').tobytes()).hexdigest()


def _binding_count(value):
    item = np.asarray(value)
    if item.shape != () or item.dtype.kind not in 'iu':
        raise ValueError('binding vertex_count must be an integer scalar')
    return int(item.item())


def load_weights(path, faces, vertex_count):
    if type(vertex_count) is not int or vertex_count <= 0:
        raise ValueError('vertex_count must be a positive integer')
    a = np.asarray(faces)
    digest = topology_digest(a)
    if a.size == 0 or a.min() < 0 or a.max() >= vertex_count:
        raise ValueError('faces out of range')
    with np.load(Path(path), allow_pickle=False) as d:
        if str(d['faces_sha256'].item()) != digest or _binding_count(d['vertex_count']) != vertex_count:
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
            if w.shape != (vertex_count,) or w.dtype.kind not in 'fiu' or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)) or not np.any(w > 0):
                raise ValueError('invalid weights')
            w = w.astype(np.float32)
    return w


def attach(optimizer, weights, known_non_arm=None):
    state = optimizer._vertices_etc
    if state.ndim != 2 or state.shape[1] != 9 or len(weights) != len(state):
        raise ValueError('unsupported optimizer state or weight count')
    part = torch.as_tensor(weights, dtype=state.dtype, device=state.device).reshape(-1, 1)
    if not bool(torch.isfinite(part).all()) or bool(((part < 0) | (part > 1)).any()):
        raise ValueError('invalid arm weights')
    channels = [state.detach(), part]
    if known_non_arm is not None:
        known = torch.as_tensor(known_non_arm, dtype=state.dtype, device=state.device).reshape(-1, 1)
        if len(known) != len(state) or not bool(torch.isfinite(known).all()) or bool(((known < 0) | (known > 1)).any()) or bool(((known > 0) & (part > 0)).any()):
            raise ValueError('invalid or overlapping known non-arm weights')
        channels.append(known)
    optimizer._vertices_etc = torch.cat(channels, dim=1)
    optimizer._split_vertices_etc()
    optimizer._vertices.requires_grad_()


def normal_confidence(renderer, vertices, faces, state, alpha, dilation=8, threshold=.75, attenuation=0.0, return_stats=False):
    if state.ndim != 2 or state.shape[1] not in (10, 11) or state.shape[0] != vertices.shape[0]:
        raise ValueError('invalid transported optimizer state')
    if not bool(torch.isfinite(state[:,9:]).all()):
        raise ValueError('nonfinite transported part state')
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid threshold')
    if type(dilation) is not int or not 0 <= dilation <= 256:
        raise ValueError('invalid dilation')
    if isinstance(attenuation, bool) or not isinstance(attenuation, (float, int)) or not np.isfinite(attenuation) or not 0 <= attenuation <= 1:
        raise ValueError('invalid attenuation')
    with torch.no_grad():
        part = state[:,9].detach().clamp(0,1)
        known = state[:,10].detach().clamp(0,1) if state.shape[1] == 11 else None
        colors = torch.stack((part, known if known is not None else torch.zeros_like(part), part), dim=1)
        projection = renderer.render(vertices.detach(),faces,colors=colors.contiguous())
        visible = alpha.detach() > .5
        arm_pixels = (projection[...,:1] > threshold) & visible
        active = arm_pixels
        if dilation:
            active = F.max_pool2d(active.float().permute(0,3,1,2),2*dilation+1,stride=1,padding=dilation).permute(0,2,3,1) > .5
            active &= visible
        confidence = torch.where(active, torch.as_tensor(float(attenuation), dtype=alpha.dtype, device=alpha.device), torch.ones_like(alpha))
        if not return_stats:
            return confidence
        known_pixels = ((projection[...,1:2] > threshold) & visible) if known is not None else None
        count = lambda x: x.sum(dim=(1,2,3)).detach().cpu().tolist()
        stats = {'maskedPixelsByView': count(active), 'knownNonArmSpillByView': count(active & known_pixels) if known_pixels is not None else None,
                 'unknownMaskedPixelsByView': count(active & ~arm_pixels & ~known_pixels) if known_pixels is not None else None}
        return confidence, stats


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


def remap_known_non_arm(path, source_faces, source_vertex_count, kept_ids, arm_weights):
    """Return optional explicit non-arm vertex labels; never infer unknown labels."""
    with np.load(Path(path), allow_pickle=False) as d:
        if str(d['faces_sha256'].item()) != topology_digest(source_faces) or _binding_count(d['vertex_count']) != source_vertex_count:
            raise ValueError('topology binding mismatch')
        if 'known_non_arm_ids' not in d:
            return None
        ids = np.asarray(d['known_non_arm_ids'])
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids.size and (ids.min() < 0 or ids.max() >= source_vertex_count)) or len(np.unique(ids)) != len(ids):
            raise ValueError('invalid known non-arm IDs')
        full = np.zeros(source_vertex_count, np.float32); full[ids] = 1
        result = full[np.asarray(kept_ids)]
        if np.any((result > 0) & (arm_weights > 0)):
            raise ValueError('arm and known non-arm labels overlap')
        return result

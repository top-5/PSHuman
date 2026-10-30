"""Shared SOMA prior byte loading and indexed carving selection."""

import os

import numpy as np
import torch

SOMA_PRIOR_GROUPS = ("left_hand", "right_hand", "left_distal_arm", "right_distal_arm", "head")


def load_soma_body_prior_npz(path, device, *, bind_prior=None):
    """The fitted SOMA body: `vertices` (V,3) float, `faces` (F,3) int, and int vertex-id groups SOMA_PRIOR_GROUPS."""
    path = str(path or "").strip()
    if not path:
        raise ValueError("PSHUMAN_BODY_MODEL=soma needs PSHUMAN_BODY_PRIOR_NPZ (the SOMA body fitted by seed)")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PSHUMAN_BODY_PRIOR_NPZ does not exist: {path}")
    loaded_prior_sha256 = None
    if bind_prior is not None and type(bind_prior) is not bool:
        raise ValueError("bind_prior must be a boolean or None")
    enabled = bool(os.environ.get("PSHUMAN_ARM_NORMAL_CONFIDENCE_NPZ", "").strip()) if bind_prior is None else bind_prior
    if enabled:
        import hashlib
        import io
        with open(path, "rb") as prior_stream:
            prior_bytes = prior_stream.read()
        loaded_prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
        data = np.load(io.BytesIO(prior_bytes), allow_pickle=False)
    else:
        data = np.load(path, allow_pickle=False)
    missing = [k for k in ("vertices", "faces") + SOMA_PRIOR_GROUPS if k not in data]
    if missing:
        raise KeyError(f"PSHUMAN_BODY_PRIOR_NPZ {path} lacks keys {missing} (has {sorted(data.files)})")
    verts = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"])
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0 or not np.isfinite(verts).all():
        raise ValueError(f"PSHUMAN_BODY_PRIOR_NPZ {path}: `vertices` must be finite (V,3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"PSHUMAN_BODY_PRIOR_NPZ {path}: `faces` must be integer (F,3), got {faces.dtype} {faces.shape}")
    n = verts.shape[0]
    if faces.min() < 0 or faces.max() >= n:
        raise ValueError(f"PSHUMAN_BODY_PRIOR_NPZ {path}: `faces` index outside 0..{n - 1}")
    out = {
        "path": path,
        "vertices": torch.as_tensor(verts, device=device),
        "faces": torch.as_tensor(faces.astype(np.int64), device=device),
    }
    if loaded_prior_sha256 is not None:
        out["loaded_sha256"] = loaded_prior_sha256
    # optional: the surface the carve starts from, when seed gives one (a closed, evenly meshed shell of the same
    # body: SOMA's own face has parted lips over an inner mouth, and the carve grew needles from it)
    if "carve_vertices" in data and "carve_faces" in data:
        out["carve_vertices"] = torch.as_tensor(np.asarray(data["carve_vertices"], dtype=np.float32), device=device)
        out["carve_faces"] = torch.as_tensor(np.asarray(data["carve_faces"]).astype(np.int64), device=device)
    for key in SOMA_PRIOR_GROUPS:
        ids = np.asarray(data[key]).reshape(-1)
        if ids.size == 0 or not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0 or ids.max() >= n:
            raise ValueError(f"PSHUMAN_BODY_PRIOR_NPZ {path}: group `{key}` must be non-empty int ids in 0..{n - 1}")
        out[key] = ids.astype(np.int64)
    return out



def outer_surface(vertices, faces):
    """The largest face-connected piece of a mesh, compacted: the SMPL-X body without its two eyeballs (separate
    546-vertex spheres in the eye sockets). Carved from with them, the remesher folded each eyeball into the eyelids
    and left a crumpled patch on both eyes (alena-bikini, 2026-10-05). Returns (vertices, faces, kept vertex ids)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    f = faces.detach().cpu().numpy()
    n = int(vertices.shape[0])
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    graph = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    used = np.zeros(n, bool)
    used[f.ravel()] = True
    counts = np.bincount(label[used], minlength=label.max() + 1)
    keep = (label == int(np.argmax(counts))) & used
    remap = -np.ones(n, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    f_keep = remap[f]
    f_keep = f_keep[(f_keep >= 0).all(axis=1)]
    ids = torch.as_tensor(np.flatnonzero(keep), device=vertices.device)
    return vertices.index_select(0, ids), torch.as_tensor(f_keep, device=faces.device, dtype=faces.dtype), ids



def select_carve_start(body_prior, fallback_vertices, fallback_faces):
    """Return the exact source surface and its largest indexed component."""
    if body_prior is not None and "carve_vertices" in body_prior:
        source_vertices, source_faces = body_prior["carve_vertices"], body_prior["carve_faces"]
    else:
        source_vertices, source_faces = fallback_vertices, fallback_faces
    vertices, faces, ids = outer_surface(source_vertices, source_faces)
    return source_vertices, source_faces, vertices, faces, ids

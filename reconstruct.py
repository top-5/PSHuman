from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from utils.func import  make_sparse_camera,  make_round_views
from utils.render import NormalsRenderer
import  torch.optim as optim
from tqdm import tqdm
from utils.video_utils import write_video
from omegaconf import OmegaConf
import numpy as np
import os
from PIL import Image
import kornia
import torch
import torch.nn as nn
import trimesh
from icecream import ic
from utils.project_mesh import multiview_color_projection, get_cameras_list
from utils.mesh_utils import  to_py3d_mesh, rot6d_to_rotmat, tensor2variable
from utils.project_mesh import  project_color, get_cameras_list
from utils.smpl_util import SMPLX
from lib.dataset.mesh_util import apply_vertex_mask, part_removal, poisson, keep_largest
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import argparse
from datetime import datetime, timezone
import json
import pickle
import sqlite3


PSHUMAN_VIEW_TO_LABEL16_INDEX = {
    "front_face": 0,
    "front_right": 2,
    "right": 4,
    "back": 8,
    "left": 12,
    "front_left": 14,
}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _load_smpl_prior_npz(path, device, dtype):
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"PSHUMAN_SMPL_PRIOR_NPZ does not exist: {path}")
    data = np.load(path, allow_pickle=True)

    def tensor(name):
        if name not in data:
            return None
        return torch.as_tensor(np.asarray(data[name], dtype=np.float32), device=device, dtype=dtype)

    return {
        "path": path,
        "betas": tensor("betas"),
        "global_orient_6d": tensor("global_orient_6d"),
        "body_pose_6d": tensor("body_pose_6d"),
        "trans": tensor("trans"),
    }


def _copy_partial_(target, source):
    """Copy source values into target, preserving unmatched trailing params."""

    if source is None:
        return 0
    target_flat = target.view(-1)
    source_flat = source.reshape(-1).to(device=target.device, dtype=target.dtype)
    n = min(int(target_flat.numel()), int(source_flat.numel()))
    if n <= 0:
        return 0
    target_flat[:n].copy_(source_flat[:n])
    return n


def _partial_l2(param, prior):
    if prior is None:
        return torch.zeros((), device=param.device, dtype=param.dtype)
    p = param.reshape(-1)
    q = prior.reshape(-1).to(device=param.device, dtype=param.dtype)
    n = min(int(p.numel()), int(q.numel()))
    if n <= 0:
        return torch.zeros((), device=param.device, dtype=param.dtype)
    return ((p[:n] - q[:n]) ** 2).mean()


def _pshuman_root():
    return os.path.dirname(os.path.abspath(__file__))


def _load_smplx_anatomy_ids():
    """Load SMPL-X anatomical vertex id sets used by fit-time constraints.

    These constraints are intentionally based on trusted SMPL-X/MANO metadata,
    not on inferred mesh surgery outputs.  The ids are cached as numpy arrays
    and converted to tensors inside the optimizer once the device/dtype is known.
    """

    root = _pshuman_root()
    out = {}
    mano_path = os.path.join(root, "smpl_related", "smpl_data", "MANO_SMPLX_vertex_ids.pkl")
    if os.path.exists(mano_path):
        with open(mano_path, "rb") as f:
            mano = pickle.load(f)
        out["mano_left_hand"] = np.asarray(mano.get("left_hand", []), dtype=np.int64).reshape(-1)
        out["mano_right_hand"] = np.asarray(mano.get("right_hand", []), dtype=np.int64).reshape(-1)

    partial_dir = os.path.join(root, "smpl_related", "HPS", "pymafx_data", "partial_mesh")
    for name in ["smplx_larm_vids", "smplx_rarm_vids", "smplx_lwrist_vids", "smplx_rwrist_vids"]:
        path = os.path.join(partial_dir, f"{name}.npz")
        if os.path.exists(path):
            out[name] = np.load(path)["vids"].astype(np.int64).reshape(-1)

    seg_path = os.path.join(root, "smpl_related", "smpl_vert_segmentation.json")
    if os.path.exists(seg_path):
        with open(seg_path, "r") as f:
            seg = json.load(f)
        for name in ["leftFoot", "rightFoot", "leftToeBase", "rightToeBase", "leftLeg", "rightLeg"]:
            if name in seg:
                out[f"smplseg_{name}"] = np.asarray(seg[name], dtype=np.int64).reshape(-1)
    return out


def _as_index_tensor(id_sets, name, device):
    ids = id_sets.get(name)
    if ids is None or len(ids) == 0:
        return None
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _paired_mirror_vertex_loss(vertices, left_ids, right_ids):
    """Pointwise mirror-consistency loss for paired left/right topology ids.

    This catches the exact failure mode we saw: one hand can be globally in the
    right silhouette while its palm/fingers are rolled about 180 degrees.  A
    silhouette term cannot see that; paired MANO ids can.
    """

    if left_ids is None or right_ids is None:
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    n = min(int(left_ids.numel()), int(right_ids.numel()))
    if n < 16:
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    left = vertices.index_select(0, left_ids[:n])
    right = vertices.index_select(0, right_ids[:n])
    left = left.clone()
    left[:, 0] = -left[:, 0]
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    return ((left - right) ** 2).mean()


def _segment_mirror_stats_loss(vertices, left_ids, right_ids):
    """Mirror-consistency loss for unpaired segment ids such as SMPL feet/toes."""

    if left_ids is None or right_ids is None:
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    if int(left_ids.numel()) < 8 or int(right_ids.numel()) < 8:
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    left = vertices.index_select(0, left_ids)
    right = vertices.index_select(0, right_ids)
    left = left.clone()
    left[:, 0] = -left[:, 0]
    lc = left.mean(dim=0)
    rc = right.mean(dim=0)
    left0 = left - lc
    right0 = right - rc
    cov_l = left0.T @ left0 / max(int(left0.shape[0]) - 1, 1)
    cov_r = right0.T @ right0 / max(int(right0.shape[0]) - 1, 1)
    # Centroid plus covariance keeps symmetric foot/toe segments from rolling
    # into incompatible outside/upside-down shapes without requiring a fake
    # point correspondence.
    return ((lc - rc) ** 2).mean() + 0.25 * ((cov_l - cov_r) ** 2).mean()


def _env_int_set(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return set(default)
    out = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def _dilate_bool_mask(mask, iterations):
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        src = out
        dst = src.copy()
        dst[1:] |= src[:-1]
        dst[:-1] |= src[1:]
        dst[:, 1:] |= src[:, :-1]
        dst[:, :-1] |= src[:, 1:]
        dst[1:, 1:] |= src[:-1, :-1]
        dst[1:, :-1] |= src[:-1, 1:]
        dst[:-1, 1:] |= src[1:, :-1]
        dst[:-1, :-1] |= src[1:, 1:]
        out = dst
    return out


def _parse_label_view_indices(default_map):
    raw = os.environ.get("PSHUMAN_SMPL_FIT_LABEL_VIEW_INDICES", "").strip()
    if not raw:
        return dict(default_map)
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("PSHUMAN_SMPL_FIT_LABEL_VIEW_INDICES must contain 6 comma-separated indices")
    return {name: vals[i] for i, name in enumerate(["front_face", "front_right", "right", "back", "left", "front_left"])}


def _load_fit_label_map(labels_dir, view, view_index, resolution):
    candidates = [
        os.path.join(labels_dir, f"{view}_labels.npy"),
        os.path.join(labels_dir, f"{view}.npy"),
        os.path.join(labels_dir, f"view_{view_index:02d}_labels.npy"),
        os.path.join(labels_dir, f"view_{view_index:02d}.npy"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"no label map for {view}; tried: {candidates}")
    labels = np.load(path)
    if labels.ndim != 2:
        raise ValueError(f"label map must be HxW, got {labels.shape}: {path}")
    if labels.shape != (resolution, resolution):
        labels_img = Image.fromarray(labels.astype(np.uint8, copy=False))
        labels_img = labels_img.resize((resolution, resolution), Image.NEAREST)
        labels = np.asarray(labels_img, dtype=np.int32)
    return labels.astype(np.int32, copy=False), path


def _masked_l1_mean(diff, fit_weight_mask):
    weighted = diff.abs() * fit_weight_mask
    denom = fit_weight_mask.expand_as(diff).sum().clamp_min(1.0)
    return weighted.sum() / denom


def _tensor_np(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1))


def _silhouette_fit_metrics(pred_alpha, target_alpha, threshold=0.5):
    pred = _tensor_np(pred_alpha).squeeze()
    target = _tensor_np(target_alpha).squeeze()
    pred_mask = pred > threshold
    target_mask = target > threshold
    inter = int(np.logical_and(pred_mask, target_mask).sum())
    union = int(np.logical_or(pred_mask, target_mask).sum())
    pred_area = int(pred_mask.sum())
    target_area = int(target_mask.sum())
    false_pos = int(np.logical_and(pred_mask, ~target_mask).sum())
    false_neg = int(np.logical_and(~pred_mask, target_mask).sum())
    pred_bbox = _bbox_from_mask(pred_mask)
    target_bbox = _bbox_from_mask(target_mask)

    def _centroid(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return [float("nan"), float("nan")]
        return [float(xs.mean()), float(ys.mean())]

    pred_centroid = _centroid(pred_mask)
    target_centroid = _centroid(target_mask)
    centroid_delta = [
        float(pred_centroid[0] - target_centroid[0]),
        float(pred_centroid[1] - target_centroid[1]),
    ]
    return {
        "iou": float(inter / max(union, 1)),
        "dice": float((2 * inter) / max(pred_area + target_area, 1)),
        "precision": float(inter / max(pred_area, 1)),
        "recall": float(inter / max(target_area, 1)),
        "mae": float(np.abs(pred - target).mean()),
        "pred_area_px": pred_area,
        "target_area_px": target_area,
        "area_ratio": float(pred_area / max(target_area, 1)),
        "false_positive_px": false_pos,
        "false_negative_px": false_neg,
        "bbox_iou": _bbox_iou(pred_bbox, target_bbox),
        "pred_bbox_xyxy": pred_bbox,
        "target_bbox_xyxy": target_bbox,
        "centroid_delta_px": centroid_delta,
    }


def _silhouette_overlay_rgb(pred_alpha, target_alpha, threshold=0.5):
    pred = _tensor_np(pred_alpha).squeeze() > threshold
    target = _tensor_np(target_alpha).squeeze() > threshold
    out = np.zeros((*target.shape, 3), dtype=np.uint8)
    out[target] = np.array([40, 180, 80], dtype=np.uint8)      # target-only: green
    out[pred] = np.array([210, 60, 210], dtype=np.uint8)       # pred-only: magenta
    out[np.logical_and(pred, target)] = np.array([245, 245, 245], dtype=np.uint8)
    return out


def _repair_severed_limbs(mesh_remeshed, min_faces=500, max_gap_m=0.30):
    """
    Reconnect disjoint limb fragments to the body by explicit triangle stitching.

    Design:
      1) For each disconnected fragment near the body in the limb-height zone,
         find closest body/fragment vertex bands.
      2) Order each band around the local gap axis (cylindrical angle order).
      3) Build a watertight bridge strip of triangles between the two ordered rings.

    Invariants:
      - Never carve/delete body or fragment faces.
      - Never drop fragments in this routine.
      - Connected input remains unchanged.
    """

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
        if len(ids) == 0:
            return np.array([], dtype=np.int64)
        pts = verts[ids] - center[None, :]
        u, v = _orthonormal_basis(axis)
        angles = np.arctan2(pts.dot(v), pts.dot(u))
        order = np.argsort(angles)
        ordered = np.asarray(ids, dtype=np.int64)[order]
        if len(ordered) <= ring_n:
            return ordered
        pick = np.linspace(0, len(ordered) - 1, ring_n).astype(np.int64)
        return ordered[pick]

    def _stitch_fragment(body_mesh, frag_mesh, dists, body_nn, ring_vertices=48):
        min_idx = int(np.argmin(dists))
        p_frag = frag_mesh.vertices[min_idx]
        p_body = body_mesh.vertices[int(body_nn[min_idx])]
        axis = p_frag - p_body
        if np.linalg.norm(axis) < 1e-6:
            axis = frag_mesh.vertices.mean(axis=0) - body_mesh.vertices.mean(axis=0)
        axis = axis / (np.linalg.norm(axis) + 1e-9)

        gap = float(dists[min_idx])
        base_radius = max(0.02, min(0.10, 1.5 * gap + 0.01))
        body_tree = KDTree(body_mesh.vertices)
        frag_tree = KDTree(frag_mesh.vertices)

        body_seed = np.array([], dtype=np.int64)
        frag_seed = np.array([], dtype=np.int64)
        for mul in [1.0, 1.5, 2.0, 2.5, 3.0]:
            r = base_radius * mul
            b_ids = body_tree.query_ball_point(p_body, r)
            f_ids = frag_tree.query_ball_point(p_frag, r)

            if len(b_ids) > 0 and len(f_ids) > 0:
                b_ids = np.asarray(b_ids, dtype=np.int64)
                f_ids = np.asarray(f_ids, dtype=np.int64)

                # Keep a narrow slab around the closest-point plane to form a ring band.
                b_slab = np.abs((body_mesh.vertices[b_ids] - p_body[None, :]).dot(axis)) < (0.45 * r)
                f_slab = np.abs((frag_mesh.vertices[f_ids] - p_frag[None, :]).dot(axis)) < (0.45 * r)
                b_ids = b_ids[b_slab]
                f_ids = f_ids[f_slab]

                if len(b_ids) >= 12 and len(f_ids) >= 12:
                    body_seed = b_ids
                    frag_seed = f_ids
                    break

        if len(body_seed) < 12 or len(frag_seed) < 12:
            # Fallback: nearest neighborhoods around closest points.
            k = int(min(max(ring_vertices * 6, 72), len(body_mesh.vertices), len(frag_mesh.vertices)))
            _, b_knn = body_tree.query(p_body, k=k)
            _, f_knn = frag_tree.query(p_frag, k=k)
            body_seed = np.asarray(b_knn, dtype=np.int64).reshape(-1)
            frag_seed = np.asarray(f_knn, dtype=np.int64).reshape(-1)

        ring_n = int(min(ring_vertices, len(body_seed), len(frag_seed)))
        if ring_n < 12:
            return trimesh.util.concatenate([body_mesh, frag_mesh]), False, gap

        c_body = body_mesh.vertices[body_seed].mean(axis=0)
        c_frag = frag_mesh.vertices[frag_seed].mean(axis=0)
        body_ring = _ordered_ring(body_seed, body_mesh.vertices, c_body, axis, ring_n)
        frag_ring = _ordered_ring(frag_seed, frag_mesh.vertices, c_frag, axis, ring_n)

        if len(body_ring) != len(frag_ring) or len(body_ring) < 12:
            return trimesh.util.concatenate([body_mesh, frag_mesh]), False, gap

        # Circularly align rings to minimize crossing and strip stretch.
        b_pts = body_mesh.vertices[body_ring]
        f_pts = frag_mesh.vertices[frag_ring]
        shifts = []
        for s in range(len(frag_ring)):
            shifts.append(np.mean(np.linalg.norm(b_pts - np.roll(f_pts, s, axis=0), axis=1)))
        best_shift = int(np.argmin(shifts))
        frag_ring = np.roll(frag_ring, best_shift)

        body_v = np.asarray(body_mesh.vertices)
        frag_v = np.asarray(frag_mesh.vertices)
        body_f = np.asarray(body_mesh.faces, dtype=np.int64)
        frag_f = np.asarray(frag_mesh.faces, dtype=np.int64)

        v_out = np.vstack([body_v, frag_v])
        frag_off = len(body_v)

        bridge_faces = []
        for i in range(len(body_ring)):
            j = (i + 1) % len(body_ring)
            b0 = int(body_ring[i])
            b1 = int(body_ring[j])
            f0 = int(frag_off + frag_ring[i])
            f1 = int(frag_off + frag_ring[j])
            bridge_faces.append([b0, f0, b1])
            bridge_faces.append([b1, f0, f1])

        f_out = np.vstack([
            body_f,
            frag_f + frag_off,
            np.asarray(bridge_faces, dtype=np.int64)
        ])

        stitched = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
        gap_after = float(KDTree(body_v).query(frag_v, k=1)[0].min())
        return stitched, True, gap_after

    parts = mesh_remeshed.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh_remeshed

    parts_by_size = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    body = parts_by_size[0]
    passthrough = []

    bb = body.bounds
    body_height = float(bb[1, 1] - bb[0, 1])
    limb_y_limit = float(bb[0, 1] + 0.78 * body_height)

    stitched_count = 0
    for frag in parts_by_size[1:]:
        body_tree = KDTree(body.vertices)
        dists3d, body_nn3d = body_tree.query(frag.vertices, k=1)
        gap = float(dists3d.min())
        c = frag.vertices.mean(axis=0)
        is_limb_zone = bool(c[1] < limb_y_limit)
        large_enough = bool(len(frag.faces) >= min_faces)

        if is_limb_zone and large_enough and gap <= max_gap_m:
            stitched_mesh, stitched_ok, _ = _stitch_fragment(body, frag, dists3d, body_nn3d)
            if stitched_ok:
                body = stitched_mesh
                stitched_count += 1
                print(f"[arm-repair] stitched fragment {len(frag.faces)} faces, gap={gap:.4f} m")
            else:
                passthrough.append(frag)
                print(f"[arm-repair] stitch-fallback passthrough fragment {len(frag.faces)} faces, gap={gap:.4f} m")
        else:
            passthrough.append(frag)
            print(
                f"[arm-repair] passthrough fragment {len(frag.faces)} faces, "
                f"gap={gap:.4f} m, limb_zone={is_limb_zone}, large_enough={large_enough}"
            )

    if stitched_count == 0 and len(passthrough) == 0:
        return body
    if len(passthrough) == 0:
        return body
    return trimesh.util.concatenate([body] + passthrough)
#### ------------------- config----------------------   
bg_color = np.array([1,1,1])


def _parse_view_indices(value, default):
    if value is None or value.strip() == "":
        return [int(x) for x in default]
    indices = []
    for part in value.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def _parse_view_pairs(value, default):
    """'0-1,0-5,3-2' -> [(0,1),(0,5),(3,2)]. Returns `default` if empty/invalid."""
    if value is None or value.strip() == "":
        return list(default)
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split("-")
            out.append((int(a), int(b)))
        except (ValueError, IndexError):
            print(f"[reconstruct] bad pair '{part}' in PSHUMAN_XVIEW_DEPTH_PAIRS, ignoring", flush=True)
    return out if out else list(default)


def cross_view_world_consistency(xyz_alpha, mvp, pairs):
    """Cross-view rasterised-XYZ consistency loss (Approach B).

    For each (a, b) pair: the 3D world position rasterised by view a at
    each pixel is reprojected into view b's image plane and compared
    against view b's rasterised XYZ at the same projected location.
    Identity on perfectly cross-view-consistent meshes (e.g. SMPL init).

    Args
    ----
    xyz_alpha : (C, H, W, 4) tensor — channels 0:3 = world XYZ, 3 = alpha.
    mvp       : (C, 4, 4) tensor — proj @ mv per view.
    pairs     : list of (int, int) view-index pairs.

    Returns
    -------
    Scalar tensor: mean of per-pair masked L1 disagreement (in world units).
    Returns 0.0 tensor when there are no valid samples.
    """
    import torch.nn.functional as F
    if not pairs:
        return xyz_alpha.new_zeros(())
    C, H, W, _ = xyz_alpha.shape
    xyz = xyz_alpha[..., :3]              # C,H,W,3 (world)
    alpha = xyz_alpha[..., 3:4]           # C,H,W,1
    total = xyz_alpha.new_zeros(())
    n = 0
    for a, b in pairs:
        if a < 0 or b < 0 or a >= C or b >= C or a == b:
            continue
        xyz_a = xyz[a]                    # H,W,3
        a_a = alpha[a, ..., 0]            # H,W
        # Project a's world XYZ into b's clip space.
        xyz_a_h = torch.cat((xyz_a, torch.ones_like(xyz_a[..., :1])), dim=-1)  # H,W,4
        clip_b = xyz_a_h @ mvp[b].T       # H,W,4
        w_b = clip_b[..., 3:4].clamp(min=1e-6)
        ndc_xy = clip_b[..., :2] / w_b    # H,W,2 in [-1,1] when in-view
        # nvdiffrast convention used in PSHuman has projection_matrix[1,1] < 0
        # so the rasterised image has y-up (row 0 = NDC y = +1). grid_sample
        # expects y-down (-1 at top). Flip y to match.
        grid = torch.stack((ndc_xy[..., 0], -ndc_xy[..., 1]), dim=-1)  # H,W,2
        grid = grid.unsqueeze(0)          # 1,H,W,2
        # Sample b's rasterised world XYZ + alpha at the projected pixel.
        xyz_b = xyz[b].permute(2, 0, 1).unsqueeze(0)        # 1,3,H,W
        alpha_b = alpha[b].permute(2, 0, 1).unsqueeze(0)    # 1,1,H,W
        sampled_xyz = F.grid_sample(
            xyz_b, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(0).permute(1, 2, 0)     # H,W,3
        sampled_a = F.grid_sample(
            alpha_b, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(0).squeeze(0)           # H,W
        in_view = (
            (ndc_xy[..., 0] > -1) & (ndc_xy[..., 0] < 1) &
            (ndc_xy[..., 1] > -1) & (ndc_xy[..., 1] < 1)
        ).to(xyz.dtype)                   # H,W
        valid = a_a * sampled_a * in_view # H,W (soft mask)
        diff = (xyz_a - sampled_xyz).abs().sum(dim=-1)  # H,W (L1 in world units)
        denom = valid.sum().clamp(min=1.0)
        total = total + (diff * valid).sum() / denom
        n += 1
    if n == 0:
        return xyz_alpha.new_zeros(())
    return total / n


class colorModel(nn.Module):
    def __init__(self, renderer, v, f, c):
        super().__init__()
        self.renderer = renderer
        self.v = v
        self.f = f
        self.colors = nn.Parameter(c, requires_grad=True)
        self.bg_color = torch.from_numpy(bg_color).float().to(self.colors.device)
    def forward(self, return_mask=False):
        rgba = self.renderer.render(self.v, self.f, colors=self.colors)
        if return_mask:
            return rgba
        else:
            mask = rgba[..., 3:]
            return rgba[..., :3] * mask + self.bg_color * (1 - mask)


def scale_mesh(vert):
    min_bbox, max_bbox = vert.min(0)[0], vert.max(0)[0]
    center = (min_bbox + max_bbox) / 2 
    offset = -center
    vert = vert + offset

    max_dist = torch.max(torch.sqrt(torch.sum(vert**2, dim=1)))
    scale = 1.0 / max_dist
    return scale, offset

def save_mesh(save_name, vertices, faces,  color=None):
    vertices_np = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
    faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    color_np = None
    if color is not None:
        color_np = color.detach().cpu().numpy() if torch.is_tensor(color) else np.asarray(color)
        color_np = (color_np * 255).astype(np.uint8)
    trimesh.Trimesh(
        vertices_np,
        faces_np,
        vertex_colors=color_np) \
    .export(save_name)
        

    

class ReMesh:
    def __init__(self, opt, econ_dataset):
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.num_view = opt.num_view
        
        self.out_path = opt.res_path
        os.makedirs(self.out_path, exist_ok=True)
        self.resolution = opt.resolution
        self.views = ['front_face', 'front_right', 'right', 'back', 'left', 'front_left' ]
        # ---- view weighting ----------------------------------------------------
        # Front (azim 0) and Back (azim 180) are the camera-anchored views: they
        # constrain the X/Y silhouette directly. Side views (azim 90/270) are
        # diffusion-imagined and are the only ones that constrain depth (Z) — so
        # when they disagree with the others (e.g. drop a hand) they create the
        # parallel-pancake artifact at the wrist.
        # Default weights now bias toward Front/Back (1.0) and downweight pure
        # sides (0.6) and quarter views (0.4). Override with PSHUMAN_VIEW_WEIGHTS
        # as a comma-separated list of 6 floats matching `self.views` ordering.
        default_w = [1.0, 0.4, 0.8, 1.0, 0.8, 0.4]
        env_w = os.environ.get("PSHUMAN_VIEW_WEIGHTS", "").strip()
        if env_w:
            try:
                vals = [float(x) for x in env_w.split(",")]
                if len(vals) == 6:
                    default_w = vals
                else:
                    print(f"[reconstruct] PSHUMAN_VIEW_WEIGHTS expects 6 floats, got {len(vals)}; using defaults", flush=True)
            except ValueError:
                print(f"[reconstruct] PSHUMAN_VIEW_WEIGHTS parse error; using defaults", flush=True)
        self.weights = torch.Tensor(default_w).view(6,1,1,1).to(self.device)
        print(f"[reconstruct] view weights = {default_w}", flush=True)
        # ---- cross-view prior mode --------------------------------------------
        # off                : original behaviour (no mask augmentation)
        # smplx_silhouette   : OR the fitted SMPL-X silhouette into side-view
        #                      masks so the optimizer cannot drop limbs the prior
        #                      says exist (fixes parallel-pancake wrist).
        # Default: off — smplx_silhouette caused front/back bulge when SMPL-X
        # arm joints were poorly initialized (zero mediapipe keypoints) and the
        # resulting wider depth profile was injected into NeuS masks.
        self.xview_mode = os.environ.get("PSHUMAN_XVIEW_MODE", "off").strip().lower()
        # Which view indices receive prior-silhouette injection. Defaults to the
        # pure side views (right=2, left=4) since front/back come from the
        # actual photo and should not be overridden.
        side_idx_env = os.environ.get("PSHUMAN_XVIEW_SIDE_INDICES", "2,4").strip()
        self.xview_side_indices = _parse_view_indices(side_idx_env, [2, 4])
        # Strength of the OR (0..1): 1.0 = full inject, 0.5 = half-weight prior.
        try:
            self.xview_strength = float(os.environ.get("PSHUMAN_XVIEW_STRENGTH", "1.0"))
        except ValueError:
            self.xview_strength = 1.0
        print(
            f"[reconstruct] xview_mode={self.xview_mode} "
            f"side_indices={self.xview_side_indices} strength={self.xview_strength}",
            flush=True,
        )
        self.smpl_exclude_hands_from_fit = _env_bool("PSHUMAN_SMPL_EXCLUDE_HANDS_FROM_FIT", False)
        self.smpl_fit_labels_dir = os.environ.get("PSHUMAN_SMPL_FIT_LABELS_DIR", "").strip()
        self.smpl_fit_exclude_classes = _env_int_set("PSHUMAN_SMPL_FIT_EXCLUDE_CLASSES", {6, 15})
        self.smpl_fit_exclude_dilate = max(0, _env_int("PSHUMAN_SMPL_FIT_EXCLUDE_DILATE", 10))
        self.smpl_fit_label_view_indices = _parse_label_view_indices(PSHUMAN_VIEW_TO_LABEL16_INDEX)
        if self.smpl_exclude_hands_from_fit:
            print(
                "[reconstruct] SMPL-X fit excludes semantic classes "
                f"{sorted(self.smpl_fit_exclude_classes)} from loss "
                f"(dilate={self.smpl_fit_exclude_dilate}, labels={self.smpl_fit_labels_dir or 'unset'})",
                flush=True,
            )
        # Per-iter rendered-normal dump (off by default). Set
        # PSHUMAN_DEBUG_DUMP_NORMALS=1 to write {case}/normals/{step:04d}.png
        self.debug_dump_normals = os.environ.get("PSHUMAN_DEBUG_DUMP_NORMALS", "0") != "0"
        try:
            self.debug_dump_every = int(os.environ.get("PSHUMAN_DEBUG_DUMP_EVERY", "50"))
        except ValueError:
            self.debug_dump_every = 50

        # ---- cross-view depth/world consistency loss --------------------------
        # Penalise per-pixel disagreement of rasterised WORLD-space XYZ between
        # paired views. Closes the parallel-pancake wrist artifact at the loss
        # level (forearm and palm cannot settle on different Z planes if any
        # view pair sees both regions). Default OFF (weight 0) to preserve old
        # behaviour; enable with e.g. PSHUMAN_XVIEW_DEPTH_W=0.5.
        try:
            self.xview_depth_w = float(os.environ.get("PSHUMAN_XVIEW_DEPTH_W", "0.0"))
        except ValueError:
            self.xview_depth_w = 0.0
        # Pairs of view indices (a,b) over which to enforce consistency.
        # Default: front<->near-quarter views and back<->near-quarter views.
        # views = [front_face=0, front_right=1, right=2, back=3, left=4, front_left=5]
        pairs_env = os.environ.get("PSHUMAN_XVIEW_DEPTH_PAIRS", "0-1,0-5,3-2,3-4").strip()
        self.xview_depth_pairs = _parse_view_pairs(pairs_env, [(0, 1), (0, 5), (3, 2), (3, 4)])
        # Optional one-shot synthetic check: rasterise SMPL init from each view,
        # apply the loss, and print per-pair values BEFORE optimization starts.
        # A correct implementation gives ~mm-scale residuals (sampling error only)
        # since SMPL init IS perfectly consistent across views by construction.
        self.xview_depth_check = os.environ.get("PSHUMAN_XVIEW_DEPTH_CHECK", "0") != "0"
        print(
            f"[reconstruct] xview_depth_w={self.xview_depth_w} "
            f"pairs={self.xview_depth_pairs} synthetic_check={self.xview_depth_check}",
            flush=True,
        )

        self.renderer = self.prepare_render()
        # pose prediction
        self.econ_dataset = econ_dataset
        self.smplx_face =  torch.Tensor(econ_dataset.faces.astype(np.int64)).long().to(self.device)
    
    def prepare_render(self):
        ### ------------------- prepare camera and renderer----------------------
        mv, proj = make_sparse_camera(self.opt.cam_path, self.opt.scale, views=[0,1,2,4,6,7], device=self.device)
        renderer = NormalsRenderer(mv, proj, [self.resolution, self.resolution], device=self.device)
        return renderer
    
    def proj_texture(self, fused_images, vertices, faces, normal_images=None):
        mesh = to_py3d_mesh(vertices, faces)
        mesh = mesh.to(self.device)
        camera_focal =  1/2
        cameras_list = get_cameras_list([0, 45, 90, 180, 270, 315], device=self.device, focal=camera_focal)
        normal_gate_threshold = float(os.environ.get("PSHUMAN_TEXTURE_NORMAL_GATE", "0.65"))
        depth_tol = float(os.environ.get("PSHUMAN_TEXTURE_DEPTH_TOL", "0.01"))
        depth_edge_tol = float(os.environ.get("PSHUMAN_TEXTURE_DEPTH_EDGE_TOL", "0.006"))
        depth_edge_dilate_px = int(os.environ.get("PSHUMAN_TEXTURE_DEPTH_EDGE_DILATE", "12"))
        alpha_erode_px = int(os.environ.get("PSHUMAN_TEXTURE_ALPHA_ERODE", "4"))
        alpha_edge_threshold = float(os.environ.get("PSHUMAN_TEXTURE_ALPHA_EDGE_THRESHOLD", "0.95"))
        side_occlusion_guard = os.environ.get("PSHUMAN_SIDE_OCCLUSION_GUARD", "1") != "0"
        side_occlusion_indices = _parse_view_indices(
            os.environ.get("PSHUMAN_SIDE_OCCLUSION_INDICES"),
            [1, 2, 4, 5],
        )
        print(f"[project_mesh] normal gate threshold: {normal_gate_threshold}", flush=True)
        print(
            f"[project_mesh] depth_tol={depth_tol} "
            f"depth_edge_tol={depth_edge_tol} "
            f"depth_edge_dilate_px={depth_edge_dilate_px} "
            f"alpha_erode_px={alpha_erode_px} alpha_edge_threshold={alpha_edge_threshold}",
            flush=True,
        )
        print(
            f"[project_mesh] side_occlusion_guard={side_occlusion_guard} "
            f"side_occlusion_indices={side_occlusion_indices}",
            flush=True,
        )
        mesh = multiview_color_projection(
            mesh,
            fused_images,
            normal_image_list=normal_images,
            camera_focal=camera_focal,
            resolution=self.resolution,
            weights=self.weights.squeeze().cpu().numpy(),
            device=self.device,
            complete_unseen=True,
            confidence_threshold=0.2,
            cameras_list=cameras_list,
            normal_gate_threshold=normal_gate_threshold,
            depth_tol=depth_tol,
            depth_edge_tol=depth_edge_tol,
            depth_edge_dilate_px=depth_edge_dilate_px,
            alpha_erode_px=alpha_erode_px,
            alpha_edge_threshold=alpha_edge_threshold,
            side_occlusion_indices=side_occlusion_indices,
            side_occlusion_guard=side_occlusion_guard,
        )
        return mesh
    
    def get_invisible_idx(self, imgs, vertices, faces):
        mesh = to_py3d_mesh(vertices, faces)
        mesh = mesh.to(self.device)
        camera_focal =  1/2
        if self.num_view == 6:
            cameras_list = get_cameras_list([0, 45, 90, 180, 270, 315], device=self.device, focal=camera_focal)
        elif self.num_view == 4:
            cameras_list = get_cameras_list([0, 90, 180, 270], device=self.device, focal=camera_focal)
        valid_vert_id = []
        vertices_colors = torch.zeros((vertices.shape[0], 3)).float().to(self.device)
        valid_cnt = torch.zeros((vertices.shape[0])).to(self.device)
        for  cam, img, weight in zip(cameras_list, imgs, self.weights.squeeze()):
            ret = project_color(mesh, cam, img, eps=0.01, resolution=self.resolution, device=self.device)
            # print(ret['valid_colors'].shape)
            valid_cnt[ret['valid_verts']] += weight
            vertices_colors[ret['valid_verts']] += ret['valid_colors']*weight
        valid_mask = valid_cnt > 1
        invalid_mask = valid_cnt < 1
        vertices_colors[valid_mask] /= valid_cnt[valid_mask][:, None] 
        
        # visibility
        invisible_vert = valid_cnt < 1
        invisible_vert_indices = torch.nonzero(invisible_vert).squeeze()
        # vertices_colors[invalid_vert] = torch.tensor([1.0, 0.0, 0.0]).float().to("cuda")
        return vertices_colors, invisible_vert_indices 
    
    def inpaint_missed_colors(self, all_vertices, all_colors, missing_indices):
        all_vertices = all_vertices.detach().cpu().numpy()
        all_colors = all_colors.detach().cpu().numpy()
        missing_indices = missing_indices.detach().cpu().numpy()


        non_missing_indices = np.setdiff1d(np.arange(len(all_vertices)), missing_indices)

        kdtree = KDTree(all_vertices[non_missing_indices])


        for missing_index in missing_indices:
            missing_vertex = all_vertices[missing_index]
        
            _, nearest_index = kdtree.query(missing_vertex.reshape(1, -1))
            
            interpolated_color = all_colors[non_missing_indices[nearest_index]]
            
            all_colors[missing_index] = interpolated_color
        
        return torch.from_numpy(all_colors).to(self.device)

    def load_training_data(self, case):
        ###------------------ load target images -------------------------------
        kernal = torch.ones(3, 3)
        erode_iters = 2
        normals = []
        masks = []
        colors = []
        for idx, view in enumerate(self.views):
        # for idx  in [0,2,3,4]:
            normal = Image.open(f'{self.opt.mv_path}/{case}/normals_{view}_masked.png')
            # normal = Image.open(f'{data_path}/{case}/normals/{idx:02d}_rgba.png')
            normal = normal.convert('RGBA').resize((self.resolution, self.resolution), Image.BILINEAR)
            normal = np.array(normal).astype(np.float32) / 255.
            mask = normal[..., 3:]  # alpha
            mask_troch = torch.from_numpy(mask).unsqueeze(0)
            for _ in range(erode_iters):
                mask_torch = kornia.morphology.erosion(mask_troch, kernal)
            mask_erode = mask_torch.squeeze(0).numpy()
            masks.append(mask_erode)
            normal = normal[..., :3] * mask_erode 
            normals.append(normal)
            
            color = Image.open(f'{self.opt.mv_path}/{case}/color_{view}_masked.png')
            color = color.convert('RGBA').resize((self.resolution, self.resolution), Image.BILINEAR)
            color = np.array(color).astype(np.float32) / 255.
            color_mask = color[..., 3:]  # alpha
            # color_dilate = color[..., :3] * color_mask  + bg_color * (1 - color_mask)
            color_dilate = color[..., :3] * mask_erode + bg_color * (1 - mask_erode)
            colors.append(color_dilate)

        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks).to(self.device)
        normals = np.stack(normals, 0) 
        target_normals = torch.from_numpy(normals).to(self.device)
        colors = np.stack(colors, 0)
        target_colors = torch.from_numpy(colors).to(self.device)
        return masks, target_colors, target_normals
    
    def preprocess(self, color_pils, normal_pils):
          ###------------------ load target images -------------------------------
        kernal = torch.ones(3, 3)
        erode_iters = 2
        normals = []
        masks = []
        colors = []
        for normal, color in zip(normal_pils, color_pils):
            normal = normal.resize((self.resolution, self.resolution), Image.BILINEAR)
            normal = np.array(normal).astype(np.float32) / 255.
            mask = normal[..., 3:]  # alpha
            mask_troch = torch.from_numpy(mask).unsqueeze(0)
            for _ in range(erode_iters):
                mask_torch = kornia.morphology.erosion(mask_troch, kernal)
            mask_erode = mask_torch.squeeze(0).numpy()
            masks.append(mask_erode)
            normal = normal[..., :3] * mask_erode 
            normals.append(normal)
            
            color = color.resize((self.resolution, self.resolution), Image.BILINEAR)
            color = np.array(color).astype(np.float32) / 255.
            color_mask = color[..., 3:]  # alpha
            # color_dilate = color[..., :3] * color_mask  + bg_color * (1 - color_mask)
            color_dilate = color[..., :3] * mask_erode + bg_color * (1 - mask_erode)
            colors.append(color_dilate)

        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks).to(self.device)
        normals = np.stack(normals, 0) 
        target_normals = torch.from_numpy(normals).to(self.device)
        colors = np.stack(colors, 0)
        target_colors = torch.from_numpy(colors).to(self.device)
        return masks, target_colors, target_normals

    def _build_smpl_fit_loss_mask(self, masks, case_path):
        if not self.smpl_exclude_hands_from_fit:
            return None, {"enabled": False}
        if not self.smpl_fit_labels_dir:
            raise RuntimeError("PSHUMAN_SMPL_EXCLUDE_HANDS_FROM_FIT=1 requires PSHUMAN_SMPL_FIT_LABELS_DIR")

        loss_masks = []
        stats = {
            "enabled": True,
            "labels_dir": self.smpl_fit_labels_dir,
            "exclude_classes": sorted(int(x) for x in self.smpl_fit_exclude_classes),
            "dilate": int(self.smpl_fit_exclude_dilate),
            "views": {},
        }
        for vi, view in enumerate(self.views):
            label_index = self.smpl_fit_label_view_indices[view]
            labels, label_path = _load_fit_label_map(
                self.smpl_fit_labels_dir,
                view,
                label_index,
                self.resolution,
            )
            excluded = np.isin(labels, list(self.smpl_fit_exclude_classes))
            excluded = _dilate_bool_mask(excluded, self.smpl_fit_exclude_dilate)
            keep = (~excluded).astype(np.float32)[..., None]
            loss_masks.append(keep)
            stats["views"][view] = {
                "label_path": label_path,
                "label_index": int(label_index),
                "excluded_px": int(excluded.sum()),
                "excluded_frac": float(excluded.mean()),
            }

        loss_mask = torch.from_numpy(np.stack(loss_masks, 0)).to(self.device)
        if self.debug_dump_normals or _env_bool("PSHUMAN_SMPL_SAVE_FIT", True):
            import imageio

            out_dir = os.path.join(case_path, "smpl_fit", "loss_masks")
            os.makedirs(out_dir, exist_ok=True)
            for vi, view in enumerate(self.views):
                img = (loss_masks[vi].squeeze(-1) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, f"{view}_fit_keep_mask.png"), img)
        return loss_mask, stats

    def _inject_prior_silhouette(self, masks, target_normals, v_smpl, case_path=None):
        """Cross-view prior: OR the fitted SMPL-X silhouette into the configured
        side-view masks (and bake matching prior normals into those pixels) so
        that the optimizer cannot drop body parts the prior says exist.

        This kills the parallel-pancake artifact at the wrist: when diffusion
        omits a hand from `right_masked` / `left_masked`, the side-view mask
        loss otherwise yanks the hand to whatever Z minimizes the contradiction
        with "empty" — typically a different plane from the forearm. With the
        prior injected, the side views again provide a depth constraint that
        agrees with front/back, so the hand stays co-planar with the forearm.

        Returns: (masks, target_normals) — possibly augmented copies.
        """
        if self.xview_mode == "off":
            return masks, target_normals
        if self.xview_mode != "smplx_silhouette":
            print(f"[reconstruct] unknown PSHUMAN_XVIEW_MODE='{self.xview_mode}', skipping", flush=True)
            return masks, target_normals
        with torch.no_grad():
            v = v_smpl.detach()
            # Exclude head vertices from the prior silhouette.  SMPL-X uses a
            # smooth sphere for the skull which has a different profile from real
            # hair, so injecting it into side views creates a mohawk / blob.
            # Strategy: drop any face whose highest vertex sits above the neck.
            # "Highest" = largest value on the dominant vertical axis (the axis
            # with the greatest peak-to-peak range across all vertices).
            ranges = v.max(dim=0).values - v.min(dim=0).values  # (3,)
            up_axis = int(ranges.argmax().item())               # 0=X,1=Y,2=Z
            v_up = v[:, up_axis]
            v_up_min, v_up_max = float(v_up.min()), float(v_up.max())
            body_height = v_up_max - v_up_min
            # Neck ≈ top 18 % of body height; clip head by dropping faces above.
            neck_cutoff = v_up_min + 0.82 * body_height
            face_vert_up = v_up[self.smplx_face]               # [F,3]
            body_face_mask = face_vert_up.max(dim=-1).values < neck_cutoff
            body_faces = self.smplx_face[body_face_mask]
            n = calc_vertex_normals(v, body_faces)
            rendered = self.renderer.render(v, body_faces, normals=n)
        prior_alpha = rendered[..., 3:]            # [V,H,W,1]
        prior_normals = rendered[..., :3]          # [V,H,W,3]
        strength = float(max(0.0, min(1.0, self.xview_strength)))
        sides = sorted({i for i in self.xview_side_indices if 0 <= i < masks.shape[0]})
        if not sides:
            return masks, target_normals

        masks_aug = masks.clone()
        normals_aug = target_normals.clone()
        added_total = 0.0
        for vi in sides:
            # OR mask: pixel becomes max(diffusion_mask, prior_alpha * strength).
            new_mask = torch.maximum(masks_aug[vi], prior_alpha[vi] * strength)
            added = float((new_mask - masks_aug[vi]).sum().item())
            added_total += added
            # In pixels gained by the prior, also seed the target normals from
            # the prior so the normal-loss does not pull the surface back to
            # zero. Where the diffusion mask was already non-zero, we keep its
            # (more accurate) normals untouched.
            gain_mask = (new_mask > masks_aug[vi]).float()  # [H,W,1]
            normals_aug[vi] = normals_aug[vi] * (1.0 - gain_mask) + prior_normals[vi] * gain_mask
            masks_aug[vi] = new_mask
        print(
            f"[reconstruct] xview prior injected {added_total:.0f} pixels of mask "
            f"across views {sides} (strength={strength})",
            flush=True,
        )
        # Optional: dump the augmented masks for diagnostic.
        if case_path is not None and self.debug_dump_normals:
            import imageio
            os.makedirs(f'{case_path}/xview_prior', exist_ok=True)
            for vi in sides:
                m = (masks_aug[vi].squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(f'{case_path}/xview_prior/mask_view{vi:02d}.png', m)
                nrm_vis = ((prior_normals[vi].cpu().numpy() * 0.5 + 0.5) * 255).clip(0,255).astype(np.uint8)
                imageio.imwrite(f'{case_path}/xview_prior/prior_normals_view{vi:02d}.png', nrm_vis)
        return masks_aug, normals_aug

    def optimize_case(self, case, pose, clr_img, nrm_img, opti_texture=True):
        case_path = f'{self.out_path}/{case}'
        os.makedirs(case_path, exist_ok=True)
        
        if clr_img is not None:
            masks, target_colors, target_normals = self.preprocess(clr_img, nrm_img)
        else:
            masks, target_colors, target_normals = self.load_training_data(case)
        
        # rotation
        rz = R.from_euler('z', 180, degrees=True).as_matrix()
        ry = R.from_euler('y', 180, degrees=True).as_matrix()
        rz = torch.from_numpy(rz).float().to(self.device)
        ry = torch.from_numpy(ry).float().to(self.device)
        
        scale, offset = None, None

        global_orient = pose["global_orient"] # pymaf_res[idx]['smplx_params']['body_pose'][:, :1, :, :2].to(device).reshape(1, 1, -1) # data["global_orient"]
        body_pose = pose["body_pose"] # pymaf_res[idx]['smplx_params']['body_pose'][:, 1:22, :, :2].to(device).reshape(1, 21, -1) # data["body_pose"]
        left_hand_pose = pose["left_hand_pose"] # pymaf_res[idx]['smplx_params']['left_hand_pose'][:, :, :, :2].to(device).reshape(1, 15, -1)
        right_hand_pose = pose["right_hand_pose"] # pymaf_res[idx]['smplx_params']['right_hand_pose'][:, :, :, :2].to(device).reshape(1, 15, -1)
        beta = pose["betas"]
        
        # The optimizer and variables
        optimed_pose = torch.tensor(body_pose,
                                device=self.device,
                                requires_grad=True)  # [1,23,3,3]
        optimed_trans = torch.tensor(pose["trans"],
                                        device=self.device,
                                        requires_grad=True)  # [3]
        optimed_betas = torch.tensor(beta,
                                        device=self.device,
                                        requires_grad=True)  # [1,200]
        optimed_orient = torch.tensor(global_orient,
                                        device=self.device,
                                        requires_grad=True)  # [1,1,3,3]
        optimed_rhand = torch.tensor(right_hand_pose,
                                        device=self.device,
                                        requires_grad=True)  
        optimed_lhand = torch.tensor(left_hand_pose,
                                        device=self.device,
                                        requires_grad=True)  
        
        smpl_reset_hands = _env_bool("PSHUMAN_SMPL_RESET_HANDS", False)
        smpl_freeze_hands = _env_bool("PSHUMAN_SMPL_FREEZE_HANDS", False)
        if smpl_reset_hands:
            with torch.no_grad():
                optimed_lhand.copy_(
                    torch.eye(3, device=self.device, dtype=optimed_lhand.dtype)
                    .view(1, 1, 3, 3)
                    .expand_as(optimed_lhand)
                )
                optimed_rhand.copy_(
                    torch.eye(3, device=self.device, dtype=optimed_rhand.dtype)
                    .view(1, 1, 3, 3)
                    .expand_as(optimed_rhand)
                )

        smpl_prior_path = os.environ.get("PSHUMAN_SMPL_PRIOR_NPZ", "").strip()
        smpl_prior = _load_smpl_prior_npz(smpl_prior_path, self.device, optimed_betas.dtype) if smpl_prior_path else None
        smpl_init_from_prior = _env_bool("PSHUMAN_SMPL_INIT_FROM_PRIOR", bool(smpl_prior is not None))
        smpl_prior_init_counts = {}
        if smpl_prior is not None and smpl_init_from_prior:
            with torch.no_grad():
                smpl_prior_init_counts["betas"] = _copy_partial_(optimed_betas, smpl_prior.get("betas"))
                smpl_prior_init_counts["global_orient_6d"] = _copy_partial_(
                    optimed_orient,
                    smpl_prior.get("global_orient_6d"),
                )
                smpl_prior_init_counts["body_pose_6d"] = _copy_partial_(
                    optimed_pose,
                    smpl_prior.get("body_pose_6d"),
                )
                smpl_prior_init_counts["trans"] = _copy_partial_(optimed_trans, smpl_prior.get("trans"))
            print(
                "[smpl-fit] initialized from prior "
                f"{smpl_prior['path']} copied={smpl_prior_init_counts}",
                flush=True,
            )

        smpl_steps = _env_int("PSHUMAN_SMPL_STEPS", 100)
        smpl_hand_lr = _env_float("PSHUMAN_SMPL_HAND_LR", 1e-3)
        smpl_beta_lr = _env_float("PSHUMAN_SMPL_BETA_LR", 3e-3)
        smpl_pose_lr = _env_float("PSHUMAN_SMPL_POSE_LR", 3e-3)
        smpl_orient_lr = _env_float("PSHUMAN_SMPL_ORIENT_LR", 3e-3)
        smpl_trans_lr = _env_float("PSHUMAN_SMPL_TRANS_LR", 3e-3)
        # Freeze global orient so hard-symmetry constraint cannot be absorbed by
        # tilting the whole body to match a slightly asymmetric silhouette.
        smpl_freeze_orient = _env_bool("PSHUMAN_SMPL_FREEZE_GLOBAL_ORIENT", False)
        if smpl_freeze_orient:
            optimed_orient.requires_grad_(False)
        smpl_beta_l2 = _env_float("PSHUMAN_SMPL_BETA_L2", 0.0)
        smpl_beta_clamp = _env_float("PSHUMAN_SMPL_BETA_CLAMP", 0.0)
        smpl_mask_weight = _env_float("PSHUMAN_SMPL_MASK_WEIGHT", 1.0)
        smpl_normal_weight = _env_float("PSHUMAN_SMPL_NORMAL_WEIGHT", 1.0)
        smpl_pose_l2 = _env_float("PSHUMAN_SMPL_POSE_L2", 0.0)
        smpl_hand_l2 = _env_float("PSHUMAN_SMPL_HAND_L2", 0.0)
        smpl_trans_l2 = _env_float("PSHUMAN_SMPL_TRANS_L2", 0.0)
        smpl_head_neck_l2 = _env_float("PSHUMAN_SMPL_HEAD_NECK_L2", 0.0)
        smpl_prior_beta_l2 = _env_float("PSHUMAN_SMPL_PRIOR_BETA_L2", _env_float("PSHUMAN_SMPL_PRIOR_BETA_WEIGHT", 0.0))
        smpl_prior_pose_l2 = _env_float("PSHUMAN_SMPL_PRIOR_POSE_L2", _env_float("PSHUMAN_SMPL_PRIOR_POSE_WEIGHT", 0.0))
        smpl_prior_orient_l2 = _env_float("PSHUMAN_SMPL_PRIOR_ORIENT_L2", _env_float("PSHUMAN_SMPL_PRIOR_ORIENT_WEIGHT", 0.0))
        smpl_prior_trans_l2 = _env_float("PSHUMAN_SMPL_PRIOR_TRANS_L2", _env_float("PSHUMAN_SMPL_PRIOR_TRANS_WEIGHT", 0.0))
        smpl_anatomy_weight = _env_float("PSHUMAN_SMPL_ANATOMY_WEIGHT", 0.0)
        smpl_symmetry_weight = _env_float("PSHUMAN_SMPL_SYMMETRY_WEIGHT", smpl_anatomy_weight)
        smpl_hand_symmetry_weight = _env_float("PSHUMAN_SMPL_HAND_SYMMETRY_WEIGHT", smpl_symmetry_weight)
        smpl_arm_symmetry_weight = _env_float("PSHUMAN_SMPL_ARM_SYMMETRY_WEIGHT", smpl_symmetry_weight)
        smpl_foot_symmetry_weight = _env_float("PSHUMAN_SMPL_FOOT_SYMMETRY_WEIGHT", smpl_symmetry_weight)
        smpl_toe_symmetry_weight = _env_float("PSHUMAN_SMPL_TOE_SYMMETRY_WEIGHT", smpl_foot_symmetry_weight)
        smpl_freeze_head_neck = _env_bool("PSHUMAN_SMPL_FREEZE_HEAD_NECK", False)
        # Hard bilateral symmetry projection (default ON). Operates directly on
        # body_pose 6D after each optimizer step using the same mirror map as
        # seed.smplx_iterfit.symmetry — averages left and mirror(right), writes
        # the symmetric pair back. Sign pattern for 6D mirror across YZ plane:
        # [r1,r2] = [a,b,c, d,e,f] -> [a,-b,-c, -d,e,f]. Pairs are 0-indexed
        # into the 21-joint body_pose.
        smpl_hard_symmetry = _env_bool("PSHUMAN_SMPL_HARD_SYMMETRY", True)
        smpl_head_neck_joint_spec = os.environ.get("PSHUMAN_SMPL_HEAD_NECK_JOINTS", "11,14")
        smpl_head_neck_joints = []
        try:
            smpl_pose_joint_count = int(optimed_pose.view(-1, 6).shape[0])
            for part in smpl_head_neck_joint_spec.split(","):
                part = part.strip()
                if not part:
                    continue
                idx = int(part)
                if 0 <= idx < smpl_pose_joint_count:
                    smpl_head_neck_joints.append(idx)
        except Exception:
            smpl_head_neck_joints = []
        smpl_log_every = max(1, _env_int("PSHUMAN_SMPL_LOG_EVERY", 10))
        smpl_active_betas = _env_int("PSHUMAN_SMPL_ACTIVE_BETAS", int(optimed_betas.shape[-1]))
        smpl_active_betas = max(0, min(int(smpl_active_betas), int(optimed_betas.shape[-1])))
        smpl_save_fit = _env_bool("PSHUMAN_SMPL_SAVE_FIT", True)
        smpl_anatomy_ids_np = _load_smplx_anatomy_ids()
        smpl_anatomy_ids = {
            name: _as_index_tensor(smpl_anatomy_ids_np, name, self.device)
            for name in smpl_anatomy_ids_np
        }

        # Local 6D bilateral symmetry projection — mirror plane = YZ in SMPL-X
        # model space (M = diag(-1,1,1), so R' = M·R·M).
        #
        # CRITICAL: ``utils.mesh_utils.rot6d_to_rotmat`` does ``x.view(-1, 3, 2)``
        # on a contiguous (B,6) tensor, which is row-major. So the 6D layout is
        # *interleaved* — [r00, r01, r10, r11, r20, r21] (pairs of (col0, col1)
        # per row), NOT the column-major [r00, r10, r20, r01, r11, r21]. The
        # mirror sign must follow the interleaved layout, otherwise four of the
        # six components flip the wrong way, Gram-Schmidt rebuilds a garbage
        # rotation, and the projection scrambles the pose every step.
        # Pairs are 0-indexed into the 21-joint body_pose.
        _SYMMETRY_PAIRS_6D = (
            (0, 1), (3, 4), (6, 7), (9, 10),
            (12, 13), (15, 16), (17, 18), (19, 20),
        )

        def _project_body_pose_symmetric(pose_tensor: torch.Tensor) -> None:
            """In-place project body_pose 6D to bilaterally symmetric subspace."""
            pose_flat = pose_tensor.view(-1, 6)
            # Interleaved layout: signs for [r00, r01, r10, r11, r20, r21]
            # under R' = diag(-1,1,1) · R · diag(-1,1,1).
            sign = torch.tensor(
                [1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                device=pose_flat.device,
                dtype=pose_flat.dtype,
            )
            for li, ri in _SYMMETRY_PAIRS_6D:
                left = pose_flat[li].clone()
                right = pose_flat[ri].clone()
                sym_left = (left + right * sign) * 0.5
                pose_flat[li] = sym_left
                pose_flat[ri] = sym_left * sign

        if smpl_hard_symmetry:
            with torch.no_grad():
                _project_body_pose_symmetric(optimed_pose)

        beta_initial = optimed_betas.detach().clone()
        pose_initial = optimed_pose.detach().clone()
        lhand_initial = optimed_lhand.detach().clone()
        rhand_initial = optimed_rhand.detach().clone()
        trans_initial = optimed_trans.detach().clone()
        smpl_fit_dir = f'{case_path}/smpl_fit'

        optimed_params = [
            {'params': [optimed_lhand, optimed_rhand], 'lr': smpl_hand_lr, 'name': 'hands'},
            {'params': [optimed_betas], 'lr': smpl_beta_lr, 'name': 'betas'},
            {'params': [optimed_pose], 'lr': smpl_pose_lr, 'name': 'body_pose'},
            {'params': [optimed_trans], 'lr': smpl_trans_lr, 'name': 'trans'},
        ]
        if not smpl_freeze_orient:
            optimed_params.append({'params': [optimed_orient], 'lr': smpl_orient_lr, 'name': 'global_orient'})
        optimizer_smpl = torch.optim.Adam(
            optimed_params,
            amsgrad=True,
        )
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            min_lr=1e-5,
            patience=5,
        )

        # Allow staged SMPL-X contour-fit experiments. More steps alone do not
        # guarantee better shape: pose/camera can absorb contour error while
        # betas barely move. Separate LR groups, beta clamp/L2, and active-beta
        # masks make attempts comparable and prevent implausible PCA drift.
        # Also dump init (pre-loop) and final (post-loop) SMPL renders when
        # fit saving is enabled, so config-driven attempts are inspectable.
        dump_smpl_diag = self.debug_dump_normals or smpl_save_fit
        if dump_smpl_diag or smpl_save_fit:
            os.makedirs(smpl_fit_dir, exist_ok=True)

        smpl_fit_history = []
        smpl_fit_config = {
            "steps": int(smpl_steps),
            "active_betas": int(smpl_active_betas),
            "beta_dim": int(optimed_betas.shape[-1]),
            "learning_rates": {
                "hands": float(smpl_hand_lr),
                "betas": float(smpl_beta_lr),
                "body_pose": float(smpl_pose_lr),
                "global_orient": float(smpl_orient_lr),
                "trans": float(smpl_trans_lr),
            },
            "reset_hands": bool(smpl_reset_hands),
            "freeze_hands": bool(smpl_freeze_hands),
            "freeze_orient": bool(smpl_freeze_orient),
            "weights": {
                "mask": float(smpl_mask_weight),
                "normal": float(smpl_normal_weight),
                "beta_l2": float(smpl_beta_l2),
                "pose_l2": float(smpl_pose_l2),
                "hand_l2": float(smpl_hand_l2),
                "trans_l2": float(smpl_trans_l2),
                "head_neck_l2": float(smpl_head_neck_l2),
                "prior_beta_l2": float(smpl_prior_beta_l2),
                "prior_pose_l2": float(smpl_prior_pose_l2),
                "prior_orient_l2": float(smpl_prior_orient_l2),
                "prior_trans_l2": float(smpl_prior_trans_l2),
                "anatomy": float(smpl_anatomy_weight),
                "symmetry": float(smpl_symmetry_weight),
                "hand_symmetry": float(smpl_hand_symmetry_weight),
                "arm_symmetry": float(smpl_arm_symmetry_weight),
                "foot_symmetry": float(smpl_foot_symmetry_weight),
                "toe_symmetry": float(smpl_toe_symmetry_weight),
            },
            "anatomy_constraints": {
                "enabled": bool(
                    smpl_anatomy_weight > 0
                    or smpl_symmetry_weight > 0
                    or smpl_hand_symmetry_weight > 0
                    or smpl_arm_symmetry_weight > 0
                    or smpl_foot_symmetry_weight > 0
                    or smpl_toe_symmetry_weight > 0
                ),
                "contract": "seed.pshuman_smplx_fit_anatomy_constraints.v1",
                "principle": "Prefer trusted SMPL-X/MANO left-right anatomical symmetry; reject 180-degree palm/toe roll by loss and post-fit gate.",
                "id_sets": {k: int(len(v)) for k, v in smpl_anatomy_ids_np.items()},
            },
            "head_neck": {
                "freeze": bool(smpl_freeze_head_neck),
                "joints": [int(j) for j in smpl_head_neck_joints],
                "joint_index_space": "body_pose_6d_zero_based",
            },
            "beta_clamp": float(smpl_beta_clamp),
            "hard_symmetry": bool(smpl_hard_symmetry),
            "prior": {
                "path": smpl_prior["path"] if smpl_prior is not None else "",
                "init_from_prior": bool(smpl_init_from_prior),
                "init_counts": dict(smpl_prior_init_counts),
            },
        }
        smpl_fit_loss_mask, smpl_fit_loss_mask_stats = self._build_smpl_fit_loss_mask(masks, case_path)
        smpl_fit_config["loss_mask"] = smpl_fit_loss_mask_stats
        if smpl_save_fit:
            with open(f'{smpl_fit_dir}/fit_config.json', 'w') as f:
                json.dump(smpl_fit_config, f, indent=2)

        for i in tqdm(range(smpl_steps)):
            optimizer_smpl.zero_grad()
            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(
                -1, 6)).unsqueeze(0)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(
                -1, 6)).unsqueeze(0)

            smpl_verts, smpl_landmarks, smpl_joints = self.econ_dataset.smpl_model(
                shape_params=optimed_betas,
                expression_params=tensor2variable(pose["exp"], self.device),
                body_pose=optimed_pose_mat,
                global_pose=optimed_orient_mat,
                jaw_pose=tensor2variable(pose["jaw_pose"], self.device),
                left_hand_pose=optimed_lhand,
                right_hand_pose=optimed_rhand,

            )

            smpl_verts = smpl_verts + optimed_trans
            
            v_smpl = torch.matmul(torch.matmul(smpl_verts.squeeze(0), rz.T), ry.T)
            if scale is None:
                scale, offset = scale_mesh(v_smpl.detach())
            v_smpl = (v_smpl + offset) * scale * 2
            # if i == 0:
            #   save_mesh(f'{case_path}/{case}_init_smpl.obj', v_smpl, self.smplx_face)
            # exit()
            normals = calc_vertex_normals(v_smpl, self.smplx_face)
            nrm = self.renderer.render(v_smpl, self.smplx_face, normals=normals)

            # Diagnostic: dump SMPL-X front-view render at iter 0 (initial HPS
            # estimate) so we can see whether the prior is wrong vs whether
            # refinement drifts to the wrong pose.
            if dump_smpl_diag and i == 0:
                import imageio
                front_nrm = (nrm.detach()[0,:,:,:3] * 255).clamp(max=255).type(torch.uint8).cpu().numpy()
                imageio.imwrite(f'{case_path}/smpl_fit/init_iter000.png', front_nrm)

            masks_ = nrm[..., 3:] 
            if smpl_fit_loss_mask is None:
                smpl_mask_loss = ((masks_ - masks) * self.weights).abs().mean()
                smpl_nrm_loss = ((nrm[..., :3] - target_normals) * self.weights).abs().mean()
            else:
                fit_weight_mask = self.weights * smpl_fit_loss_mask
                smpl_mask_loss = _masked_l1_mean(masks_ - masks, fit_weight_mask)
                smpl_nrm_loss = _masked_l1_mean(nrm[..., :3] - target_normals, fit_weight_mask)
            smpl_beta_loss = (optimed_betas ** 2).mean()
            smpl_pose_loss = ((optimed_pose - pose_initial) ** 2).mean()
            smpl_hand_loss = (
                ((optimed_lhand - lhand_initial) ** 2).mean()
                + ((optimed_rhand - rhand_initial) ** 2).mean()
            ) * 0.5
            smpl_trans_loss = ((optimed_trans - trans_initial) ** 2).mean()
            if smpl_head_neck_joints:
                pose_flat = optimed_pose.view(-1, 6)
                pose_initial_flat = pose_initial.view(-1, 6)
                head_neck_idx = torch.as_tensor(smpl_head_neck_joints, device=pose_flat.device, dtype=torch.long)
                smpl_head_neck_loss = ((pose_flat.index_select(0, head_neck_idx) - pose_initial_flat.index_select(0, head_neck_idx)) ** 2).mean()
            else:
                smpl_head_neck_loss = torch.zeros((), device=self.device, dtype=optimed_pose.dtype)
            if smpl_prior is not None:
                smpl_prior_beta_loss = _partial_l2(optimed_betas, smpl_prior.get("betas"))
                smpl_prior_pose_loss = _partial_l2(optimed_pose, smpl_prior.get("body_pose_6d"))
                smpl_prior_orient_loss = _partial_l2(optimed_orient, smpl_prior.get("global_orient_6d"))
                smpl_prior_trans_loss = _partial_l2(optimed_trans, smpl_prior.get("trans"))
            else:
                smpl_prior_beta_loss = torch.zeros((), device=self.device, dtype=optimed_pose.dtype)
                smpl_prior_pose_loss = torch.zeros((), device=self.device, dtype=optimed_pose.dtype)
                smpl_prior_orient_loss = torch.zeros((), device=self.device, dtype=optimed_pose.dtype)
                smpl_prior_trans_loss = torch.zeros((), device=self.device, dtype=optimed_pose.dtype)
            smpl_hand_symmetry_loss = _paired_mirror_vertex_loss(
                v_smpl,
                smpl_anatomy_ids.get("mano_left_hand"),
                smpl_anatomy_ids.get("mano_right_hand"),
            )
            smpl_arm_symmetry_loss = (
                _paired_mirror_vertex_loss(
                    v_smpl,
                    smpl_anatomy_ids.get("smplx_larm_vids"),
                    smpl_anatomy_ids.get("smplx_rarm_vids"),
                )
                + _paired_mirror_vertex_loss(
                    v_smpl,
                    smpl_anatomy_ids.get("smplx_lwrist_vids"),
                    smpl_anatomy_ids.get("smplx_rwrist_vids"),
                )
            ) * 0.5
            smpl_foot_symmetry_loss = _segment_mirror_stats_loss(
                v_smpl,
                smpl_anatomy_ids.get("smplseg_leftFoot"),
                smpl_anatomy_ids.get("smplseg_rightFoot"),
            )
            smpl_toe_symmetry_loss = _segment_mirror_stats_loss(
                v_smpl,
                smpl_anatomy_ids.get("smplseg_leftToeBase"),
                smpl_anatomy_ids.get("smplseg_rightToeBase"),
            )
            smpl_anatomy_loss = (
                smpl_hand_symmetry_weight * smpl_hand_symmetry_loss
                + smpl_arm_symmetry_weight * smpl_arm_symmetry_loss
                + smpl_foot_symmetry_weight * smpl_foot_symmetry_loss
                + smpl_toe_symmetry_weight * smpl_toe_symmetry_loss
            )

            smpl_loss = (
                smpl_mask_weight * smpl_mask_loss
                + smpl_normal_weight * smpl_nrm_loss
                + smpl_beta_l2 * smpl_beta_loss
                + smpl_pose_l2 * smpl_pose_loss
                + smpl_hand_l2 * smpl_hand_loss
                + smpl_trans_l2 * smpl_trans_loss
                + smpl_head_neck_l2 * smpl_head_neck_loss
                + smpl_prior_beta_l2 * smpl_prior_beta_loss
                + smpl_prior_pose_l2 * smpl_prior_pose_loss
                + smpl_prior_orient_l2 * smpl_prior_orient_loss
                + smpl_prior_trans_l2 * smpl_prior_trans_loss
                + smpl_anatomy_loss
            )
            # smpl_loss =  smpl_mask_loss 
            smpl_loss.backward()
            if smpl_freeze_hands:
                if optimed_lhand.grad is not None:
                    optimed_lhand.grad.zero_()
                if optimed_rhand.grad is not None:
                    optimed_rhand.grad.zero_()
            if smpl_freeze_head_neck and optimed_pose.grad is not None and smpl_head_neck_joints:
                grad_flat = optimed_pose.grad.view(-1, 6)
                grad_flat[smpl_head_neck_joints] = 0
            if optimed_betas.grad is not None and smpl_active_betas < optimed_betas.shape[-1]:
                optimed_betas.grad[..., smpl_active_betas:] = 0
            optimizer_smpl.step()
            if smpl_hard_symmetry:
                with torch.no_grad():
                    _project_body_pose_symmetric(optimed_pose)
            if smpl_freeze_head_neck and smpl_head_neck_joints:
                with torch.no_grad():
                    pose_flat = optimed_pose.view(-1, 6)
                    pose_initial_flat = pose_initial.view(-1, 6)
                    pose_flat[smpl_head_neck_joints] = pose_initial_flat[smpl_head_neck_joints]
            if smpl_beta_clamp > 0:
                with torch.no_grad():
                    optimed_betas.clamp_(min=-smpl_beta_clamp, max=smpl_beta_clamp)
            scheduler_smpl.step(smpl_loss)
            if smpl_save_fit and (i == 0 or (i + 1) % smpl_log_every == 0 or i + 1 == smpl_steps):
                beta_delta = (optimed_betas.detach() - beta_initial).abs()
                front_sil = _silhouette_fit_metrics(masks_[0], masks[0])
                per_view_iou = [
                    _silhouette_fit_metrics(masks_[vi], masks[vi])["iou"]
                    for vi in range(int(masks.shape[0]))
                ]
                smpl_fit_history.append({
                    "iter": int(i + 1),
                    "loss": float(smpl_loss.detach().cpu()),
                    "mask_loss": float(smpl_mask_loss.detach().cpu()),
                    "normal_loss": float(smpl_nrm_loss.detach().cpu()),
                    "beta_l2": float(smpl_beta_loss.detach().cpu()),
                    "pose_l2": float(smpl_pose_loss.detach().cpu()),
                    "hand_l2": float(smpl_hand_loss.detach().cpu()),
                    "trans_l2": float(smpl_trans_loss.detach().cpu()),
                    "head_neck_l2": float(smpl_head_neck_loss.detach().cpu()),
                    "prior_beta_l2": float(smpl_prior_beta_loss.detach().cpu()),
                    "prior_pose_l2": float(smpl_prior_pose_loss.detach().cpu()),
                    "prior_orient_l2": float(smpl_prior_orient_loss.detach().cpu()),
                    "prior_trans_l2": float(smpl_prior_trans_loss.detach().cpu()),
                    "anatomy_loss": float(smpl_anatomy_loss.detach().cpu()),
                    "hand_symmetry_loss": float(smpl_hand_symmetry_loss.detach().cpu()),
                    "arm_symmetry_loss": float(smpl_arm_symmetry_loss.detach().cpu()),
                    "foot_symmetry_loss": float(smpl_foot_symmetry_loss.detach().cpu()),
                    "toe_symmetry_loss": float(smpl_toe_symmetry_loss.detach().cpu()),
                    "front_silhouette_iou": front_sil["iou"],
                    "mean_silhouette_iou": float(np.mean(per_view_iou)),
                    "beta_norm": float(torch.linalg.norm(optimed_betas.detach()).cpu()),
                    "beta_delta_max": float(beta_delta.max().cpu()),
                    "beta_delta_mean": float(beta_delta.mean().cpu()),
                    "lr": [float(g["lr"]) for g in optimizer_smpl.param_groups],
                })

        # Diagnostic: dump SMPL-X front-view render after the refinement loop.
        if dump_smpl_diag and smpl_steps > 0:
            import imageio
            with torch.no_grad():
                _n = calc_vertex_normals(v_smpl, self.smplx_face)
                _r = self.renderer.render(v_smpl, self.smplx_face, normals=_n)
            final_nrm = (_r.detach()[0,:,:,:3] * 255).clamp(max=255).type(torch.uint8).cpu().numpy()
            imageio.imwrite(f'{case_path}/smpl_fit/final_iter{smpl_steps:03d}.png', final_nrm)
        elif smpl_steps == 0:
            # No refinement ran — materialise v_smpl once from the raw HPS pose
            # so downstream code (MeshOptimizer init, prior injection) has it.
            import imageio
            with torch.no_grad():
                optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).unsqueeze(0)
                optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).unsqueeze(0)
                smpl_verts, _, _ = self.econ_dataset.smpl_model(
                    shape_params=optimed_betas,
                    expression_params=tensor2variable(pose["exp"], self.device),
                    body_pose=optimed_pose_mat,
                    global_pose=optimed_orient_mat,
                    jaw_pose=tensor2variable(pose["jaw_pose"], self.device),
                    left_hand_pose=optimed_lhand,
                    right_hand_pose=optimed_rhand,
                )
                smpl_verts = smpl_verts + optimed_trans
                v_smpl = torch.matmul(torch.matmul(smpl_verts.squeeze(0), rz.T), ry.T)
                if scale is None:
                    scale, offset = scale_mesh(v_smpl.detach())
                v_smpl = (v_smpl + offset) * scale * 2
                _n = calc_vertex_normals(v_smpl, self.smplx_face)
                _r = self.renderer.render(v_smpl, self.smplx_face, normals=_n)
            if dump_smpl_diag:
                final_nrm = (_r.detach()[0,:,:,:3] * 255).clamp(max=255).type(torch.uint8).cpu().numpy()
                imageio.imwrite(f'{case_path}/smpl_fit/final_iter000_no_refine.png', final_nrm)
        mesh_smpl = trimesh.Trimesh(vertices=v_smpl.detach().cpu().numpy(), faces=self.smplx_face.detach().cpu().numpy())  

        if smpl_save_fit:
            with torch.no_grad():
                fit_normals = calc_vertex_normals(v_smpl, self.smplx_face)
                fit_render = self.renderer.render(v_smpl, self.smplx_face, normals=fit_normals)
                final_hand_symmetry_loss = _paired_mirror_vertex_loss(
                    v_smpl,
                    smpl_anatomy_ids.get("mano_left_hand"),
                    smpl_anatomy_ids.get("mano_right_hand"),
                )
                final_arm_symmetry_loss = (
                    _paired_mirror_vertex_loss(
                        v_smpl,
                        smpl_anatomy_ids.get("smplx_larm_vids"),
                        smpl_anatomy_ids.get("smplx_rarm_vids"),
                    )
                    + _paired_mirror_vertex_loss(
                        v_smpl,
                        smpl_anatomy_ids.get("smplx_lwrist_vids"),
                        smpl_anatomy_ids.get("smplx_rwrist_vids"),
                    )
                ) * 0.5
                final_foot_symmetry_loss = _segment_mirror_stats_loss(
                    v_smpl,
                    smpl_anatomy_ids.get("smplseg_leftFoot"),
                    smpl_anatomy_ids.get("smplseg_rightFoot"),
                )
                final_toe_symmetry_loss = _segment_mirror_stats_loss(
                    v_smpl,
                    smpl_anatomy_ids.get("smplseg_leftToeBase"),
                    smpl_anatomy_ids.get("smplseg_rightToeBase"),
                )
            per_view_silhouette = {}
            for vi, view_name in enumerate(self.views[: int(masks.shape[0])]):
                per_view_silhouette[view_name] = _silhouette_fit_metrics(
                    fit_render[vi, ..., 3:],
                    masks[vi],
                )
            front_silhouette = per_view_silhouette.get("front_face", _silhouette_fit_metrics(fit_render[0, ..., 3:], masks[0]))
            mean_silhouette_iou = float(np.mean([m["iou"] for m in per_view_silhouette.values()])) if per_view_silhouette else 0.0
            mean_silhouette_bbox_iou = float(np.mean([m["bbox_iou"] for m in per_view_silhouette.values()])) if per_view_silhouette else 0.0
            beta_delta = (optimed_betas.detach() - beta_initial).abs()
            final_summary = {
                "case": case,
                "config": smpl_fit_config,
                "history": smpl_fit_history,
                "final": {
                    "front_silhouette_iou": front_silhouette["iou"],
                    "mean_silhouette_iou": mean_silhouette_iou,
                    "mean_silhouette_bbox_iou": mean_silhouette_bbox_iou,
                    "beta_norm": float(torch.linalg.norm(optimed_betas.detach()).cpu()),
                    "beta_delta_max": float(beta_delta.max().cpu()),
                    "beta_delta_mean": float(beta_delta.mean().cpu()),
                    "active_beta_delta_max": float(beta_delta[..., :smpl_active_betas].max().cpu()) if smpl_active_betas else 0.0,
                    "inactive_beta_delta_max": float(beta_delta[..., smpl_active_betas:].max().cpu()) if smpl_active_betas < optimed_betas.shape[-1] else 0.0,
                },
                "silhouette_fit": {
                    "threshold": 0.5,
                    "score": front_silhouette["iou"],
                    "front": front_silhouette,
                    "mean_iou": mean_silhouette_iou,
                    "mean_bbox_iou": mean_silhouette_bbox_iou,
                    "per_view": per_view_silhouette,
                },
                "anatomy_constraints": {
                    "contract": "seed.pshuman_smplx_fit_anatomy_constraints.v1",
                    "enabled": smpl_fit_config["anatomy_constraints"]["enabled"],
                    "final_losses": {
                        "hand_symmetry_loss": float(final_hand_symmetry_loss.detach().cpu()),
                        "arm_symmetry_loss": float(final_arm_symmetry_loss.detach().cpu()),
                        "foot_symmetry_loss": float(final_foot_symmetry_loss.detach().cpu()),
                        "toe_symmetry_loss": float(final_toe_symmetry_loss.detach().cpu()),
                    },
                    "weights": {
                        "hand_symmetry": float(smpl_hand_symmetry_weight),
                        "arm_symmetry": float(smpl_arm_symmetry_weight),
                        "foot_symmetry": float(smpl_foot_symmetry_weight),
                        "toe_symmetry": float(smpl_toe_symmetry_weight),
                    },
                    "policy": "If post-fit anatomical validation fails, do not graft; refit or search parameters instead.",
                },
            }
            attempt_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            attempt_tag = f"{attempt_tag}_steps{int(smpl_steps)}_betas{int(smpl_active_betas)}"
            attempt_dir = os.path.join(smpl_fit_dir, "attempts", attempt_tag)
            os.makedirs(attempt_dir, exist_ok=True)

            canonical_report_path = f'{smpl_fit_dir}/fit_report.json'
            canonical_overlay_path = f'{smpl_fit_dir}/front_silhouette_fit_overlay.png'
            canonical_params_path = f'{smpl_fit_dir}/smplx_fit_params.npz'
            canonical_mesh_path = f'{smpl_fit_dir}/smplx_fit_mesh.obj'
            attempt_report_path = os.path.join(attempt_dir, "fit_report.json")
            attempt_overlay_path = os.path.join(attempt_dir, "front_silhouette_fit_overlay.png")
            attempt_params_path = os.path.join(attempt_dir, "smplx_fit_params.npz")
            attempt_mesh_path = os.path.join(attempt_dir, "smplx_fit_mesh.obj")

            final_summary["artifacts"] = {
                "attempt_dir": attempt_dir,
                "report_json": attempt_report_path,
                "front_silhouette_fit_overlay": attempt_overlay_path,
                "params_path": attempt_params_path,
                "mesh_path": attempt_mesh_path,
                "canonical_report_json": canonical_report_path,
                "canonical_params_path": canonical_params_path,
                "canonical_mesh_path": canonical_mesh_path,
            }
            with open(canonical_report_path, 'w') as f:
                json.dump(final_summary, f, indent=2)
            with open(attempt_report_path, 'w') as f:
                json.dump(final_summary, f, indent=2)
            try:
                import imageio
                overlay = _silhouette_overlay_rgb(fit_render[0, ..., 3:], masks[0])
                imageio.imwrite(canonical_overlay_path, overlay)
                imageio.imwrite(attempt_overlay_path, overlay)
            except Exception as exc:
                print(f"[smpl-fit] silhouette overlay write failed: {exc}", flush=True)
            smpl_fit_npz = {
                "betas": _tensor_np(optimed_betas),
                "betas_initial": _tensor_np(beta_initial),
                "trans": _tensor_np(optimed_trans),
                "global_orient_6d": _tensor_np(optimed_orient),
                "body_pose_6d": _tensor_np(optimed_pose),
                "left_hand_pose": _tensor_np(optimed_lhand),
                "right_hand_pose": _tensor_np(optimed_rhand),
                "v_smpl": _tensor_np(v_smpl),
                "faces": _tensor_np(self.smplx_face),
            }
            np.savez_compressed(canonical_params_path, **smpl_fit_npz)
            np.savez_compressed(attempt_params_path, **smpl_fit_npz)
            save_mesh(canonical_mesh_path, v_smpl.detach().cpu().numpy(), self.smplx_face.detach().cpu().numpy())
            save_mesh(attempt_mesh_path, v_smpl.detach().cpu().numpy(), self.smplx_face.detach().cpu().numpy())
            sqlite_path = os.environ.get("PSHUMAN_SMPL_SQLITE", f'{smpl_fit_dir}/smpl_fit_attempts.sqlite')
            try:
                conn = sqlite3.connect(sqlite_path)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS smpl_fit_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_name TEXT NOT NULL,
                        steps INTEGER NOT NULL,
                        active_betas INTEGER NOT NULL,
                        beta_lr REAL NOT NULL,
                        pose_lr REAL NOT NULL,
                        trans_lr REAL NOT NULL,
                        mask_weight REAL NOT NULL,
                        normal_weight REAL NOT NULL,
                        beta_l2 REAL NOT NULL,
                        beta_clamp REAL NOT NULL,
                        front_silhouette_iou REAL NOT NULL DEFAULT 0,
                        mean_silhouette_iou REAL NOT NULL DEFAULT 0,
                        beta_norm REAL NOT NULL,
                        beta_delta_max REAL NOT NULL,
                        attempt_dir TEXT NOT NULL DEFAULT '',
                        overlay_path TEXT NOT NULL DEFAULT '',
                        report_json TEXT NOT NULL,
                        params_path TEXT NOT NULL,
                        mesh_path TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                for col in [
                    ("front_silhouette_iou", "REAL NOT NULL DEFAULT 0"),
                    ("mean_silhouette_iou", "REAL NOT NULL DEFAULT 0"),
                    ("attempt_dir", "TEXT NOT NULL DEFAULT ''"),
                    ("overlay_path", "TEXT NOT NULL DEFAULT ''"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE smpl_fit_attempts ADD COLUMN {col[0]} {col[1]}")
                    except sqlite3.OperationalError:
                        pass
                conn.execute(
                    """
                    INSERT INTO smpl_fit_attempts (
                        case_name, steps, active_betas, beta_lr, pose_lr, trans_lr,
                        mask_weight, normal_weight, beta_l2, beta_clamp,
                        front_silhouette_iou, mean_silhouette_iou,
                        beta_norm, beta_delta_max, attempt_dir, overlay_path,
                        report_json, params_path, mesh_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case,
                        int(smpl_steps),
                        int(smpl_active_betas),
                        float(smpl_beta_lr),
                        float(smpl_pose_lr),
                        float(smpl_trans_lr),
                        float(smpl_mask_weight),
                        float(smpl_normal_weight),
                        float(smpl_beta_l2),
                        float(smpl_beta_clamp),
                        final_summary["final"]["front_silhouette_iou"],
                        final_summary["final"]["mean_silhouette_iou"],
                        final_summary["final"]["beta_norm"],
                        final_summary["final"]["beta_delta_max"],
                        attempt_dir,
                        attempt_overlay_path,
                        attempt_report_path,
                        attempt_params_path,
                        attempt_mesh_path,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as exc:
                print(f"[smpl-fit] sqlite logging failed: {exc}", flush=True)

        # Cross-view prior: inject SMPL-X silhouette into side-view masks.
        # See _inject_prior_silhouette docstring. No-op when PSHUMAN_XVIEW_MODE=off.
        masks, target_normals = self._inject_prior_silhouette(
            masks, target_normals, v_smpl.detach(), case_path=case_path
        )

        nrm_opt = MeshOptimizer(v_smpl.detach(), self.smplx_face.detach(), edge_len_lims=[0.01, 0.1])
        vertices, faces = nrm_opt.vertices, nrm_opt.faces

        # ---- synthetic acceptance check for cross-view depth loss ------------
        # Run the loss on the SMPL init mesh BEFORE any optimization. Since the
        # SMPL mesh is, by construction, perfectly cross-view-consistent (it's
        # one mesh viewed from 6 cameras), the per-pair loss must be small —
        # only sub-pixel sampling residual. Anything > ~5mm means the
        # projection plumbing is wrong and the loss would corrupt the geometry.
        if self.xview_depth_check or self.xview_depth_w > 0:
            with torch.no_grad():
                xyz_init = self.renderer.render_world_xyz(vertices, faces)
                v_init = cross_view_world_consistency(
                    xyz_init, self.renderer.mvp, self.xview_depth_pairs
                )
                # also per-pair breakdown for diagnosis
                per_pair = []
                for pa, pb in self.xview_depth_pairs:
                    pv = cross_view_world_consistency(
                        xyz_init, self.renderer.mvp, [(pa, pb)]
                    )
                    per_pair.append((pa, pb, float(pv.item())))
            print(
                f"[xview-depth] synthetic check on SMPL init: mean={float(v_init.item()):.5f} "
                f"per_pair={[(a, b, round(v, 5)) for a, b, v in per_pair]}",
                flush=True,
            )
            if self.xview_depth_w > 0 and float(v_init.item()) > 0.02:
                print(
                    "[xview-depth] WARNING synthetic residual > 20mm — "
                    "projection plumbing likely wrong; consider PSHUMAN_XVIEW_DEPTH_W=0",
                    flush=True,
                )

        # ###----------------------- optimization iterations-------------------------------------
        for i in tqdm(range(self.opt.iters)):
            nrm_opt.zero_grad()

            normals = calc_vertex_normals(vertices,faces)
            nrm = self.renderer.render(vertices,faces, normals=normals)
            normals = nrm[..., :3]   
            # if i < 800:
            loss = ((normals-target_normals) * self.weights).abs().mean()
            # else:
            #     loss = ((normals-target_images) * masks).abs().mean()
            
            alpha = nrm[..., 3:]
            loss += ((alpha - masks) * self.weights).abs().mean()

            # Cross-view world-XYZ consistency — closes the parallel-pancake
            # wrist artifact at the loss level. No-op when weight == 0.
            if self.xview_depth_w > 0:
                xyz_alpha = self.renderer.render_world_xyz(vertices, faces)
                xview_loss = cross_view_world_consistency(
                    xyz_alpha, self.renderer.mvp, self.xview_depth_pairs
                )
                loss = loss + self.xview_depth_w * xview_loss
                if self.debug_dump_normals and (i % max(1, self.debug_dump_every) == 0):
                    print(f"[xview-depth] iter {i:04d}  L_xview={float(xview_loss.item()):.5f}", flush=True)

            loss.backward()
            
            nrm_opt.step()
            
            vertices,faces = nrm_opt.remesh()

            # Per-iter rendered-normal dump (opt-in via PSHUMAN_DEBUG_DUMP_NORMALS=1).
            # Writes {case_path}/normals/{step:04d}.png so we can watch pancake
            # formation / verify the cross-view prior fixed it.
            dump_now = (
                self.debug_dump_normals
                and (i % max(1, self.debug_dump_every) == 0)
            )
            if self.opt.debug or dump_now:
                import imageio
                os.makedirs(f'{case_path}/normals', exist_ok=True)
                imageio.imwrite(f'{case_path}/normals/{i:04d}.png',(nrm.detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
                # mesh_remeshed = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                # mesh_remeshed.export(f'{case_path}/{case}_remeshed_step{i}.obj')
            torch.cuda.empty_cache() 
            
        mesh_remeshed = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
        mesh_remeshed.export(f'{case_path}/{case}_remeshed.obj')
        # save_mesh(case, vertices, faces)
        vertices = vertices.detach()
        faces = faces.detach()

        # Permanently disabled in this fork.
        #
        # This local arm-stitch routine was never part of the upstream PSHuman
        # contract. In practice it bridges foreground/noise islands into the
        # torso, wrists, arms, and nearby props, creating the connected triangle
        # artifacts seen in recent project bakes. Keep the implementation above
        # only for forensic comparison; do not call it from production runs.
        if _env_bool("PSHUMAN_ARM_REPAIR", False):
            print(
                "[reconstruct] PSHUMAN_ARM_REPAIR requested but permanently disabled "
                "because it creates bridge artifacts",
                flush=True,
            )

        #### replace hand
        smpl_data = SMPLX()
        if self.opt.replace_hand  and True in pose['hands_visibility'][0]:
            hand_mask = torch.zeros(smpl_data.smplx_verts.shape[0], )
            if pose['hands_visibility'][0][0]:
                hand_mask.index_fill_(
                    0, torch.tensor(smpl_data.smplx_mano_vid_dict["left_hand"]), 1.0
                )
            if pose['hands_visibility'][0][1]:
                hand_mask.index_fill_(
                    0, torch.tensor(smpl_data.smplx_mano_vid_dict["right_hand"]), 1.0
                )

            hand_mesh = apply_vertex_mask(mesh_smpl.copy(), hand_mask)
            body_mesh = part_removal(
                mesh_remeshed.copy(),
                hand_mesh,
                0.08,
                self.device,
                mesh_smpl.copy(),
                region="hand"
            )
            final = poisson(sum([hand_mesh, body_mesh]), f'{case_path}/{case}_final.obj', 10, False)
        else:
            final = poisson(mesh_remeshed, f'{case_path}/{case}_final.obj', 10, False)
        vertices = torch.from_numpy(final.vertices).float().to(self.device)
        faces = torch.from_numpy(final.faces).long().to(self.device)
        # Differing from paper, we use the texturing method in Unique3D
        masked_color = []
        masked_normals = []
        for tmp in clr_img:
            # tmp = Image.open(f'{self.opt.mv_path}/{case}/color_{view}_masked.png')
            tmp = tmp.resize((self.resolution, self.resolution), Image.BILINEAR)
            tmp = np.array(tmp).astype(np.float32) / 255.
            masked_color.append(torch.from_numpy(tmp).permute(2, 0, 1).to(self.device))
        for tmp in nrm_img:
            tmp = tmp.resize((self.resolution, self.resolution), Image.BILINEAR)
            tmp = np.array(tmp).astype(np.float32) / 255.
            masked_normals.append(torch.from_numpy(tmp).permute(2, 0, 1).to(self.device))

        meshes = self.proj_texture(masked_color, vertices, faces, normal_images=masked_normals)
        vertices = meshes.verts_packed().float()
        faces = meshes.faces_packed().long()
        colors = meshes.textures.verts_features_packed().float()
        save_mesh(f'{case_path}/result_clr_scale{self.opt.scale}_{case}.obj', vertices, faces, colors)
        self.evaluate(vertices, colors, faces,  save_path=f'{case_path}/result_clr_scale{self.opt.scale}_{case}.mp4', save_nrm=True)
        

    def evaluate(self, target_vertices, target_colors, target_faces, save_path=None, save_nrm=False):
        mv, proj = make_round_views(60, self.opt.scale, device=self.device)
        renderer = NormalsRenderer(mv, proj, [512, 512], device=self.device)
        
        target_images = renderer.render(target_vertices,target_faces, colors=target_colors)
        target_images = target_images.detach().cpu().numpy()
        target_images = target_images[..., :3] * target_images[..., 3:4]  + bg_color * (1 - target_images[..., 3:4])
        target_images = (target_images.clip(0, 1) * 255).astype(np.uint8)
        
        if save_nrm:
            target_normals = calc_vertex_normals(target_vertices, target_faces)
            # target_normals[:, 2] *= -1
            target_normals = renderer.render(target_vertices, target_faces, normals=target_normals)
            target_normals = target_normals.detach().cpu().numpy()
            target_normals = target_normals[..., :3] * target_normals[..., 3:4]  + bg_color * (1 - target_normals[..., 3:4])
            target_normals = (target_normals.clip(0, 1) * 255).astype(np.uint8)
            frames = [np.concatenate([img, nrm], 1) for img, nrm in zip(target_images, target_normals)]
        else:
            frames = [img for img in target_images]
        if save_path is not None:
            write_video(frames, fps=25, save_path=save_path)
        return frames
    
    def run(self):
        cases = sorted(os.listdir(self.opt.imgs_path))  
        for idx in range(len(cases)):
            case = cases[idx].split('.')[0]
            print(f'Processing {case}')
            pose = self.econ_dataset.__getitem__(idx)
            v, f, c =  self.optimize_case(case, pose, None, None, opti_texture=True)
            self.evaluate(v, c, f,  save_path=f'{self.opt.res_path}/{case}/result_clr_scale{self.opt.scale}_{case}.mp4', save_nrm=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  help="path to the yaml configs file", default='config.yaml')
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    from econdataset import SMPLDataset
    dataset_param = {'image_dir': opt.imgs_path, 'seg_dir': None, 'colab': False, 'has_det': True, 'hps_type': 'pixie'}
    econdata = SMPLDataset(dataset_param, device='cuda')
    EHuman = ReMesh(opt, econdata)
    EHuman.run()

   
    

"""Source-support sampling with an independent cutout-edge safety mask."""


def sample_source_color(image, silhouette, points, *, alpha_erode_px=0, alpha_edge_threshold=0.95):
    """Sample supported RGB without mixing unsupported RGB into a contribution.

    ``image`` is CHW RGB plus sparse support alpha; ``silhouette`` is the
    original same-view CHW RGBA cutout; ``points`` are Nx2 grid coordinates.
    The sampled support determines contribution weight. Only silhouette alpha
    is eroded for edge safety. No geometry/normal/depth gate is relaxed.
    """
    import torch
    import torch.nn.functional as F

    if image.ndim != 3 or image.shape[0] != 4 or silhouette.shape != image.shape:
        raise ValueError('source and silhouette must be matching CHW RGBA')
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('points must be Nx2 grid coordinates')
    if type(alpha_erode_px) is not int or alpha_erode_px < 0:
        raise ValueError('alpha erosion radius must be a nonnegative integer')
    if not 0 <= float(alpha_edge_threshold) <= 1:
        raise ValueError('invalid silhouette threshold')
    support = image[3:4].clamp(0, 1)
    premultiplied = torch.cat((image[:3] * support, support), dim=0)
    grid = points[None, :, None, :]
    sample = F.grid_sample(premultiplied[None].flip((-1, -2)), grid,
                           align_corners=False, padding_mode='border', mode='bilinear')[0, :, :, 0].T.clamp(0, 1)
    alpha = sample[:, 3:4]
    rgb = torch.where(alpha > 1e-8, sample[:, :3] / alpha.clamp_min(1e-8), torch.zeros_like(sample[:, :3]))
    if alpha_erode_px:
        edge_alpha = silhouette[3:4][None]
        for _ in range(alpha_erode_px):
            edge_alpha = -F.max_pool2d(-edge_alpha, kernel_size=3, stride=1, padding=1)
        safe = F.grid_sample(edge_alpha.flip((-1, -2)), grid, align_corners=False,
                             padding_mode='zeros', mode='bilinear')[0, :, :, 0].T.clamp(0, 1)
        alpha = alpha * (safe >= float(alpha_edge_threshold)).to(alpha.dtype)
    return rgb, alpha


def validate_baseline_colors(vertices, faces, baseline):
    """Accept only colors explicitly bound to this exact indexed mesh."""
    import torch
    if not isinstance(baseline, dict) or set(baseline) != {"vertices", "faces", "colors"}:
        raise ValueError("baseline must contain vertices, faces and colors")
    bv, bf, colors = baseline["vertices"], baseline["faces"], baseline["colors"]
    if not all(isinstance(v, torch.Tensor) for v in (bv, bf, colors)):
        raise ValueError("baseline arrays must be tensors")
    if bv.dtype != vertices.dtype or bf.dtype != faces.dtype or not torch.equal(bv.to(vertices.device), vertices) or not torch.equal(bf.to(faces.device), faces):
        raise ValueError("baseline geometry and topology must exactly match the mesh")
    if colors.shape != vertices.shape or not colors.is_floating_point() or not torch.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
        raise ValueError("baseline colors must be finite RGB values in [0, 1]")
    return colors.to(device=vertices.device, dtype=vertices.dtype).clone()

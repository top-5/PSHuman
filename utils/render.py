"""DRTK normal and world-coordinate buffers for PSHuman reconstruction.

The existing GL clip matrices define output orientation (the sparse-camera
builder already flips Y). DRTK supplies hard-forward visibility with its explicit
edge-gradient estimator; this is not nvdiffrast's fractional antialias filter.
"""

import torch

from seed.render import drtk_ops


class NormalsRenderer:
    backend = "drtk"
    coverage = "hard forward with DRTK visibility-boundary gradients"

    def __init__(
        self,
        mv: torch.Tensor,
        proj: torch.Tensor,
        image_size: tuple[int, int],
        device: str,
    ):
        if len(image_size) != 2 or any(
            int(value) != value or value <= 0 for value in image_size
        ):
            raise ValueError("image_size must contain positive height and width")
        mv = torch.as_tensor(mv, dtype=torch.float32, device=device)
        proj = torch.as_tensor(proj, dtype=torch.float32, device=device)
        if mv.shape[-2:] != (4, 4) or proj.shape[-2:] != (4, 4):
            raise ValueError("mv and proj must be batched 4x4 matrices")
        if mv.ndim == 2:
            mv = mv[None]
        if proj.ndim == 2:
            proj = proj[None]
        last = proj[..., 3, :]
        orthographic = torch.allclose(
            last, last.new_tensor([0, 0, 0, 1]).expand_as(last)
        )
        perspective = bool(
            (last[..., :2].abs() < 1e-7).all()
            and (last[..., 2].abs() > 1e-7).all()
            and (last[..., 3].abs() < 1e-7).all()
        )
        if not orthographic and not perspective:
            raise NotImplementedError(
                "DRTK renderer supports standard GL orthographic or perspective projections"
            )
        self._projection = "orthographic" if orthographic else "perspective"
        self._mvp = proj @ mv
        self._image_size = tuple(map(int, image_size))
        self.runtime = drtk_ops.runtime_identity()
        self._glctx = None  # compatibility only: DRTK creates no OpenGL context

    def _raster(self, vertices, faces):
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError("vertices must be [V,3]")
        homogeneous = torch.cat((vertices, torch.ones_like(vertices[:, :1])), -1)
        clip = homogeneous @ self._mvp.transpose(-2, -1)
        return drtk_ops.rasterize(
            clip.contiguous(),
            faces.to(torch.int32),
            self._image_size,
            projection=self._projection,
        )

    def render(
        self, vertices, faces, colors=None, normals=None, return_triangles=False
    ):
        """Return [C,H,W,4] encoded normal/color and alpha buffers."""
        attributes = (normals + 1) / 2 if normals is not None else colors
        if attributes is None or attributes.shape != vertices.shape:
            raise ValueError("normals or colors must be [V,3] matching vertices")
        raster = self._raster(vertices, faces)
        color = drtk_ops.interpolate(attributes[None].contiguous(), raster)
        rgba = torch.cat((color, raster.valid[..., None].to(color.dtype)), -1)
        rgba = drtk_ops.edge_gradients(rgba, raster)
        if return_triangles:
            return rgba, (raster.index + 1).to(rgba.dtype)
        return rgba

    def render_world_xyz(self, vertices, faces):
        """Return same-view hard world XYZ/alpha for cross-view geometry losses."""
        raster = self._raster(vertices, faces)
        xyz = drtk_ops.interpolate(vertices[None].contiguous(), raster)
        return torch.cat((xyz, raster.valid[..., None].to(xyz.dtype)), -1)

    @property
    def mvp(self):
        """[C,4,4] GL projection@view matrices for cross-view operations."""
        return self._mvp

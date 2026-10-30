"""Separate registered source color from inferred fallback in a six-view bundle."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def load_support(path, expected_sha256, images):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Source-color support hash differs from the bound bridge")
    with np.load(path, allow_pickle=False) as data:
        observed = data["color_observed"]
        inferred = data["parent_color_inferred"]
        foreground = data["baseline_foreground"]
    if len(images) != 6 or observed.dtype != bool or inferred.dtype != bool or foreground.dtype != bool:
        raise ValueError("Six boolean observed, inferred, and foreground masks required")
    if observed.shape != inferred.shape or observed.shape != foreground.shape or observed.ndim != 3 or observed.shape[0] != 6:
        raise ValueError("Source-color support dimensions differ")
    if (observed & inferred).any() or (inferred & ~foreground).any():
        raise ValueError("Color support must be disjoint and inferred support inside baseline foreground")
    if any(image.size != (observed.shape[2], observed.shape[1]) for image in images):
        raise ValueError("Source-color support and view dimensions differ")
    if any(((observed[i] | inferred[i]) & (np.asarray(image.convert("RGBA"))[..., 3] == 0)).any()
           for i, image in enumerate(images)):
        raise ValueError("Color support lies outside view foreground")
    return observed, inferred


def masked_images(images, masks):
    """Give projection its own support alpha; never promote neutral fill."""
    result = []
    for image, mask in zip(images, masks, strict=True):
        rgba = np.asarray(image.convert("RGBA")).copy()
        rgba[..., 3] = np.minimum(rgba[..., 3], mask.astype(np.uint8) * 255)
        result.append(Image.fromarray(rgba, "RGBA"))
    return result

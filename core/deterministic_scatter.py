"""Deterministic drop-in replacements for the torch_scatter calls in opt.py and remesh.py.

WHY. torch_scatter's CUDA kernels reduce with atomic ops: the thread that lands first wins the
race, so the floating-point summation/comparison order -- and therefore the result -- varies run
to run on the same input. PSHuman's reconstruction is otherwise deterministic (the diffusion step
is); this scatter_mean (opt.py, Laplacian neighbor-smoothing) and the two scatter_max calls
(remesh.py, edge-collapse and edge-flip priority ranks) are the two remaining sources of
run-to-run mesh variance (seed infra-memory 234c8394).

WHAT. Both reduce via sort instead of atomics: group every (src, index) pair by its target index
through one deterministic `torch.argsort` (stable, so ties keep their input order), then read each
segment's extent from `torch.bincount` + `torch.cumsum` -- both are themselves order-independent,
not accumulated element-by-element, so there is nothing left for thread-completion order to
affect. The result is bit-identical across runs on the same input and device.

Matches torch_scatter.scatter_mean/scatter_max's own call signature (`src`, `index`, `dim`, `out`)
and semantics exactly, including dim=0-only, index broadcasting from (N,1) to (N,D) when `index`
has fewer trailing dims than `src`, and leaving an output row untouched when no `src` row maps to
it (scatter_max's `out[idx] = max(out[idx], src)` also keeps combining with whatever `out` already
held, since remesh.py reuses one `out` tensor across a loop of scatter_max calls).
"""

from __future__ import annotations

import torch


def _broadcast_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """torch_scatter broadcasts an index with fewer trailing dims than src across them (dim=0
    only here): index (N, 1) against src (N, D) becomes index (N, D), same index per row."""
    if index.dim() == 1:
        index = index.view(-1, *([1] * (src.dim() - 1)))
    return index.expand_as(src)


def _segment_sum_and_count(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-index sum and count of a 1-D `values`/`index` pair, via sort + cumsum."""
    order = torch.argsort(index, stable=True)
    sorted_values = values[order]
    cumsum = torch.cat([sorted_values.new_zeros(1), torch.cumsum(sorted_values, dim=0)])
    counts = torch.bincount(index, minlength=dim_size)
    offsets = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    sums = cumsum[offsets[1:]] - cumsum[offsets[:-1]]
    return sums, counts


def _segment_max(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-index max of a 1-D `values`/`index` pair.

    Two stable sorts, no atomics: first group by index (as in `_segment_sum_and_count`), then a
    stable sort by descending value re-sorted back into index-ascending order -- a stable sort
    preserves relative order among equal keys, so within each index's block the entries end up in
    descending-value order, and the block's first entry is its max. `dim_size` entries with no
    matching index get `-inf` (the identity element for max, so `torch.maximum(out, result)`
    leaves `out` unchanged there, matching torch_scatter.scatter_max's own convention).
    """
    order = torch.argsort(index, stable=True)
    sorted_index, sorted_values = index[order], values[order]
    counts = torch.bincount(index, minlength=dim_size)
    offsets = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

    value_order = torch.argsort(sorted_values, descending=True, stable=True)
    regrouped_index = sorted_index[value_order]
    group_order = torch.argsort(regrouped_index, stable=True)
    descending_within_group = sorted_values[value_order][group_order]

    starts = offsets[:-1].clamp(max=max(len(descending_within_group) - 1, 0))
    maxima = descending_within_group[starts] if len(descending_within_group) else descending_within_group.new_zeros(dim_size)
    empty_sentinel = torch.iinfo(maxima.dtype).min if not maxima.dtype.is_floating_point else float("-inf")
    maxima = torch.where(counts > 0, maxima, torch.full_like(maxima, empty_sentinel))
    return maxima, counts


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    """Deterministic replacement for `torch_scatter.scatter_mean(src, index, dim, out)`."""
    if dim != 0:
        raise NotImplementedError("only dim=0 is used by opt.py/remesh.py")
    index = _broadcast_index(index, src)
    dim_size = out.shape[0]
    flat_src, flat_index = src.reshape(src.shape[0], -1), index.reshape(index.shape[0], -1)
    columns = []
    for c in range(flat_src.shape[1]):
        sums, counts = _segment_sum_and_count(flat_src[:, c], flat_index[:, c], dim_size)
        means = sums / counts.clamp(min=1).to(sums.dtype)
        columns.append(torch.where(counts > 0, means, out.reshape(out.shape[0], -1)[:, c]))
    result = torch.stack(columns, dim=1).reshape(out.shape)
    out.copy_(result)
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    """Deterministic replacement for `torch_scatter.scatter_max(src, index, dim, out)`."""
    if dim != 0:
        raise NotImplementedError("only dim=0 is used by opt.py/remesh.py")
    index = _broadcast_index(index, src)
    dim_size = out.shape[0]
    flat_src, flat_index = src.reshape(src.shape[0], -1), index.reshape(index.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    columns = []
    for c in range(flat_src.shape[1]):
        maxima, _ = _segment_max(flat_src[:, c], flat_index[:, c], dim_size)
        columns.append(torch.maximum(flat_out[:, c], maxima))
    result = torch.stack(columns, dim=1).reshape(out.shape)
    out.copy_(result)
    return out

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
import os


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
    # Subtracting global float32 prefix sums loses the small local coordinate
    # differences the mesh optimizer needs. At 100k groups even a constant
    # 1.001 field became 1.0..1.0078125. Accumulate AND subtract in float64;
    # scatter_mean casts only after dividing, when copying to the caller's out.
    # This preserves deterministic ordering without CUDA atomics.
    if sorted_values.dtype in (torch.float16, torch.bfloat16, torch.float32):
        sorted_values = sorted_values.to(torch.float64)
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


def _reference_scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
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


_verified_calls = 0
#: why PSHUMAN_FAST_MESH=1 runs the reference reduction instead, once known (None = not fallen back)
fast_mesh_fallback = None


def _fast_integer_scatter_max():
    """seed's native integer rank max, or None after recording why it is unavailable.

    The fast kernel lives in seed (seed.mesh_ops), which this fork's own Python environment reaches only through the
    runner's PYTHONPATH; a runner without it crashed every 10-27 suprim build in remesh (#394). The reference reduction
    gives the same integers, so the run continues, but never silently: a slow path must not hide inside a speed
    baseline. The first fallback warns and leaves a "fast_mesh: reference fallback (<reason>)" span in the run's trace.
    """
    global fast_mesh_fallback
    if fast_mesh_fallback is not None:
        return None
    try:
        from seed.mesh_ops.gpu_reductions import integer_scatter_max
    except ImportError as exc:
        fast_mesh_fallback = f"{type(exc).__name__}: {exc}"
        message = f"fast_mesh: reference fallback ({fast_mesh_fallback})"
        import warnings

        warnings.warn(message + "; PSHUMAN_FAST_MESH=1 asked for seed's native integer rank max", RuntimeWarning)
        print(f"[remesh] {message}", flush=True)
        try:
            import seed_spans
        except ImportError:  # a fork checkout used without seed: the warning above is the record
            return None
        import time

        now = time.time_ns()
        seed_spans.emit(message, now, now, fast_mesh="reference_fallback", reason=fast_mesh_fallback)
        return None
    return integer_scatter_max


def scatter_max(src, index, dim, out):
    global _verified_calls
    if dim != 0:
        raise NotImplementedError("only dim=0 is used")
    index = _broadcast_index(index, src)
    integer_scatter_max = None
    if (os.environ.get('PSHUMAN_FAST_MESH') == '1' and src.is_cuda
            and src.dtype in (torch.int32, torch.int64)):
        integer_scatter_max = _fast_integer_scatter_max()
    if integer_scatter_max is not None:
        verify = os.environ.get('PSHUMAN_FAST_VERIFY') == '1'
        reference = _reference_scatter_max(src, index, dim, out.clone()) if verify else None
        result = integer_scatter_max(src, index, out)
        if verify:
            if not torch.equal(reference, result):
                raise RuntimeError('Integer rank max failed exact same-input parity')
            _verified_calls += 1
        return result
    return _reference_scatter_max(src, index, dim, out)


def _write_verification():
    path = os.environ.get('PSHUMAN_FAST_VERIFY_RECEIPT')
    if path:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(dict(exact_checked_calls=_verified_calls, reference_fallback=fast_mesh_fallback,
            scope='integer rank arrays on identical live inputs; not full mesh parity'), indent=2)+'\n')

import atexit
atexit.register(_write_verification)

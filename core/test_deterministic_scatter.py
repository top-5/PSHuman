"""deterministic_scatter must match torch_scatter's own output (correctness) and must not vary
run to run on the same input (the actual bug: torch_scatter's CUDA atomics do, see
deterministic_scatter.py's module docstring; seed infra-memory 234c8394)."""

import pytest
import torch
import torch_scatter

from core import deterministic_scatter as det

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_mean_matches_torch_scatter_with_broadcast_index(device: str) -> None:
    # opt.py's call shape: src (E*2, S), index (E*2, 1) broadcast across S columns.
    torch.manual_seed(0)
    E, V, S = 500, 120, 8
    edges = torch.randint(0, V, (E, 2), device=device)
    src = torch.randn(E * 2, S, device=device)
    index = edges.reshape(E * 2, 1)

    ref = torch.zeros(V, S, device=device)
    torch_scatter.scatter_mean(src=src, index=index, dim=0, out=ref)

    out = torch.zeros(V, S, device=device)
    det.scatter_mean(src=src, index=index, dim=0, out=out)

    assert torch.allclose(ref, out, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_max_matches_torch_scatter_across_a_reused_out(device: str) -> None:
    # remesh.py's collapse_edges shape: src/index both 1-D, out reused (combined, not overwritten)
    # across repeated calls in a loop.
    torch.manual_seed(0)
    E, V = 500, 120
    edges = torch.randint(0, V, (E, 2), device=device)
    rank = torch.randperm(E, device=device)

    ref, out = torch.zeros(V, dtype=torch.long, device=device), torch.zeros(V, dtype=torch.long, device=device)
    rank_ref, rank_out = rank.clone(), rank.clone()
    for _ in range(3):
        torch_scatter.scatter_max(src=rank_ref[:, None].expand(-1, 2).reshape(-1), index=edges.reshape(-1), dim=0, out=ref)
        rank_ref, _ = ref[edges].max(dim=-1)
        det.scatter_max(src=rank_out[:, None].expand(-1, 2).reshape(-1), index=edges.reshape(-1), dim=0, out=out)
        rank_out, _ = out[edges].max(dim=-1)

    assert torch.equal(ref, out)


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_max_matches_torch_scatter_with_matched_shape_index(device: str) -> None:
    # flip_edges's shape: src and index both (N, 4) -- 4 independent column-wise scatters.
    torch.manual_seed(0)
    N, V = 500, 120
    edges_neighbors = torch.randint(0, V, (N, 4), device=device)
    rank = torch.randperm(N, device=device)

    ref = torch.zeros(V, 4, dtype=torch.long, device=device)
    torch_scatter.scatter_max(src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=ref)

    out = torch.zeros(V, 4, dtype=torch.long, device=device)
    det.scatter_max(src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=out)

    assert torch.equal(ref, out)


def test_scatter_max_leaves_an_untouched_row_unchanged() -> None:
    # An index with no matching src row must keep out's pre-existing value, not get overwritten.
    out = torch.tensor([7, 7, 7], dtype=torch.long)
    det.scatter_max(src=torch.tensor([3, 9]), index=torch.tensor([0, 0]), dim=0, out=out)
    assert out.tolist() == [9, 7, 7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the bug this guards is CUDA-only")
def test_torch_scatter_itself_is_nondeterministic_on_cuda_but_the_replacement_is_not() -> None:
    """Documents the actual defect being fixed, not just the replacement's correctness: repeated
    calls to torch_scatter.scatter_mean on the SAME input, same device, vary; the deterministic
    replacement does not."""
    torch.manual_seed(1)
    device = "cuda"
    E, V, S = 20000, 3000, 8
    edges = torch.randint(0, V, (E, 2), device=device)
    src = torch.randn(E * 2, S, device=device)
    index = edges.reshape(E * 2, 1)

    det_results = []
    for _ in range(8):
        out = torch.zeros(V, S, device=device)
        det.scatter_mean(src=src, index=index, dim=0, out=out)
        det_results.append(out.clone())
    assert all(torch.equal(det_results[0], r) for r in det_results[1:])

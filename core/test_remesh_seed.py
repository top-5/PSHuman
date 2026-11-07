import pytest
import torch

from core.opt import MeshOptimizer, remesh
from core.remesh import calc_face_collapses


def _collapse_inputs():
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=torch.long
    )
    edges = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long
    )
    face_to_edge = torch.tensor(
        [[0, 3, 1], [1, 5, 2], [2, 4, 0], [4, 5, 3]], dtype=torch.long
    )
    vertices = torch.zeros((4, 3))
    edge_length = torch.tensor([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    face_normals = -torch.ones((4, 3))
    vertex_normals = torch.ones((4, 3))
    return vertices, faces, edges, face_to_edge, edge_length, face_normals, vertex_normals


def _collapse_with(generator):
    return calc_face_collapses(
        *_collapse_inputs(),
        shortest_probability=0.5,
        generator=generator,
    )


def test_dedicated_generator_is_independent_of_global_rng():
    first = torch.Generator().manual_seed(314159)
    expected = _collapse_with(first)

    torch.manual_seed(999)
    torch.randint(0, 1000, (10000,))
    second = torch.Generator().manual_seed(314159)
    actual = _collapse_with(second)
    assert torch.equal(actual, expected)


def test_mesh_optimizer_seed_is_explicit_and_rejects_invalid_value(monkeypatch):
    vertices = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = torch.tensor([[0, 2, 3], [2, 1, 3], [1, 0, 3], [0, 1, 2]])

    monkeypatch.setenv("PSHUMAN_REMESH_SEED", "42")
    optimizer = MeshOptimizer(vertices.clone(), faces.clone())
    assert optimizer._remesh_generator is not None
    assert optimizer._remesh_generator.initial_seed() == 42

    monkeypatch.setenv("PSHUMAN_REMESH_SEED", "not-an-int")
    with pytest.raises(ValueError, match="must be an integer"):
        MeshOptimizer(vertices.clone(), faces.clone())


def test_unset_seed_preserves_global_rng_mode(monkeypatch):
    monkeypatch.delenv("PSHUMAN_REMESH_SEED", raising=False)
    vertices = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = torch.tensor([[0, 2, 3], [2, 1, 3], [1, 0, 3], [0, 1, 2]])
    assert MeshOptimizer(vertices, faces)._remesh_generator is None


def test_full_remesh_replay_is_exact_with_restored_generator_state():
    vertices = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=torch.long,
    )
    state = torch.zeros((len(vertices), 9), dtype=torch.float32)
    state[:, :3] = vertices
    state[:, 8] = 0.5
    minimum = torch.full((len(vertices),), 0.2)
    maximum = torch.full((len(vertices),), 0.8)

    first_generator = torch.Generator().manual_seed(7)
    first_state, first_faces = remesh(
        state.clone(), faces.clone(), minimum.clone(), maximum.clone(), True,
        generator=first_generator,
    )
    first_post_state = first_generator.get_state()

    torch.manual_seed(123456)
    torch.randint(0, 1000, (10000,))
    second_generator = torch.Generator().manual_seed(7)
    second_state, second_faces = remesh(
        state.clone(), faces.clone(), minimum.clone(), maximum.clone(), True,
        generator=second_generator,
    )
    assert torch.equal(second_state, first_state)
    assert torch.equal(second_faces, first_faces)
    assert torch.equal(second_generator.get_state(), first_post_state)

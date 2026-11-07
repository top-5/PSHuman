import torch

from core.opt import remesh
from core.remesh import calc_edges, prepend_dummies


def _octahedron():
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
    return state, faces


def test_cached_adjacency_transforms_exactly_to_dummy_topology():
    state, faces = _octahedron()
    cached_edges, cached_face_to_edge = calc_edges(faces)
    _, dummy_faces = prepend_dummies(state, faces)
    expected_edges, expected_face_to_edge = calc_edges(dummy_faces)
    transformed_edges = torch.cat((torch.zeros((1, 2), dtype=torch.long), cached_edges + 1))
    transformed_face_to_edge = torch.cat((torch.zeros((1, 3), dtype=torch.long), cached_face_to_edge + 1))
    assert torch.equal(transformed_edges, expected_edges)
    assert torch.equal(transformed_face_to_edge, expected_face_to_edge)


def test_cached_adjacency_preserves_full_remesh_result():
    state, faces = _octahedron()
    minimum = torch.full((len(state),), 0.2)
    maximum = torch.full((len(state),), 0.8)
    adjacency = calc_edges(faces)

    torch.manual_seed(17)
    expected_state, expected_faces = remesh(
        state.clone(), faces.clone(), minimum.clone(), maximum.clone(), True
    )
    torch.manual_seed(17)
    actual_state, actual_faces = remesh(
        state.clone(),
        faces.clone(),
        minimum.clone(),
        maximum.clone(),
        True,
        collapse_adjacency=adjacency,
    )
    assert torch.equal(actual_faces, expected_faces)
    assert torch.equal(actual_state, expected_state)

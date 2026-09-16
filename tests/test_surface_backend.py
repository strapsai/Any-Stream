"""Geometric and transport invariants for the deployable surface backend."""
import copy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from loop_utils.metric_io import read_bundle, validate_graph, write_bundle
from loop_utils.metric_surface import solve, transform


def problem():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(24, 3))
    C = np.tile(np.eye(4), (6, 1, 1))
    C[:, :3, 3] = rng.normal(size=(6, 3))
    true = (2., Rotation.from_rotvec([.1, -.2, .3]).as_matrix(), np.array([20., -10., 3.]))
    initial = [(true[0], true[1].copy(), true[2] + [1., 0., 0.]),
               (true[0], true[1].copy(), true[2] + [-1., .5, 0.])]
    graph = dict(absolutes=initial, state=dict(local_c2w=[C.copy(), C.copy()], chunk_indices=[(0, 6), (6, 12)]),
                 gps=np.tile(transform(C[:, :3, 3], true), (2, 1)), valid_gps=np.ones(12, bool),
                 seams=[(a.copy(), a.copy())], loops=[],
                 anchors=[dict(chunk=0, local=a.copy(), target=transform(a, true), sigma=.1)])
    return graph


def test_transport_preserves_numeric_observations_and_detects_corruption(tmp_path):
    graph = problem()
    graph['seams'][0] = tuple(a.astype(np.float32) for a in graph['seams'][0])
    path = write_bundle(tmp_path/'graph', dict(graph=graph), dict(epoch=1))
    payload, meta = read_bundle(path)
    assert meta == dict(epoch=1)
    assert payload['graph']['seams'][0][0].dtype == np.float32
    assert np.array_equal(payload['graph']['seams'][0][0], graph['seams'][0][0])
    validate_graph(payload['graph'])
    blob = path.parent/'arrays.npz'
    blob.write_bytes(blob.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        read_bundle(path)


def test_warm_start_preserves_the_original_objective_and_base():
    graph = problem()
    original = copy.deepcopy(graph)
    poses, first = solve(graph, max_nfev=200)
    _, warm = solve(graph, initial_absolutes=poses, max_nfev=2)
    assert abs(first['cost'] - warm['initial_cost']) < 1e-7
    assert warm['cost'] <= warm['initial_cost'] + 1e-8
    assert warm['warm_start_chunks'] == 2
    for before, after in zip(original['absolutes'], graph['absolutes']):
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
    assert first['cost'] < first['initial_cost'] / 10


def test_surface_objective_respects_a_rigid_change_of_world_frame():
    graph = problem()
    a, da = solve(graph, max_nfev=200)
    Q = Rotation.from_rotvec([.4, -.2, .1]).as_matrix()
    t = np.array([100., -200., 40.])
    other = copy.deepcopy(graph)
    other['absolutes'] = [(s, Q @ R, Q @ p + t) for s, R, p in graph['absolutes']]
    other['gps'] = graph['gps'] @ Q.T + t
    other['anchors'][0]['target'] = graph['anchors'][0]['target'] @ Q.T + t
    b, db = solve(other, max_nfev=200)
    assert abs(da['cost'] - db['cost']) < 1e-6
    for pa, pb in zip(a, b):
        np.testing.assert_allclose(transform(graph['seams'][0][0], pa) @ Q.T + t,
                                   transform(graph['seams'][0][0], pb), atol=2e-4)


def test_bad_rotation_and_invalid_factor_are_rejected():
    graph = problem()
    graph['absolutes'][0][1][:] *= 2
    with pytest.raises(ValueError, match='orthonormal'):
        solve(graph)
    graph = problem()
    graph['anchors'][0]['chunk'] = 9
    with pytest.raises(ValueError, match='invalid chunk'):
        solve(graph)


def test_single_chunk_anchored_graph_is_supported():
    graph = problem()
    graph['absolutes'] = graph['absolutes'][:1]
    graph['state'] = dict(local_c2w=graph['state']['local_c2w'][:1],chunk_indices=[(0,6)])
    graph['gps'], graph['valid_gps'], graph['seams'] = graph['gps'][:6], graph['valid_gps'][:6], []
    poses, diag = solve(graph, max_nfev=100)
    assert len(poses) == 1 and diag['cost'] < diag['initial_cost']

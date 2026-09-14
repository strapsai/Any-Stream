"""Portable synthetic tests for the experimental objective and graph interchange."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.gps_pgo import optimizer, reference
from experiments.gps_pgo.problem import load_problem, save_problem


def make_problem():
    rng = np.random.default_rng(410)
    base, cameras, gps, true = [], [], [], []
    for k in range(3):
        scale = 2. + k
        translation = np.array([5. * k, 0., 0.])
        true.append((scale, np.eye(3), translation))
        base.append((scale, np.eye(3), translation + np.array([0., 0., k * .5])))
        poses = np.tile(np.eye(4), (10, 1, 1))
        poses[:, :3, 3] = rng.normal(size=(10, 3))
        cameras.append(poses)
        gps.append(optimizer.transform(poses[:, :3, 3], true[-1]))

    def pairs(i, j):
        world = rng.normal(size=(60, 3)) * 3
        return ((world - true[i][2]) / true[i][0],
                (world - true[j][2]) / true[j][0])

    a, b = pairs(0, 2)
    return dict(absolutes=base, gps=np.concatenate(gps), valid_gps=np.ones(30, bool),
                state=dict(local_c2w=cameras, chunk_indices=[(0, 10), (10, 20), (20, 30)]),
                seams=[pairs(0, 1), pairs(1, 2)], loops=[dict(i=0, j=2, a=a, b=b)])


def residual_function(module, data, **options):
    captured = []

    def capture(function, z, **kwargs):
        captured.append(function)
        return SimpleNamespace(x=z, success=True, status=1, message="test capture",
                               nfev=0, cost=0., optimality=0.)

    with patch.object(module, "least_squares", capture):
        module.solve(data, **options)
    return captured[0]


class MetricGraphTests(unittest.TestCase):
    def test_batched_residual_matches_independent_reference(self):
        data = make_problem()
        rng = np.random.default_rng(411)
        cases = [dict(), dict(compress=True, vector_loss=False),
                 dict(compress=True, block_loss=True, vector_loss=False),
                 dict(compress=True, block_loss=True, gps_only_block=True, vector_loss=False)]
        for options in cases:
            slow = residual_function(reference, data, **options)
            fast = residual_function(optimizer, data, **options)
            for _ in range(20):
                z = rng.normal(size=(3, 7))
                z[:, :3] *= .1
                z[:, 6] *= .02
                np.testing.assert_allclose(slow(z.ravel()), fast(z.ravel()), atol=1e-10)

    def test_full_objective_is_invariant_to_rigid_coordinate_change(self):
        data = make_problem()
        changed = copy.deepcopy(data)
        G = Rotation.from_euler("xyz", [37., -15., 60.], degrees=True).as_matrix()
        offset = np.array([100., -50., 20.])
        changed["absolutes"] = [(s, G @ R, G @ t + offset)
                                for s, R, t in data["absolutes"]]
        changed["gps"] = data["gps"] @ G.T + offset
        original = residual_function(optimizer, data)
        moved = residual_function(optimizer, changed)
        rng = np.random.default_rng(412)
        for _ in range(20):
            z = rng.normal(size=(3, 7)) * .1
            zz = z.copy()
            zz[:, :3] = z[:, :3] @ G.T
            zz[:, 3:6] = z[:, 3:6] @ G.T
            a, b = original(z.ravel()), moved(zz.ravel())
            self.assertAlmostEqual(float(a @ a), float(b @ b), places=7)

    def test_pointwise_robust_moment_compression_is_rejected(self):
        for module in [optimizer, reference]:
            with self.assertRaisesRegex(ValueError, "Moment summaries"):
                module.solve(make_problem(), compress=True)

    def test_solver_corrects_known_chunk_displacement_without_rescaling(self):
        data = make_problem()
        result, diagnostics = optimizer.solve(data)
        self.assertTrue(diagnostics["success"], diagnostics)
        np.testing.assert_array_equal([x[0] for x in result],
                                      [x[0] for x in data["absolutes"]])
        errors = np.concatenate([optimizer.transform(a, result[k])
                                 - optimizer.transform(b, result[k + 1])
                                 for k, (a, b) in enumerate(data["seams"])])
        self.assertLess(float(np.quantile(np.linalg.norm(errors, axis=1), .9)), .01)

    def test_npz_round_trip_preserves_residuals(self):
        data = make_problem()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.npz"
            save_problem(path, data)
            loaded = load_problem(path)
        z = np.arange(21) * .001
        np.testing.assert_array_equal(residual_function(optimizer, data)(z),
                                      residual_function(optimizer, loaded)(z))

    def test_invalid_geometry_is_rejected(self):
        data = make_problem()
        data["seams"][0][0][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.npz"
            save_problem(path, data)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_problem(path)


if __name__ == "__main__":
    unittest.main()

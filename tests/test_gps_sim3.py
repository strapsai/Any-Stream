"""GPS graph regressions; run with da3_streaming on PYTHONPATH and GTSAM 4.3a1."""
import unittest
from unittest.mock import patch

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from loop_utils.sim3loop import Sim3GPSFactor, Sim3LoopOptimizer


def transform(points, pose):
    s, rotation, translation = pose
    return s * (points @ rotation.T) + translation


def fixture(**options):
    config = dict(lang_version="python", max_iterations=60, lambda_init="1e-6",
                  init_method="umeyama", gps_sigma_t=2.0, heading_sigma=1e6)
    config.update(options)
    optimizer = Sim3LoopOptimizer({"Loop": {"SIM3_Optimizer": config}})
    sequential = [(2.0, np.eye(3), np.array([3., 1., 0.])),
                  (0.5, np.eye(3), np.array([1., 2., 0.]))]
    measurements = []
    for k in range(3):
        for j in range(k + 1):
            measurements.append(dict(chunk_k=k, c_loc=np.array([j, 0., 0.]),
                                     p_obs=np.array([20. + 7 * k + j, -5., 3.]),
                                     v_loc=np.array([1., 0., 0.]),
                                     v_obs=np.array([1., 0., 0.])))
    alignment = (3.0, np.eye(3), np.array([20., -5., 3.]))
    return optimizer, sequential, measurements, alignment


def capture_graph(optimizer, sequential, measurements, alignment):
    """Capture the real graph, bypassing optimization to inspect its measurements."""
    captured = {}

    class Capture:
        def __init__(self, graph, initial, params):
            captured.update(graph=graph, initial=initial)
            self.graph, self.initial = graph, initial

        def error(self):
            return self.graph.error(self.initial)

        def optimize(self):
            return self.initial

        def iterations(self):
            return 0

    with patch("loop_utils.sim3loop.gtsam.LevenbergMarquardtOptimizer", Capture):
        captured["output"] = optimizer.optimize_gps_sim3(sequential, measurements, alignment)
    return captured


class SimilarityConventionTests(unittest.TestCase):
    def test_action_composition_between_and_gps_residual(self):
        rng = np.random.default_rng(75)
        encode = Sim3LoopOptimizer._gtsam_sim3_from_srt
        decode = Sim3LoopOptimizer._srt_from_gtsam_sim3
        for _ in range(50):
            a, b = [(float(rng.uniform(.2, 50)),
                     Rotation.random(random_state=rng).as_matrix(),
                     rng.normal(size=3) * 30) for _ in range(2)]
            p = rng.normal(size=3) * 10
            A, B = encode(*a), encode(*b)
            np.testing.assert_allclose(A.transformFrom(p), transform(p, a), atol=1e-9)
            np.testing.assert_allclose(transform(p, decode(A.compose(B))),
                                       transform(transform(p, b), a), atol=1e-8)
            np.testing.assert_allclose(A.compose(A.between(B)).transformFrom(p),
                                       B.transformFrom(p), atol=1e-8)
            direction = np.array([1., 0., 0.])
            factor = Sim3GPSFactor(gtsam.symbol("x", 0), transform(p, a),
                                   a[1] @ direction, p, direction,
                                   gtsam.noiseModel.Isotropic.Sigma(5, 1.))
            np.testing.assert_allclose(factor._residual(A), np.zeros(5), atol=1e-9)

    def test_initialization_and_output_use_metric_translation(self):
        args = fixture()
        result = capture_graph(*args)
        optimizer, sequential, _, alignment = args
        for actual, model in zip(result["output"],
                                 optimizer.sequential_to_absolute_poses(sequential)):
            from loop_utils.sim3loop import pp
            model = optimizer.pypose_sim3_to_numpy(pp.Sim3(model))
            points = np.array([[0., 0., 0.], [1., 2., 3.]])
            np.testing.assert_allclose(transform(points, actual),
                                       transform(transform(points, model), alignment), atol=1e-8)


if __name__ == "__main__":
    unittest.main()

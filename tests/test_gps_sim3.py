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


class VisualMeasurementTests(unittest.TestCase):
    def test_edges_do_not_follow_gps_anchoring(self):
        graphs = []
        for method in ["umeyama", "anchored"]:
            args = fixture(init_method=method)
            graphs.append(capture_graph(*args))
        # Anchoring really changed the guess; the test is not vacuous.
        self.assertGreater(np.linalg.norm(graphs[0]["output"][1][2]
                                          - graphs[1]["output"][1][2]), 1.)
        points = np.array([[0., 0., 0.], [1., 2., 3.]])
        for k, expected in enumerate(fixture()[1]):
            measurements = []
            for captured in graphs:
                graph = captured["graph"]
                edges = [graph.at(i) for i in range(graph.size())
                         if isinstance(graph.at(i), gtsam.BetweenFactorSimilarity3)]
                measurements.append(edges[k].measured())
            for measured in measurements:
                np.testing.assert_allclose(
                    [measured.transformFrom(p) for p in points],
                    transform(points, expected), atol=1e-7)


class MetricWeightTests(unittest.TestCase):
    def test_between_translation_units(self):
        rng = np.random.default_rng(31)
        encode = Sim3LoopOptimizer._gtsam_sim3_from_srt
        for _ in range(50):
            s, u = rng.uniform(.2, 80, 2)
            R, Q = Rotation.random(2, random_state=rng).as_matrix()
            t, v = rng.normal(size=(2, 3)) * 20
            delta = rng.normal(size=3) * .3
            A, B, C = encode(s, R, t), encode(u, Q, v), encode(u, Q, v + delta)
            error = gtsam.Similarity3.Logmap(A.between(B).between(A.between(C)))
            np.testing.assert_allclose(error[:3], 0., atol=1e-9)
            self.assertAlmostEqual(error[6], 0.)
            self.assertAlmostEqual(np.linalg.norm(error[3:6]) * u, np.linalg.norm(delta))

    def test_metric_sigmas_and_scale_priors(self):
        result = capture_graph(*fixture(seq_sigma_t_m=.1, per_chunk_scale_prior_sigma=1e-5))
        graph = result["graph"]
        edges = [graph.at(i) for i in range(graph.size())
                 if isinstance(graph.at(i), gtsam.BetweenFactorSimilarity3)]
        for k, edge in enumerate(edges):
            destination = result["initial"].atSimilarity3(gtsam.symbol("x", k + 1))
            np.testing.assert_allclose(edge.noiseModel().sigmas()[3:6], .1 / destination.scale())
        priors = [graph.at(i) for i in range(graph.size())
                  if isinstance(graph.at(i), gtsam.PriorFactorSimilarity3)]
        self.assertEqual(len(priors), 3)
        for prior in priors:
            self.assertAlmostEqual(prior.noiseModel().sigmas()[-1], 1e-5)

    def test_grouping_counts_only_accepted_factors(self):
        args = list(fixture(gps_chunk_information_normalization=True))
        valid = args[2]
        from copy import deepcopy
        bad = deepcopy(valid[-1])
        bad["v_obs"] = np.zeros(3)
        invalid = deepcopy(valid[-1])
        invalid["c_loc"][0] = np.nan
        outside = deepcopy(valid[-1])
        outside["chunk_k"] = 99
        args[2] = valid + [bad, invalid, outside, {"chunk_k": 99}]
        grouped = capture_graph(*args)["graph"]
        plain = capture_graph(*fixture())["graph"]
        self.assertEqual(grouped.size(), plain.size())
        for i, measurement in enumerate(valid):
            np.testing.assert_allclose(grouped.at(i).noiseModel().covariance(),
                                       plain.at(i).noiseModel().covariance()
                                       * (measurement["chunk_k"] + 1))

    def test_disabled_controls_preserve_graph(self):
        original = capture_graph(*fixture())["graph"]
        disabled = capture_graph(*fixture(gps_chunk_information_normalization=False,
                                          per_chunk_scale_prior_sigma=0.))["graph"]
        # CustomFactor equality also compares Python callback identity. Compare
        # actual factor residuals and covariance at perturbed poses instead.
        self.assertEqual(original.size(), disabled.size())
        rng = np.random.default_rng(10)
        for _ in range(10):
            values = gtsam.Values()
            for k in range(3):
                pose = Sim3LoopOptimizer._gtsam_sim3_from_srt(
                    2., np.eye(3), rng.normal(size=3))
                values.insert(gtsam.symbol("x", k), pose)
            for i in range(original.size()):
                a, b = original.at(i), disabled.at(i)
                self.assertEqual(list(a.keys()), list(b.keys()))
                np.testing.assert_array_equal(a.noiseModel().covariance(),
                                              b.noiseModel().covariance())
                np.testing.assert_array_equal(a.unwhitenedError(values),
                                              b.unwhitenedError(values))

    def test_invalid_sigmas_fail_early(self):
        for options in [dict(seq_sigma_t_m=0.), dict(seq_sigma_t_m=-1.),
                        dict(seq_sigma_t_m=float("nan")),
                        dict(per_chunk_scale_prior_sigma=-1.)]:
            with self.assertRaises(ValueError):
                capture_graph(*fixture(**options))


if __name__ == "__main__":
    unittest.main()

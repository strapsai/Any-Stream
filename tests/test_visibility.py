import unittest
import numpy as np
from experiments.gps_pgo.visibility import classify_camera, retention

class VisibilityTests(unittest.TestCase):

    def test_visibility_semantics(self):
        K = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        depth = np.full((5, 5), 10.0)
        valid = np.ones((5, 5), bool)
        p = np.array([[10.0, 10.0, 5.0], [20.0, 20.0, 10.0], [30.0, 30.0, 15.0], [100.0, 100.0, 5.0], [-2.0, -2.0, -1.0]])
        r = classify_camera(p, K, depth, valid, grid_origin=0, grid_step=1)
        assert r['free'].tolist() == [True, False, False, False, False]
        assert r['support'].tolist() == [False, True, False, False, False]
        assert r['occluded'].tolist() == [False, False, True, False, False]
        assert r['observed'].tolist() == [True, True, True, False, False]
        depth[:, :3] = 5.0
        roof = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0], [35.0, 20.0, 10.0]])
        r = classify_camera(roof, K, depth, valid, grid_origin=0, grid_step=1)
        assert r['support'][0] and r['occluded'][1] and r['support'][2]
        edge = np.array([[10.0, 10.0, 5.0]])
        r = classify_camera(edge, K, depth, valid, grid_origin=0, grid_step=1)
        assert not r['observed'][0]
        valid[:] = False
        r = classify_camera(p, K, depth, valid, grid_origin=0, grid_step=1)
        assert not r['observed'].any()
        e = {'free': np.array([1, 3, 3, 2]), 'support': np.array([0, 0, 2, 0]), 'last_free': np.array([5, 5, 5, 5]), 'last_support': np.array([-1, -1, 6, 6])}
        assert retention(e).tolist() == [True, False, True, True]
        print('PASS visibility: free space, real roof, occluded ground, neighboring ground, FOV, behind-camera, discontinuity, invalid-depth and temporal evidence cases')

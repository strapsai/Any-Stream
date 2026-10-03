import numpy as np
import pytest
from loop_utils.metric_ground import GroundConfig, GroundSurface


def cloud(fn, extent=40):
    # 64 samples/cell, exact sloped surfaces and deterministic XY medians.
    a = np.arange(-extent + .25, extent, .5)
    x, y = np.meshgrid(a, a)
    return np.column_stack((x.ravel(), y.ravel(), fn(x, y).ravel()))


def surface(points, **kw):
    return GroundSurface(points, GroundConfig(enabled=True, **kw))


def test_sloped_ground_beneath_roof():
    points = cloud(lambda x,y: 7 + .1*x + .04*y + np.where((abs(x)<8)&(abs(y)<8), 6, 0))
    r = surface(points).query(0, 0)
    assert r['status'] == 'supported'
    assert r['ground_z'] == pytest.approx(7, abs=.05)
    assert r['plane_dx'] == pytest.approx(.1, abs=.01)


def test_low_outliers_do_not_dig_pit():
    points = cloud(lambda x,y: np.where((x>10)&(x<14)&(y>10)&(y<14), -20, 3.))
    assert surface(points).query(0,0)['ground_z'] == pytest.approx(3, abs=.05)


def test_missing_and_degenerate_evidence_is_unknown():
    assert surface(np.empty((0,3))).query(0,0)['ground_z'] is None
    assert surface(cloud(lambda x,y: x*0, extent=8)).query(0,0)['ground_z'] is None
    x = np.repeat(np.arange(-160,160,4), 45)
    p = np.column_stack((x, x*0, x*0))
    assert surface(p).query(0,0)['ground_z'] is None
    assert surface(cloud(lambda x,y: x*0)).query(500,500)['ground_z'] is None


def test_vertical_datum_translation_and_missing_points():
    p = cloud(lambda x,y: .1*x)
    a = surface(p).query(3, 1)
    p[:,2] += 150
    p = np.concatenate((p, [[np.nan,0,0], [1,2,np.inf]]))
    b = surface(p).query(3,1)
    assert b['ground_z'] - a['ground_z'] == pytest.approx(150)
    assert b['ground_z'] == pytest.approx(150.3, abs=.05)


def test_disabled_and_configuration_validation():
    assert GroundSurface(cloud(lambda x,y: x*0)).query(0,0)['status'] == 'disabled'
    for params in ({'radius_m':0}, {'max_slope':float('nan')}, {'min_support_cells':2},
                   {'hypotheses':1.5}, {'hypotheses':128.0}, {'hypotheses':float('inf')},
                   {'enabled':'true'}, {'typo':2}):
        with pytest.raises(ValueError):
            GroundConfig.from_dict(params)


def test_roof_only_limitation_is_explicit():
    # Geometric support is NOT proof that this horizontal surface is terrain.
    assert surface(cloud(lambda x,y: x*0+8)).query(0,0)['ground_z'] == pytest.approx(8)


def test_two_slabs_cannot_invent_a_ground_between_them():
    a = cloud(lambda x,y: x*0)
    b = a.copy(); b[:,2] = 6
    g = surface(np.concatenate((a,b)))
    assert g.rejected_mixed_cells > 0
    assert g.query(0,0)['ground_z'] is None


def test_steep_plane_is_not_accepted_and_cell_boundary_is_continuous():
    assert surface(cloud(lambda x,y: .8*x), max_cell_height_span_m=10).query(0,0)['ground_z'] is None
    g = surface(cloud(lambda x,y: .1*x+.05*y))
    a = g.query(3.99,0); b = g.query(4.01,0)
    assert b['ground_z']-a['ground_z'] == pytest.approx(.002,abs=.002)

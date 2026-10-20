"""A fixed visual reference can own height without a manually fitted GPS datum."""
import copy
import numpy as np
import pytest
from loop_utils.metric_surface import solve, transform


def problem(shift=0., anchors=True):
    rng=np.random.default_rng(3)
    cameras=np.tile(np.eye(4),(10,1,1));cameras[:,:3,3]=rng.normal(size=(10,3))
    points=rng.normal(size=(60,3))
    return dict(absolutes=[(1.,np.eye(3),np.array([0.,0.,shift]))],
                gps=cameras[:,:3,3]+[0,0,shift],valid_gps=np.ones(10,dtype=bool),
                state=dict(local_c2w=[cameras],chunk_indices=[(0,10)]),seams=[],
                anchors=[dict(chunk=0,local=points,target=points,sigma=.1)] if anchors else [])


@pytest.mark.parametrize('jacobian_mode',['analytic','finite_difference'])
@pytest.mark.parametrize('shift',[-40.,0.,37.27])
def test_reference_height_recovers_despite_gps_datum_shift(jacobian_mode,shift):
    data=problem(shift)
    poses,diag=solve(data,gps_altitude_mode='reference',jacobian_mode=jacobian_mode,x_scale_mode='jac',max_nfev=200)
    assert diag['success']
    assert diag['parameters']['reference_altitude_active']
    np.testing.assert_allclose(transform(data['anchors'][0]['local'],poses[0]),data['anchors'][0]['target'],atol=1e-4)


def test_without_visual_anchor_reference_mode_preserves_gps_height():
    data=problem(37.27,anchors=False)
    poses,diag=solve(data,gps_altitude_mode='reference',jacobian_mode='analytic')
    assert not diag['parameters']['reference_altitude_active']
    np.testing.assert_allclose(poses[0][2],[0,0,37.27],atol=1e-7)


def test_reference_mode_supports_previous_accepted_pose_warm_start():
    data=problem(37.27)
    poses,diag=solve(data,gps_altitude_mode='reference',jacobian_mode='analytic',initial_absolutes=[(1.,np.eye(3),np.zeros(3))])
    assert diag['warm_start_chunks']==1
    np.testing.assert_allclose(poses[0][2],[0,0,0],atol=1e-7)


def test_unknown_altitude_mode_is_rejected():
    with pytest.raises(ValueError,match='gps_altitude_mode'):
        solve(problem(),gps_altitude_mode='guess')

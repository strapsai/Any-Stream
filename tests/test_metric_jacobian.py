"""Check analytic derivatives against the exact production residual closure."""
import copy
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from test_surface_backend import problem
from loop_utils import metric_surface
from loop_utils.metric_surface import transform


@pytest.mark.parametrize('mode',['radial','quadratic','quadratic_compressed','anchor_normalized','gps_bias','gravity','fixed_scale','single_chunk'])
def test_every_jacobian_column_against_central_differences(monkeypatch,mode):
    graph=problem();kwargs={};graph['loops']=[dict(i=0,j=1,a=graph['seams'][0][0],b=graph['seams'][0][1])]
    if mode.startswith('quadratic'):kwargs.update(vector_loss=False,compress=mode.endswith('compressed'))
    if mode=='anchor_normalized':kwargs['anchor_delta_m']=None
    if mode=='gps_bias':kwargs['gps_bias_sigma']=2.
    if mode=='gravity':
        kwargs.update(gravity_sigma=.2,global_scale_sigma=.5)
        graph['gravity_targets']=[np.eye(3),np.eye(3)]
    if mode=='fixed_scale':kwargs['fix_scale']=True
    if mode=='single_chunk':
        graph['absolutes']=graph['absolutes'][:1]
        graph['state']=dict(local_c2w=graph['state']['local_c2w'][:1],chunk_indices=[(0,6)])
        graph['gps']=graph['gps'][:6];graph['valid_gps']=graph['valid_gps'][:6];graph['seams']=[];graph['loops']=[]
    def check(fun,x0,**options):
        analytic=options['jac'];rng=np.random.default_rng(73)
        for magnitude in [0.,1e-7,.4]:
            z=x0+rng.normal(size=x0.shape)*magnitude
            actual=analytic(z).toarray();expected=np.zeros_like(actual)
            for j in range(len(z)):
                delta=np.zeros_like(z);delta[j]=2e-6
                expected[:,j]=(fun(z+delta)-fun(z-delta))/(4e-6)
            np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=2e-5)
        return least_squares(fun,x0,**options)
    monkeypatch.setattr(metric_surface,'least_squares',check)
    metric_surface.solve(graph,jacobian_mode='analytic',x_scale_mode='jac',max_nfev=2,**kwargs)


@pytest.mark.parametrize('x_scale_mode',['unit','jac'])
def test_analytic_and_finite_difference_solution_and_world_frame_agree(x_scale_mode):
    graph=problem()
    a,da=metric_surface.solve(graph,jacobian_mode='finite_difference',x_scale_mode=x_scale_mode,max_nfev=200)
    b,db=metric_surface.solve(graph,jacobian_mode='analytic',x_scale_mode=x_scale_mode,max_nfev=200)
    assert da['success'] and db['success'];assert da['initial_cost']==db['initial_cost']
    assert abs(da['cost']-db['cost'])<1e-7
    for p,q in zip(a,b):np.testing.assert_allclose(transform(graph['seams'][0][0],p),transform(graph['seams'][0][0],q),atol=2e-4)
    Q=Rotation.from_rotvec([.4,-.2,.1]).as_matrix();t=np.array([100.,-200.,40.]);other=copy.deepcopy(graph)
    other['absolutes']=[(s,Q@R,Q@p+t) for s,R,p in graph['absolutes']]
    other['gps']=graph['gps']@Q.T+t;other['anchors'][0]['target']=graph['anchors'][0]['target']@Q.T+t
    c,dc=metric_surface.solve(other,jacobian_mode='analytic',x_scale_mode=x_scale_mode,max_nfev=200)
    assert abs(db['cost']-dc['cost'])<1e-6
    for p,q in zip(b,c):np.testing.assert_allclose(transform(graph['seams'][0][0],p)@Q.T+t,transform(graph['seams'][0][0],q),atol=2e-4)


def test_unsupported_group_coupling_is_rejected():
    with pytest.raises(ValueError,match='group-norm'):
        metric_surface.solve(problem(),jacobian_mode='analytic',vector_loss=False,block_loss=True)

"""Metric surface graph with shared-pixel, GPS, return-loop and fixed-reference factors.

All parameters are corrections around each chunk's camera centroid in the initial
ENU gauge. GPS and overlap terms therefore both use metres. Shared-pixel training
and reporting sets are disjoint; no ground-plane residual is used.
"""
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def transform(points, pose):
    """Apply a mapper Sim3 tuple, with translation outside scale."""
    scale, rotation, translation = pose
    return float(scale) * (np.asarray(points) @ np.asarray(rotation).T) + translation


def solve(
    data,
    gps_sigma=4.0,
    overlap_sigma=0.3,
    gps_group=True,
    rot_sigma=0.005,
    scale_sigma=0.03,
    loss='linear',
    max_nfev=1000,
    global_scale_sigma=0.0,
    fix_scale=False,
    loop_sigma=0.1,
    gravity_sigma=None,
    compress=False,
    block_loss=False,
    vector_loss=True,
    gps_only_block=False,
    tolerance=1e-07,
    anchor_delta_m=1.0,
    gps_bias_sigma=None,
    initial_absolutes=None,
):
    """Solve cached chunk geometry; defaults reproduce the selected radial-loss graph.

    Overlap/loop even-indexed pairs supply factors; odd-indexed pairs are
    reserved. Fixed-reference anchors use all supplied pairs. Warm starts are
    expressed relative to the original base; they never replace the priors.
    No ground-plane constraint, correspondence discovery, or ROS integration is
    performed here. Return (absolute mapper Sim3 tuples, solver diagnostics).

    Moment compression is valid for quadratic/group-norm loss, not per-point
    radial loss, and the incompatible combination raises ValueError."""
    from .metric_io import validate_graph
    validate_graph(data)
    for name, value in dict(gps_sigma=gps_sigma, overlap_sigma=overlap_sigma,
                            rot_sigma=rot_sigma, scale_sigma=scale_sigma,
                            loop_sigma=loop_sigma, tolerance=tolerance).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if anchor_delta_m is not None and (not np.isfinite(anchor_delta_m) or anchor_delta_m <= 0):
        raise ValueError("anchor_delta_m must be finite and positive")
    if compress and vector_loss:
        raise ValueError('Moment summaries do not preserve per-point radial robust loss')
    n = len(data['absolutes'])
    nbias = 3 if gps_bias_sigma is not None else 0
    base = data['absolutes']
    cam = [transform(c[:, :3, 3], x) for c, x in zip(data['state']['local_c2w'], base)]
    pivots = np.array([np.mean(c, axis=0) for c in cam])
    terms = []
    for k, (start, end) in enumerate(data['state']['chunk_indices']):
        good = np.flatnonzero(data['valid_gps'][start:end])[::3]
        if len(good):
            terms.append(('gps',
                k,
                cam[k][good] - pivots[k],
                data['gps'][start:end][good],
                gps_sigma * np.sqrt(len(good)) if gps_group else gps_sigma))
    for k, (a, b) in enumerate(data['seams']):
        if not len(a):
            continue
        a = a[::2]
        b = b[::2]
        step = max(1, len(a) // 80)
        a = a[::step]
        b = b[::step]
        terms.append(('overlap',
            k,
            transform(a, base[k]) - pivots[k],
            transform(b, base[k + 1]) - pivots[k + 1],
            overlap_sigma * np.sqrt(len(a))))
    for loop in data.get('loops', []):
        i, j = (loop['i'], loop['j'])
        a = loop['a'][::2]
        b = loop['b'][::2]
        step = max(1, len(a) // 60)
        a = a[::step]
        b = b[::step]
        terms.append(('loop',
            i,
            transform(a, base[i]) - pivots[i],
            transform(b, base[j]) - pivots[j],
            loop_sigma * np.sqrt(len(a)),
            j))
    for anchor in data.get('anchors', []):
        k = anchor['chunk']
        a = np.asarray(anchor['local'])
        b = np.asarray(anchor['target'])
        terms.append(('anchor', k, transform(a, base[k]) - pivots[k], b,
                      anchor['sigma'] * np.sqrt(len(a))))
    # Optional moment summaries preserve group squared error only.
    if compress:
        compressed = []
        for t in terms:
            a, b, sigma = t[2:5]
            joint = np.c_[a, b]
            mean = joint.mean(0)
            center = joint - mean
            C = center.T @ center / len(joint)
            w, V = np.linalg.eigh(C)
            w = np.maximum(w, 0.0)
            virtual = (V * np.sqrt(6 * w)).T
            paired = np.concatenate([mean + virtual, mean - virtual])
            compressed.append((t[0],
                t[1],
                paired[:, :3],
                paired[:, 3:],
                sigma * np.sqrt(len(paired) / len(joint)),
                *t[5:]))
        terms = compressed
    # Sparsity follows the chunks incident on each residual group.
    sizes = [len(t[2]) * 3 for t in terms]
    nres = sum(sizes) + (n - 1) * 6 + int(global_scale_sigma > 0) + (3 * n if gravity_sigma is not None else 0) + nbias
    sparsity = lil_matrix((nres, n * 7 + nbias), dtype=int)
    off = 0
    for t, size in zip(terms, sizes):
        _, k, *_ = t
        sparsity[off:off + size, k * 7:(k + 1) * 7] = 1
        if nbias and t[0] == 'gps':
            sparsity[off:off + size, n * 7:] = 1
        if t[0] not in ('gps', 'anchor'):
            j = t[5] if len(t) > 5 else k + 1
            sparsity[off:off + size, j * 7:(j + 1) * 7] = 1
        off += size
    for k in range(n - 1):
        sparsity[off:off + 6, k * 7:(k + 2) * 7] = 1
        off += 6
    if gravity_sigma is not None:
        for k in range(n):
            sparsity[off:off + 3, k * 7:k * 7 + 3] = 1
            off += 3
    if global_scale_sigma > 0:
        sparsity[off, 6:n * 7:7] = 1
        off += 1
    if nbias:
        sparsity[off:off + 3, n * 7:] = 1
    A = np.concatenate([t[2] for t in terms])
    B = np.concatenate([t[3] for t in terms])
    I = np.concatenate([np.full(len(t[2]), t[1], int) for t in terms])
    J = np.concatenate([np.full(len(t[2]),
        t[5] if len(t) > 5 else min(t[1] + 1, n - 1),
        int) for t in terms])
    group = np.concatenate([np.full(len(t[2]), k, int) for k, t in enumerate(terms)])
    is_gps = np.array([t[0] in ('gps', 'anchor') for t in terms])
    gps_points = is_gps[group]
    anchor_points = np.array([t[0] == 'anchor' for t in terms])[group]
    gps_only_points = np.array([t[0] == 'gps' for t in terms])[group]
    sigma = np.concatenate([np.broadcast_to(np.asarray(t[4]), (len(t[2]), 3)) for t in terms])

    # Corrections act about initial camera centroids in metric world coordinates.
    def residual(z):
        bias = z[n * 7:]
        z = z[:n * 7].reshape(n, 7)
        R = Rotation.from_rotvec(z[:, :3]).as_matrix()
        s = np.ones(n) if fix_scale else np.exp(z[:, 6])
        t = pivots + z[:, 3:6]
        pred = s[I, None] * np.einsum('nij,nj->ni', R[I], A) + t[I]
        target = s[J, None] * np.einsum('nij,nj->ni', R[J], B) + t[J]
        target[gps_points] = B[gps_points]
        if nbias: target[gps_only_points] += bias
        r = (pred - target) / sigma
        if vector_loss:
            q = np.sum(r * r, axis=1)
            if anchor_delta_m is not None:
                q[anchor_points] = np.sum((pred[anchor_points] - target[anchor_points]) ** 2, axis=1) / anchor_delta_m ** 2
            r *= np.sqrt(2 / (np.sqrt(1 + q) + 1))[:, None]
        elif block_loss:
            q = np.bincount(group, weights=np.sum(r * r, axis=1), minlength=len(terms))
            weights = np.sqrt(2 / (np.sqrt(1 + q) + 1))
            if gps_only_block:
                weights[~is_gps] = 1.0
            r *= weights[group, None]
        rel = np.einsum('nji,njk->nik', R[:-1], R[1:])
        regular = np.zeros((n - 1, 6))
        if n > 1:
            regular[:, :3] = Rotation.from_matrix(rel).as_rotvec() / rot_sigma
        regular[:, 3] = np.diff(z[:, 6]) / scale_sigma
        out = [r.ravel(), regular.ravel()]
        if gravity_sigma is not None:
            for k, target in enumerate(data['gravity_targets']):
                g = R[k] @ base[k][1] @ target[2, :]
                out.append((g - np.array([0.0, 0.0, 1.0])) / gravity_sigma)
        if global_scale_sigma > 0:
            out.append(np.array([np.mean(z[:, 6]) / global_scale_sigma]))
        if nbias: out.append(bias / gps_bias_sigma)
        return np.concatenate(out)

    initial = np.zeros(n * 7 + nbias)
    if initial_absolutes is not None:
        if nbias:
            raise ValueError("GPS-bias warm starts require an explicit bias state")
        if len(initial_absolutes) > n:
            raise ValueError("warm start exceeds graph coverage")
        from .metric_io import validate_poses
        validate_poses(initial_absolutes)
        for k, (s1, R1, t1) in enumerate(initial_absolutes):
            s0, R0, t0 = base[k]
            Q = np.asarray(R1) @ np.asarray(R0).T
            ratio = float(s1) / float(s0)
            if fix_scale and abs(ratio - 1.) > 1e-8:
                raise ValueError("fixed-scale warm start changes the base scale")
            z = initial[k * 7:(k + 1) * 7]
            z[:3] = Rotation.from_matrix(Q).as_rotvec()
            z[3:6] = np.asarray(t1) - ratio * (Q @ (np.asarray(t0) - pivots[k])) - pivots[k]
            z[6] = 0. if fix_scale else np.log(ratio)
    initial_residual = residual(initial)
    # Report an actual SciPy objective cost, including its optional outer loss.
    # The deployment recipe uses linear outer loss and radial loss internally.
    initial_cost = .5 * float(initial_residual @ initial_residual) if loss == "linear" else None
    t0 = time.time()
    r = least_squares(residual,
        initial,
        jac_sparsity=sparsity.tocsr(),
        loss=loss,
        f_scale=1.0,
        max_nfev=max_nfev,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance)
    z = r.x[:n * 7].reshape(n, 7)
    out = []
    for k, (s, R, t) in enumerate(base):
        Q = Rotation.from_rotvec(z[k, :3]).as_matrix()
        u = 1.0 if fix_scale else np.exp(z[k, 6])
        p = pivots[k]
        out.append((s * u, Q @ R, u * (Q @ (t - p)) + p + z[k, 3:6]))
    diag = dict(success=bool(r.success),
        status=int(r.status),
        message=r.message,
        nfev=r.nfev,
        cost=float(r.cost),
        initial_cost=initial_cost,
        warm_start_chunks=0 if initial_absolutes is None else len(initial_absolutes),
        optimality=float(r.optimality),
        seconds=time.time() - t0,
        gps_bias_m=r.x[n * 7:].tolist() if nbias else None,
        parameters=dict(gps_sigma=gps_sigma, overlap_sigma=overlap_sigma, gps_group=gps_group, rot_sigma=rot_sigma, scale_sigma=scale_sigma, loss=loss, global_scale_sigma=global_scale_sigma, fix_scale=fix_scale, loop_sigma=loop_sigma, loop_count=len(data.get('loops', [])), anchor_count=len(data.get('anchors', [])), anchor_delta_m=anchor_delta_m, gps_bias_sigma=gps_bias_sigma, gravity_sigma=gravity_sigma, compress=compress, block_loss=block_loss, vector_loss=vector_loss, gps_only_block=gps_only_block, tolerance=tolerance, stored_factor_points=sum((len(t[2]) for t in terms))))
    return (out, diag)

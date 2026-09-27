"""Experimental graph with metric shared-pixel factors and correlated GPS groups.

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
    gps_sigma=2.0,
    overlap_sigma=0.3,
    gps_group=True,
    rot_sigma=0.005,
    scale_sigma=0.03,
    loss='linear',
    max_nfev=150,
    global_scale_sigma=0.0,
    fix_scale=True,
    loop_sigma=0.1,
    gravity_sigma=None,
    compress=False,
    block_loss=False,
    vector_loss=True,
    gps_only_block=False,
    tolerance=1e-07,
):
    """Solve cached chunk geometry; defaults reproduce the selected radial-loss graph.

    Input schema and sampling rules are documented in README.md. Even-indexed
    correspondences supply factors; odd-indexed correspondences remain unused.
    No ground-plane constraint, correspondence discovery, or ROS integration is
    performed here. Return (absolute mapper Sim3 tuples, solver diagnostics).

    Moment compression is valid for quadratic/group-norm loss, not per-point
    radial loss, and the incompatible combination raises ValueError."""
    if compress and vector_loss:
        raise ValueError('Moment summaries do not preserve per-point radial robust loss')
    n = len(data['absolutes'])
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
    nres = sum(sizes) + (n - 1) * 6 + int(global_scale_sigma > 0) + (3 * n if gravity_sigma is not None else 0)
    sparsity = lil_matrix((nres, n * 7), dtype=int)
    off = 0
    for t, size in zip(terms, sizes):
        _, k, *_ = t
        sparsity[off:off + size, k * 7:(k + 1) * 7] = 1
        if t[0] != 'gps':
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
        sparsity[-1, 6::7] = 1
    A = np.concatenate([t[2] for t in terms])
    B = np.concatenate([t[3] for t in terms])
    I = np.concatenate([np.full(len(t[2]), t[1], int) for t in terms])
    J = np.concatenate([np.full(len(t[2]),
        t[5] if len(t) > 5 else min(t[1] + 1, n - 1),
        int) for t in terms])
    group = np.concatenate([np.full(len(t[2]), k, int) for k, t in enumerate(terms)])
    is_gps = np.array([t[0] == 'gps' for t in terms])
    gps_points = is_gps[group]
    sigma = np.concatenate([np.broadcast_to(np.asarray(t[4]), (len(t[2]), 3)) for t in terms])

    # Corrections act about initial camera centroids in metric world coordinates.
    def residual(z):
        z = z.reshape(n, 7)
        R = Rotation.from_rotvec(z[:, :3]).as_matrix()
        s = np.ones(n) if fix_scale else np.exp(z[:, 6])
        t = pivots + z[:, 3:6]
        pred = s[I, None] * np.einsum('nij,nj->ni', R[I], A) + t[I]
        target = s[J, None] * np.einsum('nij,nj->ni', R[J], B) + t[J]
        target[gps_points] = B[gps_points]
        r = (pred - target) / sigma
        if vector_loss:
            q = np.sum(r * r, axis=1)
            r *= np.sqrt(2 / (np.sqrt(1 + q) + 1))[:, None]
        elif block_loss:
            q = np.bincount(group, weights=np.sum(r * r, axis=1), minlength=len(terms))
            weights = np.sqrt(2 / (np.sqrt(1 + q) + 1))
            if gps_only_block:
                weights[~is_gps] = 1.0
            r *= weights[group, None]
        rel = np.einsum('nji,njk->nik', R[:-1], R[1:])
        regular = np.zeros((n - 1, 6))
        regular[:, :3] = Rotation.from_matrix(rel).as_rotvec() / rot_sigma
        regular[:, 3] = np.diff(z[:, 6]) / scale_sigma
        out = [r.ravel(), regular.ravel()]
        if gravity_sigma is not None:
            for k, target in enumerate(data['gravity_targets']):
                g = R[k] @ base[k][1] @ target[2, :]
                out.append((g - np.array([0.0, 0.0, 1.0])) / gravity_sigma)
        if global_scale_sigma > 0:
            out.append(np.array([np.mean(z[:, 6]) / global_scale_sigma]))
        return np.concatenate(out)

    t0 = time.time()
    r = least_squares(residual,
        np.zeros(n * 7),
        jac_sparsity=sparsity.tocsr(),
        loss=loss,
        f_scale=1.0,
        max_nfev=max_nfev,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance)
    z = r.x.reshape(n, 7)
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
        optimality=float(r.optimality),
        seconds=time.time() - t0,
        parameters=dict(gps_sigma=gps_sigma, overlap_sigma=overlap_sigma, gps_group=gps_group, rot_sigma=rot_sigma, scale_sigma=scale_sigma, loss=loss, global_scale_sigma=global_scale_sigma, fix_scale=fix_scale, loop_sigma=loop_sigma, loop_count=len(data.get('loops', [])), gravity_sigma=gravity_sigma, compress=compress, block_loss=block_loss, vector_loss=vector_loss, gps_only_block=gps_only_block, tolerance=tolerance, stored_factor_points=sum((len(t[2]) for t in terms))))
    return (out, diag)

"""Conservative depth visibility evidence; a retention policy, not pose correction.

Votes are counted at most once per later chunk. A point behind a visible surface
is occluded and is never classified as free space by that view. Only points in
front of all four neighboring valid depths can receive a contradiction vote.
"""
import numpy as np

def classify_camera(points, K, depth, valid, min_margin=0.75, relative_margin=0.02, max_edge_absolute=0.75, max_edge_relative=0.05, grid_origin=4.0, grid_step=8.0):
    count = len(points)
    free = np.zeros(count, bool)
    support = np.zeros(count, bool)
    occluded = np.zeros(count, bool)
    observed = np.zeros(count, bool)
    uv = points @ K.T
    z = points[:, 2]
    uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-12)
    xy = (uv - grid_origin) / grid_step
    ij = np.floor(np.clip(xy, -1000000.0, 1000000.0)).astype(int)
    x, y = ij.T
    h, w = depth.shape
    ids = np.flatnonzero((z > 0.1) & (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1) & np.isfinite(points).all(1))
    if not len(ids):
        return dict(free=free, support=support, occluded=occluded, observed=observed)
    xx = x[ids]
    yy = y[ids]
    ds = np.stack([depth[yy, xx], depth[yy, xx + 1], depth[yy + 1, xx], depth[yy + 1, xx + 1]], 1)
    vs = np.stack([valid[yy, xx], valid[yy, xx + 1], valid[yy + 1, xx], valid[yy + 1, xx + 1]], 1)
    lo = np.min(ds, axis=1)
    hi = np.max(ds, axis=1)
    mid = (lo + hi) / 2
    good = vs.all(1) & np.isfinite(ds).all(1) & (lo > 0.1) & (hi - lo <= np.maximum(max_edge_absolute, max_edge_relative * mid))
    ids = ids[good]
    lo = lo[good]
    hi = hi[good]
    mid = mid[good]
    margin = np.maximum(min_margin, relative_margin * mid)
    observed[ids] = True
    free[ids] = z[ids] < lo - margin
    occluded[ids] = z[ids] > hi + margin
    support[ids] = (z[ids] >= lo - margin) & (z[ids] <= hi + margin)
    return dict(free=free, support=support, occluded=occluded, observed=observed)

def evidence(data, absolutes, points, ids, min_margin=0.75, relative_margin=0.02):
    n = len(points)
    free = np.zeros(n, np.uint16)
    support = np.zeros(n, np.uint16)
    occluded = np.zeros(n, np.uint16)
    observed = np.zeros(n, np.uint16)
    last_free = np.full(n, -1, np.int16)
    last_support = np.full(n, -1, np.int16)
    for k, (chunk, x) in enumerate(zip(data['chunks'], absolutes)):
        sel = np.flatnonzero(ids < k - 1)
        if not len(sel):
            continue
        s, R, t = x
        local = (points[sel] - t) @ R / s
        flags = {name: np.zeros(len(sel), bool) for name in ['free', 'support', 'occluded', 'observed']}
        ov = data['state']['config']['Model']['overlap']
        frames = np.unique(np.linspace(ov, len(chunk['points']) - 1, 3).astype(int))
        for f in frames:
            C = chunk['c2w'][f]
            camera = (local - C[:3, 3]) @ C[:3, :3] * s
            pc = (chunk['points'][f] - C[:3, 3]) @ C[:3, :3] * s
            depth = pc[..., 2]
            valid = chunk['valid'][f] & np.isfinite(pc).all(-1) & (np.linalg.norm(pc, axis=-1) < 120)
            result = classify_camera(camera, chunk['intrinsics'][f], depth, valid, min_margin, relative_margin)
            for name in flags:
                flags[name] |= result[name]
        flags['free'] &= ~flags['support']
        for name, arr in [('free', free), ('support', support), ('occluded', occluded), ('observed', observed)]:
            arr[sel] += flags[name]
        last_free[sel[flags['free']]] = k
        last_support[sel[flags['support']]] = k
        print('CHUNK', k, 'older_points', len(sel), 'free_votes', int(flags['free'].sum()), 'support_votes', int(flags['support'].sum()), flush=True)
    return dict(free=free, support=support, occluded=occluded, observed=observed, last_free=last_free, last_support=last_support)

def retention(e, min_votes=2, ratio=2.0, protect_reconfirmed=True):
    remove = (e['free'] >= min_votes) & (e['free'] >= ratio * np.maximum(e['support'], 1))
    if protect_reconfirmed:
        remove &= e['last_free'] > e['last_support']
    return ~remove

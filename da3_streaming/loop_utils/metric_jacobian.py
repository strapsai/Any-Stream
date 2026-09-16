"""Analytic sparse derivatives for the metric surface graph's pointwise losses.

SO(3) uses right-trivialized exponential-coordinate derivatives. This module
changes derivative evaluation only; the residual remains in metric_surface.
Group-norm robust loss has cross-point coupling and is intentionally unsupported.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def skew(v):
    v = np.asarray(v)
    out = np.zeros((*v.shape[:-1], 3, 3), dtype=float)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def right_jacobian(v, inverse=False):
    v = np.asarray(v, float)
    theta2 = np.sum(v * v, axis=-1)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    safe = np.where(small, 1., theta)
    K = skew(v)
    I = np.broadcast_to(np.eye(3), K.shape)
    if inverse:
        a = np.where(small, 1 / 12 + theta2 / 720 + theta2**2 / 30240,
                     (1 - .5 * safe / np.tan(.5 * safe)) / (safe * safe))
        return I + .5 * K + a[..., None, None] * (K @ K)
    a = np.where(small, .5 - theta2 / 24 + theta2**2 / 720, (1 - np.cos(safe)) / (safe * safe))
    b = np.where(small, 1 / 6 - theta2 / 120 + theta2**2 / 5040, (safe - np.sin(safe)) / (safe**3))
    return I - a[..., None, None] * K + b[..., None, None] * (K @ K)


def make_surface_jacobian(*,
                          A,
                          B,
                          I,
                          J,
                          pivots,
                          sigma,
                          gps_points,
                          anchor_points,
                          gps_only_points,
                          nres,
                          rot_sigma,
                          scale_sigma,
                          vector_loss,
                          anchor_delta_m,
                          fix_scale=False,
                          global_scale_sigma=0.,
                          gravity_vectors=None,
                          gravity_sigma=None,
                          gps_bias_sigma=None):
    n = len(pivots)
    m = len(A)
    nbias = 3 if gps_bias_sigma is not None else 0
    free = ~gps_points
    # Row/column structure is constant through a solve. Values are vectorized.
    point_rows = np.arange(m * 3).reshape(m, 3, 1)
    columns = I[:, None, None] * 7 + np.arange(7)[None, None, :]
    rows = [np.broadcast_to(point_rows, (m, 3, 7)).ravel()]
    cols = [np.broadcast_to(columns, (m, 3, 7)).ravel()]
    rows.append(np.broadcast_to(point_rows[free], (int(free.sum()), 3, 7)).ravel())
    cols.append(
        np.broadcast_to(J[free, None, None] * 7 + np.arange(7), (int(free.sum()), 3, 7)).ravel())
    if nbias:
        ng = int(gps_only_points.sum())
        rows.append(np.broadcast_to(point_rows[gps_only_points], (ng, 3, 3)).ravel())
        cols.append(np.broadcast_to(n * 7 + np.arange(3), (ng, 3, 3)).ravel())
    offset = m * 3
    prior_rows = offset + np.arange(n - 1)[:, None, None] * 6 + np.arange(3)[None, :, None]
    for shift in [0, 1]:
        rows.append(np.broadcast_to(prior_rows, (n - 1, 3, 3)).ravel())
        cols.append(
            np.broadcast_to((np.arange(n - 1) + shift)[:, None, None] * 7 + np.arange(3),
                            (n - 1, 3, 3)).ravel())
    for shift in [0, 1]:
        rows.append(offset + np.arange(n - 1) * 6 + 3)
        cols.append((np.arange(n - 1) + shift) * 7 + 6)
    offset += (n - 1) * 6
    if gravity_sigma is not None:
        g_rows = offset + np.arange(n)[:, None, None] * 3 + np.arange(3)[None, :, None]
        rows.append(np.broadcast_to(g_rows, (n, 3, 3)).ravel())
        cols.append(
            np.broadcast_to(np.arange(n)[:, None, None] * 7 + np.arange(3), (n, 3, 3)).ravel())
        offset += 3 * n
    if global_scale_sigma > 0:
        rows.append(np.full(n, offset))
        cols.append(np.arange(n) * 7 + 6)
        offset += 1
    if nbias:
        rows.append(offset + np.arange(3))
        cols.append(n * 7 + np.arange(3))
        offset += 3
    assert offset == nres
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    skew_a = skew(A)
    skew_b = skew(B[free])
    eye = np.eye(3)

    def jacobian(z):
        zz = np.asarray(z[:n * 7]).reshape(n, 7)
        phi = zz[:, :3]
        R = Rotation.from_rotvec(phi).as_matrix()
        Jr = right_jacobian(phi)
        s = np.ones(n) if fix_scale else np.exp(zz[:, 6])
        t = pivots + zz[:, 3:6]
        pa = s[I, None] * np.einsum('nij,nj->ni', R[I], A)
        pb = s[J, None] * np.einsum('nij,nj->ni', R[J], B)
        target = pb + t[J]
        target[gps_points] = B[gps_points]
        if nbias: target[gps_only_points] += z[n * 7:]
        raw = (pa + t[I] - target) / sigma
        left = np.zeros((m, 3, 7))
        left[:, :, :3] = -s[I, None, None] * (R[I] @ skew_a @ Jr[I])
        left[:, :, 3:6] = eye
        if not fix_scale: left[:, :, 6] = pa
        right = np.zeros((int(free.sum()), 3, 7))
        right[:, :, :3] = s[J[free], None, None] * (R[J[free]] @ skew_b @ Jr[J[free]])
        right[:, :, 3:6] = -eye
        if not fix_scale: right[:, :, 6] = -pb[free]
        # d(normalized residual)/d(metric point difference), including the
        # radial pseudo-Huber reweighting derivative, not frozen IRLS weights.
        if vector_loss:
            alpha = np.ones_like(raw)
            if anchor_delta_m is not None:
                alpha[anchor_points] = (sigma[anchor_points] / anchor_delta_m)**2
            q = np.sum(alpha * raw * raw, axis=1)
            h = np.sqrt(1 + q)
            w = np.sqrt(2 / (h + 1))
            dw = -w / (4 * h * (h + 1))
            M = w[:, None, None] * eye + raw[:, :, None] * (2 * dw[:, None] * alpha * raw)[:,
                                                                                           None, :]
            M = M / sigma[:, None, :]
        else:
            M = np.broadcast_to(eye, (m, 3, 3)) / sigma[:, None, :]
        values = [(M @ left).ravel(), (M[free] @ right).ravel()]
        if nbias: values.append((-M[gps_only_points]).ravel())
        rel = np.einsum('nji,njk->nik', R[:-1], R[1:])
        e = Rotation.from_matrix(rel).as_rotvec() if n > 1 else np.empty((0, 3))
        inv = right_jacobian(e, inverse=True) / rot_sigma
        values.extend([(-inv @ rel.transpose(0, 2, 1) @ Jr[:-1]).ravel(), (inv @ Jr[1:]).ravel(),
                       np.full(n - 1, -1 / scale_sigma),
                       np.full(n - 1, 1 / scale_sigma)])
        if gravity_sigma is not None:
            values.append((-R @ skew(gravity_vectors) @ Jr / gravity_sigma).ravel())
        if global_scale_sigma > 0: values.append(np.full(n, 1 / (n * global_scale_sigma)))
        if nbias: values.append(np.full(3, 1 / gps_bias_sigma))
        return coo_matrix((np.concatenate(values), (row, col)), shape=(nres, n * 7 + nbias)).tocsr()

    return jacobian

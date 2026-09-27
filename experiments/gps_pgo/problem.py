"""Portable, numeric-only NPZ interchange for the experimental geometry graph."""
from pathlib import Path

import numpy as np


def _pack_pairs(pairs):
    sizes = [len(a) for a, _ in pairs]
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    a = np.concatenate([a for a, _ in pairs]) if pairs else np.empty((0, 3))
    b = np.concatenate([b for _, b in pairs]) if pairs else np.empty((0, 3))
    return a, b, offsets


def save_problem(path, data):
    """Export the minimal solver input from an in-memory experiment cache."""
    base = data["absolutes"]
    cameras = [c[:, :3, 3] for c in data["state"]["local_c2w"]]
    seams = _pack_pairs(data["seams"])
    loops = data.get("loops", [])
    loop_points = _pack_pairs([(loop["a"], loop["b"]) for loop in loops])
    np.savez_compressed(
        path, format_version=np.array(1),
        scales=np.array([x[0] for x in base]),
        rotations=np.array([x[1] for x in base]),
        translations=np.array([x[2] for x in base]),
        camera_positions=np.concatenate(cameras),
        camera_offsets=np.r_[0, np.cumsum([len(c) for c in cameras])],
        gps=data["gps"], valid_gps=data["valid_gps"],
        seam_a=seams[0], seam_b=seams[1], seam_offsets=seams[2],
        loop_a=loop_points[0], loop_b=loop_points[1], loop_offsets=loop_points[2],
        loop_i=np.array([loop["i"] for loop in loops], dtype=np.int64),
        loop_j=np.array([loop["j"] for loop in loops], dtype=np.int64),
    )


def _offsets(value, groups, total, minimum):
    if (value.shape != (groups + 1,) or value.dtype.kind not in "iu"
            or value[0] != 0 or value[-1] != total
            or np.any(np.diff(value) < minimum)):
        raise ValueError("Invalid group offsets or insufficient group support")


def _points(value):
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("Geometry must contain finite Nx3 point arrays")


def load_problem(path):
    """Load and validate a format-v1 graph without enabling pickle deserialization."""
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if arrays["format_version"].item() != 1:
        raise ValueError("Unsupported graph format version")
    scales, rotations, translations = [arrays[key] for key in
                                       ("scales", "rotations", "translations")]
    n = len(scales)
    if (n < 2 or scales.shape != (n,) or not np.isfinite(scales).all()
            or np.any(scales <= 0) or rotations.shape != (n, 3, 3)
            or translations.shape != (n, 3)):
        raise ValueError("Expected at least two valid initial Sim3 transforms")
    _points(translations)
    if (not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)):
        raise ValueError("Initial rotations must belong to SO(3)")
    cameras = arrays["camera_positions"]
    _points(cameras)
    offsets = arrays["camera_offsets"]
    _offsets(offsets, n, len(cameras), 1)
    gps, valid = arrays["gps"], arrays["valid_gps"]
    if (gps.shape != cameras.shape or valid.shape != (len(cameras),)
            or valid.dtype.kind != "b" or not valid.any()
            or not np.isfinite(gps[valid]).all()):
        raise ValueError("GPS and its boolean validity mask must align with camera frames")
    seam_a, seam_b, seam_offsets = [arrays[key] for key in
                                   ("seam_a", "seam_b", "seam_offsets")]
    _points(seam_a)
    _points(seam_b)
    if seam_a.shape != seam_b.shape:
        raise ValueError("Paired seam arrays must have matching shapes")
    _offsets(seam_offsets, n - 1, len(seam_a), 2)
    loop_i, loop_j = arrays["loop_i"], arrays["loop_j"]
    if (loop_i.ndim != 1 or loop_i.shape != loop_j.shape
            or loop_i.dtype.kind not in "iu" or loop_j.dtype.kind not in "iu"
            or np.any(loop_i < 0) or np.any(loop_j < 0)
            or np.any(loop_i >= n) or np.any(loop_j >= n) or np.any(loop_i == loop_j)):
        raise ValueError("Invalid loop chunk indices")
    loop_a, loop_b, loop_offsets = [arrays[key] for key in
                                   ("loop_a", "loop_b", "loop_offsets")]
    _points(loop_a)
    _points(loop_b)
    if loop_a.shape != loop_b.shape:
        raise ValueError("Paired loop arrays must have matching shapes")
    _offsets(loop_offsets, len(loop_i), len(loop_a), 2)
    local_c2w = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        poses = np.tile(np.eye(4), (end - start, 1, 1))
        poses[:, :3, 3] = cameras[start:end]
        local_c2w.append(poses)
    return dict(
        absolutes=list(zip(scales, rotations, translations)), gps=gps, valid_gps=valid,
        state=dict(local_c2w=local_c2w, chunk_indices=list(zip(offsets[:-1], offsets[1:]))),
        seams=[(seam_a[a:b], seam_b[a:b]) for a, b in
               zip(seam_offsets[:-1], seam_offsets[1:])],
        loops=[dict(i=int(i), j=int(j), a=loop_a[a:b], b=loop_b[a:b])
               for i, j, a, b in zip(loop_i, loop_j, loop_offsets[:-1], loop_offsets[1:])],
    )

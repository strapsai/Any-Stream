"""Validated JSON/NPZ transport for a metric graph; no executable pickle input.

The absolute pose convention is x_world = s * R @ x_local + t. Local cameras
and factor endpoints never change when optimization changes absolute poses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "anystream-metric-graph/v1"


def validate_poses(poses):
    if not len(poses):
        raise ValueError("pose list is empty")
    for i, pose in enumerate(poses):
        if len(pose) != 3:
            raise ValueError(f"pose {i}: expected scale, rotation, translation")
        s, R, t = pose
        R, t = np.asarray(R), np.asarray(t)
        if not np.isfinite(s) or s <= 0 or R.shape != (3, 3) or t.shape != (3, ):
            raise ValueError(f"pose {i}: invalid shape or scale")
        if not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError(f"pose {i}: nonfinite pose")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or abs(np.linalg.det(R) - 1) > 1e-4:
            raise ValueError(f"pose {i}: rotation is not proper orthonormal")


def _pairs(a, b, name, allow_empty=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or a.shape[1:] != (3, ) or a.shape != b.shape:
        raise ValueError(f"{name}: expected paired [N,3] arrays")
    if (not allow_empty and len(a) < 2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{name}: empty or nonfinite point support")


def validate_graph(data):
    """Reject malformed factors before a worker or ROS publication can use them."""
    validate_poses(data["absolutes"])
    n = len(data["absolutes"])
    state = data["state"]
    if len(state["local_c2w"]) != n or len(state["chunk_indices"]) != n:
        raise ValueError("camera/chunk ownership differs from absolute pose coverage")
    end = 0
    for k, (bounds, cameras) in enumerate(zip(state["chunk_indices"], state["local_c2w"])):
        start, stop = bounds
        cameras = np.asarray(cameras)
        if start != end or stop <= start or cameras.shape != (stop - start, 4, 4):
            raise ValueError(f"chunk {k}: camera ownership is not contiguous and exact")
        if not np.isfinite(cameras).all() or not np.allclose(cameras[:, 3], [0, 0, 0, 1]):
            raise ValueError(f"chunk {k}: invalid homogeneous cameras")
        rot = cameras[:, :3, :3]
        if not np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-4) or not np.allclose(
                np.linalg.det(rot), 1, atol=1e-4):
            raise ValueError(f"chunk {k}: camera rotations contain scale/reflection")
        end = stop
    gps, valid = np.asarray(data["gps"]), np.asarray(data["valid_gps"])
    if gps.shape != (end, 3) or valid.shape != (end, ) or valid.dtype != np.bool_:
        raise ValueError("GPS arrays do not match owned frames")
    if not np.isfinite(gps[valid]).all():
        raise ValueError("valid GPS samples must be finite")
    if len(data["seams"]) != n - 1:
        raise ValueError("each neighboring chunk boundary needs an explicit seam entry")
    for k, (a, b) in enumerate(data["seams"]):
        _pairs(a, b, f"seam {k}", allow_empty=True)
    for row in data.get("loops", []):
        i, j = row["i"], row["j"]
        if not isinstance(i, (int, np.integer)) or not isinstance(
                j, (int, np.integer)) or not 0 <= i < n or not 0 <= j < n or i == j:
            raise ValueError("loop references an invalid chunk pair")
        _pairs(row["a"], row["b"], "loop")
    for row in data.get("anchors", []):
        k = row["chunk"]
        if not isinstance(k, (int, np.integer)) or not 0 <= k < n:
            raise ValueError("anchor references an invalid chunk")
        _pairs(row["local"], row["target"], "anchor")
        if not np.isfinite(row["sigma"]) or row["sigma"] <= 0:
            raise ValueError("anchor sigma must be finite and positive")
    if not valid.any() and not data.get("anchors"):
        raise ValueError("graph has no fixed world-frame evidence")
    return data


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("only numeric and boolean arrays are supported")
        key = f"a{len(arrays):05d}"
        arrays[key] = value
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if "__array__" in value:
            raise ValueError("reserved transport key")
        return {str(k): _encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v, arrays) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported payload type {type(value).__name__}")


def write_bundle(folder, payload, metadata, *, compressed=True):
    """Create an immutable local bundle, committing the manifest after its blob."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {}
    tree = _encode(payload, arrays)
    blob = folder / "arrays.npz"
    (np.savez_compressed if compressed else np.savez)(blob, **arrays)
    manifest = dict(schema_version=SCHEMA,
                    complete=True,
                    metadata=metadata,
                    arrays_file=blob.name,
                    arrays_sha256=hashlib.sha256(blob.read_bytes()).hexdigest(),
                    payload=tree)
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    tmp.replace(folder / "manifest.json")
    return folder / "manifest.json"


def read_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("unsupported or incomplete graph bundle")
    name = manifest["arrays_file"]
    if Path(name).name != name:
        raise ValueError("bundle arrays must be a sibling file")
    blob = path.parent / name
    if hashlib.sha256(blob.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("graph blob SHA-256 mismatch")
    with np.load(blob, allow_pickle=False) as arrays:

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return arrays[value["__array__"]].copy()
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value

        payload = decode(manifest["payload"])
    return payload, manifest["metadata"]

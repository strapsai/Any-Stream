# Conservative visibility retention experiment

`visibility.py` is an offline point-retention experiment. It changes neither poses
nor correspondences. On the selected Starling map it removes about 0.42% of points,
retains all occupied XY/low-surface cells, and barely changes repeated surface spread.
It is useful evidence for conservative cleanup, not a remedy for split planes.

`classify_camera(points, K, depth, valid, ...)` is the standalone primitive:

- `points`: `N×3` camera-frame positions in metres; positive Z faces forward.
- `K`: pixel-space `3×3` intrinsics.
- `depth`: positive camera-Z depth in metres on a sampled `H×W` grid.
- `valid`: boolean `H×W` depth validity.
- `grid_origin=4`, `grid_step=8`: grid sample `(x, y)` represents original pixel
  `(4+8*x, 4+8*y)`, matching the experiment's cached grid. Set origin 0 and step 1
  for a full-resolution depth image.

An observation requires four finite, valid neighboring depth samples away from a
depth discontinuity. A point in front of every sample by more than the uncertainty
margin is in observed free space; one behind the surface is occluded and survives.
Outside-image and behind-camera points are unobserved. A matching depth supports
the point. Defaults use a 0.75 m or 2% depth margin and a 0.75 m or 5% edge guard.
These are experiment settings rather than a calibrated sensor uncertainty model.

`evidence(data, absolutes, points, ids)` applies that primitive to cached later
chunks, counts at most one vote per later chunk, excludes adjacent chunks sharing
images, and lets any supporting view veto that chunk's free-space vote. `points`
are candidate world points and `ids` are their origin chunk indices. The cache
requires `state.config.Model.overlap` and, for each `chunks` entry:
`points[F,H,W,3]` in chunk-local coordinates, `valid[F,H,W]`, local camera-to-world
`c2w[F,4,4]`, and `intrinsics[F,3,3]`. It samples three frames per eligible chunk.
This depth cache is separate from the geometric solver's minimal NPZ format.

`retention(evidence)` returns a keep mask. Defaults require two free-space votes,
at least twice the support count (with a one-vote floor), and contradiction newer
than the latest support. Thus one contradiction, balanced evidence, or later
confirmation preserves the point. This is not probabilistic surface fusion, and
repeated pose/depth errors may remain correlated.

The synthetic regression covers a spurious front plane, visible roof, occluded
ground, neighboring ground, field of view, behind-camera points, depth edges,
invalid samples, and temporal evidence disagreement. It runs in the repository's
normal `unittest` discovery alongside the geometric graph checks.

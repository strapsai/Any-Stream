# Metric surface backend

This backend reproduces the selected ACFA `metric_surface_v2` graph: GPS, adjacent shared-image surfaces, verified return surfaces, and optional fixed-reference surface anchors in one solve. It returns **absolute** chunk-local-to-world Sim3 transforms, using `x_world = s R x_local + t`. Do not pass these as corrections onto a separately optimized graph.

`da3_streaming/loop_utils/metric_surface.py` promotes the selected offline objective into a reusable module. The source of that objective is preserved at `/mnt/data/airlab_ws/map-merge/anystr/evals/acfa/20260914_identity_eval/deployment_merged_20260916/source/optimizer.py`, SHA-256 `25b2932ac476ce05194d833dbf2a79a72cc6a0a01e0ce5a2d2f28e6c31ded0be`. The additions are input validation, a one-chunk case, and an explicit warm start that retains the original prior frame.

## What the graph means

`data['absolutes']` is the initial local-to-world transform for each chunk. `state.local_c2w` contains owned, rigid local camera poses; `state.chunk_indices` gives their exact contiguous frame intervals. GPS positions and validity flags index those same owned frames. `seams[k]` contains paired local surface points for chunks k and k+1. A loop contains `i`, `j`, `a`, `b`; an anchor contains `chunk`, `local`, `target`, `sigma`, with target already in the fixed reference world frame.

The optimizer samples even-indexed seam/loop correspondences for training and keeps the odd-indexed points unused. It uses all supplied reference-anchor points. Group normalization, sampling and radial robust loss are part of the objective; copying the sigmas into another solver does not reproduce it. The selected recipe uses grouped GPS sigma 4 m, overlap sigma 0.3 m, loop sigma 0.1 m, reference-anchor sigma 0.1 m, free regularized scales, and a 1 m robust transition for reference anchors. These are experiment settings, not universal sensor calibration.

A warm start is `initial_absolutes=previous_result`. It is converted into correction variables about the original camera-centroid pivots. Replacing `data['absolutes']` with an earlier answer would change the regularization and is not this operation. Warm starts with a separate GPS-bias state are rejected until the bias itself has an explicit input contract.

## Process and file boundary

`metric_io.py` writes immutable JSON + numeric NPZ bundles, with a SHA-256-checked blob and a manifest committed last. It does not deserialize pickle. Graph validation checks finite poses, proper rotations, camera/frame ownership, GPS coverage, endpoint dimensions, chunk references and positive anchor sigma.

`python -m loop_utils.metric_worker --request REQUEST/manifest.json --output NEW_DIRECTORY` runs one numerical solve. The request payload contains `graph`, `parameters`, and optionally `initial_absolutes`; its metadata identifies reset, graph revision, chunk identities, reference epoch and world frame. The result preserves that metadata and contains `absolutes` and `diagnostics`. The ROS coordinator supplies a wall-clock deadline and manages the child process.

The tested scientific worker uses Python 3.11.14 / NumPy 1.26.4 / SciPy 1.17.1. It is isolated from ROS Python 3.10, whose installed SciPy differs. No package installation is required in the current sandbox. Dependency versions are recorded in every worker result.

## Evidence and limits

The maintained backend reproduced all 91 selected ACFA transforms exactly in 426 evaluations. With Spirit anchors removed, continuation from the saved 1,000-evaluation Starling-only state reproduced the converged standalone control exactly in 137 evaluations, preserving its initial objective cost. Tests exercise transport corruption, world-frame equivariance, prior-preserving warm starts and invalid graph inputs.

This module does not discover image matches, judge whether a geometrically plausible match is semantically correct, provide a DROID database publisher, or certify online deployment. The real ROS frozen-observation test and complete provenance are documented at `/mnt/data/airlab_ws/map-merge/anystr/evals/acfa/20260914_identity_eval/deployment_merged_20260916/IMPLEMENTATION.md`.

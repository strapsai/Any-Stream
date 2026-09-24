# Experimental metric geometry graph

This is the selected **offline prototype** from the Starling GPS PGO investigation.
It has no ROS node integration or loop detector. Its input is already inferred,
aligned chunk geometry plus explicitly verified point correspondences. Existing
production entry points do not import this directory.

`optimizer.py` evaluates factors in batches. `reference.py` independently evaluates
the same factors in a Python loop, making residual comparisons possible. Defaults
reproduce the selected objective: GPS sigma 2 m, overlap sigma 0.3 m, verified-loop
sigma 0.1 m, rotation regularization 0.005 rad, fixed initial chunk scales, grouped
GPS information, and a radial pseudo-Huber loss on each 3D residual vector. SciPy
receives `loss="linear"` because radial robustification is already in the residual.
Component-wise robust loss is axis-dependent and is not the selected configuration.

Corrections act about each initial world-space camera centroid. GPS and geometric
point residuals use metres. The graph has no plane-flattening residual and does not
filter the displayed cloud. It returns one absolute `(scale, rotation, translation)`
tuple per chunk, acting as `scale * rotation @ point + translation`.

## Portable execution

From this checkout, using an existing environment with NumPy and SciPy:

```bash
python -m experiments.gps_pgo.run /path/to/problem.npz /path/to/output
# Independent reference evaluator:
python -m experiments.gps_pgo.run /path/to/problem.npz /path/to/output_reference --reference
```

Outputs are `transforms.npz` and `solver.json`, including held-out adjacent P90 and
convergence diagnostics. An iteration-limited run exits unsuccessfully after saving
diagnostic output; it can be retried explicitly with `--max-nfev 500`. The tested
runtime was Python 3.11, NumPy 1.26.4, SciPy 1.17. No weights, inference caches,
recordings, or dependencies are distributed in this directory.

`problem.save_problem(path, data)` converts the original in-memory experiment
schema to a numeric-only NPZ archive; `load_problem` validates it with pickle disabled.
Format version 1 contains:

| Arrays | Shape / meaning |
| --- | --- |
| `format_version` | scalar integer, `1` |
| `scales`, `rotations`, `translations` | initial local-to-world Sim3, `C`, `C×3×3`, `C×3` |
| `camera_positions`, `camera_offsets` | chunk-local camera centres concatenated as `F×3`; `C+1` offsets |
| `gps`, `valid_gps` | world-frame GPS `F×3` and boolean `F`, in the same frame order |
| `seam_a`, `seam_b`, `seam_offsets` | matching local points for each adjacent chunk pair; `C` offsets |
| `loop_a`, `loop_b`, `loop_offsets` | matching local points for each verified loop; `L+1` offsets |
| `loop_i`, `loop_j` | source/destination chunk indices for each loop, each `L` |

Each geometry group has at least two finite pairs. There are at least two chunks,
each with at least one camera, and at least one valid GPS fix. Empty loop arrays are
allowed. NPZ inputs contain neither raw images nor arbitrary Python objects.
Camera orientations are unnecessary for the selected objective. Experimental gravity
options in the Python API additionally require `gravity_targets` in memory and are
not part of this format.

Sampling is intentional and preserved from the experiment: select every third valid
GPS frame per chunk; use even-indexed overlap/loop pairs, then stride by
`max(1, count//80)` or `max(1, count//60)` respectively. These are approximate sampling
budgets, not strict caps. Odd-indexed adjacent points are reserved for reporting.
Corresponding images and the initial dense alignment may already have seen those
pixels: the holdout tests downstream consistency, not unseen-scene generalization.
GPS groups and geometric pair groups normalize residuals by the square root of their
sample count. These are experimental information weights, not calibrated independent
per-point measurement uncertainties.

The first verified image pair supplied the reported flight's only loop constraint.
Three other image pairs supplied its primary revisit validation; that validation is
not automatically reconstructed from an arbitrary NPZ. Do not infer a verified loop
from proximity in the current, possibly drifting map.

## Validation and alternatives

```bash
PYTHONPATH="$PWD/da3_streaming:$PYTHONPATH" python -m unittest discover -s tests -v
```

Synthetic tests compare 80 randomized full residual vectors between batched and
reference evaluators, rotate/translate the full objective, solve a known displacement
without rescaling, and verify numeric archive round-trips. The delivery's full-flight
NPZ run also matches the audited batched solver exactly; finite solver differences
from the slower reported reference map are about 3.4 mm P90.

`compress=True, vector_loss=False` enables paired-point moment summaries for quadratic
loss. Adding `block_loss=True` robustifies a whole group's norm. Twelve virtual pairs
preserve mean squared Sim3 correspondence error; they **do not** preserve independent
per-point radial robust loss. The incompatible option combination raises an error.
These options preserve the tested research alternatives and are disabled by default.

See [the investigation handoff](../../docs/gps_pgo/investigation.md) for results,
limitations, cache locations, and next integration steps.

The separate [visibility retention experiment](VISIBILITY.md) describes conservative
free-space voting and why it barely changed the final map.

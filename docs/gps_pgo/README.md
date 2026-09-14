# GPS Sim3 consistency and optional weights

`Sim3LoopOptimizer.optimize_gps_sim3` accepts and returns mapper transforms
`p_world = s * R @ p_local + t`. GTSAM Similarity3 stores its translation inside
the scale operation, so the adapter divides translation by scale on input and
multiplies on output. GPS factors use `transformFrom` to match the group action.
Sequential measurements always come from visual alignment: GPS anchoring changes
only the initial guess. The fixes apply to the Sim3 GPS path.

The following **optional** keys live under `Loop.SIM3_Optimizer`:

| Key | Default | Meaning |
| --- | --- | --- |
| `seq_sigma_t_m` | absent | Translation sigma in metres, divided by the destination chunk's **initial** global scale for each between factor. Overrides `seq_sigma_t`. |
| `gps_chunk_information_normalization` | `false` | Multiply each accepted GPS factor's covariance by that chunk's accepted factor count. This downweights correlated fixes to roughly one aggregate observation per chunk. |
| `per_chunk_scale_prior_sigma` | `0` | Positive log-scale sigma adds a prior about every chunk's initial scale. Zero disables it. |

The legacy `seq_sigma_t` is in destination-chunk local units, not metres. The
metric conversion is exact for pure translation error at the initial scale; it
is an approximation if scale changes during optimization. Group normalization
scales the complete 5D position/heading covariance, including supplied covariance.
It is a correlation heuristic, not a fitted temporal GPS noise model. The existing
eligibility rules still exclude fixes without usable velocity directions.

Scale priors have very weak pose components (sigma `1e6`); a small scale sigma
approximately preserves scale rather than imposing an exact fixed-scale constraint.
The existing `scale_prior_sigma` on chunk zero remains independent. Absent new keys,
the corrected graph and its numerical output are unchanged.

[starling_example.yaml](starling_example.yaml) is an ablation configuration fragment
for the 787-frame Starling flight investigated in September 2026. Merge its keys
into an existing model configuration. It is not a complete launch configuration or
a generally calibrated default. See the experiment report for quality tradeoffs.

Run the synthetic graph tests in an environment with the project's existing
requirements (including GTSAM 4.3a1):

```bash
PYTHONPATH="$PWD/da3_streaming:$PYTHONPATH" python -m unittest discover -s tests -v
```

These exercise group action/composition, production GPS residuals, graph
initialization/output, visual edge independence from anchoring, translation units,
accepted-factor normalization, optional priors, and default graph equivalence.

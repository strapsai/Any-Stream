# Starling GPS PGO investigation — delivery handoff

The investigation ran on 2026-09-13/14 in `reid-dev`, with existing dependencies
and local model weights. These branches package the validated corrections and
selected research code. They do not deploy a new ROS optimizer or change model
configuration defaults.

## Source and input provenance

- Any-Stream baseline: `4039f9544705e865f40b3a8541afeb8e6d61c47f` (`origin/dev`).
- anystream_ros baseline: `73c5b68b096336931750f66cc620e0bce5b2d047` (`origin/dev`).
- Delivery branches: `codex/gps-pgo-consistency` in both repositories.
- Source MCAP: `/airlab-storage/chiron/datasets/year3/starling/mcaps-03-22-2026/day3/starling2_2B2ZEO_cdr_pp.mcap`.
- Image topic: `/starling2/hires_small_encoded/decoded_pp`.
- All 787 timestamps from `/mnt/data/xstream/mar17/s2_jpg` occur in that topic.
  JPEG recompression means the bytes are not identical. Message dates are March 17,
  despite the parent directory's March 22 name.
- The full MCAP lasts 1,160.56 s; its 839 GPS messages end about 279 s into the recording.
  Main comparisons use the GPS-covered flight. Full-recording processing also ran,
  without extrapolating GPS into the stationary/handling remainder.

Fresh inference called the ROS `AnyStreamer` core on deterministically extracted
images. This exercised inference/alignment rather than ROS replay scheduling or
publication. The controlled Omega comparison used 787 frames, 39 chunks (30 frames,
10 overlap), and the same 728,283 evaluation points. The separate ROS coverage fix
recovers omitted frames but did not change support in these controlled comparisons.

## What is ready to review

1. **Sim3 convention:** mapper tuples act as `s R p + t`; GTSAM stores translation
   inside scale. The adapter converts at construction/output; GPS factors use the
   same group action. Origin-shift regressions exposed metre-scale changes in the
   legacy implementation and numerical invariance after correction.
2. **Visual edges:** sequential measurements use immutable visual alignment.
   GPS anchoring only changes initialization. The supplied bad run used Umeyama
   initialization, so this second defect alone does not explain that run.
3. **Optional weights:** [documented keys](README.md) normalize GPS information per
   chunk, accept a translation sigma in metres, and add per-chunk scale priors.
   Their defaults preserve the corrected graph. The new common conversion helper
   performs division in float64; full-flight delivery results differ from the original
   reviewed patch by at most 2.7 micrometres in sampled point coordinates.
4. **Offline geometric solver:** [portable experimental runner](../../experiments/gps_pgo/README.md),
   batched/reference evaluators, fixed scales, radial robust point residuals and
   explicitly supplied verified loops. The portable full-flight result is exactly
   equal to the audited batched implementation. It differs from the slower reported
   reference by 3.41 mm P90 at finite optimization tolerance.
5. **Conservative visibility research:** a separately documented experiment counts
   depth free-space contradictions; it does not fix poses or erase entire XY columns.
6. **ROS companion:** unique point-frame ownership and a standalone reader that
   recomputes adjacent/revisit metrics directly from embedded Rerun endpoints.

## Metrics and controlled results

Primary adjacent error matches the same timestamp/pixel across independently inferred
chunks. Geometry is sampled on a deterministic grid with frozen validity/range masks.
Even sample indices supply new factors; odd indices report errors. The original depth
inference and visual alignment still processed their images, so this is a downstream
consistency holdout, not a test of unseen-scene generalization.

Nonlocal revisit error uses three image pairs absent from the selected graph. A
fourth pair supplies its only loop. The auxiliary low-surface proxy freezes support
in the initial visual gauge, then reports the P90 height range of per-chunk medians
within repeated 2 m XY cells. Support is selected near the first surface height;
it is not semantic ground segmentation or suitable for every hilly site. Report
coverage, retained points and chunk scales alongside it. Empty repeated support
is undefined, never perfect quality. GPS is a noisy reference rather than truth.

All distances below are metres, with fixed support and initial scales for the
selected geometric graph:

| Map | GPS RMSE | Adjacent P90 | Unused revisit P90 | Low-surface spread P90 |
| --- | ---: | ---: | ---: | ---: |
| Legacy sigma 4 | 1.645 | 4.653 | 7.355 | 5.197 |
| Corrected Sim3 sigma 4 | 0.422 | 3.012 | 2.904 | 3.672 |
| Earlier grouped geometry | 2.597 | 1.703 | 7.247 | 1.657 |
| Selected invariant geometry + verified loop | 2.385 | 1.745 | 0.750 | 1.684 |

Smaller GPS sigma alone worsened the legacy map: sigma 0.1 m gave 7.31 m seam P90.
A small GTSAM-only extension, with 2 m GPS, 0.1 m metric sequential sigma, 0.005 rad
rotation, weak heading and approximately fixed scales, yielded 1.705 m seam P90 and
1.648 m low-surface spread without the dense solver. The example YAML records that
ablation; it is not a universally calibrated configuration.

The verified return loop required a 12.4 m correction and had been rejected by an
arbitrary 10 m map-distance gate. Reciprocal SIFT, epipolar checks, spatial extent,
and held-out 3D checks supported it instead. Changing which of the four verified
image pairs supplies the edge preserved the improvement. A wider search over 2,361
candidate pairs did not establish another edge. Generic loop retrieval is unfinished.

## Why multiple planes appeared

Independently inferred chunks have imperfect relative geometry, scale and tilt.
Small orientation errors amplify at long range: 2 degrees at 50 m is about 1.75 m.
Thin individual surfaces can therefore become separated surfaces when chunks move.
The Sim3 convention mismatch, correlated GPS counted repeatedly, sequential weights
with varying metric meaning, and absent long-baseline constraints compounded this.
The old `seq_sigma_t: 0.05` meant roughly 0.51–3.13 m across this flight's scales.
Position-derived heading is also weakly justified when adjacent displacement is
smaller than GPS noise. Real roofs and terrain must not be mistaken for map defects.

## Robustness and approaches that did not win

Six-seed perturbations added independent noise, smooth bias, or sustained offsets,
each bounded by ±3 m **per axis**, refitting initialization each time. After removing
one rigid alignment, mean map-deformation P90 changed as follows:

| Perturbation | Corrected GPS graph | Selected graph + loop |
| --- | ---: | ---: |
| Independent noise | 1.203 | 0.210 |
| Smooth correlated bias | 3.392 | 1.708 |
| Sustained offset | 2.370 | 1.548 |

Removing each quarter's GPS from both initialization and objective showed a tradeoff:
the selected method predicted the endpoint outages better; the GPS-heavy corrected
graph did better on the two interior outages. These are comparisons to withheld noisy
GPS, not proof of absolute accuracy. Smooth bias still creates metre-scale deformation.

- The first chunk's motion was nearly a line and did not constrain orientation well.
  The existing configuration already waits five chunks; even that fit differed by
  12.6 degrees and 3.78% scale from the complete-flight alignment.
- Causal tests covered 35 prefixes with no future GPS. The loop only arrived at the
  landing chunk and revised old geometry by 3.21 m P90 after rigid alignment.
- Freezing all but five recent poses made GPS/seam quality worse. Ten was less harmful;
  keeping old poses adjustable best handled the late correction.
- Paired-point moments compress quadratic/group-norm factors to 12 virtual pairs.
  They do not preserve the selected per-point radial robust loss, so that combination
  is rejected. No constant-memory online system was demonstrated.
- Conservative two-vote visibility removal discarded about 0.42% of final points,
  retained all occupied XY/low-surface cells, and barely changed surface spread.
  Aggressive XY recency looked clean partly by removing repeated observations.
- Two additional 787-frame MapAnything runs, with and without recorded intrinsics,
  did not displace Omega in this comparison. The existing checkpoint passed a complete
  strict load; no random encoder or downloaded weights were used.
- Anisotropic GPS, flexible-scale and range-weighted alignment trials did not produce
  a combined improvement sufficient to replace the selected result.
- The original component-wise robust prototype failed coordinate-rotation tests.
  Radial vector loss replaced it; the rejected implementation is not promoted here.

The final batched solve took about 3.5 s excluding inference/evaluation. One outage
case needed 170 evaluations and 11.3 s on retry. These are prototype measurements,
not a real-time guarantee.

## Reproduction and retained research record

The sandbox remains `/mnt/data/cache/tmp/gps_pgo_20260913`, linked from
`/home/ubuntu/airlab_ws/experiments/gps_pgo_20260913`. Large inputs and outputs stay
there; no system-wide packages or configuration were changed. Original worktrees
remain on `dev`, including the user's preexisting Optional annotation edit.

| Local sandbox path | Contents |
| --- | --- |
| `REPORT.md`, `phase2/REPORT.md`, `WORK_LOG.md` | Full reports, all significant decisions and failed trials |
| `SOURCE_STATE.json`, `ENVIRONMENT.json` | Source state, runtime versions, model hashes |
| `phase2/results/experiment_index.csv` | Selected result index |
| `phase2/results/final_validation.json` | Original complete validation |
| `phase2/artifacts/verified_loop_comparison.rrd` | Four maps and complete metric endpoints |
| `phase2/artifacts/causal_prefixes.rrd` | Growing-prefix audit |
| `phase2/artifacts/invariant_radial.ply` | Selected full point cloud |
| `delivery/starling_problem.npz` | Portable numeric graph for this branch's CLI |
| `delivery/*validation.json`, `delivery/*tests.log` | Delivery reproduction and regression checks |

From the host, reuse the existing container environment:

```bash
docker exec -w /mnt/data/cache/tmp/gps_pgo_20260913/delivery/Any-Stream reid-dev \
  bash /mnt/data/cache/tmp/gps_pgo_20260913/delivery/env.sh \
  -m experiments.gps_pgo.run \
  /mnt/data/cache/tmp/gps_pgo_20260913/delivery/starling_problem.npz \
  /mnt/data/cache/tmp/gps_pgo_20260913/delivery/reproduce
```

The checked-in [delivery validation](delivery_validation.json) records numeric
agreement and test coverage. The next implementation task is a ROS adapter that
builds and retains appropriate geometric constraints, accepts verified loop edges,
and updates old map chunks. Calibration, loop discovery, motion-conditioned startup,
and transfer to other flights remain open. Those are separate integration tasks.

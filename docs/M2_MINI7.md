# M2 `mini_7` adapter, evidence, and cache

M2 is an independently written official-data development slice. It does not
consume or compare any historical source tree, executable, CSV, or generated
output. `mini_7` remains regression/development data and cannot support a
generalization, calibration, detector, or final-test claim.

## Dataset contract observed locally

- Raw root: configurable with `--data-root`; the local value is `data`.
- Archive SHA-256:
  `46e9dbe9f6e952be9186a6271abbcaaf11c965a5589d785fc278afd3eb235663`.
- Hugging Face revision:
  `1a61aea747aa6bc45da2c2f085a1f1d3abc41f91`.
- Five cloud agents: `003`, `004`, `dome`, `laser`, and `top`, each with
  contiguous cloud sync IDs 0–298. Odometry exists for the three vehicles.
- Archive label filenames are one-based (`mini_7_1.txt` through
  `mini_7_290.txt`). The configured offset binds them to top-anchor semantic
  sync IDs 0–289. Therefore sync IDs 290–298 are label-unavailable, rather than
  empty scenes.
- A short timestamp fraction omits leading zeroes. For example,
  `1712121167.80167605` maps to integer nanoseconds
  `1712121167080167605`, exactly matching both odometry time columns. The
  adapter audits every vehicle cloud timestamp against its CSV.
- PCD XYZI fields are scalar float32 and the local encoding is ASCII. Raw
  intensity is retained; the documented diagnostic normalization is
  `clip(raw / 3500, 0, 1)`.

## Calibration boundary

The raw infrastructure clouds are map-like and the vehicle clouds are local.
The official sensor-to-top/map-to-top matrices, axis/origin conventions,
nominal coverage, and the reported 3.25 m vehicle vertical-origin adjustment
remain unresolved. No matrix or vertical adjustment is guessed or inferred
from labels. All five real point-to-GT projections, visibility decisions,
per-box counts, count surprise, and peer corroboration therefore carry explicit
`unresolved_transform`, `visibility_unknown`, or `insufficient_peers` outcomes.

Known-coordinate synthetic fixtures prove forward/inverse SE(3), `top` and
`dome` PCD production-loader round trips, yaw-box boundaries/counts, two-sided
surprise, and leave-one-sender-out invariants independently of this unresolved
real calibration. Vehicle kinematic self-consistency remains available after
the first sample because it compares self-reported map position with
self-reported body velocity rotated by ten midpoint-SLERP orientations. It is
not independent proof of honesty.

## Cache and D1 commands

Frame ranges are half-open. A two-frame cache and D1 playback are:

```bash
uv run lidar-shield cache build \
  --data-root data \
  --artifact-root artifacts \
  --experiment-id m2-mini7-d1 \
  --frames 0:2

uv run lidar-shield demo d1 \
  --data-root data \
  --artifact-root artifacts \
  --experiment-id m2-mini7-d1 \
  --frames 0:2

uv run lidar-shield cache verify \
  --cache-dir artifacts/m2-mini7-d1/sequences/mini_7/clean
```

The clean directory contains `frame.parquet`, `object.parquet`,
`evidence.parquet`, `source_manifest.json`, and `resume.complete.json`.
Manifest provenance binds the archive and selected-source hashes, configured
and resolved source roots, selected frames/agents, registry transforms,
configuration hashes, package/contract/cache/code versions, split role,
producer command, output row counts/sizes/hashes, and the official revision.
The completion marker is written only after Parquet row-count/readability and
hash verification. A marker/config mismatch, output mutation, or partial
directory is rejected rather than reused.

The historical 0.40/0.30/0.30 score is emitted only as
`historical_fixed_risk_diagnostic`; its perfect missing-kinematics choice and
raw share are intentionally preserved as named limitations. The separate
unknown-aware score uses kinematic, two-sided support, and peer evidence. It
renormalizes only with at least 0.60 original weight, requires kinematics when
applicable, and otherwise abstains. Both risks are uncalibrated in M2.

## Deliberately deferred

- Authoritative real-data extrinsics, coverage, coordinate axes/origins, and
  visibility acceptance.
- M5 multi-sequence benign normalizer fitting and contextual count model.
- Attacks, temporal state, communication/fusion, learned risk/calibration,
  final-test evaluation, detector claims, and BEV work.

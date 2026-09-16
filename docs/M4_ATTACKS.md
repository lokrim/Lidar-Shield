# M4 immutable attacks and incremental evaluation

M4 is the first attack framework in `lidar-shield`. Schema `1.0.0` binds a
full episode/variant identity, registered source and split role, exact frame,
source time, sender, object-or-interval target, raw/archive/clean-cache hashes,
family parameters and units, seed, attacker knowledge, expected modality, and
code/config versions. M3's disposable corruption stays outside this namespace.

## Overlay and dependency contract

Velocity spike and drift overlays modify only declared body-frame velocity
samples. A spike adds the full `delta_v_mps`; drift adds
`(j + 1) / K * delta_v_mps`. Position and orientation remain unchanged. Both
intervals touching every changed endpoint are recomputed with M2's
ten-midpoint-SLERP evidence function.

Spatial addition samples seeded uniform box-local points. Non-coordinate fields
use the declared source-median policy. Removal selects exactly
`floor(f * n_inside)` seeded clean eligible points. Derived PCDs retain field
order, scalar dtypes/counts, and encoding, then reload through the production
adapter. Every overlapping oracle box and all leave-one-sender-out peer
consequences are dependency-expanded, so affected honest companions remain.

Each variant directory contains:

```text
attack_manifest.json
overlay.json | overlay.pcd
evidence_delta.parquet
realized_effect.json
```

Raw and clean-cache hashes are checked before generation and after success or
failure. Existing variant directories are collisions and are not overwritten.
Failed, zero-effect, low-effect, and interrupted attempts retain explicit
records and remain coverage rows; effect size is never a retention gate.

## D3 command

With the repository-relative `data/mini_7` input available, run:

```bash
uv run lidar-shield demo d3 \
  --config configs/attacks/mini7_d3_velocity_spike.yaml \
  --data-root data \
  --artifact-root artifacts \
  --frames 0:3
```

Use a fresh experiment/variant ID before rerunning because variants are
immutable. The command builds or verifies the clean cache, resolves and
validates the manifest, writes the overlay and evidence delta, prints the
manifest/effect preview, and replays clean and attacked evidence through fixed
risk, EWMA/state, full-share/hard-gate byte accounting, and proxy fusion.
If the immutable clean cache already exists but the source archive has been
removed after extraction, pass
`--clean-dir artifacts/<clean-experiment>/sequences/mini_7/clean`; its manifest
still supplies the recorded archive hash while the targeted raw file is checked
directly.

`configs/attacks/stage1_matrix.yaml` records newly authored M4 development
seeds/severities. It authorizes only `mini_7`; M5 must freeze new scientific
sequence IDs before those sources can be attacked.

## Verification and boundary

`scripts/verify_m4.sh` runs D3 when `mini_7` is present, then all tests, Ruff,
format, and strict mypy. Dataset-free tests cover deterministic reruns,
production PCD reload (including `dome` and an extra integer field), source
hash preservation, incremental/full equivalence, endpoint spillover,
overlapping boxes, collisions, unsuccessful/interrupted attempts, and honest
peer retention.

M4 does not add new-sequence attacks, contextual normalizer fitting, learned
models/calibration, replay/delay/drop/burst attacks, sustained/on-off strategy
generation, collusion, score-aware search, or robust fusion.

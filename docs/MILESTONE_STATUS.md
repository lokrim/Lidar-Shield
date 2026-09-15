# Milestone status

## Implemented: M0 and M1

- Distribution/import package and editable CLI installation.
- Strict YAML configuration loading with explicit, config-relative paths.
- Domain-neutral JSON provenance and file-integrity foundation.
- Deterministic synthetic TXT label, empty-label, odometry, and PCD examples.
- Locked Stage 1 environment, optional trust-neural/visualization extras,
  quality tooling, CI, local verification, and Colab bootstrap script.
- Canonical typed frame, object, and evidence-envelope schema version `1.0.0`.
- Duplicate/null/key checks and cardinality/overlap-safe exact joins.
- Causal and declared-buffered source matching with signed source/pose ages.
- Bounded pose interpolation and rigid-transform round trips.
- Five-agent capability registry with explicit dynamic membership/dropout.
- Stage 1 split skeleton, unresolved-source guards, and provenance leakage checks.
- Hash-bound source/experiment/schema manifests and deterministic M1 fixtures.
- D0 nine-event synthetic timeline with no unavailable-event attrition.

## Intentionally not implemented

M2 and later own official raw data, evidence engines, attacks, models, temporal state,
communication/fusion, replay, evaluation, and reports. The CLI therefore exposes
no commands for those capabilities in M1.

Stage 2 detector dependencies remain unselected and isolated until M10's
maintained-stack/checkpoint gate. Open3D is optional visualization only; PyTorch
is in the `trust-neural` extra.

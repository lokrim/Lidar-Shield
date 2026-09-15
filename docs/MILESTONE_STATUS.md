# Milestone status

## Implemented: M0

- Distribution/import package and editable CLI installation.
- Strict YAML configuration loading with explicit, config-relative paths.
- Domain-neutral JSON provenance and file-integrity foundation.
- Deterministic synthetic TXT label, empty-label, odometry, and PCD examples.
- Locked Stage 1 environment, optional trust-neural/visualization extras,
  quality tooling, CI, local verification, and Colab bootstrap script.

## Intentionally not implemented

M1 owns canonical frame/object/evidence contracts, synchronization, pose
interpolation, geometry, split registry, and full contract fixtures. M2 and
later own official raw data, evidence engines, attacks, models, temporal state,
communication/fusion, replay, evaluation, and reports. The CLI therefore exposes
no commands for those capabilities in M0.

Stage 2 detector dependencies remain unselected and isolated until M10's
maintained-stack/checkpoint gate. Open3D is optional visualization only; PyTorch
is in the `trust-neural` extra.

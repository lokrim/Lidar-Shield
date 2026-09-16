# Manifest and artifact convention

The domain-neutral JSON envelope is implemented by
`lidar_shield.manifest.Manifest`. M1 adds typed `SourceManifest` and
`ExperimentManifest` contracts without conflating them with frame/object rows.

Required manifest data includes manifest and fixture schema versions, a stable
experiment ID, artifact kind, an allowed origin (`synthetic`,
`official_dataset`, or `generated`), a precise origin description, UTC creation
time, seed, configuration hash, software/schema versions, the producer command,
and an ordered list of explicitly named files. Each file record contains a
relative path, byte count, media type, SHA-256 hash, and optional expected
values. Absolute paths, parent traversal, duplicate paths, and unstable file
ordering are rejected.

Later milestones extend the envelope without weakening these requirements. The
stable experiment layout is:

```text
artifacts/<experiment_id>/
├── experiment.json
├── splits.json
├── sequences/<sequence_id>/
│   ├── clean/{source_manifest.json,resume.complete.json}
│   ├── clean/{frame,object,evidence}.parquet
│   └── variants/<variant_id>/...
├── models/<model_id>/...
└── evaluation/{predictions,states,schedules,fusion}.parquet
```

Only `artifacts/README.md` is committed. Generated experiment contents are
ignored. The synthetic fixture manifest demonstrates loading, stable ordering,
expected values, exact byte sizes, and integrity checks without raw data.
`manifests/contracts/m1_schema_registry.json`, `manifests/datasets/m1_*.json`,
and `manifests/experiments/m1_contract_fixtures.json` bind the M1 schema,
configuration registries, two synthetic sources, and producer command.

M2 stores its cache-local `source_manifest.json` beside the three clean tables
and completion marker. It additionally binds the configured/resolved data root,
official archive and selected-file SHA-256 hashes, Hugging Face revision, exact
frame/agent selection, declared transforms (including unresolved values), split
role, dataset/agent/split configuration hashes, package/contract/cache/code
versions, producer command, and verified output hashes. The
`resume.complete.json` marker binds the manifest and output hashes and is
published only after Parquet verification.

M4 introduces `attack_manifest.json` schema `1.0.0`. It binds complete
episode/variant/source/target identity, split role, archive/selected-source/raw/
clean-cache hashes, family parameters and units, seed, attacker knowledge,
expected modality, and code/config/schema versions. Each immutable variant also
contains one JSON or derived-PCD overlay, `evidence_delta.parquet`, and
`realized_effect.json`. Failed and interrupted attempts use the same identity;
completed variant IDs are never overwritten.

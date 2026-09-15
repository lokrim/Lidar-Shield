# Manifest and artifact convention

The M0 JSON envelope is implemented by `lidar_shield.manifest.Manifest`. It is a
domain-neutral foundation, not the M1 canonical frame/object schema.

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
│   ├── source_manifest.json
│   ├── clean/{frame,object,evidence}.parquet
│   └── variants/<variant_id>/...
├── models/<model_id>/...
└── evaluation/{predictions,states,schedules,fusion}.parquet
```

Only `artifacts/README.md` is committed. Generated experiment contents are
ignored. The synthetic fixture manifest demonstrates loading, stable ordering,
expected values, exact byte sizes, and integrity checks without raw data.

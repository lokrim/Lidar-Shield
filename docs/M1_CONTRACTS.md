# M1 canonical contract

Contract schema `1.0.0` uses this immutable identity, in order:

```text
experiment_id, variant_id, sequence_id, session_id, sync_frame_id,
decision_time_ns, source_time_ns, agent_id, agent_type,
object_id_or_local_index
```

Identifiers are strings, including `"003"`; frame indices and nanoseconds are
integers. Frame tables and object tables are distinct. A frame row always has a
null object key. `source_time_ns`, `match_age_ns`, `pose_source_time_ns`, and
`pose_age_ns` are null only when their source is unavailable. An object row has
non-null source and object keys. Empty scenes therefore produce one frame row
with `empty_scene` and zero object rows; they never receive a fabricated object.
Evidence envelopes declare `frame` or `object` granularity, applicability,
availability, and a reason. An unavailable value is null, not a favorable or
unfavorable score.

The signed-age convention is:

```text
age_ns = decision_time_ns - matched_source_time_ns
```

A positive age is a past source, zero is exact, and a negative age is a future
source admitted only by `buffered_nearest` with a positive recorded look-ahead.
Primary `causal_backward` matching selects the newest arrived source at or
before the decision within its age limit. Buffered matching selects the nearest
source available within both the age limit and declared buffer. Unmatched,
late, missing-message, empty-scene, invalid-pose, out-of-range, stale, and
non-member events remain rows.

Pose interpolation never extrapolates. It requires finite translations, unit
`x,y,z,w` quaternions, strictly increasing integer sample times, a maximum
bracketing gap, and an availability-compatible future bracket. Its pose age is
computed from the newer interpolation bracket so buffered look-ahead remains
auditable.

Exact table joins use `merge_checked`: callers must state cardinality and rename
every overlapping non-key column before merging. Default suffixes are never
allowed to conceal or replace identity. Typed table builders use pandas nullable
`Int64` for nullable nanoseconds.

## Capabilities, splits, and provenance

`configs/agents/mixed_signals.yaml` declares five agents without runtime
hardcoding. Infrastructure odometry is `not_applicable`; it is not assigned a
perfect observation. Unverified nominal coverage and dataset calibration are
explicit nulls. The `top -> top` identity frame transform is the only verified
entry; remaining official extrinsics and sensor coverage are M2 prerequisites.

`configs/data/stage1_split.yaml` freezes `mini_7` as regression/development
only. Four training, one calibration, and two final-test source IDs remain
explicit unresolved slots pending the M5 metadata-only selection. Unassigned
sources and `mini_7` cannot be used for scientific fitting or attack generation.
Assigned roles must be frozen before variants, and leakage checks keep every
agent, object, neighboring-frame block, clean counterpart, and variant of a
source episode in one role. A frame-local object index is scoped by sequence,
session, and frame; it is not a track ID.

The committed schema, source, experiment, agent-config, and split-config hashes
under `manifests/` demonstrate the roadmap's
`experiment -> sequence -> clean/variant` artifact binding without producing
raw-data caches or attack artifacts.

## D0 and verification

Run the deterministic nine-event, five-agent replay:

```bash
uv run lidar-shield demo d0 --agents configs/agents/mixed_signals.yaml
```

Run the full M1 gate:

```bash
scripts/verify_m1.sh
```

Both commands require no raw dataset or historical material.

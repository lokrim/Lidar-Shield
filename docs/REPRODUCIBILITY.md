# Reproducibility contract

M0 establishes these conventions for every later producer:

- Use a named `experiment_id`; write only below
  `artifacts/<experiment_id>/`, and never mix two experiment runs.
- Record the exact producer command, package/software versions, configuration
  hash, schema versions, seed, source IDs and content hashes in JSON manifests.
- Use the configured unsigned 32-bit seed as the root seed. Later producers
  must derive named child seeds deterministically and record them rather than
  relying on process-global random state.
- Sort identifiers using UTF-8 lexicographic ordering before serialization or
  aggregation. Preserve string identifiers such as `"003"` and integer
  nanoseconds; do not pass them through floating point.
- Hash configuration files as their exact source bytes so the same checked-in
  YAML has the same digest on every machine. Serialize small JSON records as
  UTF-8 with sorted keys, compact separators,
  no NaN/Infinity, and exactly one trailing newline
  (`json-sort-keys-utf8-v1`). Parquet becomes the typed table format in M1.
- Hash explicitly named project inputs and outputs with SHA-256. M1 source and
  experiment manifests bind the schema, agent, split, and source hashes. Never scan,
  inventory, or hash historical repositories or teammate artifacts.
- Treat completed artifact roots as immutable. A changed configuration, source,
  code revision, schema, or seed receives a new experiment ID.

Python, NumPy, PyTorch, and worker-specific deterministic controls will be set by
the milestone that first uses each runtime. M0 records the policy but does not
claim later numerical kernels are implemented.

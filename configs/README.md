# Configuration boundary

`default.yaml` is the bootstrap configuration. Every path is explicit and is
resolved relative to the configuration file. Unknown keys are rejected.

M1 adds `agents/mixed_signals.yaml` and `data/stage1_split.yaml`. Unknown
calibration/FOV/range facts remain explicit nulls, and scientific sequence slots
remain unresolved until M5. Later milestones add `evidence/`, `attacks/`,
`models/`, and `policies/` only when their behavior is implemented.

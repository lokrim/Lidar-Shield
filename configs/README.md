# Configuration boundary

`default.yaml` is the bootstrap configuration. Every path is explicit and is
resolved relative to the configuration file. Unknown keys are rejected.

M1 adds `agents/mixed_signals.yaml` and `data/stage1_split.yaml`. Unknown
calibration/FOV/range facts remain explicit nulls, and scientific sequence slots
remain unresolved until M5. M3's disposable fixture contains its own explicit
EWMA threshold and packet deadline so D2 needs no scientific policy
configuration or artifact. M4 owns
`attacks/mini7_d3_velocity_spike.yaml` and the newly authored
development-only `attacks/stage1_matrix.yaml`; M5 still owns every new-sequence
scientific assignment. Later milestones add `evidence/`, `models/`, and
`policies/` only when their behavior is implemented.

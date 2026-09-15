# Configuration boundary

`default.yaml` is the M0 bootstrap configuration. Every path is explicit and is
resolved relative to the configuration file. Unknown keys are rejected.

Later milestones will add the roadmap's `agents/`, `data/`, `evidence/`,
`attacks/`, `models/`, and `policies/` configurations when their schemas and
behavior are implemented. Their absence in M0 is intentional; this repository
does not advertise non-working commands or premature domain contracts.

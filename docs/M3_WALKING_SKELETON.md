# M3 thin end-to-end walking skeleton

M3 connects the existing unknown-aware fixed baseline to an independently
implemented EWMA, a threshold-only auditable state label, full-share and
hard-gate communication, deterministic proxy packets, mean fusion, ego-only
fallback, and integration metrics. Each component remains directly callable.

The fixed output is an **uncalibrated risk score**. Its complementary
operational reliability is separately named. `calibrated_p_attack` remains
null, predictive uncertainty remains explicitly unavailable, and abstention is
separate from both. Learned calibration belongs to M6. The M3 state machine has
only `normal`, `quarantined`, and `unknown` labels around one EWMA threshold;
the elapsed-time accumulator, hysteresis, recovery dwell, and calibrated
controller remain deferred to M7.

## D2 verification command

Run the complete M3 gate with one command:

```bash
scripts/verify_m3.sh
```

The command prints D2 JSON and then runs all tests, Ruff checks, and strict
mypy. To print only D2:

```bash
uv run lidar-shield demo d2 \
  --fixture tests/fixtures/m3/integration_fixture.json
```

Both commands are dataset-free and write no scientific artifact. The output is
stdout only.

## Fixture boundary and analytical result

`tests/fixtures/m3/integration_fixture.json` defines one immutable
`fixture_run_id`, source/sequence/session identity, seed `1729`, clean rows, and
one exact `integration_corruption` replacement. It changes the selected
sender's three fixed-baseline inputs from fully benign values to their bounded
suspicious endpoints and changes its aligned proxy vector from `[1, 1]` to
`[8, 8]`. Therefore its fixed risk changes analytically from `0` to `1`; after
the preceding clean value, alpha `0.30` produces EWMA risk `0.30`, which crosses
the fixture threshold `0.25`.

Full share admits the changed packet and has positive proxy reconstruction
error. Hard gate rejects that exact packet, admits the unchanged companion, and
has zero corrupted-source bytes and zero reconstruction error for the target
frame. A later frame preserves two missing remote observation rows and returns
the ego vector exactly with degraded metadata. The final frame has no ego and
explicitly abstains.

This changed input is a disposable **integration corruption fixture**, not a
simulated attack or scientific result. It has no attack manifest, attack
registry entry, episode/variant identity, split membership, coverage
denominator, attack-effectiveness metric, or training/calibration/test
eligibility. Its clean reference enters only `evaluation.metrics`; risk,
scheduling, and fusion do not receive it. M4 must create its own scientific
attack schemas and artifacts and must not promote this fixture.

## Packet, policy, and fusion contracts

Proxy packets use canonical sorted-key UTF-8 JSON with a trailing newline. The
descriptor includes fixture/source identity, sender and source time, pose,
region key, payload type, local confidence, SHA-256 payload integrity metadata,
and a measured byte count that includes the descriptor itself. Full share and
hard gate consume the same packet objects and use the same integrity check,
source-age deadline, and measured cost.

Fusion accepts framework-neutral aligned tuples. It normalizes over the valid
ego plus admitted remotes, reports contributor count and normalized remote
weight, and keeps quality separate from unavailable predictive uncertainty.
No remote contributor returns exact ego with `degraded=true`; no valid ego
returns an abstention.

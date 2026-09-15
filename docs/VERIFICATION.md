# M0/M1 verification record

Verification is run with `scripts/verify_m0.sh`, which performs a frozen editable
sync, CLI version/config smoke checks, dataset-free tests, Ruff lint/format, and
mypy strict type checking.

M1 adds `scripts/verify_m1.sh`, which also runs the D0 timeline and every M1
contract, synchronization, geometry, split, leakage, fixture, and manifest check.

- Local macOS arm64, CPython 3.12.13, 2026-09-15 (M1): passed. Observed 58
  tests passing with 94.70% branch-aware coverage; frozen dependency check, D0,
  Ruff lint/format, and strict mypy all passed. `uv build` produced the wheel and
  source distribution with the M1 modules and no planning/historical tree.

- Local macOS arm64, CPython 3.12.13, 2026-09-15: passed. Observed 22 tests
  passing with 94.58% branch-aware coverage; Ruff lint/format, strict mypy,
  editable import, CLI version, valid configuration, and invalid-configuration
  exit-code checks passed. `uv build` also produced and inspected a wheel and
  source distribution; neither contains the ignored historical planning tree.
- GitHub Actions, CPython 3.10 and 3.12: configured, not yet observed in CI.
- Google Colab: reproducible bootstrap is documented in
  `scripts/colab_bootstrap.sh`; not executed from this local environment.

No line above equates a configured but unavailable environment with an executed
smoke test. Update this record only with observed results.

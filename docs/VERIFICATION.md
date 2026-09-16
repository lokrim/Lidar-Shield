# M0–M4 verification record

Verification is run with `scripts/verify_m0.sh`, which performs a frozen editable
sync, CLI version/config smoke checks, dataset-free tests, Ruff lint/format, and
mypy strict type checking.

M1 adds `scripts/verify_m1.sh`, which also runs the D0 timeline and every M1
contract, synchronization, geometry, split, leakage, fixture, and manifest check.

M2 adds `scripts/verify_m2.sh`, which runs all static/test gates and, when the
official archive is present, the two-frame real D1 cache/playback and cache
verification.

M3 adds `scripts/verify_m3.sh`, which prints the dataset-free D2 clean versus
integration-corruption result and runs the complete test, lint, format, and
strict-type gate. It writes no scientific experiment artifact.

M4 adds `scripts/verify_m4.sh`, which runs D3 against registered local `mini_7`
when available and always runs synthetic overlay, reload, dependency,
source-integrity, episode, coverage, and quality gates.

- Local macOS arm64, CPython 3.12.13, 2026-09-16 (M4): passed. Observed 107
  tests passing and one optional archive-presence test skipped, with 90.45%
  branch-aware coverage; Ruff and strict mypy passed. D3 reused the verified
  two-frame `mini_7` clean cache because the extracted source remained present
  while `data/train/mini_7.tar` was no longer locally available. It generated
  a successful registered velocity-spike episode with manifest SHA-256
  `7834ccfadfc034b9cc32bd8fde89025dc06255212a4e25065463780147f7984b`,
  realized residual change `0.2135470861893952 m`, reproducible attacked
  evidence SHA-256
  `a0706a2edd3cc114d8318ea81b92c3b2e463881a9fccd73cf660cd7391eb5958`,
  one unscreened coverage attempt, and four clean plus four attacked fusion
  outcomes. Offline package build produced the wheel and source distribution;
  the wheel contains M4 modules and no tests, raw data, artifacts, planning
  tree, or historical dependency.

- Local macOS arm64, CPython 3.12.13, 2026-09-15 (M3): passed. Observed 97
  tests passing and one optional real-data test skipped, with 91.45%
  branch-aware coverage; Ruff and strict mypy passed. D2 reproduced target risk
  `0.0 → 1.0`, EWMA risk `0.0 → 0.30`, state `normal → quarantined`, full-share
  corrupted-source bytes `725`, hard-gate corrupted-source bytes `0`,
  full-share aggregate proxy error `3.299831645537222`, and hard-gate aggregate
  proxy error `0.0`. Both policies reported ego-only fallback and ego-absent
  abstention from the same immutable fixture identity. `uv build` produced the
  source distribution and wheel; the wheel contains all M3 runtime modules and
  no tests, raw data, planning tree, or historical dependency.

- Local macOS arm64, CPython 3.12.13, 2026-09-15 (M2): passed. Observed 74
  tests passing with 90.10% branch-aware coverage; Ruff and strict mypy passed.
  Official `mini_7` indexing found five agents × 299 clouds, three odometry
  CSVs with complete filename/time agreement, and semantic labels 0–289. Real
  `top` and `dome` PCDs decoded as ASCII float32 XYZI. D1 produced ten
  frame-agent rows and verified three clean Parquet outputs; real
  transform-dependent evidence correctly remained unavailable. `uv build`
  produced the source distribution and wheel with all M2 modules and no raw or
  historical tree.

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

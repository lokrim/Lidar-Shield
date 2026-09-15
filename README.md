# lidar-shield

`lidar-shield` is a fresh, independently implemented coursework research project
studying sender trust, temporal assurance, communication, and robust fusion in
the Mixed Signals V2X setting. The distribution and CLI are `lidar-shield`; the
Python import package is `lidar_shield`.

Milestone 0 is implemented. It provides packaging, a locked Stage 1 environment,
strict configuration loading, provenance/integrity foundations, tiny synthetic
fixtures, quality checks, and CI. It does **not** implement dataset indexing,
canonical domain schemas, geometry/evidence, attacks, trust models, temporal
control, communication/fusion, replay, evaluation, or reporting commands.

## Clean setup

Install [uv](https://docs.astral.sh/uv/), then create the pinned Python 3.12
editable environment:

```bash
uv sync --frozen
uv run python -c "import lidar_shield; print(lidar_shield.__version__)"
uv run lidar-shield --help
uv run lidar-shield --version
uv run lidar-shield config validate --config configs/default.yaml
```

Python 3.10–3.13 is supported. `uv.lock` is the authoritative complete lock;
`.python-version` selects Python 3.12 for local and Colab reproducibility. The
base environment includes NumPy, pandas, PyArrow, SciPy, pypcd4,
numpy-quaternion, PyYAML/Pydantic, scikit-learn, Matplotlib, Seaborn, and Jinja.
PyTorch is isolated in the `trust-neural` extra and Open3D in `visualization`:

```bash
uv sync --frozen --extra trust-neural
uv sync --frozen --extra visualization
```

No detector dependency is selected in M0. Stage 2 receives a separate optional
environment only after M10 verifies a maintained detector stack and checkpoint.

## Checks

Run the complete local gate:

```bash
scripts/verify_m0.sh
```

Or run its parts:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

These checks require no raw dataset, GPU, historical source tree, script, CSV,
output, or teammate artifact. The synthetic fixture set was newly authored from
the written contracts and can be regenerated deterministically:

```bash
uv run python scripts/generate_synthetic_fixtures.py \
  --output tests/fixtures/synthetic
```

## Configuration and artifacts

Every CLI configuration path is explicit. Relative values inside YAML are
resolved relative to that YAML file; unknown fields and invalid values produce a
nonzero exit with a field-specific message. The default seed is `1729`, schemas
and software are separately versioned, identifiers use stable UTF-8
lexicographic ordering, small JSON uses stable sorted-key serialization, and
content integrity uses SHA-256.

Generated work belongs under the ignored
`artifacts/<experiment_id>/` hierarchy. Producers must record their command,
configuration hash, source/output hashes, schema/software versions, seed, and
stable ordering. Completed experiments are immutable and isolated. See the
[reproducibility contract](docs/REPRODUCIBILITY.md) and
[manifest/artifact convention](docs/MANIFESTS.md). The
[dependency record](docs/DEPENDENCIES.md) records the selected local versions
and which APIs were independently checked.

## Colab

After cloning the repository into a Colab runtime, run:

```bash
bash scripts/colab_bootstrap.sh
```

The script pins uv, performs a frozen editable sync, validates the default
configuration, and runs dataset-free tests. It does not mount Drive, download
Mixed Signals, or request a GPU. The [verification record](docs/VERIFICATION.md)
distinguishes locally observed results from configured but unavailable CI/Colab
environments.

## Scope and provenance

M1 will add canonical frame/object/evidence contracts, synchronization, pose
handling, split registration, and their full fixtures. Official raw archive
processing starts in M2. Later milestones add attacks, models, temporal control,
communication/fusion consumers, evaluation, and reports. Detector/BEV work is an
optional, gated M10 extension. See [milestone status](docs/MILESTONE_STATUS.md).

Every production module is written in this repository from the current
specifications and maintained-library APIs. Historical/reference material is
optional documentary research context only: it is never copied, ported,
adapted, imported, executed, scanned, hashed, packaged, or required at runtime.
Missing historical material cannot fail any gate. The current specification
wins over historical behavior; independently derived mathematical equivalence is
allowed and does not require artificial differences.

- [Implementation boundary](planning/IMPLEMENTATION_BOUNDARY.md)
- [Architecture](planning/NEW_ARCHITECTURE_SPEC.md)
- [Roadmap](planning/IMPLEMENTATION_ROADMAP.md)
- [Feasibility plan](planning/FEASIBILITY_PLAN.md)
- [Research notes](planning/RESEARCH_NOTES.md)

`planning/CURRENT_STATE.md` and `planning/CRITICAL_REVIEW.md` describe historical
research only. They are not the current implementation state or executable
requirements.

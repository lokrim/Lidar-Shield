# Dependency selection record

`uv.lock` was resolved on 2026-09-15 with uv 0.11.17 for the declared Python
3.10–3.13 range. A frozen local CPython 3.12.13 sync selected:

| Purpose | Package | Local selected version |
|---|---|---:|
| arrays | NumPy | 2.5.3 |
| tables | pandas | 3.0.5 |
| typed columnar data | PyArrow | 23.0.1 |
| scientific numerics | SciPy | 1.18.1 |
| PCD I/O | pypcd4 | 1.5.0 |
| quaternions | numpy-quaternion | 2024.0.13 |
| YAML | PyYAML | 6.0.3 |
| validation | Pydantic | 2.13.5 |
| classical ML | scikit-learn | 1.9.1 |
| plotting | Matplotlib / Seaborn | 3.11.2 / 0.13.2 |
| report templates | Jinja2 | 3.1.6 |

The universal lock may select compatible marker-specific versions on another
supported interpreter or platform; `uv sync --frozen` must not re-resolve it.

Context7 was checked on 2026-09-15 against the official Pydantic and pypcd4
documentation before implementation. M0 uses Pydantic v2 `ConfigDict`,
`model_validate`, `ValidationError`, and field-level non-strict parsing for YAML
paths/JSON tuples. The PCD smoke uses the documented
`PointCloud.from_path(path).numpy(("x", "y", "z", "intensity"))` API. Context7
also records pypcd4 support for CPython 3.8–3.13.

M1 rechecked the official pandas and SciPy references through Context7 before
implementation. The pandas contract confirms sorted-key `merge_asof` semantics,
`backward`/`nearest`, `by`, `tolerance`, merge cardinality validation, and
overlap suffix behavior. M1 implements synchronization directly for explicit
event accounting and uses pandas exact merges only through a wrapper requiring
cardinality and deliberate overlap renames. The SciPy contract confirms
`Rotation.from_quat` in `x,y,z,w` order, rotation composition/inversion, and
strictly increasing `Slerp` key times. M1 validates finiteness and unit
quaternions before invoking SciPy and bounds interpolation separately.

`torch` is locked only through the `trust-neural` extra and `open3d` only through
the `visualization` extra; neither was installed in the local M0 base smoke.
Detector dependencies are deliberately unselected until the M10 gate.

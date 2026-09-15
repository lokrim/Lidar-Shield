import json
from pathlib import Path

import pytest

from lidar_shield.manifest import (
    Manifest,
    ManifestError,
    canonical_json_bytes,
    load_manifest,
    sha256_bytes,
    sha256_file,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def test_fixture_manifest_validates_and_verifies_all_listed_files() -> None:
    manifest = load_manifest(MANIFEST_PATH, verify_files=True)

    assert manifest.origin == "synthetic"
    assert manifest.experiment_id == "m1-synthetic-fixtures"
    assert [record.path for record in manifest.files] == sorted(
        record.path for record in manifest.files
    )


def test_canonical_json_is_stable_and_rejects_nan() -> None:
    assert canonical_json_bytes({"z": 1, "a": "003"}) == b'{"a":"003","z":1}\n'
    assert sha256_bytes(b"fixture") == (
        "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"
    )
    with pytest.raises(ManifestError, match="canonical JSON"):
        canonical_json_bytes({"value": float("nan")})


def test_file_hash_matches_manifest() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    record = next(item for item in manifest.files if item.path.endswith(".pcd"))
    assert sha256_file(FIXTURE_ROOT / record.path) == record.sha256


def test_manifest_rejects_unstable_or_unsafe_paths() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["files"] = list(reversed(raw["files"]))
    with pytest.raises(ValueError, match="lexicographic"):
        Manifest.model_validate(raw)

    raw["files"] = [dict(json.loads(MANIFEST_PATH.read_text())["files"][0])]
    raw["files"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="relative"):
        Manifest.model_validate(raw)


def test_integrity_failure_names_the_file(tmp_path: Path) -> None:
    copied_manifest = tmp_path / "manifest.json"
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["files"] = [raw["files"][0]]
    copied_manifest.write_text(json.dumps(raw), encoding="utf-8")
    target = tmp_path / raw["files"][0]["path"]
    target.parent.mkdir(parents=True)
    target.write_text("tampered", encoding="utf-8")

    with pytest.raises(ManifestError, match="size mismatch"):
        load_manifest(copied_manifest, verify_files=True)


def test_invalid_manifest_is_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="invalid manifest"):
        load_manifest(path)

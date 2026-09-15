"""Small, domain-neutral provenance manifests for M0 inputs and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ManifestError(ValueError):
    """A manifest is invalid or an integrity check failed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and small artifact files."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"value is not canonical JSON data: {exc}") from exc
    return f"{text}\n".encode()


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one explicitly named file without scanning any surrounding tree."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Producer(_ManifestModel):
    command: str = Field(min_length=1)
    package_version: str = Field(min_length=1)


class FileRecord(_ManifestModel):
    path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected: dict[str, Any] = Field(default_factory=dict)


class Manifest(_ManifestModel):
    manifest_schema_version: Literal["0.1.0", "1.0.0"]
    fixture_schema_version: Literal["0.1.0", "1.0.0"]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    artifact_kind: Literal["fixture_set", "experiment", "dataset", "model"]
    origin: Literal["synthetic", "official_dataset", "generated"]
    origin_description: str = Field(min_length=1)
    created_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    seed: int = Field(ge=0, le=4_294_967_295)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    software_versions: dict[str, str] = Field(min_length=1)
    schema_versions: dict[str, str] = Field(min_length=1)
    producer: Producer
    files: tuple[FileRecord, ...] = Field(strict=False)

    @model_validator(mode="after")
    def require_stable_unique_paths(self) -> Manifest:
        paths = [record.path for record in self.files]
        if paths != sorted(paths):
            raise ValueError("manifest files must use lexicographic path ordering")
        if len(paths) != len(set(paths)):
            raise ValueError("manifest file paths must be unique")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
            raise ValueError("manifest file paths must stay relative to the manifest")
        return self


def load_manifest(path: str | Path, *, verify_files: bool = False) -> Manifest:
    """Load a JSON manifest and optionally verify its explicitly listed files."""

    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = Manifest.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ManifestError(f"invalid manifest {manifest_path}: {exc}") from exc

    if verify_files:
        for record in manifest.files:
            file_path = manifest_path.parent / record.path
            if not file_path.is_file():
                raise ManifestError(f"manifest file does not exist: {file_path}")
            actual_size = file_path.stat().st_size
            if actual_size != record.bytes:
                raise ManifestError(
                    f"size mismatch for {record.path}: "
                    f"expected {record.bytes}, got {actual_size}"
                )
            actual_hash = sha256_file(file_path)
            if actual_hash != record.sha256:
                raise ManifestError(
                    f"sha256 mismatch for {record.path}: "
                    f"expected {record.sha256}, got {actual_hash}"
                )
    return manifest

"""Strict, path-explicit configuration loading for the M0 foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lidar_shield.manifest import sha256_file


class ConfigurationError(ValueError):
    """A configuration could not be read or did not satisfy its schema."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProjectIdentity(_StrictModel):
    distribution: Literal["lidar-shield"]
    package: Literal["lidar_shield"]


class VersionContract(_StrictModel):
    software: str = Field(min_length=1)
    config_schema: str = Field(min_length=1)
    manifest_schema: str = Field(min_length=1)
    fixture_schema: str = Field(min_length=1)
    contract_schema: str = Field(min_length=1)


class PathContract(_StrictModel):
    fixture_root: Path = Field(strict=False)
    artifact_root: Path = Field(strict=False)
    report_root: Path = Field(strict=False)


class ReproducibilityContract(_StrictModel):
    seed: int = Field(ge=0, le=4_294_967_295)
    hash_algorithm: Literal["sha256"]
    stable_serialization: Literal["json-sort-keys-utf8-v1"]
    stable_ordering: Literal["lexicographic-utf8-v1"]
    deterministic_algorithms: bool


class ProjectConfig(_StrictModel):
    schema_version: Literal["1.0.0"]
    project: ProjectIdentity
    versions: VersionContract
    paths: PathContract
    reproducibility: ReproducibilityContract


def _resolve_paths(config: ProjectConfig, config_path: Path) -> ProjectConfig:
    base = config_path.parent

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else (base / value).resolve()

    return config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "fixture_root": resolve(config.paths.fixture_root),
                    "artifact_root": resolve(config.paths.artifact_root),
                    "report_root": resolve(config.paths.report_root),
                }
            )
        }
    )


def load_config(path: str | Path) -> ProjectConfig:
    """Load YAML from *path*, reject unknown fields, and resolve declared paths."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")

    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read YAML {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {config_path}")

    try:
        config = ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(
            f"invalid configuration {config_path}:\n{exc}"
        ) from exc

    resolved = _resolve_paths(config, config_path)
    if not resolved.paths.fixture_root.is_dir():
        raise ConfigurationError(
            f"configured fixture_root is not a directory: {resolved.paths.fixture_root}"
        )
    return resolved


def config_hash(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact configuration source bytes."""

    return sha256_file(path)

"""First scientific attack identity and provenance contracts (M4)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from lidar_shield.evaluation.splits import SplitRegistry, SplitRole
from lidar_shield.manifest import canonical_json_bytes, sha256_bytes, sha256_file

ATTACK_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
Identifier = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Vector3 = tuple[float, float, float]


class AttackManifestError(ValueError):
    """An M4 attack manifest is malformed or not bound to its clean source."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceHashes(_StrictModel):
    archive_sha256: Sha256
    selected_source_sha256: Sha256
    clean_manifest_sha256: Sha256
    clean_frame_sha256: Sha256
    clean_object_sha256: Sha256
    clean_evidence_sha256: Sha256
    target_raw_sha256: Sha256
    split_registry_sha256: Sha256


class AttackTarget(_StrictModel):
    source_id: Identifier
    sequence_id: Identifier
    session_id: Identifier
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    source_time_ns: int = Field(ge=0)
    sender_id: Identifier
    agent_type: Literal["vehicle", "infrastructure"]
    object_id_or_local_index: str | int | None = None
    interval_sample_times_ns: tuple[int, ...] = Field(default=(), strict=False)
    raw_relative_path: Identifier

    @field_validator("raw_relative_path")
    @classmethod
    def safe_raw_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("target raw path must be safe and relative")
        return value

    @field_validator("interval_sample_times_ns")
    @classmethod
    def ordered_sample_times(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 0 for item in value) or tuple(sorted(set(value))) != value:
            raise ValueError("interval sample times must be sorted and unique")
        return value


class BoxTarget(_StrictModel):
    center_m: Vector3 = Field(strict=False)
    size_m: Vector3 = Field(strict=False)
    yaw_rad: float
    frame: Identifier

    @model_validator(mode="after")
    def finite_positive_box(self) -> BoxTarget:
        values = (*self.center_m, *self.size_m, self.yaw_rad)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("box values must be finite")
        if any(item <= 0 for item in self.size_m):
            raise ValueError("box sizes must be positive")
        return self


class KinematicParameters(_StrictModel):
    attack_kind: Literal["velocity_spike", "velocity_drift"]
    delta_v_mps: Vector3 = Field(strict=False)
    duration_samples: int = Field(gt=0)
    units: Literal["m/s_body_frame"] = "m/s_body_frame"
    position_policy: Literal["unchanged"] = "unchanged"
    orientation_policy: Literal["unchanged"] = "unchanged"

    @field_validator("delta_v_mps")
    @classmethod
    def finite_delta(cls, value: Vector3) -> Vector3:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("velocity delta must be finite")
        return value


class SpatialParameters(_StrictModel):
    attack_kind: Literal["point_addition", "point_removal"]
    box: BoxTarget
    n_add: int | None = Field(default=None, ge=0)
    removal_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_tolerance_m: float = Field(default=1e-6, ge=0.0)
    added_extra_field_policy: Literal["source_median"] = "source_median"
    empty_source_extra_field_policy: Literal["zero"] = "zero"
    preserve_encoding: Literal[True] = True
    preserve_fields_and_dtypes: Literal[True] = True

    @model_validator(mode="after")
    def kind_specific_parameters(self) -> SpatialParameters:
        if self.attack_kind == "point_addition":
            if self.n_add is None or self.removal_fraction is not None:
                raise ValueError("point addition requires only n_add")
        elif self.removal_fraction is None or self.n_add is not None:
            raise ValueError("point removal requires only removal_fraction")
        return self


AttackParameters = KinematicParameters | SpatialParameters


class AttackVersions(_StrictModel):
    attack_schema: Literal["1.0.0"] = ATTACK_SCHEMA_VERSION
    contract_schema: Identifier
    package_version: Identifier
    code_sha256: Sha256
    config_sha256: Sha256


class AttackManifest(_StrictModel):
    """Immutable identity for one attempted M4 attack variant."""

    schema_version: Literal["1.0.0"] = ATTACK_SCHEMA_VERSION
    experiment_id: Identifier
    episode_id: Identifier
    variant_id: Identifier
    origin: Literal["official_dataset", "synthetic"]
    split_role: SplitRole
    target: AttackTarget
    source_hashes: SourceHashes
    family: Literal["kinematic", "spatial_support"]
    severity: Identifier
    parameters: AttackParameters
    seed: int = Field(ge=0)
    attacker_knowledge: Literal["source_only", "box_aware", "white_box_fixture"]
    expected_modality: Literal["kinematic", "spatial"]
    versions: AttackVersions

    @model_validator(mode="after")
    def coherent_family_and_target(self) -> AttackManifest:
        if self.family == "kinematic":
            if not isinstance(self.parameters, KinematicParameters):
                raise ValueError("kinematic family requires kinematic parameters")
            if self.expected_modality != "kinematic":
                raise ValueError("kinematic family requires kinematic modality")
            times = self.target.interval_sample_times_ns
            if len(times) != self.parameters.duration_samples:
                raise ValueError("kinematic target must enumerate exactly K samples")
            if self.target.object_id_or_local_index is not None:
                raise ValueError("kinematic targets cannot carry an object key")
        else:
            if not isinstance(self.parameters, SpatialParameters):
                raise ValueError("spatial family requires spatial parameters")
            if self.expected_modality != "spatial":
                raise ValueError("spatial family requires spatial modality")
            if self.target.object_id_or_local_index is None:
                raise ValueError("spatial targets require a frame-local object key")
            if self.target.interval_sample_times_ns:
                raise ValueError("spatial targets cannot enumerate odometry samples")
        return self

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.experiment_id,
            self.episode_id,
            self.target.source_id,
            self.target.sequence_id,
            self.variant_id,
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes())


def load_attack_manifest(path: str | Path) -> AttackManifest:
    source = Path(path).expanduser().resolve()
    try:
        return AttackManifest.model_validate_json(source.read_bytes(), strict=False)
    except (OSError, ValidationError) as exc:
        raise AttackManifestError(f"invalid attack manifest {source}: {exc}") from exc


def validate_manifest_source(
    manifest: AttackManifest,
    *,
    split_registry: SplitRegistry,
    split_registry_sha256: str,
    clean_dir: str | Path,
    data_root: str | Path,
) -> None:
    """Reject unregistered sources and any stale raw/cache binding."""

    if split_registry_sha256 != manifest.source_hashes.split_registry_sha256:
        raise AttackManifestError("manifest split registry hash is stale")
    assignment = split_registry.assignment_for(manifest.target.sequence_id)
    split_registry.assert_variant_role(manifest.target.sequence_id, manifest.split_role)
    if assignment.source_id != manifest.target.source_id:
        raise AttackManifestError(
            "manifest source ID conflicts with split registration"
        )
    clean = Path(clean_dir).expanduser().resolve()
    raw = Path(data_root).expanduser().resolve() / manifest.target.raw_relative_path
    checks = {
        clean / "source_manifest.json": manifest.source_hashes.clean_manifest_sha256,
        clean / "frame.parquet": manifest.source_hashes.clean_frame_sha256,
        clean / "object.parquet": manifest.source_hashes.clean_object_sha256,
        clean / "evidence.parquet": manifest.source_hashes.clean_evidence_sha256,
        raw: manifest.source_hashes.target_raw_sha256,
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise AttackManifestError(f"source hash mismatch: {path}")

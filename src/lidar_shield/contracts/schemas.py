"""Pydantic contracts for M1 records, capabilities, and manifests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from lidar_shield.contracts.keys import TableLevel, validate_canonical_table
from lidar_shield.contracts.reasons import (
    MATCHED_SOURCE_REASONS,
    UNAVAILABLE_SOURCE_REASONS,
    AgentType,
    Applicability,
    MatchPolicy,
    ReasonCode,
)

CONTRACT_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
Identifier = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ObjectKey = str | int


def _validate_match_metadata(
    *,
    decision_time_ns: int,
    source_time_ns: int | None,
    match_age_ns: int | None,
    match_policy: MatchPolicy,
    lookahead_buffer_ns: int,
) -> None:
    if (source_time_ns is None) != (match_age_ns is None):
        raise ValueError("source time and match age must be present together")
    if source_time_ns is not None:
        if match_age_ns != decision_time_ns - source_time_ns:
            raise ValueError(
                "match_age_ns must equal decision_time_ns - source_time_ns"
            )
    if match_policy is MatchPolicy.CAUSAL_BACKWARD:
        if lookahead_buffer_ns != 0:
            raise ValueError("causal policy cannot declare look-ahead")
        if match_age_ns is not None and match_age_ns < 0:
            raise ValueError("causal matching cannot admit future sources")
    elif match_policy is MatchPolicy.BUFFERED_NEAREST:
        if lookahead_buffer_ns <= 0:
            raise ValueError("buffered matching requires positive look-ahead")
        if match_age_ns is not None and -match_age_ns > lookahead_buffer_ns:
            raise ValueError("matched future source exceeds the declared look-ahead")
    elif source_time_ns is not None:
        raise ValueError("a matched source requires causal or buffered policy")


def _validate_pose_age(
    *,
    decision_time_ns: int,
    pose_source_time_ns: int | None,
    pose_age_ns: int | None,
) -> None:
    if (pose_source_time_ns is None) != (pose_age_ns is None):
        raise ValueError("pose source time and pose age must be present together")
    if pose_source_time_ns is not None:
        if pose_age_ns != decision_time_ns - pose_source_time_ns:
            raise ValueError("pose_age_ns uses the canonical signed-age convention")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Extrinsics(_StrictModel):
    from_frame: Identifier
    to_frame: Identifier
    matrix_4x4: tuple[tuple[float, float, float, float], ...] | None = Field(
        strict=False
    )
    status: Literal["verified", "unresolved"]
    note: Identifier

    @model_validator(mode="after")
    def validate_matrix_status(self) -> Extrinsics:
        if self.status == "verified" and self.matrix_4x4 is None:
            raise ValueError("verified extrinsics require matrix_4x4")
        if self.status == "unresolved" and self.matrix_4x4 is not None:
            raise ValueError("unresolved extrinsics must not fabricate a matrix")
        if self.matrix_4x4 is not None:
            if len(self.matrix_4x4) != 4 or any(
                len(row) != 4 for row in self.matrix_4x4
            ):
                raise ValueError("matrix_4x4 must be exactly 4 by 4")
            if not all(
                math.isfinite(value) for row in self.matrix_4x4 for value in row
            ):
                raise ValueError("matrix_4x4 values must be finite")
        return self


class NominalCoverage(_StrictModel):
    horizontal_fov_deg: float | None
    vertical_fov_deg: float | None
    range_m: float | None
    status: Literal["verified", "unresolved"]

    @model_validator(mode="after")
    def validate_coverage(self) -> NominalCoverage:
        values = (self.horizontal_fov_deg, self.vertical_fov_deg, self.range_m)
        if self.status == "verified" and any(value is None for value in values):
            raise ValueError("verified nominal coverage requires all values")
        if self.status == "unresolved" and any(value is not None for value in values):
            raise ValueError("unresolved coverage values must remain null")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in values
        ):
            raise ValueError("nominal coverage values must be finite and positive")
        return self


class AgentCapability(_StrictModel):
    agent_id: Identifier
    agent_type: AgentType
    sensor_type: Identifier
    odometry: Applicability
    extrinsics: Extrinsics
    nominal_coverage: NominalCoverage
    expected_point_fields: tuple[Identifier, ...] = Field(strict=False)
    expected_odometry_fields: tuple[Identifier, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_odometry_fields(self) -> AgentCapability:
        if self.odometry is Applicability.NOT_APPLICABLE:
            if self.expected_odometry_fields:
                raise ValueError("non-applicable odometry must not declare fields")
        elif not self.expected_odometry_fields:
            raise ValueError("applicable odometry requires expected fields")
        return self


class AgentRegistry(_StrictModel):
    schema_version: Literal["1.0.0"]
    registry_id: Identifier
    dataset: Literal["mixed_signals"]
    agents: tuple[AgentCapability, ...] = Field(strict=False)

    @model_validator(mode="after")
    def unique_agents(self) -> AgentRegistry:
        identifiers = [agent.agent_id for agent in self.agents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("agent registry IDs must be unique")
        if identifiers != sorted(identifiers):
            raise ValueError("agent registry IDs must use stable ordering")
        return self


class FrameIdentity(_StrictModel):
    schema_version: Literal["1.0.0"] = CONTRACT_SCHEMA_VERSION
    experiment_id: Identifier
    variant_id: Identifier
    sequence_id: Identifier
    session_id: Identifier
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    source_time_ns: int | None = Field(default=None, ge=0)
    agent_id: Identifier
    agent_type: AgentType
    object_id_or_local_index: None = None


class FrameRecord(FrameIdentity):
    """One expected frame/agent event, including unavailable observations."""

    member: bool
    message_present: bool
    source_matched: bool
    evidence_available: bool
    reason_code: ReasonCode
    match_policy: MatchPolicy
    match_age_ns: int | None = None
    lookahead_buffer_ns: int = Field(ge=0)
    pose_source_time_ns: int | None = Field(default=None, ge=0)
    pose_age_ns: int | None = None
    oracle_gt: bool

    @model_validator(mode="after")
    def validate_availability(self) -> FrameRecord:
        _validate_match_metadata(
            decision_time_ns=self.decision_time_ns,
            source_time_ns=self.source_time_ns,
            match_age_ns=self.match_age_ns,
            match_policy=self.match_policy,
            lookahead_buffer_ns=self.lookahead_buffer_ns,
        )
        has_source_fields = self.source_time_ns is not None
        if self.source_matched != has_source_fields:
            raise ValueError(
                "matched rows require source_time_ns and match_age_ns together"
            )
        if self.source_matched:
            if self.reason_code not in MATCHED_SOURCE_REASONS:
                raise ValueError("matched source has an incompatible reason_code")
        elif self.reason_code not in UNAVAILABLE_SOURCE_REASONS:
            raise ValueError("unmatched source has an incompatible reason_code")
        if self.evidence_available and self.reason_code is not ReasonCode.MATCHED:
            raise ValueError("only fully matched rows may expose evidence")
        if self.evidence_available and not self.source_matched:
            raise ValueError("evidence requires a matched source")
        if self.reason_code is ReasonCode.NOT_MEMBER and self.member:
            raise ValueError("NOT_MEMBER rows must set member=false")
        if self.message_present and self.reason_code is ReasonCode.MISSING_MESSAGE:
            raise ValueError("missing-message rows cannot report a message")
        _validate_pose_age(
            decision_time_ns=self.decision_time_ns,
            pose_source_time_ns=self.pose_source_time_ns,
            pose_age_ns=self.pose_age_ns,
        )
        return self


class ObjectRecord(_StrictModel):
    """One present object; empty/unavailable scenes stay in the frame table."""

    schema_version: Literal["1.0.0"] = CONTRACT_SCHEMA_VERSION
    experiment_id: Identifier
    variant_id: Identifier
    sequence_id: Identifier
    session_id: Identifier
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    source_time_ns: int = Field(ge=0)
    agent_id: Identifier
    agent_type: AgentType
    object_id_or_local_index: ObjectKey
    match_policy: MatchPolicy
    match_age_ns: int
    lookahead_buffer_ns: int = Field(ge=0)
    pose_source_time_ns: int | None = Field(default=None, ge=0)
    pose_age_ns: int | None = None
    object_key_kind: Literal["stable_id", "frame_local_index"]
    source_kind: Literal["oracle_gt", "received_claim"]
    oracle_gt: bool
    class_id: int | None = None

    @model_validator(mode="after")
    def validate_source_kind(self) -> ObjectRecord:
        _validate_match_metadata(
            decision_time_ns=self.decision_time_ns,
            source_time_ns=self.source_time_ns,
            match_age_ns=self.match_age_ns,
            match_policy=self.match_policy,
            lookahead_buffer_ns=self.lookahead_buffer_ns,
        )
        _validate_pose_age(
            decision_time_ns=self.decision_time_ns,
            pose_source_time_ns=self.pose_source_time_ns,
            pose_age_ns=self.pose_age_ns,
        )
        if self.source_kind == "oracle_gt" and not self.oracle_gt:
            raise ValueError("oracle_gt source_kind requires oracle_gt=true")
        if self.source_kind == "received_claim" and self.oracle_gt:
            raise ValueError("received claims cannot be marked oracle GT")
        return self


class EvidenceEnvelope(_StrictModel):
    """A typed evidence value whose unavailability is never imputed."""

    schema_version: Literal["1.0.0"] = CONTRACT_SCHEMA_VERSION
    experiment_id: Identifier
    variant_id: Identifier
    sequence_id: Identifier
    session_id: Identifier
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    source_time_ns: int | None = Field(default=None, ge=0)
    agent_id: Identifier
    agent_type: AgentType
    object_id_or_local_index: ObjectKey | None = None
    match_policy: MatchPolicy
    match_age_ns: int | None = None
    lookahead_buffer_ns: int = Field(ge=0)
    pose_source_time_ns: int | None = Field(default=None, ge=0)
    pose_age_ns: int | None = None
    evidence_level: Literal["frame", "object"]
    evidence_name: Identifier
    value: float | int | bool | None
    applicable: bool
    available: bool
    reason_code: ReasonCode
    oracle_gt: bool

    @field_validator("value")
    @classmethod
    def finite_value(
        cls, value: float | int | bool | None
    ) -> float | int | bool | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("evidence values must be finite")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> EvidenceEnvelope:
        _validate_match_metadata(
            decision_time_ns=self.decision_time_ns,
            source_time_ns=self.source_time_ns,
            match_age_ns=self.match_age_ns,
            match_policy=self.match_policy,
            lookahead_buffer_ns=self.lookahead_buffer_ns,
        )
        _validate_pose_age(
            decision_time_ns=self.decision_time_ns,
            pose_source_time_ns=self.pose_source_time_ns,
            pose_age_ns=self.pose_age_ns,
        )
        if self.evidence_level == "frame" and self.object_id_or_local_index is not None:
            raise ValueError("frame evidence has no object key")
        if self.evidence_level == "object" and self.object_id_or_local_index is None:
            raise ValueError("object evidence requires an object key")
        if self.available != (self.value is not None):
            raise ValueError(
                "available evidence requires a value and unavailable evidence null"
            )
        if self.available and self.source_time_ns is None:
            raise ValueError("available evidence requires a matched source")
        if self.available and self.reason_code is not ReasonCode.MATCHED:
            raise ValueError("available evidence requires MATCHED reason")
        if not self.applicable and self.reason_code is not ReasonCode.NOT_APPLICABLE:
            raise ValueError("inapplicable evidence requires NOT_APPLICABLE")
        return self


class SourceFile(_StrictModel):
    path: Identifier
    bytes: int = Field(ge=0)
    sha256: Sha256


class SourceManifest(_StrictModel):
    schema_version: Literal["1.0.0"]
    source_id: Identifier
    origin: Literal["synthetic", "official_dataset"]
    sequence_id: Identifier
    session_id: Identifier
    split_role: Identifier
    split_registry_sha256: Sha256
    agent_registry_sha256: Sha256
    config_sha256: Sha256
    files: tuple[SourceFile, ...] = Field(strict=False)
    source_sha256: Sha256

    @model_validator(mode="after")
    def stable_files(self) -> SourceManifest:
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("source files must have stable unique path ordering")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
            raise ValueError("source file paths must be safe relative paths")
        return self


class ExperimentManifest(_StrictModel):
    schema_version: Literal["1.0.0"]
    experiment_id: Identifier
    contract_schema_version: Literal["1.0.0"]
    config_sha256: Sha256
    agent_registry_sha256: Sha256
    split_registry_sha256: Sha256
    schema_registry_sha256: Sha256
    source_manifest_sha256: tuple[Sha256, ...] = Field(strict=False)
    artifact_layout: Literal["experiment-sequence-variant-v1"]
    producer_command: Identifier

    @model_validator(mode="after")
    def stable_source_manifests(self) -> ExperimentManifest:
        hashes = list(self.source_manifest_sha256)
        if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
            raise ValueError("source manifest hashes must use stable unique ordering")
        return self


class ObjectSource(Protocol):
    """Common Stage 1/Stage 2 object-source boundary."""

    oracle_gt: bool

    def objects_for_frame(self, frame: FrameRecord) -> tuple[ObjectRecord, ...]:
        """Return zero or more object rows for a matched frame."""


JsonObject = dict[str, Any]


def _records_to_table(
    records: tuple[FrameRecord | ObjectRecord | EvidenceEnvelope, ...],
    *,
    model: type[FrameRecord] | type[ObjectRecord] | type[EvidenceEnvelope],
    level: TableLevel,
) -> pd.DataFrame:
    columns = list(model.model_fields)
    table = pd.DataFrame(
        [record.model_dump(mode="json") for record in records], columns=columns
    )
    for column in (
        "experiment_id",
        "variant_id",
        "sequence_id",
        "session_id",
        "agent_id",
        "agent_type",
    ):
        table[column] = table[column].astype("string")
    table["sync_frame_id"] = table["sync_frame_id"].astype("int64")
    table["decision_time_ns"] = table["decision_time_ns"].astype("int64")
    table["source_time_ns"] = table["source_time_ns"].astype("Int64")
    validate_canonical_table(table, level=level)
    return table


def frame_records_to_table(records: tuple[FrameRecord, ...]) -> pd.DataFrame:
    """Create a typed frame table without floating nullable timestamps."""

    return _records_to_table(records, model=FrameRecord, level=TableLevel.FRAME)


def object_records_to_table(records: tuple[ObjectRecord, ...]) -> pd.DataFrame:
    """Create a typed, distinct object table."""

    return _records_to_table(records, model=ObjectRecord, level=TableLevel.OBJECT)


def evidence_records_to_table(records: tuple[EvidenceEnvelope, ...]) -> pd.DataFrame:
    """Create a typed evidence table with frame/object granularity preserved."""

    return _records_to_table(records, model=EvidenceEnvelope, level=TableLevel.EVIDENCE)

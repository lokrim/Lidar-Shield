from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from lidar_shield.contracts.keys import (
    ContractError,
    JoinCardinality,
    TableLevel,
    merge_checked,
    validate_canonical_table,
)
from lidar_shield.contracts.reasons import AgentType, MatchPolicy, ReasonCode
from lidar_shield.contracts.schemas import (
    AgentRegistry,
    EvidenceEnvelope,
    Extrinsics,
    FrameRecord,
    NominalCoverage,
    ObjectRecord,
    SourceFile,
    SourceManifest,
    evidence_records_to_table,
    frame_records_to_table,
    object_records_to_table,
)
from lidar_shield.data.adapters.oracle_objects import InMemoryOracleObjectSource
from lidar_shield.data.index import load_agent_registry


def frame(**changes: object) -> FrameRecord:
    values: dict[str, object] = {
        "experiment_id": "exp",
        "variant_id": "clean",
        "sequence_id": "sequence-a",
        "session_id": "session-a",
        "sync_frame_id": 7,
        "decision_time_ns": 1_700_000_000_000_000_003,
        "source_time_ns": 1_700_000_000_000_000_001,
        "agent_id": "003",
        "agent_type": AgentType.VEHICLE,
        "member": True,
        "message_present": True,
        "source_matched": True,
        "evidence_available": True,
        "reason_code": ReasonCode.MATCHED,
        "match_policy": MatchPolicy.CAUSAL_BACKWARD,
        "match_age_ns": 2,
        "lookahead_buffer_ns": 0,
        "pose_source_time_ns": 1_700_000_000_000_000_000,
        "pose_age_ns": 3,
        "oracle_gt": True,
    }
    values.update(changes)
    return FrameRecord.model_validate(values)


def object_row(**changes: object) -> ObjectRecord:
    values: dict[str, object] = {
        "experiment_id": "exp",
        "variant_id": "clean",
        "sequence_id": "sequence-a",
        "session_id": "session-a",
        "sync_frame_id": 7,
        "decision_time_ns": 1_700_000_000_000_000_003,
        "source_time_ns": 1_700_000_000_000_000_001,
        "agent_id": "003",
        "agent_type": AgentType.VEHICLE,
        "object_id_or_local_index": 0,
        "match_policy": MatchPolicy.CAUSAL_BACKWARD,
        "match_age_ns": 2,
        "lookahead_buffer_ns": 0,
        "pose_source_time_ns": 1_700_000_000_000_000_000,
        "pose_age_ns": 3,
        "object_key_kind": "frame_local_index",
        "source_kind": "oracle_gt",
        "oracle_gt": True,
        "class_id": 1,
    }
    values.update(changes)
    return ObjectRecord.model_validate(values)


def test_frame_table_preserves_integer_nanoseconds_and_leading_zero_id() -> None:
    table = frame_records_to_table((frame(),))
    assert table.loc[0, "agent_id"] == "003"
    assert table.loc[0, "decision_time_ns"] == 1_700_000_000_000_000_003
    assert str(table["source_time_ns"].dtype) == "Int64"
    assert table.loc[0, "object_id_or_local_index"] is None


def test_unavailable_frame_has_nullable_source_not_a_fabricated_observation() -> None:
    missing = frame(
        source_time_ns=None,
        source_matched=False,
        evidence_available=False,
        reason_code=ReasonCode.MISSING_MESSAGE,
        match_age_ns=None,
        message_present=False,
        pose_source_time_ns=None,
        pose_age_ns=None,
    )
    table = frame_records_to_table((missing,))
    assert pd.isna(table.loc[0, "source_time_ns"])
    with pytest.raises(ValidationError):
        frame(source_time_ns=None)


def test_object_and_evidence_granularity_contracts() -> None:
    obj = object_row()
    object_table = object_records_to_table((obj,))
    assert object_table.loc[0, "object_id_or_local_index"] == 0
    source = InMemoryOracleObjectSource((obj,))
    assert source.objects_for_frame(frame()) == (obj,)
    assert (
        source.objects_for_frame(
            frame(
                source_time_ns=None,
                source_matched=False,
                evidence_available=False,
                reason_code=ReasonCode.UNMATCHED,
                match_age_ns=None,
                message_present=True,
                pose_source_time_ns=None,
                pose_age_ns=None,
            )
        )
        == ()
    )

    evidence = EvidenceEnvelope(
        experiment_id="exp",
        variant_id="clean",
        sequence_id="sequence-a",
        session_id="session-a",
        sync_frame_id=7,
        decision_time_ns=1_700_000_000_000_000_003,
        source_time_ns=1_700_000_000_000_000_001,
        agent_id="003",
        agent_type=AgentType.VEHICLE,
        object_id_or_local_index=0,
        match_policy=MatchPolicy.CAUSAL_BACKWARD,
        match_age_ns=2,
        lookahead_buffer_ns=0,
        pose_source_time_ns=1_700_000_000_000_000_000,
        pose_age_ns=3,
        evidence_level="object",
        evidence_name="point_count",
        value=3,
        applicable=True,
        available=True,
        reason_code=ReasonCode.MATCHED,
        oracle_gt=True,
    )
    assert evidence_records_to_table((evidence,)).shape == (1, 23)
    invalid_evidence = evidence.model_dump()
    invalid_evidence["value"] = None
    with pytest.raises(ValidationError):
        EvidenceEnvelope.model_validate(invalid_evidence)


def test_record_validation_rejects_inconsistent_source_and_buffer_fields() -> None:
    with pytest.raises(ValidationError, match="future sources"):
        frame(
            source_time_ns=1_700_000_000_000_000_004,
            match_age_ns=-1,
        )
    with pytest.raises(ValidationError, match="positive look-ahead"):
        frame(
            match_policy=MatchPolicy.BUFFERED_NEAREST,
            lookahead_buffer_ns=0,
        )
    with pytest.raises(ValidationError, match="oracle_gt"):
        object_row(oracle_gt=False)
    with pytest.raises(ValueError, match="oracle_gt"):
        InMemoryOracleObjectSource(
            (object_row(source_kind="received_claim", oracle_gt=False),)
        )


def test_duplicate_and_non_null_key_violations_are_rejected() -> None:
    table = frame_records_to_table((frame(),))
    duplicated = pd.concat([table, table], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate frame keys"):
        validate_canonical_table(duplicated, level=TableLevel.FRAME)

    invalid = table.copy()
    invalid.loc[0, "decision_time_ns"] = None
    with pytest.raises(ContractError, match="decision_time_ns"):
        validate_canonical_table(invalid, level=TableLevel.FRAME)

    float_time = table.copy()
    float_time["decision_time_ns"] = float_time["decision_time_ns"].astype(float)
    with pytest.raises(ContractError, match="float nanoseconds"):
        validate_canonical_table(float_time, level=TableLevel.FRAME)


def test_merge_requires_cardinality_and_deliberate_overlap_names() -> None:
    left = pd.DataFrame({"id": ["003", "004"], "identity": [1, 2]})
    right = pd.DataFrame({"id": ["003", "004"], "identity": [10, 20]})
    with pytest.raises(ContractError, match="deliberately renamed"):
        merge_checked(
            left,
            right,
            on=("id",),
            cardinality=JoinCardinality.ONE_TO_ONE,
        )
    merged = merge_checked(
        left,
        right,
        on=("id",),
        cardinality=JoinCardinality.ONE_TO_ONE,
        right_rename={"identity": "source_identity"},
    )
    assert list(merged.columns) == ["id", "identity", "source_identity"]

    duplicate_right = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="one_to_one"):
        merge_checked(
            left,
            duplicate_right,
            on=("id",),
            cardinality=JoinCardinality.ONE_TO_ONE,
            right_rename={"identity": "source_identity"},
        )


def test_detector_claims_module_is_interface_only() -> None:
    from lidar_shield.data.adapters.detector_claims import DetectorClaimSource

    assert DetectorClaimSource.__name__ == "DetectorClaimSource"
    path = (
        Path(__file__).parents[2]
        / "src"
        / "lidar_shield"
        / "data"
        / "adapters"
        / "detector_claims.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "class DetectorClaimSource(Protocol)" in text
    assert "oracle_gt: bool" in text


def test_capability_schema_rejects_fabricated_or_malformed_configuration() -> None:
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValidationError, match="require matrix"):
        Extrinsics(
            from_frame="a", to_frame="b", matrix_4x4=None, status="verified", note="x"
        )
    with pytest.raises(ValidationError, match="must not fabricate"):
        Extrinsics(
            from_frame="a",
            to_frame="b",
            matrix_4x4=identity,
            status="unresolved",
            note="x",
        )
    with pytest.raises(ValidationError, match="4 by 4"):
        Extrinsics(
            from_frame="a",
            to_frame="b",
            matrix_4x4=identity[:3],
            status="verified",
            note="x",
        )
    malformed = [list(row) for row in identity]
    malformed[0][0] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        Extrinsics.model_validate(
            {
                "from_frame": "a",
                "to_frame": "b",
                "matrix_4x4": malformed,
                "status": "verified",
                "note": "x",
            },
            strict=False,
        )
    with pytest.raises(ValidationError, match="requires all"):
        NominalCoverage(
            horizontal_fov_deg=None,
            vertical_fov_deg=30.0,
            range_m=100.0,
            status="verified",
        )
    with pytest.raises(ValidationError, match="remain null"):
        NominalCoverage(
            horizontal_fov_deg=360.0,
            vertical_fov_deg=None,
            range_m=None,
            status="unresolved",
        )
    with pytest.raises(ValidationError, match="positive"):
        NominalCoverage(
            horizontal_fov_deg=360.0,
            vertical_fov_deg=30.0,
            range_m=0.0,
            status="verified",
        )


def test_frame_and_evidence_consistency_guards() -> None:
    invalid_updates = (
        {"reason_code": ReasonCode.STALE},
        {"evidence_available": True, "reason_code": ReasonCode.EMPTY_SCENE},
        {"reason_code": ReasonCode.NOT_MEMBER},
        {
            "source_time_ns": None,
            "source_matched": False,
            "match_age_ns": None,
            "reason_code": ReasonCode.MISSING_MESSAGE,
            "message_present": True,
            "evidence_available": False,
        },
        {"pose_age_ns": None},
        {"pose_age_ns": 4},
    )
    base = frame().model_dump()
    for update in invalid_updates:
        with pytest.raises(ValidationError):
            FrameRecord.model_validate({**base, **update})

    evidence_base = {
        "experiment_id": "exp",
        "variant_id": "clean",
        "sequence_id": "s",
        "session_id": "session",
        "sync_frame_id": 0,
        "decision_time_ns": 1,
        "source_time_ns": None,
        "agent_id": "003",
        "agent_type": AgentType.VEHICLE,
        "object_id_or_local_index": None,
        "match_policy": MatchPolicy.NONE,
        "match_age_ns": None,
        "lookahead_buffer_ns": 0,
        "pose_source_time_ns": None,
        "pose_age_ns": None,
        "evidence_level": "frame",
        "evidence_name": "availability",
        "value": None,
        "applicable": False,
        "available": False,
        "reason_code": ReasonCode.NOT_APPLICABLE,
        "oracle_gt": True,
    }
    assert EvidenceEnvelope.model_validate(evidence_base).value is None
    invalid_evidence_updates: tuple[dict[str, object], ...] = (
        {"object_id_or_local_index": 0},
        {"evidence_level": "object"},
        {"value": float("nan"), "available": True},
        {"applicable": False, "reason_code": ReasonCode.UNMATCHED},
    )
    for update in invalid_evidence_updates:
        with pytest.raises(ValidationError):
            EvidenceEnvelope.model_validate({**evidence_base, **update})


def test_registry_and_source_manifest_require_unique_stable_entries() -> None:
    registry = load_agent_registry(
        Path(__file__).parents[2] / "configs" / "agents" / "mixed_signals.yaml"
    )
    raw_registry = registry.model_dump()
    raw_registry["agents"] = registry.agents + (registry.agents[0],)
    with pytest.raises(ValidationError, match="unique"):
        AgentRegistry.model_validate(raw_registry)
    digest = "0" * 64
    files = (
        SourceFile(path="b", bytes=0, sha256=digest),
        SourceFile(path="a", bytes=0, sha256=digest),
    )
    with pytest.raises(ValidationError, match="stable unique"):
        SourceManifest(
            schema_version="1.0.0",
            source_id="source",
            origin="synthetic",
            sequence_id="s",
            session_id="session",
            split_role="synthetic",
            split_registry_sha256=digest,
            agent_registry_sha256=digest,
            config_sha256=digest,
            files=files,
            source_sha256=digest,
        )


def test_table_and_join_defensive_guards() -> None:
    table = frame_records_to_table((frame(),))
    with pytest.raises(ContractError, match="missing canonical"):
        validate_canonical_table(
            table.drop(columns=["agent_id"]), level=TableLevel.FRAME
        )
    empty = table.iloc[:0]
    validate_canonical_table(empty, level=TableLevel.FRAME)
    invalid_string = table.copy()
    invalid_string.loc[0, "agent_id"] = ""
    with pytest.raises(ContractError, match="non-empty strings"):
        validate_canonical_table(invalid_string, level=TableLevel.FRAME)
    invalid_frame = table.copy()
    invalid_frame.loc[0, "object_id_or_local_index"] = 0
    with pytest.raises(ContractError, match="frame rows"):
        validate_canonical_table(invalid_frame, level=TableLevel.FRAME)

    object_table = object_records_to_table((object_row(),))
    missing_object = object_table.copy()
    missing_object.loc[0, "object_id_or_local_index"] = None
    with pytest.raises(ContractError, match="object rows require"):
        validate_canonical_table(missing_object, level=TableLevel.OBJECT)
    missing_source = object_table.copy()
    missing_source.loc[0, "source_time_ns"] = None
    with pytest.raises(ContractError, match="require source_time"):
        validate_canonical_table(missing_source, level=TableLevel.OBJECT)

    left = pd.DataFrame({"id": [1], "a": [1], "b": [2]})
    right = pd.DataFrame({"id": [1], "a": [3], "b": [4]})
    with pytest.raises(ContractError, match="must be declared"):
        merge_checked(left, right, on=(), cardinality=JoinCardinality.ONE_TO_ONE)
    with pytest.raises(ContractError, match="absent"):
        merge_checked(
            left, right, on=("missing",), cardinality=JoinCardinality.ONE_TO_ONE
        )
    with pytest.raises(ContractError, match="targets must be unique"):
        merge_checked(
            left,
            right,
            on=("id",),
            cardinality=JoinCardinality.ONE_TO_ONE,
            right_rename={"a": "x", "b": "x"},
        )
    with pytest.raises(ContractError, match="collide"):
        merge_checked(
            left,
            right,
            on=("id",),
            cardinality=JoinCardinality.ONE_TO_ONE,
            right_rename={"a": "b", "b": "source_b"},
        )

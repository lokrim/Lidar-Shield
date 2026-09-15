from pathlib import Path

import pytest

from lidar_shield.contracts.reasons import Applicability, MatchPolicy, ReasonCode
from lidar_shield.data.index import (
    IndexError,
    MembershipDeclaration,
    load_agent_registry,
    validate_membership,
)
from lidar_shield.data.sync import SourceMessage, SyncConfig, Synchronizer

ROOT = Path(__file__).parents[2]
AGENTS = ROOT / "configs" / "agents" / "mixed_signals.yaml"


def member(
    agent: str,
    time_ns: int = 1_000,
    *,
    sequence: str = "sequence-a",
    session: str = "session-a",
    frame: int = 0,
    active: bool = True,
) -> MembershipDeclaration:
    return MembershipDeclaration(
        sequence_id=sequence,
        session_id=session,
        sync_frame_id=frame,
        decision_time_ns=time_ns,
        agent_id=agent,
        member=active,
        message_expected=active,
    )


def test_registry_has_capabilities_and_unknown_infrastructure_odometry() -> None:
    registry = load_agent_registry(AGENTS)
    assert [agent.agent_id for agent in registry.agents] == [
        "003",
        "004",
        "dome",
        "laser",
        "top",
    ]
    dome = next(agent for agent in registry.agents if agent.agent_id == "dome")
    assert dome.odometry is Applicability.NOT_APPLICABLE
    assert dome.expected_odometry_fields == ()
    assert dome.nominal_coverage.range_m is None
    assert dome.extrinsics.matrix_4x4 is None


def test_membership_is_explicit_sorted_and_isolated_across_sequences() -> None:
    registry = load_agent_registry(AGENTS)
    declarations = (
        member("laser", sequence="sequence-b", session="session-b"),
        member("003", sequence="sequence-a", session="session-a"),
        member("top", sequence="sequence-b", session="session-b", active=False),
    )
    ordered = validate_membership(declarations, registry)
    assert [(row.sequence_id, row.agent_id) for row in ordered] == [
        ("sequence-a", "003"),
        ("sequence-b", "laser"),
        ("sequence-b", "top"),
    ]
    with pytest.raises(IndexError, match="duplicate membership"):
        validate_membership((declarations[0], declarations[0]), registry)
    with pytest.raises(IndexError, match="unknown agent"):
        validate_membership((member("invented"),), registry)
    with pytest.raises(ValueError, match="non-member"):
        MembershipDeclaration(
            sequence_id="s",
            session_id="session",
            sync_frame_id=0,
            decision_time_ns=1,
            agent_id="003",
            member=False,
            message_expected=True,
        )


def test_invalid_agent_registry_is_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text("agents: [", encoding="utf-8")
    with pytest.raises(IndexError, match="invalid agent registry"):
        load_agent_registry(path)


def test_primary_causal_matching_and_signed_age() -> None:
    registry = load_agent_registry(AGENTS)
    synchronizer = Synchronizer(
        registry, SyncConfig(MatchPolicy.CAUSAL_BACKWARD, max_age_ns=100)
    )
    rows = synchronizer.synchronize(
        (member("003"),),
        (
            SourceMessage("sequence-a", "session-a", "003", 900, 900),
            SourceMessage(
                "sequence-a",
                "session-a",
                "003",
                950,
                960,
                pose_source_time_ns=940,
            ),
            SourceMessage("sequence-a", "session-a", "003", 1_010, 1_010),
        ),
        experiment_id="exp",
        variant_id="clean",
        oracle_gt=True,
    )
    assert len(rows) == 1
    assert rows[0].source_time_ns == 950
    assert rows[0].match_age_ns == 50
    assert rows[0].pose_age_ns == 60


def test_synchronization_state_cannot_cross_sequence_or_session() -> None:
    registry = load_agent_registry(AGENTS)
    synchronizer = Synchronizer(
        registry, SyncConfig(MatchPolicy.CAUSAL_BACKWARD, max_age_ns=100)
    )
    rows = synchronizer.synchronize(
        (
            member("003", sequence="sequence-a", session="session-a"),
            member("003", sequence="sequence-b", session="session-b"),
            member(
                "004",
                sequence="sequence-b",
                session="session-b",
                active=False,
            ),
        ),
        (
            SourceMessage("sequence-a", "session-a", "003", 910, 920),
            SourceMessage("sequence-b", "session-b", "003", 990, 995),
        ),
        experiment_id="exp",
        variant_id="clean",
        oracle_gt=True,
    )
    assert [(row.sequence_id, row.source_time_ns) for row in rows[:2]] == [
        ("sequence-a", 910),
        ("sequence-b", 990),
    ]
    assert rows[2].reason_code is ReasonCode.NOT_MEMBER


def test_buffered_nearest_records_negative_age_and_lookahead() -> None:
    registry = load_agent_registry(AGENTS)
    synchronizer = Synchronizer(
        registry,
        SyncConfig(
            MatchPolicy.BUFFERED_NEAREST,
            max_age_ns=100,
            lookahead_buffer_ns=50,
        ),
    )
    row = synchronizer.synchronize(
        (member("003"),),
        (
            SourceMessage("sequence-a", "session-a", "003", 970, 970),
            SourceMessage("sequence-a", "session-a", "003", 1_010, 1_010),
        ),
        experiment_id="exp",
        variant_id="clean",
        oracle_gt=True,
    )[0]
    assert row.source_time_ns == 1_010
    assert row.match_age_ns == -10
    assert row.lookahead_buffer_ns == 50


def test_all_unavailable_and_special_events_remain_rows() -> None:
    registry = load_agent_registry(AGENTS)
    sync = Synchronizer(
        registry, SyncConfig(MatchPolicy.CAUSAL_BACKWARD, max_age_ns=100)
    )
    memberships = (
        member("003"),
        member("004"),
        member("dome"),
        member("laser"),
        member("top"),
        member("003", time_ns=2_000, frame=1),
        member("top", time_ns=2_000, frame=1, active=False),
    )
    messages = (
        SourceMessage("sequence-a", "session-a", "003", 950, 960),
        SourceMessage("sequence-a", "session-a", "004", 990, 1_100),
        SourceMessage("sequence-a", "session-a", "laser", 980, 990, pose_valid=False),
        SourceMessage("sequence-a", "session-a", "top", 1_000, 1_000, empty_scene=True),
    )
    rows = sync.synchronize(
        memberships,
        messages,
        experiment_id="exp",
        variant_id="clean",
        oracle_gt=True,
    )
    assert len(rows) == len(memberships)
    reasons = {row.reason_code for row in rows}
    assert {
        ReasonCode.MATCHED,
        ReasonCode.LATE,
        ReasonCode.MISSING_MESSAGE,
        ReasonCode.INVALID_POSE,
        ReasonCode.EMPTY_SCENE,
        ReasonCode.STALE,
        ReasonCode.NOT_MEMBER,
    } <= reasons
    assert all(row.source_time_ns is None for row in rows if not row.source_matched)


def test_sync_configuration_and_duplicate_sources_are_rejected() -> None:
    with pytest.raises(ValueError, match="look-ahead"):
        SyncConfig(MatchPolicy.CAUSAL_BACKWARD, 1, 1)
    with pytest.raises(ValueError, match="positive look-ahead"):
        SyncConfig(MatchPolicy.BUFFERED_NEAREST, 1)
    registry = load_agent_registry(AGENTS)
    sync = Synchronizer(
        registry, SyncConfig(MatchPolicy.CAUSAL_BACKWARD, max_age_ns=100)
    )
    message = SourceMessage("sequence-a", "session-a", "003", 950, 960)
    with pytest.raises(ValueError, match="duplicate source"):
        sync.synchronize(
            (member("003"),),
            (message, message),
            experiment_id="exp",
            variant_id="clean",
            oracle_gt=True,
        )


def test_sync_input_type_and_identity_guards() -> None:
    with pytest.raises(TypeError, match="max_age_ns"):
        SyncConfig(MatchPolicy.CAUSAL_BACKWARD, True)
    with pytest.raises(ValueError, match="nonnegative"):
        SyncConfig(MatchPolicy.CAUSAL_BACKWARD, -1)
    with pytest.raises(TypeError, match="lookahead_buffer_ns"):
        SyncConfig(MatchPolicy.CAUSAL_BACKWARD, 1, False)
    with pytest.raises(ValueError, match="requires causal"):
        SyncConfig(MatchPolicy.NONE, 1)
    with pytest.raises(ValueError, match="sequence_id"):
        SourceMessage("", "session", "003", 1, 1)
    with pytest.raises(TypeError, match="source_time_ns"):
        SourceMessage("s", "session", "003", True, 1)
    with pytest.raises(ValueError, match="arrival_time_ns"):
        SourceMessage("s", "session", "003", 1, -1)
    with pytest.raises(ValueError, match="cannot precede"):
        SourceMessage("s", "session", "003", 2, 1)
    with pytest.raises(TypeError, match="pose_source_time_ns"):
        SourceMessage("s", "session", "003", 1, 1, pose_source_time_ns=True)
    with pytest.raises(ValueError, match="pose_source_time_ns"):
        SourceMessage("s", "session", "003", 1, 1, pose_source_time_ns=-1)

    registry = load_agent_registry(AGENTS)
    sync = Synchronizer(
        registry, SyncConfig(MatchPolicy.CAUSAL_BACKWARD, max_age_ns=100)
    )
    with pytest.raises(ValueError, match="unknown agent"):
        sync.synchronize(
            (member("003"),),
            (SourceMessage("sequence-a", "session-a", "unknown", 1, 1),),
            experiment_id="exp",
            variant_id="clean",
            oracle_gt=True,
        )


def test_buffered_past_tie_and_out_of_range_classification() -> None:
    registry = load_agent_registry(AGENTS)
    sync = Synchronizer(
        registry,
        SyncConfig(
            MatchPolicy.BUFFERED_NEAREST,
            max_age_ns=10,
            lookahead_buffer_ns=100,
        ),
    )
    rows = sync.synchronize(
        (
            member("003"),
            member("004"),
        ),
        (
            SourceMessage("sequence-a", "session-a", "003", 990, 990),
            SourceMessage("sequence-a", "session-a", "003", 1_010, 1_010),
            SourceMessage("sequence-a", "session-a", "004", 1_050, 1_050),
        ),
        experiment_id="exp",
        variant_id="clean",
        oracle_gt=True,
    )
    assert rows[0].source_time_ns == 990
    assert rows[1].reason_code is ReasonCode.OUT_OF_RANGE

"""D0: deterministic five-agent synchronization timeline."""

from __future__ import annotations

from collections import Counter
from typing import Any

from lidar_shield.contracts.reasons import MatchPolicy
from lidar_shield.contracts.schemas import AgentRegistry
from lidar_shield.data.index import MembershipDeclaration
from lidar_shield.data.sync import SourceMessage, SyncConfig, Synchronizer


def _membership(
    agent_id: str,
    frame: int,
    time_ns: int,
    *,
    session: str = "synthetic-session-a",
    member: bool = True,
) -> MembershipDeclaration:
    return MembershipDeclaration(
        sequence_id="synthetic-sequence-a",
        session_id=session,
        sync_frame_id=frame,
        decision_time_ns=time_ns,
        agent_id=agent_id,
        member=member,
        message_expected=member,
    )


def build_d0_payload(registry: AgentRegistry) -> dict[str, Any]:
    """Replay matched, unavailable, invalid, empty, and stale fixture events."""

    causal = Synchronizer(
        registry,
        SyncConfig(policy=MatchPolicy.CAUSAL_BACKWARD, max_age_ns=100),
    )
    causal_memberships = (
        _membership("003", 0, 1_000),
        _membership("004", 0, 1_000),
        _membership("dome", 0, 1_000),
        _membership("laser", 0, 1_000),
        _membership("top", 0, 1_000),
        _membership("003", 1, 2_000),
        _membership("dome", 1, 2_000, session="synthetic-session-b"),
        _membership("top", 1, 2_000, member=False),
    )
    causal_messages = (
        SourceMessage(
            "synthetic-sequence-a",
            "synthetic-session-a",
            "003",
            950,
            960,
            pose_source_time_ns=940,
        ),
        SourceMessage("synthetic-sequence-a", "synthetic-session-a", "004", 990, 1_100),
        SourceMessage(
            "synthetic-sequence-a",
            "synthetic-session-a",
            "laser",
            980,
            990,
            pose_valid=False,
        ),
        SourceMessage(
            "synthetic-sequence-a",
            "synthetic-session-a",
            "top",
            1_000,
            1_000,
            empty_scene=True,
        ),
        SourceMessage(
            "synthetic-sequence-a", "synthetic-session-b", "dome", 2_100, 2_100
        ),
    )
    rows = list(
        causal.synchronize(
            causal_memberships,
            causal_messages,
            experiment_id="m1-d0",
            variant_id="clean",
            oracle_gt=True,
        )
    )

    buffered = Synchronizer(
        registry,
        SyncConfig(
            policy=MatchPolicy.BUFFERED_NEAREST,
            max_age_ns=100,
            lookahead_buffer_ns=200,
        ),
    )
    rows.extend(
        buffered.synchronize(
            (_membership("top", 2, 3_000, session="synthetic-session-c"),),
            (
                SourceMessage(
                    "synthetic-sequence-a",
                    "synthetic-session-c",
                    "top",
                    3_150,
                    3_150,
                ),
            ),
            experiment_id="m1-d0",
            variant_id="clean",
            oracle_gt=True,
        )
    )
    serialized = [row.model_dump(mode="json") for row in rows]
    serialized.sort(
        key=lambda row: (
            str(row["sequence_id"]),
            str(row["session_id"]),
            int(row["sync_frame_id"]),
            str(row["agent_id"]),
        )
    )
    counts = Counter(str(row["reason_code"]) for row in serialized)
    return {
        "contract_schema_version": "1.0.0",
        "event_count": len(serialized),
        "events": serialized,
        "reason_counts": dict(sorted(counts.items())),
        "signed_age_convention": "decision_time_ns - source_time_ns",
    }
